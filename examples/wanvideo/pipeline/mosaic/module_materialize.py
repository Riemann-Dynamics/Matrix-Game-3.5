from .common import *
from .cleanup import _release_cached_memory
from .config import _is_recent_first_fuse_mode, _normalize_mosaic_fuse_mode
from .history import (
    _PER3_CONTEXT_MODES,
    _build_train_clean_history_rope_time_indices,
    _sample_train_anchor_context_plan,
    _select_train_clean_history_frame_blocks,
    _train_clean_history_actual_latent_time,
)

# Surface the per_3latentframe non-divisible-window warning exactly once.
_PER3_TRAIN_NON_DIV_WARNED = [False]
from .timing import TimingReporter, get_timing_reporter
from .video_io import _read_video_gt_frames
from diffsynth.core.data.unified_dataset import (
    _derive_seed,
    _dynamic_mask_postprocess_module,
)


class _MaterializeMixin:
    def _get_trainer_frustum_handler(self):
        """Lazy-build a single FrustumHandler for fuse_candidates calls.

        The dataset workers each have their own handler (one per worker);
        here we want one in the main process for the GPU-side fuse step.
        """
        cached = getattr(self, "_trainer_frustum_handler", None)
        if cached is not None:
            if hasattr(cached, "set_timing_reporter"):
                cached.set_timing_reporter(get_timing_reporter(self))
            return cached
        from frustum.frustum_handler import FrustumHandler

        # Trainer only ever calls fuse_candidates with explicit per-batch K
        # (via fused_inputs). The placeholder eye(3) is intentional and
        # never read. latent_stride=16 matches the WAN VAE.
        cached = FrustumHandler(np.eye(3), latent_stride=16)
        if hasattr(cached, "set_timing_reporter"):
            cached.set_timing_reporter(get_timing_reporter(self))
        self._trainer_frustum_handler = cached
        return cached

    @staticmethod
    def _atomic_dump_latent(target_path, tensor):
        """Write ``tensor`` to ``target_path`` via tempfile + os.replace.

        Multi-rank safe: each rank writes its own tempfile in the same
        target directory, then atomically renames into place. Concurrent
        renames just race to the last writer; the resulting file is
        always a complete, valid torch payload.
        """
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        import tempfile as _tempfile

        fd, tmp_path = _tempfile.mkstemp(
            prefix=".tmp_",
            suffix=f".{os.getpid()}.pt",
            dir=target_dir,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(tensor.detach().to("cpu"), f)
            os.replace(tmp_path, target_path)
        except Exception:
            # Best-effort cleanup; never let cache IO bring down training.
            try:
                os.replace(tmp_path, tmp_path + ".dump")
            except OSError:
                pass
            raise

    def _write_cache_load_failure_log(self, *, frame_id, cache_path, error, data):
        if self.log_dir is None:
            return
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
        info = data.get("info") or {}
        payload = {
            "time_s": time.time(),
            "rank": int(rank),
            "step": int(self.step_count),
            "frame_id": int(frame_id),
            "cache_path": str(cache_path),
            "error": str(error),
            "clean_name": info.get("clean_name"),
            "video_path": info.get("video_path"),
        }
        log_dir = os.path.join(self.log_dir, "cache_failures")
        os.makedirs(log_dir, exist_ok=True)
        ts = time.time_ns()
        log_path = os.path.join(
            log_dir, f"memory_cache_load_failures_rank{rank:03d}_{ts}.json"
        )
        print(f"dumping log to {log_path}")
        TimingReporter._append_json_record(log_path, payload)

    @staticmethod
    def _frames_to_vae_input(frames, *, device, dtype, pixel_range=None):
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)
        if frames.dtype == torch.uint8 or str(pixel_range or "").lower() in {
            "uint8",
            "uint8_or_255",
            "0_255",
        }:
            # Match the training repository exactly: RGB decoding normalized
            # in float32 before the VAE input was cast to BF16. Normalizing
            # after the cast changes many pixels by one BF16 quantization bin.
            return (
                frames.to(device=device, dtype=torch.float32)
                .mul_(1.0 / 127.5)
                .sub_(1.0)
                .to(dtype=dtype)
            )
        return frames.to(device=device, dtype=dtype)

    @staticmethod
    def _build_memory_cache_diagnostic(
        *,
        data,
        memory_frames_by_id,
        cached_memory_paths_by_id,
        memory_cache_hash_by_id,
        memory_total_frames,
        phase,
        loaded_cache_frame_ids=None,
        cache_load_failures=None,
        encoded_miss_frame_ids=None,
        dumped_frame_ids=None,
        dump_failures=None,
    ):
        debug = dict(data.get("memory_cache_debug") or {})
        info = data.get("info") or {}

        def _int_list(values):
            return sorted(int(v) for v in (values or []))

        def _string_key_dict(values):
            return {str(int(k)): str(v) for k, v in (values or {}).items()}

        target_paths = _string_key_dict(
            debug.get("target_paths") or memory_cache_hash_by_id
        )
        path_states = {}
        for fid_str, path in target_paths.items():
            exists = os.path.exists(path)
            state = {"path": path, "exists": bool(exists)}
            if exists:
                try:
                    state["size_bytes"] = int(os.path.getsize(path))
                except OSError as exc:
                    state["stat_error"] = str(exc)
            path_states[fid_str] = state

        hit_ids = _int_list(debug.get("hit_frame_ids") or cached_memory_paths_by_id)
        miss_ids = _int_list(debug.get("miss_frame_ids") or memory_frames_by_id)
        selected_ids = _int_list(
            debug.get("selected_frame_ids") or (hit_ids + miss_ids)
        )
        payload = {
            "type": "memory_latent_cache",
            "phase": str(phase),
            "clean_name": info.get("clean_name"),
            "cache_enabled": bool(debug.get("cache_enabled", bool(target_paths))),
            "memory_total_frames": int(memory_total_frames),
            "selected_frame_ids": selected_ids,
            "hit_frame_ids": hit_ids,
            "miss_frame_ids": miss_ids,
            "hit_count": len(hit_ids),
            "miss_count": len(miss_ids),
            "loaded_cache_frame_ids": _int_list(loaded_cache_frame_ids),
            "cache_load_failures": list(cache_load_failures or []),
            "encoded_miss_frame_ids": _int_list(encoded_miss_frame_ids),
            "dumped_frame_ids": _int_list(dumped_frame_ids),
            "dump_failures": list(dump_failures or []),
            "target_paths": target_paths,
            "cached_paths": _string_key_dict(cached_memory_paths_by_id),
            "path_states": path_states,
            "cn_vae_encode_cacheable": False,
            "cn_vae_encode_note": (
                "materialize.cn_vae_encode encodes the current clean+noisy "
                "training window and is not served by memory_latent_cache_dir; "
                "cache hits only skip materialize.memory_miss_vae_encode."
            ),
        }
        read_frame_ids = debug.get("read_frame_ids")
        if read_frame_ids is not None:
            payload["read_frame_ids"] = _int_list(read_frame_ids)
        return payload

    def _sample_train_clean_history_plan_for_step(self):
        return _sample_train_anchor_context_plan(
            anchor_block_prob=float(getattr(self, "train_anchor_block_prob", 0.8)),
            block_anchor_frames=int(getattr(self, "train_block_anchor_frames", 13)),
            context_count_max=int(getattr(self, "train_context_count_max", 0)),
            base_seed=self._schedule_base_seed,
            step_index=self.step_count,
        )

    def _get_clean_history_plan_for_materialize(self):
        if bool(getattr(self, "training", True)):
            return self._sample_train_clean_history_plan_for_step()
        plan = getattr(self, "eval_clean_history_plan", None)
        if not plan:
            return None
        out = dict(plan)
        out.setdefault("enabled", True)
        out.setdefault("plan_sampler", "eval_fixed_context")
        out["anchor_form"] = "single"
        out["anchor_depth"] = 0
        out["context_count"] = max(0, int(out.get("context_count") or 0))
        out.setdefault(
            "context_selection",
            str(getattr(self, "train_context_selection", "oldest_spread")),
        )
        out["bucket"] = "single" + ("_ctx" if out["context_count"] else "")
        return out

    def _read_train_clean_history_frames(self, video_path, frame_ids, height, width):
        return _read_video_gt_frames(video_path, frame_ids, height, width)

    def _maybe_prepend_anchor_warmup_frames(self, data, cn_frames, cn_drop_count, plan):
        """Resolve the dataset-shipped warm-up prefix for this step's plan.

        The DATASET prepends ``4*depth`` GT frames ``[G-1-4d .. G-2]`` into
        the SAME single decode that produced the cn window
        (``cn_anchor_warmup_frames`` records the count) -- no second video
        read happens in the training process. Block-anchor steps KEEP the
        prefix: the joint encode covers ``[G-1-4d .. G+79]``, slot 0 plus
        ``d-1`` warm-up latents are dropped post-encode, and slot ``d`` is
        the block anchor at causal chain depth ``d`` (default 13 frames ->
        d=3, where the encode-transient channel statistics saturate against
        a deep-chain reference). Single-anchor steps and eval slice the
        prefix OFF so their encode window stays ``[G-1 .. G+79]``.

        Either way ``data["cn_frames"]`` is rewritten to the unprefixed
        view, so every downstream consumer (qwen prompt frame sampling,
        previews) sees the exact ``[G-1..G+79]`` tensor it always did.
        """
        if cn_drop_count != 0:
            raise RuntimeError(
                "train clean-history expects the dataset to ship cn windows "
                "without legacy context blocks (cn_drop_count == 0), got "
                f"{cn_drop_count}."
            )
        warmup = int(data.get("cn_anchor_warmup_frames", 0) or 0)
        if warmup:
            data["cn_frames"] = cn_frames[:, warmup:]
        use_block_anchor = (
            plan is not None
            and plan.get("enabled")
            and plan.get("anchor_form") == "block"
            and bool(getattr(self, "training", True))
        )
        if not use_block_anchor:
            return (cn_frames[:, warmup:] if warmup else cn_frames), 0, False
        depth = max(1, int(plan.get("anchor_depth") or 1))
        if warmup != 4 * depth:
            raise ValueError(
                f"block anchor at depth {depth} needs a {4 * depth}-frame "
                "dataset warm-up prefix in the cn window, got "
                f"cn_anchor_warmup_frames={warmup}. Dataset and trainer "
                "disagree -- check datasets.py "
                "train_anchor_warmup_prepend_frames vs "
                "history._required_context_blocks_min."
            )
        return cn_frames, depth, True

    def _encode_train_clean_history_windows(
        self,
        *,
        data,
        frame_windows,
        device,
        dtype,
    ):
        if not frame_windows:
            raise ValueError("frame_windows must not be empty.")
        info = data.get("info") or {}
        video_path = info.get("video_path")
        if not video_path:
            raise ValueError("train clean-history requires data['info']['video_path'].")
        height = int(data["cn_frames"].shape[-2])
        width = int(data["cn_frames"].shape[-1])
        flat_frame_ids = sorted(
            {int(fid) for window in frame_windows for fid in window}
        )
        frames_np = self._read_train_clean_history_frames(
            video_path,
            flat_frame_ids,
            height,
            width,
        )
        id_to_pos = {fid: pos for pos, fid in enumerate(flat_frame_ids)}
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
        frames_t = frames_t.to(device=device, dtype=dtype) / 127.5 - 1.0
        videos = []
        for window in frame_windows:
            positions = [id_to_pos[int(fid)] for fid in window]
            videos.append(
                frames_t.index_select(0, torch.as_tensor(positions, device=device))
                .permute(1, 0, 2, 3)
                .contiguous()
            )
        encoded = self.pipe.vae.encode(videos, device=device, tiled=False)
        if encoded.ndim != 5 or int(encoded.shape[2]) < 1:
            raise RuntimeError(
                "train clean-history VAE encode returned unexpected shape "
                f"{tuple(encoded.shape)}."
            )
        encoded = encoded[:, :, -1:].contiguous()
        return encoded.permute(1, 0, 2, 3, 4).reshape(
            1,
            int(encoded.shape[1]),
            int(encoded.shape[0]),
            int(encoded.shape[3]),
            int(encoded.shape[4]),
        )

    @staticmethod
    def _subject_mask_source_hw_from_record(row):
        mask_rle = row.get("mask_rle") if isinstance(row, dict) else None
        if isinstance(mask_rle, dict):
            size = mask_rle.get("size")
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                try:
                    return int(size[0]), int(size[1])
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _resize_subject_bool_mask_for_training(mask, *, height, width):
        """Center-crop and resize a bool mask like DA3 video preprocessing."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape={mask.shape}.")
        src_h, src_w = int(mask.shape[0]), int(mask.shape[1])
        height = int(height)
        width = int(width)
        crop_h = min(src_h, height)
        crop_w = min(src_w, width)
        top = max((src_h - crop_h) // 2, 0)
        left = max((src_w - crop_w) // 2, 0)
        mask = mask[top : top + crop_h, left : left + crop_w]
        if mask.shape == (height, width):
            return mask.astype(bool, copy=False)
        import cv2

        if mask.shape[0] >= height and mask.shape[1] >= width:
            resized = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            return resized > 0.0
        resized = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized > 0

    def _dynamic_mask_path_for_subject_context(self, data):
        info = data.get("info") or {}
        mask_path = info.get("dynamic_mask_path")
        if mask_path:
            return str(mask_path)
        scene_dir = info.get("dirname") or info.get("scene_dir")
        if not scene_dir:
            video_path = info.get("video_path")
            if video_path:
                scene_dir = os.path.dirname(os.path.dirname(str(video_path)))
        if not scene_dir:
            return ""
        subdir = str(getattr(self, "dynamic_object_mask_subdir", "objects") or "objects")
        filename = str(
            getattr(self, "dynamic_object_mask_filename", "dynamic_masks.jsonl")
            or "dynamic_masks.jsonl"
        )
        return os.path.join(str(scene_dir), subdir, filename)

    def _load_subject_context_dynamic_mask_records(self, mask_path):
        if not mask_path or not os.path.exists(mask_path):
            return {}
        cache = getattr(self, "_subject_context_dynamic_mask_records_cache", None)
        if cache is None:
            cache = {}
            self._subject_context_dynamic_mask_records_cache = cache
        cached = cache.get(mask_path)
        if cached is not None:
            return cached
        records_by_frame = {}
        with open(mask_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    frame_idx = int(row.get("frame_idx"))
                except (TypeError, ValueError):
                    continue
                records_by_frame.setdefault(frame_idx, []).append(row)
        cache[mask_path] = records_by_frame
        if len(cache) > 16:
            for key in list(cache.keys())[:-16]:
                cache.pop(key, None)
        return records_by_frame

    @staticmethod
    def _latent_mask_to_dit_patch_mask_np(latent_mask):
        latent_mask = np.asarray(latent_mask, dtype=bool)
        h_lat, w_lat = latent_mask.shape[-2:]
        h_patch = int(h_lat) // 2
        w_patch = int(w_lat) // 2
        cropped = latent_mask[: h_patch * 2, : w_patch * 2]
        if cropped.size == 0:
            return np.zeros((h_patch, w_patch), dtype=bool)
        return cropped.reshape(h_patch, 2, w_patch, 2).any(axis=(1, 3))

    def _build_subject_context_patch_masks_for_frame_windows(
        self,
        data,
        *,
        frame_windows,
        h_lat,
        w_lat,
    ):
        """Build dynamic-object patch masks for arbitrary source-frame windows."""
        if not frame_windows:
            return None, {"status": "no_windows"}
        mask_path = self._dynamic_mask_path_for_subject_context(data)
        if not mask_path:
            return None, {"status": "no_mask_path"}
        records_by_frame = self._load_subject_context_dynamic_mask_records(mask_path)
        if not records_by_frame:
            return None, {"status": "no_mask_records", "mask_path": mask_path}

        cn_frames = data.get("cn_frames")
        if torch.is_tensor(cn_frames) and cn_frames.ndim >= 4:
            height = int(cn_frames.shape[-2])
            width = int(cn_frames.shape[-1])
        else:
            height = int(getattr(self, "height"))
            width = int(getattr(self, "width"))
        postprocess = _dynamic_mask_postprocess_module()
        patch_masks = []
        dynamic_counts = []
        for window in frame_windows:
            image_mask = np.zeros((height, width), dtype=bool)
            for frame_idx in [int(v) for v in window]:
                for row in records_by_frame.get(frame_idx, ()):
                    source_hw = self._subject_mask_source_hw_from_record(row) or (
                        height,
                        width,
                    )
                    one = postprocess.mask_from_record(
                        row,
                        height=int(source_hw[0]),
                        width=int(source_hw[1]),
                        use_bbox_fallback=False,
                    )
                    if one is None:
                        continue
                    one = self._resize_subject_bool_mask_for_training(
                        one,
                        height=height,
                        width=width,
                    )
                    image_mask |= one
            latent_mask = postprocess.image_mask_to_latent_mask(
                image_mask,
                int(h_lat),
                int(w_lat),
            )
            patch_mask = self._latent_mask_to_dit_patch_mask_np(latent_mask)
            patch_masks.append(patch_mask.astype(bool, copy=False))
            dynamic_counts.append(int(patch_mask.sum()))
        patch_masks = np.stack(patch_masks, axis=0).astype(bool, copy=False)
        if not patch_masks.any():
            return None, {
                "status": "empty_masks",
                "mask_path": mask_path,
                "frame_windows": [[int(v) for v in w] for w in frame_windows],
                "dynamic_patch_counts": dynamic_counts,
            }
        return torch.from_numpy(patch_masks).bool(), {
            "status": "ok",
            "mask_path": mask_path,
            "frame_windows": [[int(v) for v in w] for w in frame_windows],
            "dynamic_patch_counts": dynamic_counts,
        }

    def _build_subject_context_window_patch_masks(
        self,
        data,
        *,
        frame_windows,
        h_lat,
        w_lat,
    ):
        """Build full dynamic-object masks for separately encoded context latents.

        Each clean-history pool latent comes from an isolated 5-frame VAE encode
        and only the last latent is kept. Unioning all source frames in that
        window is conservative and prevents appearance leakage through any
        dynamic object that contributed to the latent.
        """
        return self._build_subject_context_patch_masks_for_frame_windows(
            data,
            frame_windows=frame_windows,
            h_lat=h_lat,
            w_lat=w_lat,
        )

    def _apply_subject_context_patch_masks_to_latents(
        self,
        data,
        latents,
        patch_masks,
        *,
        info_key,
        full_dynamic_erase=False,
        fill_mode=None,
        donor_exclude_patch_masks=None,
    ):
        if patch_masks is None:
            return latents, {"materialized": False, "status": "no_patch_masks"}
        if not torch.is_tensor(patch_masks):
            patch_masks = torch.as_tensor(patch_masks)
        if patch_masks.ndim != 3 or not bool(patch_masks.any()):
            return latents, {"materialized": False, "status": "empty_patch_masks"}
        if donor_exclude_patch_masks is not None:
            if not torch.is_tensor(donor_exclude_patch_masks):
                donor_exclude_patch_masks = torch.as_tensor(donor_exclude_patch_masks)
            if donor_exclude_patch_masks.ndim != 3:
                donor_exclude_patch_masks = None

        if fill_mode is None:
            fill_mode = (data.get("subject_context_patch_mask_info") or {}).get(
                "fill", "background_patch"
            )
        fill_mode = str(fill_mode or "background_patch").strip().lower()
        patch_masks = patch_masks.to(device=latents.device, dtype=torch.bool)
        if donor_exclude_patch_masks is not None:
            donor_exclude_patch_masks = donor_exclude_patch_masks.to(
                device=latents.device,
                dtype=torch.bool,
            )
        h_lat = int(latents.shape[-2])
        w_lat = int(latents.shape[-1])
        context_count = min(int(patch_masks.shape[0]), int(latents.shape[1]))
        if context_count <= 0:
            return latents, {"materialized": False, "status": "zero_context"}

        out = latents.clone()
        debug_frames = []
        total_patches = 0
        background_pool_min = None
        for t in range(context_count):
            patch_mask = patch_masks[t]
            if not bool(patch_mask.any()):
                continue
            latent_mask = self._dit_patch_mask_to_latent_mask(
                patch_mask,
                h_lat=h_lat,
                w_lat=w_lat,
            )
            frame_lat = out[:, t]
            donor_frame_lat = frame_lat.clone()
            donor_exclude_patch_mask = patch_mask
            if donor_exclude_patch_masks is not None and t < int(
                donor_exclude_patch_masks.shape[0]
            ):
                donor_exclude_patch_mask = donor_exclude_patch_masks[t]
                if donor_exclude_patch_mask.shape != patch_mask.shape:
                    raise ValueError(
                        "donor_exclude_patch_masks must match patch_masks shape, "
                        f"got {tuple(donor_exclude_patch_mask.shape)} vs "
                        f"{tuple(patch_mask.shape)}."
                    )
            bg_patch_mask = ~donor_exclude_patch_mask
            bg_coords = torch.nonzero(bg_patch_mask, as_tuple=False)
            masked_coords = torch.nonzero(patch_mask, as_tuple=False)
            if int(masked_coords.shape[0]) <= 0:
                continue
            if background_pool_min is None:
                background_pool_min = int(bg_coords.shape[0])
            else:
                background_pool_min = min(background_pool_min, int(bg_coords.shape[0]))

            if fill_mode == "background_mean" or int(bg_coords.shape[0]) <= 0:
                donor_exclude_latent_mask = self._dit_patch_mask_to_latent_mask(
                    donor_exclude_patch_mask,
                    h_lat=h_lat,
                    w_lat=w_lat,
                )
                bg_latent_mask = ~donor_exclude_latent_mask
                if bool(bg_latent_mask.any()):
                    mean = donor_frame_lat[:, bg_latent_mask].mean(dim=1).view(
                        -1, 1, 1
                    )
                else:
                    mean = donor_frame_lat.mean(dim=(1, 2), keepdim=True)
                frame_lat[:, latent_mask] = mean.view(-1, 1).expand(
                    -1, int(latent_mask.sum())
                )
                donor_mode = "background_mean"
            else:
                donor_mode = "background_patch"
                if int(bg_coords.shape[0]) < int(masked_coords.shape[0]):
                    choice = torch.randint(
                        0,
                        int(bg_coords.shape[0]),
                        (int(masked_coords.shape[0]),),
                        device=latents.device,
                    )
                else:
                    perm = torch.randperm(int(bg_coords.shape[0]), device=latents.device)
                    choice = perm[: int(masked_coords.shape[0])]
                donor_coords = bg_coords.index_select(0, choice)
                for dst, src in zip(masked_coords, donor_coords):
                    dy = int(dst[0].item()) * 2
                    dx = int(dst[1].item()) * 2
                    sy = int(src[0].item()) * 2
                    sx = int(src[1].item()) * 2
                    frame_lat[:, dy : dy + 2, dx : dx + 2] = donor_frame_lat[
                        :, sy : sy + 2, sx : sx + 2
                    ]
            total_patches += int(masked_coords.shape[0])
            debug_frames.append(
                {
                    "latent_index": int(t),
                    "masked_patch_count": int(masked_coords.shape[0]),
                    "background_patch_count": int(bg_coords.shape[0]),
                    "donor_excluded_patch_count": int(patch_mask.sum().item()),
                    "fill_mode": donor_mode,
                    "full_dynamic_erase": bool(full_dynamic_erase),
                }
            )

        return out, {
            "materialized": True,
            "info_key": str(info_key),
            "masked_patch_count_materialized": int(total_patches),
            "background_patch_pool_min": int(background_pool_min or 0),
            "materialized_frames": debug_frames,
        }

    def _sample_subject_context_patch_subset(self, patch_mask, data):
        coords = torch.nonzero(patch_mask.to(dtype=torch.bool), as_tuple=False)
        if int(coords.shape[0]) <= 0:
            return torch.zeros_like(patch_mask, dtype=torch.bool), 0.0
        ratio_min = float(getattr(self, "subject_context_patch_mask_ratio_min", 0.25))
        ratio_max = float(getattr(self, "subject_context_patch_mask_ratio_max", 0.75))
        ratio_min = min(1.0, max(0.0, ratio_min))
        ratio_max = min(1.0, max(0.0, ratio_max))
        if ratio_max < ratio_min:
            ratio_min, ratio_max = ratio_max, ratio_min
        seed = _derive_seed(
            "subject_context_anchor_mask",
            getattr(self, "_schedule_base_seed", 0),
            getattr(self, "step_count", 0),
            (data.get("info") or {}).get("clean_name", ""),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        ratio = float(
            torch.empty((), dtype=torch.float32)
            .uniform_(ratio_min, ratio_max, generator=generator)
            .item()
        )
        count = max(1, int(round(int(coords.shape[0]) * ratio)))
        count = min(count, int(coords.shape[0]))
        perm = torch.randperm(int(coords.shape[0]), generator=generator)[:count]
        chosen = coords.index_select(0, perm)
        out = torch.zeros_like(patch_mask, dtype=torch.bool)
        out[chosen[:, 0], chosen[:, 1]] = True
        return out, float(count / max(1, int(coords.shape[0])))

    def _sample_subject_pool_context_patch_subset(self, patch_masks, data):
        if patch_masks is None:
            return None, {"status": "no_patch_masks", "sampled_ratio_mean": 0.0}
        if not torch.is_tensor(patch_masks):
            patch_masks = torch.as_tensor(patch_masks)
        patch_masks = patch_masks.to(dtype=torch.bool)
        if patch_masks.ndim != 3 or not bool(patch_masks.any()):
            return patch_masks, {"status": "empty_patch_masks", "sampled_ratio_mean": 0.0}

        ratio_min = float(
            getattr(self, "subject_pool_context_dynamic_erase_ratio_min", 1.0)
        )
        ratio_max = float(
            getattr(self, "subject_pool_context_dynamic_erase_ratio_max", 1.0)
        )
        ratio_min = min(1.0, max(0.0, ratio_min))
        ratio_max = min(1.0, max(0.0, ratio_max))
        if ratio_max < ratio_min:
            ratio_min, ratio_max = ratio_max, ratio_min

        seed = _derive_seed(
            "subject_pool_context_dynamic_erase_mask",
            getattr(self, "_schedule_base_seed", 0),
            getattr(self, "step_count", 0),
            (data.get("info") or {}).get("clean_name", ""),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        sampled_masks = torch.zeros_like(patch_masks, dtype=torch.bool)
        ratios = []
        dynamic_counts = []
        sampled_counts = []
        for t in range(int(patch_masks.shape[0])):
            coords = torch.nonzero(patch_masks[t], as_tuple=False)
            dynamic_count = int(coords.shape[0])
            dynamic_counts.append(dynamic_count)
            if dynamic_count <= 0:
                sampled_counts.append(0)
                ratios.append(0.0)
                continue
            ratio = float(
                torch.empty((), dtype=torch.float32)
                .uniform_(ratio_min, ratio_max, generator=generator)
                .item()
            )
            count = int(round(dynamic_count * ratio))
            count = min(count, dynamic_count)
            if count <= 0:
                sampled_counts.append(0)
                ratios.append(0.0)
                continue
            perm = torch.randperm(dynamic_count, generator=generator)[:count]
            chosen = coords.index_select(0, perm)
            sampled_masks[t, chosen[:, 0], chosen[:, 1]] = True
            sampled_counts.append(int(count))
            ratios.append(float(count / max(1, dynamic_count)))

        return sampled_masks, {
            "status": "ok",
            "ratio_min": float(ratio_min),
            "ratio_max": float(ratio_max),
            "sampled_ratio_mean": float(sum(ratios) / max(1, len(ratios))),
            "dynamic_patch_counts": dynamic_counts,
            "sampled_patch_counts": sampled_counts,
        }

    def _anchor_source_frame_ids_for_subject_mask(self, data, plan):
        info = data.get("info") or {}
        g = int(info.get("generating_first_frame_idx", 0) or 0)
        if g <= 0:
            return []
        use_block_anchor = (
            plan is not None
            and plan.get("enabled")
            and plan.get("anchor_form") == "block"
            and bool(getattr(self, "training", True))
        )
        if use_block_anchor:
            return [int(v) for v in range(max(0, g - 4), g)]
        return [int(g - 1)]

    def _refresh_subject_anchor_context_patch_mask_for_plan(
        self,
        data,
        *,
        cn_latents,
        plan,
    ):
        ref_info = data.get("subject_ref_info") or {}
        base_info = dict(data.get("subject_context_patch_mask_info") or {})
        def _clear_anchor_masks():
            data.pop("subject_context_patch_mask", None)
            data.pop("subject_context_dynamic_patch_mask", None)
            data.pop("subject_context_dynamic_latent_mask", None)

        if not bool(getattr(self, "subject_ref_memory", False)):
            _clear_anchor_masks()
            return data
        if not bool(ref_info.get("enabled", False)):
            base_info.update(
                {
                    "enabled": False,
                    "status": "no_refs",
                    "trainer_anchor_refresh": False,
                }
            )
            data["subject_context_patch_mask_info"] = base_info
            _clear_anchor_masks()
            return data
        mask_prob = float(getattr(self, "subject_context_patch_mask_prob", 0.0))
        if mask_prob <= 0.0:
            base_info.update(
                {
                    "enabled": False,
                    "status": "prob_zero",
                    "trainer_anchor_refresh": False,
                }
            )
            data["subject_context_patch_mask_info"] = base_info
            _clear_anchor_masks()
            return data
        if mask_prob < 1.0:
            seed = _derive_seed(
                "subject_context_anchor_mask_enable",
                getattr(self, "_schedule_base_seed", 0),
                getattr(self, "step_count", 0),
                (data.get("info") or {}).get("clean_name", ""),
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            enabled = bool(torch.rand((), generator=generator).item() < mask_prob)
            if not enabled:
                base_info.update(
                    {
                        "enabled": False,
                        "status": "sampled_off",
                        "prob": float(mask_prob),
                        "trainer_anchor_refresh": True,
                    }
                )
                data["subject_context_patch_mask_info"] = base_info
                _clear_anchor_masks()
                return data

        frame_ids = self._anchor_source_frame_ids_for_subject_mask(data, plan)
        if not frame_ids:
            base_info.update(
                {
                    "enabled": False,
                    "status": "no_anchor_source_frames",
                    "trainer_anchor_refresh": False,
                }
            )
            data["subject_context_patch_mask_info"] = base_info
            _clear_anchor_masks()
            return data

        patch_masks, mask_debug = self._build_subject_context_patch_masks_for_frame_windows(
            data,
            frame_windows=[frame_ids],
            h_lat=int(cn_latents.shape[-2]),
            w_lat=int(cn_latents.shape[-1]),
        )
        if patch_masks is None or not bool(patch_masks.any()):
            base_info.update(
                {
                    **mask_debug,
                    "enabled": False,
                    "trainer_anchor_refresh": True,
                    "anchor_source_frame_ids": [int(v) for v in frame_ids],
                    "anchor_form": str((plan or {}).get("anchor_form") or "single"),
                }
            )
            data["subject_context_patch_mask_info"] = base_info
            _clear_anchor_masks()
            return data

        dynamic_patch_mask = patch_masks[0].to(dtype=torch.bool)
        sampled, sampled_ratio = self._sample_subject_context_patch_subset(
            dynamic_patch_mask,
            data,
        )
        data["subject_context_patch_mask"] = sampled[None, ...].cpu().bool()
        data["subject_context_dynamic_patch_mask"] = (
            dynamic_patch_mask[None, ...].cpu().bool()
        )
        data.pop("subject_context_dynamic_latent_mask", None)
        base_info.update(
            {
                **mask_debug,
                "enabled": True,
                "status": "ok",
                "fill": str(
                    getattr(
                        self,
                        "subject_context_patch_mask_fill",
                        base_info.get("fill", "background_patch"),
                    )
                ),
                "trainer_anchor_refresh": True,
                "anchor_source_frame_ids": [int(v) for v in frame_ids],
                "anchor_form": str((plan or {}).get("anchor_form") or "single"),
                "prob": float(mask_prob),
                "masked_patch_count": int(sampled.sum().item()),
                "dynamic_patch_count": int(dynamic_patch_mask.sum().item()),
                "sampled_ratio_mean": float(sampled_ratio),
            }
        )
        data["subject_context_patch_mask_info"] = base_info
        return data

    def _apply_train_clean_latent_history(
        self,
        data,
        *,
        clean_latents,
        noisy_latents,
        device,
        dtype,
        plan=None,
        anchor_is_joint=False,
    ):
        if plan is None:
            if not bool(getattr(self, "training", True)):
                data["train_clean_latent_history_debug"] = {
                    "enabled": False,
                    "applied": False,
                    "plan_sampler": "eval_disabled",
                }
                return clean_latents
            plan = self._sample_train_clean_history_plan_for_step()
        # ``cn_drop_count`` from the DATASET is always 0 now (the dataset-side
        # block draw is retired for training); the trainer-side warm-up
        # prepend reports through ``anchor_is_joint`` instead. Kept as a
        # witness for diagnostics.
        dataset_blocks = int(data.get("cn_drop_count", 0) or 0)
        debug = dict(plan)
        debug["dataset_blocks"] = dataset_blocks
        debug["applied"] = False
        data["train_clean_latent_history_debug"] = debug
        if not plan.get("enabled") or not data.get("needs_vae_materialization"):
            return clean_latents

        anchor_form = str(plan.get("anchor_form") or "single")
        context_count = max(0, int(plan.get("context_count") or 0))
        # per_3latentframe forces the context count to n_gen // 3 (one covering
        # context latent per consecutive group of 3 generating latents),
        # superseding the planned/randomised count. Derive it HERE -- the single
        # point the count is born for this step -- so BOTH the early-exit gate
        # below AND the metrics snapshot observe the derived value. ``debug`` is
        # a copy of ``plan`` (taken above) and is what metrics_logger reads for
        # train/history_count and the loss_mode_<bucket> buckets, so we update it
        # in place; updating only the selection arg would desync those readers.
        if (
            str(getattr(self, "train_context_selection", "")).strip().lower()
            in _PER3_CONTEXT_MODES
            and noisy_latents is not None
        ):
            n_gen = int(noisy_latents.shape[2])
            if n_gen % 3 != 0 and not _PER3_TRAIN_NON_DIV_WARNED[0]:
                print(
                    "[mosaic] WARNING: train_context_selection='per_3latentframe'"
                    f" but the generating latent count {n_gen} is not divisible "
                    "by 3; the remainder is dropped (floor). (warned once)",
                    flush=True,
                )
                _PER3_TRAIN_NON_DIV_WARNED[0] = True
            context_count = n_gen // 3
            debug["context_count"] = int(context_count)
            debug["bucket"] = anchor_form + ("_ctx" if context_count else "")
            debug["per3_context"] = True
        if anchor_form == "block" and not anchor_is_joint:
            raise RuntimeError(
                "train block anchor must come from the joint clean+noisy "
                "encode (anchor re-encode is banned); "
                "_maybe_prepend_anchor_warmup_frames was not applied at "
                f"step {self.step_count}."
            )
        if anchor_form == "single" and context_count == 0:
            # Dataset slot-0 single-frame anchor, joint by construction.
            debug["applied"] = True
            debug["anchor_encode"] = "joint_with_targets"
            return clean_latents

        fused_inputs = data.get("fused_query_inputs") or {}
        info = data.get("info") or {}
        G = int(info.get("generating_first_frame_idx", 0) or 0)
        if G <= 0:
            raise ValueError(
                "train clean-history requires info['generating_first_frame_idx']."
            )
        # Per-frame world-to-camera extrinsics for the candidate POOL plus the
        # GENERATING-WINDOW target poses, so segment_coverage / per_3 selection
        # can rank context blocks by camera coverage. oldest_spread / recent
        # ignore these (passed but unused).
        train_history_extr = None
        train_target_extr = None
        _w2c = fused_inputs.get("w2c")
        if _w2c is not None:
            train_history_extr = np.asarray(_w2c)
            # Target = the REAL generating-window per-frame poses. The dataset
            # ships these as ``query_extrinsics`` (== extrinsics[G : G+4*lws]) in
            # the SAME GT world frame as ``w2c`` (== extrinsics[:G]); the shared
            # selector groups them itself (per_3: n_gen//3 contiguous chunks;
            # segment_coverage: linspace-subsampled). Do NOT index ``w2c`` at
            # G+4k+1 -- ``w2c`` has only G rows, so every such index clamps to
            # G-1, collapsing the target to a SINGLE repeated pose and defeating
            # the coverage objective (the per-group costs become identical, so
            # picks degrade to the recency tiebreak). When query poses are
            # absent, leave the target as None so the selector degrades to
            # oldest_spread rather than ranking against a bogus single pose.
            _query_extr = fused_inputs.get("query_extrinsics")
            if _query_extr is not None:
                train_target_extr = np.asarray(_query_extr)
        selection = _select_train_clean_history_frame_blocks(
            candidate_frame_ids=fused_inputs.get("candidate_frame_ids") or [],
            generating_first_frame_idx=G,
            history_count=context_count + 1,
            selection_mode=str(
                plan.get(
                    "context_selection",
                    getattr(self, "train_context_selection", "oldest_spread"),
                )
            ),
            clean_history_extr=train_history_extr,
            target_extrinsics=train_target_extr,
        )
        # The anchor is ``clean_latents`` itself, ALWAYS from the joint
        # encode (block: [G-4..G-1] at chain depth d via the warm-up
        # prepend; single: dataset slot 0). Only the discrete CONTEXT
        # blocks use isolated window encodes -- far-past frames with no
        # contiguous pixel bridge to the generating window.
        pool_frame_windows = selection["encode_frame_windows"][:-1]
        if pool_frame_windows:
            pool_latents = self._encode_train_clean_history_windows(
                data=data,
                frame_windows=pool_frame_windows,
                device=device,
                dtype=dtype,
            )
            ref_info = data.get("subject_ref_info") or {}
            pool_dynamic_erase_enabled = bool(
                getattr(self, "subject_pool_context_dynamic_erase", True)
            )
            pool_dynamic_erase_prob = min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            self,
                            "subject_pool_context_dynamic_erase_prob",
                            1.0,
                        )
                    ),
                ),
            )
            if bool(getattr(self, "subject_ref_memory", False)) and bool(
                ref_info.get("enabled", False)
            ):
                data["subject_pool_context_frame_windows"] = [
                    [int(v) for v in window] for window in pool_frame_windows
                ]
                context_mask_info = dict(
                    data.get("subject_context_patch_mask_info") or {}
                )
                if pool_dynamic_erase_enabled and pool_dynamic_erase_prob > 0.0:
                    erase_applied = True
                    if pool_dynamic_erase_prob < 1.0:
                        seed = _derive_seed(
                            "subject_pool_context_dynamic_erase_enable",
                            getattr(self, "_schedule_base_seed", 0),
                            getattr(self, "step_count", 0),
                            (data.get("info") or {}).get("clean_name", ""),
                        )
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(int(seed))
                        erase_applied = bool(
                            torch.rand((), generator=generator).item()
                            < pool_dynamic_erase_prob
                        )
                    if not erase_applied:
                        data.pop("subject_pool_context_dynamic_patch_mask", None)
                        context_mask_info["pool_context_dynamic_erase"] = {
                            "enabled": False,
                            "applied": False,
                            "status": "sampled_off",
                            "prob": float(pool_dynamic_erase_prob),
                            "frame_windows": [
                                [int(v) for v in window] for window in pool_frame_windows
                            ],
                        }
                    else:
                        patch_masks, mask_debug = (
                            self._build_subject_context_window_patch_masks(
                                data,
                                frame_windows=pool_frame_windows,
                                h_lat=int(pool_latents.shape[-2]),
                                w_lat=int(pool_latents.shape[-1]),
                            )
                        )
                        sampled_patch_masks, sample_debug = (
                            self._sample_subject_pool_context_patch_subset(
                                patch_masks,
                                data,
                            )
                        )
                        if patch_masks is not None:
                            data["subject_pool_context_dynamic_patch_mask"] = (
                                sampled_patch_masks.detach().cpu().bool()
                            )
                        else:
                            data.pop("subject_pool_context_dynamic_patch_mask", None)
                        pool_latents_single, erase_debug = (
                            self._apply_subject_context_patch_masks_to_latents(
                                data,
                                pool_latents[0],
                                sampled_patch_masks,
                                info_key="pool_context_dynamic_erase",
                                full_dynamic_erase=True,
                                fill_mode=getattr(
                                    self,
                                    "subject_pool_context_dynamic_erase_fill",
                                    "background_patch",
                                ),
                                donor_exclude_patch_masks=patch_masks,
                            )
                        )
                        pool_latents = pool_latents_single.unsqueeze(0).contiguous()
                        context_mask_info["pool_context_dynamic_erase"] = {
                            "enabled": True,
                            "applied": True,
                            "prob": float(pool_dynamic_erase_prob),
                            "fill": str(
                                getattr(
                                    self,
                                    "subject_pool_context_dynamic_erase_fill",
                                    "background_patch",
                                )
                            ),
                            **erase_debug,
                            "mask_build": mask_debug,
                            "mask_sample": sample_debug,
                        }
                else:
                    data.pop("subject_pool_context_dynamic_patch_mask", None)
                    context_mask_info["pool_context_dynamic_erase"] = {
                        "enabled": False,
                        "applied": False,
                        "status": (
                            "disabled_by_subject_pool_context_dynamic_erase"
                            if not pool_dynamic_erase_enabled
                            else "prob_zero"
                        ),
                        "prob": float(pool_dynamic_erase_prob),
                        "frame_windows": [
                            [int(v) for v in window] for window in pool_frame_windows
                        ],
                    }
                data["subject_context_patch_mask_info"] = context_mask_info
            history_latents = torch.cat(
                [
                    pool_latents.to(
                        device=clean_latents.device, dtype=clean_latents.dtype
                    ),
                    clean_latents,
                ],
                dim=2,
            )
        else:
            history_latents = clean_latents

        extrinsics = np.asarray(fused_inputs["w2c"])
        memory_K = np.asarray(fused_inputs["memory_K"])
        pool_camera_frame_indices = [
            int(fid) for block in selection["frame_blocks"][:-1] for fid in block
        ]
        pool_extr = torch.from_numpy(
            extrinsics[np.asarray(pool_camera_frame_indices, dtype=np.int64)]
        ).float()
        pool_intr = torch.from_numpy(
            memory_K[np.asarray(pool_camera_frame_indices, dtype=np.int64)]
        ).float()
        if anchor_form == "block":
            anchor_frame_indices = [int(fid) for fid in selection["frame_blocks"][-1]]
            anchor_extr = torch.from_numpy(
                extrinsics[np.asarray(anchor_frame_indices, dtype=np.int64)]
            ).float()
            anchor_intr = torch.from_numpy(
                memory_K[np.asarray(anchor_frame_indices, dtype=np.int64)]
            ).float()
            anchor_time = int(selection["actual_latent_times"][-1])
            debug["anchor_cam_source"] = "block_distinct_frames"
        else:
            # Single-frame anchor: keep the dataset's replicated [G-1]*4
            # cameras exactly as shipped (mirrors inference, whose anchor
            # camera is the last frame replicated 4x). The dataset tensor
            # may live on either device by now -- align it to the
            # numpy-sourced pool cameras (CPU) for the concat; the forward
            # re-transfers these keys to the GPU in fp32 anyway.
            anchor_extr = (
                torch.as_tensor(data["clean_latent_indices_prope_extrinsic"])
                .detach()
                .float()
                .to(pool_extr.device)
            )
            anchor_intr = (
                torch.as_tensor(data["clean_latent_indices_prope_intrinsic"])
                .detach()
                .float()
                .to(pool_intr.device)
            )
            anchor_time = _train_clean_history_actual_latent_time(max(0, G - 4))
            debug["anchor_cam_source"] = "dataset_single_frame"
        data["clean_latent_indices_prope_extrinsic"] = torch.cat(
            [pool_extr, anchor_extr], dim=0
        )
        data["clean_latent_indices_prope_intrinsic"] = torch.cat(
            [pool_intr, anchor_intr], dim=0
        )

        clean_times = [int(t) for t in selection["actual_latent_times"][:-1]] + [
            int(anchor_time)
        ]
        data["clean_latents"] = history_latents
        data["latent_rope_time_indices"] = _build_train_clean_history_rope_time_indices(
            clean_actual_latent_times=clean_times,
            generating_first_frame_idx=G,
            noisy_latent_count=int(noisy_latents.shape[2]),
            mosaic_frame_indices=None,
        )
        debug.update(
            {
                "applied": True,
                "anchor_encode": "joint_with_targets",
                "context_count_actual": len(pool_frame_windows),
                "selected_block_starts": selection["selected_block_starts"],
                "pool_block_starts": selection["pool_block_starts"],
                "fallback_block_starts": selection["fallback_block_starts"],
                "frame_blocks": selection["frame_blocks"],
                "encode_frame_windows": selection["encode_frame_windows"],
                "actual_latent_times": clean_times,
                "clean_latents_shape": tuple(history_latents.shape),
                "context_group_indices": selection.get("context_group_indices"),
                "context_is_recent": selection.get("context_is_recent"),
            }
        )
        if self.debug:
            print(
                "[train][clean-history] "
                f"step={self.step_count} anchor={anchor_form} "
                f"depth={plan.get('anchor_depth')} ctx={len(pool_frame_windows)} "
                f"blocks={selection['frame_blocks']} rope_actual={clean_times}",
                flush=True,
            )
        return history_latents

    def _refresh_train_clean_history_rope_indices(
        self,
        data,
        noisy_latents,
        *,
        include_mosaic=None,
    ):
        debug = data.get("train_clean_latent_history_debug") or {}
        actual_latent_times = debug.get("actual_latent_times")
        if not actual_latent_times:
            return data

        if include_mosaic is None:
            include_mosaic = (
                getattr(self, "enable_mosaic", True)
                and not getattr(self, "only_prope", False)
                and not bool(data.get("_no_mosaic_this_step"))
            )
        mosaic_frame_indices = (
            data.get("mosaic_frame_indices") if include_mosaic else None
        )
        info = data.get("info") or {}
        G = int(info.get("generating_first_frame_idx", 0) or 0)
        if G <= 0:
            raise ValueError(
                "train clean-history requires info['generating_first_frame_idx'] "
                "to refresh RoPE indices."
            )
        noisy_count = int(
            noisy_latents.shape[2]
            if noisy_latents.ndim == 5
            else noisy_latents.shape[1]
        )
        rope_time_indices = _build_train_clean_history_rope_time_indices(
            clean_actual_latent_times=actual_latent_times,
            generating_first_frame_idx=G,
            noisy_latent_count=noisy_count,
            mosaic_frame_indices=mosaic_frame_indices,
        )
        data["latent_rope_time_indices"] = rope_time_indices
        debug = dict(debug)
        debug["latent_rope_time_indices"] = rope_time_indices.detach().cpu().tolist()
        if mosaic_frame_indices is None:
            debug["mosaic_frame_indices_for_rope"] = []
        elif torch.is_tensor(mosaic_frame_indices):
            debug["mosaic_frame_indices_for_rope"] = (
                mosaic_frame_indices.detach().cpu().reshape(-1).tolist()
            )
        else:
            debug["mosaic_frame_indices_for_rope"] = (
                np.asarray(mosaic_frame_indices).reshape(-1).tolist()
            )
        data["train_clean_latent_history_debug"] = debug
        return data

    def _resolve_train_fuse_mode(self):
        """Resolve the configured training ``mosaic_fuse_mode`` into a
        concrete fuse mode for this materialize call.

        ``random`` picks uniformly between ``fill_stop_zbuffer`` and
        ``recent_first_smooth`` per call; every other value is returned
        unchanged. The picked mode is used consistently within a single
        materialize call (for the ``fuse_candidates`` fuse_mode arg, the
        ``recent_first_cfg`` gating, and the recent-first RGB memory
        gating) so we never feed mismatched flags to the fuse path.

        Note: the dataset side prepares the *superset* of inputs needed
        by both modes (see ``unified_dataset.py``: candidate budget and
        ``recent_first_memory_frames_by_id``), so either resolved mode
        is servable from the same batch.

        Validation paths bypass this entirely and use
        ``args.mosaic_fuse_mode`` (the non-train arg) directly.
        """
        if self.mosaic_fuse_mode == "random":
            return random.choice(("fill_stop_zbuffer", "recent_first_smooth"))
        return _normalize_mosaic_fuse_mode(self.mosaic_fuse_mode)

    @staticmethod
    def _dit_patch_mask_to_latent_mask(patch_mask, *, h_lat, w_lat):
        patch_mask = patch_mask.to(dtype=torch.bool)
        h_patch = int(h_lat) // 2
        w_patch = int(w_lat) // 2
        if patch_mask.shape[-2:] != (h_patch, w_patch):
            raise ValueError(
                "subject_context_patch_mask must have DiT patch-grid shape "
                f"({h_patch}, {w_patch}), got {tuple(patch_mask.shape[-2:])}."
            )
        return (
            patch_mask.repeat_interleave(2, dim=-2)
            .repeat_interleave(2, dim=-1)
            .to(dtype=torch.bool)
        )

    def _apply_subject_context_patch_mask_to_cn_latents(self, data, cn_latents):
        """Replace selected clean-context latent blocks with background blocks.

        The replacement runs before DiT patch embedding / RoPE / PRoPE. Donor
        background content therefore receives the target position's positional
        information after it is pasted into the protagonist region.
        """
        patch_masks = data.get("subject_context_patch_mask")
        if patch_masks is None:
            return cn_latents
        if not torch.is_tensor(patch_masks):
            patch_masks = torch.as_tensor(patch_masks)
        if patch_masks.ndim != 3 or not bool(patch_masks.any()):
            return cn_latents

        dynamic_patch_masks = data.get("subject_context_dynamic_patch_mask")
        if dynamic_patch_masks is not None:
            if not torch.is_tensor(dynamic_patch_masks):
                dynamic_patch_masks = torch.as_tensor(dynamic_patch_masks)
            if dynamic_patch_masks.ndim != 3:
                dynamic_patch_masks = None

        fill_mode = str(
            (data.get("subject_context_patch_mask_info") or {}).get(
                "fill", "background_patch"
            )
        ).strip().lower()
        patch_masks = patch_masks.to(device=cn_latents.device, dtype=torch.bool)
        if dynamic_patch_masks is not None:
            dynamic_patch_masks = dynamic_patch_masks.to(
                device=cn_latents.device,
                dtype=torch.bool,
            )
        h_lat = int(cn_latents.shape[-2])
        w_lat = int(cn_latents.shape[-1])
        context_count = min(int(patch_masks.shape[0]), int(cn_latents.shape[1]))
        if context_count <= 0:
            return cn_latents

        data["subject_context_clean_latents"] = (
            cn_latents[:, :context_count].detach().clone()
        )
        out = cn_latents.clone()
        debug_frames = []
        total_patches = 0
        background_pool_min = None
        for t in range(context_count):
            patch_mask = patch_masks[t]
            if not bool(patch_mask.any()):
                continue
            latent_mask = self._dit_patch_mask_to_latent_mask(
                patch_mask, h_lat=h_lat, w_lat=w_lat
            )
            frame_lat = out[:, t]
            donor_frame_lat = frame_lat.clone()
            if dynamic_patch_masks is not None and t < int(dynamic_patch_masks.shape[0]):
                donor_exclude_patch_mask = dynamic_patch_masks[t]
                if donor_exclude_patch_mask.shape != patch_mask.shape:
                    raise ValueError(
                        "subject_context_dynamic_patch_mask must match "
                        "subject_context_patch_mask shape, got "
                        f"{tuple(donor_exclude_patch_mask.shape)} vs "
                        f"{tuple(patch_mask.shape)}."
                    )
            else:
                donor_exclude_patch_mask = patch_mask
            bg_patch_mask = ~donor_exclude_patch_mask
            bg_coords = torch.nonzero(bg_patch_mask, as_tuple=False)
            masked_coords = torch.nonzero(patch_mask, as_tuple=False)
            if int(masked_coords.shape[0]) <= 0:
                continue
            if background_pool_min is None:
                background_pool_min = int(bg_coords.shape[0])
            else:
                background_pool_min = min(background_pool_min, int(bg_coords.shape[0]))

            if fill_mode == "background_mean" or int(bg_coords.shape[0]) <= 0:
                donor_exclude_latent_mask = self._dit_patch_mask_to_latent_mask(
                    donor_exclude_patch_mask,
                    h_lat=h_lat,
                    w_lat=w_lat,
                )
                bg_latent_mask = ~donor_exclude_latent_mask
                if bool(bg_latent_mask.any()):
                    mean = donor_frame_lat[:, bg_latent_mask].mean(dim=1).view(
                        -1, 1, 1
                    )
                else:
                    mean = donor_frame_lat.mean(dim=(1, 2), keepdim=True)
                frame_lat[:, latent_mask] = mean.view(-1, 1).expand(
                    -1, int(latent_mask.sum())
                )
                donor_mode = "background_mean"
            else:
                donor_mode = "background_patch"
                if int(bg_coords.shape[0]) < int(masked_coords.shape[0]):
                    choice = torch.randint(
                        0,
                        int(bg_coords.shape[0]),
                        (int(masked_coords.shape[0]),),
                        device=cn_latents.device,
                    )
                else:
                    perm = torch.randperm(
                        int(bg_coords.shape[0]), device=cn_latents.device
                    )
                    choice = perm[: int(masked_coords.shape[0])]
                donor_coords = bg_coords.index_select(0, choice)
                for dst, src in zip(masked_coords, donor_coords):
                    dy = int(dst[0].item()) * 2
                    dx = int(dst[1].item()) * 2
                    sy = int(src[0].item()) * 2
                    sx = int(src[1].item()) * 2
                    frame_lat[:, dy : dy + 2, dx : dx + 2] = donor_frame_lat[
                        :, sy : sy + 2, sx : sx + 2
                    ]
            total_patches += int(masked_coords.shape[0])
            debug_frames.append(
                {
                    "latent_index": int(t),
                    "masked_patch_count": int(masked_coords.shape[0]),
                    "background_patch_count": int(bg_coords.shape[0]),
                    "donor_excluded_patch_count": int(
                        donor_exclude_patch_mask.sum().item()
                    ),
                    "fill_mode": donor_mode,
                }
            )

        info = dict(data.get("subject_context_patch_mask_info") or {})
        info.update(
            {
                "materialized": True,
                "masked_patch_count_materialized": int(total_patches),
                "background_patch_pool_min": int(background_pool_min or 0),
                "materialized_frames": debug_frames,
            }
        )
        data["subject_context_patch_mask_info"] = info
        data["subject_context_masked_clean_latents"] = out[:, :context_count].detach()
        return out

    def _materialize_subject_ref_latents(self, data, *, device, dtype):
        ref_images = data.get("subject_ref_images")
        if ref_images is None:
            return data
        if not torch.is_tensor(ref_images):
            ref_images = torch.as_tensor(ref_images)
        info = dict(data.get("subject_ref_latent_info") or {})
        if ref_images.ndim != 4 or int(ref_images.shape[0]) <= 0:
            data.pop("subject_ref_latents", None)
            info.update(
                {
                    "enabled": False,
                    "status": "no_refs",
                    "ref_count": 0,
                }
            )
            data["subject_ref_latent_info"] = info
            get_timing_reporter(self).set_metadata(subject_ref_latent_info=dict(info))
            return data

        ref_count = int(ref_images.shape[0])
        ref_videos = [
            ref_images[idx : idx + 1]
            .squeeze(0)
            .to(device=device, dtype=dtype)
            .unsqueeze(1)
            for idx in range(ref_count)
        ]
        ref_latents = self.pipe.vae.encode(
            ref_videos,
            device=device,
            tiled=False,
        )
        data["subject_ref_latents"] = ref_latents.to(device=device, dtype=dtype)
        data["subject_ref_count"] = int(ref_count)
        info.update(
            {
                "enabled": True,
                "status": "ok",
                "ref_count": int(ref_count),
                "ref_images_shape": tuple(ref_images.shape),
                "ref_latents_shape": tuple(ref_latents.shape),
                "dtype": str(ref_latents.dtype),
                "device": str(ref_latents.device),
            }
        )
        data["subject_ref_latent_info"] = info
        get_timing_reporter(self).set_metadata(subject_ref_latent_info=dict(info))
        return data

    @torch.no_grad()
    def _materialize_data(self, data):
        """Materialize the trainer-side latent / queried_latent slots.

        DA3 video datasets ship raw RGB frames + a fuse plan. This method
        encodes clean/noisy/memory latents on the GPU rank, then runs
        ``fuse_candidates``. It is a no-op for already-materialized payloads.
        """
        if not data.get("needs_vae_materialization"):
            return data

        timing = get_timing_reporter(self)
        device = self.pipe.device
        dtype = self.pipe.torch_dtype
        resolved_fuse_mode = self._resolve_train_fuse_mode()
        # Recorded for the dataset-preview banner so the per-call resolved
        # fuse mode (e.g. the 'random' draw) is visible in the saved mp4.
        data["_resolved_train_fuse_mode"] = str(resolved_fuse_mode)
        # Force VAE onto GPU once. With --offload_models vae this happens
        # on the very first materialize call and the load_models_to_device
        # state machine keeps it resident afterwards.
        with timing.scope("materialize.load_vae"):
            self.pipe.load_models_to_device(["vae"])

        # (C, warmup+1+80, H, W). Dataset may keep this as uint8 and defer
        # normalization until the GPU VAE encode path.
        cn_frames = data["cn_frames"]
        cn_drop_count = int(data["cn_drop_count"])

        # Per-step clean-history plan, sampled ONCE here (deterministic in
        # (seed, step_count), so the later consumers see the same draw).
        # Block-anchor steps extend the cn window before the encode so the
        # anchor is jointly encoded with the noisy targets.
        clean_history_plan = None
        clean_history_plan = self._get_clean_history_plan_for_materialize()
        with timing.scope("materialize.anchor_warmup_prepend"):
            cn_frames, cn_drop_count, anchor_is_joint = (
                self._maybe_prepend_anchor_warmup_frames(
                    data, cn_frames, cn_drop_count, clean_history_plan
                )
            )

        memory_frames_by_id = data.get("memory_frames_by_id", {}) or {}
        cached_memory_paths_by_id = data.get("cached_memory_paths_by_id", {}) or {}
        memory_cache_hash_by_id = data.get("memory_cache_hash_by_id", {}) or {}
        frame_pixel_range = data.get("frame_pixel_range")

        fused_inputs = data["fused_query_inputs"]
        memory_total_frames = int(fused_inputs["memory_total_frames"])
        H_lat = int(fused_inputs["H_lat"])
        W_lat = int(fused_inputs["W_lat"])
        timing.write_cache_diagnostic(
            self._build_memory_cache_diagnostic(
                data=data,
                memory_frames_by_id=memory_frames_by_id,
                cached_memory_paths_by_id=cached_memory_paths_by_id,
                memory_cache_hash_by_id=memory_cache_hash_by_id,
                memory_total_frames=memory_total_frames,
                phase="before_memory_cache_load",
            )
        )

        # ---- clean+noisy joint encode -----------------------------------
        # encode() takes a list; it returns (N, C, T, H, W) where N==1
        # for our single-element list. Slice the (1, C, ...) tensor.
        with timing.scope("materialize.cn_to_device"):
            cn_input = self._frames_to_vae_input(
                cn_frames,
                device=device,
                dtype=dtype,
                pixel_range=frame_pixel_range,
            )
        with timing.scope("materialize.cn_vae_encode"):
            cn_latents_batched = self.pipe.vae.encode(
                [cn_input], device=device, tiled=False
            )
        cn_latents = cn_latents_batched[0]  # (C, 21+blocks, H_lat, W_lat)
        expected_t = (cn_frames.shape[1] - 1) // 4 + 1
        if cn_latents.shape[1] != expected_t:
            raise RuntimeError(
                f"_materialize_data: cn_frames T={cn_frames.shape[1]} "
                f"encoded to {cn_latents.shape[1]} latent steps, expected "
                f"{expected_t} (4N+1 rule). Check VAE setup."
            )
        if cn_drop_count > 0:
            cn_latents = cn_latents[:, cn_drop_count:]
        data = self._refresh_subject_anchor_context_patch_mask_for_plan(
            data,
            cn_latents=cn_latents,
            plan=clean_history_plan,
        )
        cn_latents = self._apply_subject_context_patch_mask_to_cn_latents(
            data, cn_latents
        )
        # First slot -> clean_latents (1 latent), next 20 -> noisy latents.
        clean_latents = cn_latents[:, 0:1].unsqueeze(0).contiguous()
        noisy_latents = cn_latents[:, 1:].unsqueeze(0).contiguous()
        clean_latents = self._apply_train_clean_latent_history(
            data,
            clean_latents=clean_latents,
            noisy_latents=noisy_latents,
            device=device,
            dtype=dtype,
            plan=clean_history_plan,
            anchor_is_joint=anchor_is_joint,
        )
        data["clean_latents"] = clean_latents
        data["latents"] = noisy_latents
        with timing.scope("materialize.subject_ref_vae_encode"):
            data = self._materialize_subject_ref_latents(
                data,
                device=device,
                dtype=dtype,
            )

        # Optional per-step no-mosaic short-circuit (training only). Mirrors
        # the zero_memory_fallback below: write zero queried_latent /
        # ql_rope_indice_pef and skip memory cache + fuse.
        if bool(data.get("_no_mosaic_this_step")):
            with timing.scope("materialize.no_mosaic_short_circuit"):
                data["queried_latent"] = torch.zeros_like(noisy_latents)
                data["ql_rope_indice_pef"] = torch.zeros(
                    noisy_latents.shape[2],
                    noisy_latents.shape[-2],
                    noisy_latents.shape[-1],
                    2,
                    dtype=torch.long,
                    device=device,
                )
                if getattr(self, "mosaic_view_change_prope", False):
                    data["ql_view_change_pef"] = (
                        self._canonical_view_change_like_latents(
                            noisy_latents, device=device, dtype=noisy_latents.dtype
                        )
                    )
                data = self._apply_mosaic_interval_to_data(
                    data, noisy_latents, phase="no_mosaic_short_circuit"
                )
                data = self._refresh_train_clean_history_rope_indices(
                    data, noisy_latents, include_mosaic=False
                )
            data["needs_vae_materialization"] = False
            return data

        # ---- Memory latent buffer (fill cached + miss frames) -----------
        C_lat = int(cn_latents.shape[0])
        with timing.scope("materialize.memory_buffer_alloc"):
            memory_latents_full = torch.zeros(
                C_lat, memory_total_frames, H_lat, W_lat, device=device, dtype=dtype
            )

        loaded_cache_frame_ids = []
        cache_load_failures = []
        if memory_total_frames > 0:
            # Cache hits: torch.load each shard directly into the buffer.
            with timing.scope(
                "materialize.memory_cache_load",
                metadata={"hits": len(cached_memory_paths_by_id)},
            ):
                for fid, cache_path in cached_memory_paths_by_id.items():
                    try:
                        tensor = torch.load(
                            cache_path, map_location=device, weights_only=True
                        )
                    except Exception as exc:
                        print("***********\n***********\n***********\n***********\n")
                        print(f"Error loading cache file {cache_path}: {exc}")
                        print("***********\n***********\n***********\n***********\n")
                        self._write_cache_load_failure_log(
                            frame_id=fid,
                            cache_path=cache_path,
                            error=exc,
                            data=data,
                        )
                        cache_load_failures.append(
                            {
                                "frame_id": int(fid),
                                "path": str(cache_path),
                                "error": str(exc),
                            }
                        )
                        # Cache file is corrupt / unreadable; treat as a miss so
                        # we re-encode from the mp4 frames the dataset still
                        # shipped (worker that produced this batch will have
                        # dropped the frame from memory_frames_by_id only when
                        # the path existed *and* was readable enough to call
                        # os.path.exists; corruption mid-training is rare but we
                        # don't want to crash on it).
                        if int(fid) in memory_frames_by_id:
                            continue
                        raise
                    memory_latents_full[:, int(fid) : int(fid) + 1] = tensor.to(
                        device=device, dtype=dtype
                    )
                    loaded_cache_frame_ids.append(int(fid))
        timing.write_cache_diagnostic(
            self._build_memory_cache_diagnostic(
                data=data,
                memory_frames_by_id=memory_frames_by_id,
                cached_memory_paths_by_id=cached_memory_paths_by_id,
                memory_cache_hash_by_id=memory_cache_hash_by_id,
                memory_total_frames=memory_total_frames,
                phase="after_memory_cache_load",
                loaded_cache_frame_ids=loaded_cache_frame_ids,
                cache_load_failures=cache_load_failures,
            )
        )

        # Cache misses: encode in one batch (list of (C,1,H,W) tensors),
        # then dump each result back to the cache.
        encoded_miss_frame_ids = []
        dumped_frame_ids = []
        dump_failures = []
        if memory_frames_by_id:
            with timing.scope(
                "materialize.memory_cache_miss_prepare",
                metadata={"misses": len(memory_frames_by_id)},
            ):
                self.pipe.load_models_to_device(["vae"])
                miss_fids = sorted(int(fid) for fid in memory_frames_by_id.keys())
            # Encode each cache-miss source into a single
            # (C, 1, H_lat, W_lat) latent step, keyed by frame id. Two
            # independent channels feed this:
            #   * neighbour-window (``memory_neighbor_window_train``): the
            #     dataset already shipped either a real 5-frame temporal
            #     window (low camera motion) or a single frame (large motion
            #     / fallback) per slot, so we group by the per-slot frame
            #     count -- ``WanVideoVAE.encode`` ``torch.stack``s its
            #     outputs and therefore needs a uniform time dim per call.
            #   * legacy replicate: every slot is (C, 1, H, W), optionally
            #     replicated ``n_in`` times along time before one batched
            #     encode. We always keep the last latent step.
            mem_lat_by_fid = {}
            with timing.scope("materialize.memory_miss_vae_encode"):
                if self.memory_neighbor_window_train:
                    fids_by_t = {}
                    for fid in miss_fids:
                        t_in = int(memory_frames_by_id[fid].shape[1])
                        fids_by_t.setdefault(t_in, []).append(fid)
                    for _t_in, group_fids in sorted(fids_by_t.items()):
                        group_videos = [
                            self._frames_to_vae_input(
                                memory_frames_by_id[fid],
                                device=device,
                                dtype=dtype,
                                pixel_range=frame_pixel_range,
                            )
                            for fid in group_fids
                        ]
                        group_lat = self.pipe.vae.encode(
                            group_videos, device=device, tiled=False
                        )  # (N, C, T_lat, H_lat, W_lat)
                        if group_lat.shape[2] > 1:
                            group_lat = group_lat[:, :, -1:].contiguous()
                        for i, fid in enumerate(group_fids):
                            mem_lat_by_fid[fid] = group_lat[i]  # (C, 1, H_lat, W_lat)
                else:
                    miss_videos = [
                        self._frames_to_vae_input(
                            memory_frames_by_id[fid],
                            device=device,
                            dtype=dtype,
                            pixel_range=frame_pixel_range,
                        )
                        for fid in miss_fids
                    ]
                    # Optionally replicate each (C, 1, H, W) source frame N
                    # times along the time axis so the VAE produces
                    # 1+(N-1)//4 latent steps; keep the last latent step to
                    # recover a (C, 1, H_lat, W_lat) tensor per source frame.
                    n_in = self.memory_vae_encode_input_frames
                    if n_in > 1:
                        miss_videos = [v.repeat(1, n_in, 1, 1) for v in miss_videos]
                    mem_latents_batched = self.pipe.vae.encode(
                        miss_videos, device=device, tiled=False
                    )  # (N, C, T_lat, H_lat, W_lat); T_lat = (n_in-1)//4 + 1
                    if mem_latents_batched.shape[2] > 1:
                        mem_latents_batched = mem_latents_batched[
                            :, :, -1:
                        ].contiguous()
                    for batch_idx, fid in enumerate(miss_fids):
                        mem_lat_by_fid[fid] = mem_latents_batched[batch_idx]
                encoded_miss_frame_ids = list(miss_fids)
            with timing.scope("materialize.memory_cache_dump"):
                for fid in miss_fids:
                    lat_full = mem_lat_by_fid[fid]  # (C, 1, H_lat, W_lat)
                    memory_latents_full[:, fid] = lat_full[:, 0]
                    target_path = memory_cache_hash_by_id.get(fid)
                    if target_path:
                        try:
                            self._atomic_dump_latent(target_path, lat_full)
                            dumped_frame_ids.append(int(fid))
                        except OSError as exc:
                            dump_failures.append(
                                {
                                    "frame_id": int(fid),
                                    "path": str(target_path),
                                    "error": str(exc),
                                }
                            )
                            print(
                                f"[WanMosaicPipelineModule] memory-latent cache "
                                f"dump to {target_path} failed: {exc}"
                            )
        timing.write_cache_diagnostic(
            self._build_memory_cache_diagnostic(
                data=data,
                memory_frames_by_id=memory_frames_by_id,
                cached_memory_paths_by_id=cached_memory_paths_by_id,
                memory_cache_hash_by_id=memory_cache_hash_by_id,
                memory_total_frames=memory_total_frames,
                phase="after_memory_cache_miss_encode",
                loaded_cache_frame_ids=loaded_cache_frame_ids,
                cache_load_failures=cache_load_failures,
                encoded_miss_frame_ids=encoded_miss_frame_ids,
                dumped_frame_ids=dumped_frame_ids,
                dump_failures=dump_failures,
            )
        )

        if memory_total_frames == 0:
            with timing.scope("materialize.zero_memory_fallback"):
                data["queried_latent"] = torch.zeros_like(noisy_latents)
                data["ql_rope_indice_pef"] = torch.zeros(
                    noisy_latents.shape[2],
                    noisy_latents.shape[-2],
                    noisy_latents.shape[-1],
                    2,
                    dtype=torch.long,
                    device=device,
                )
                if getattr(self, "mosaic_view_change_prope", False):
                    data["ql_view_change_pef"] = (
                        self._canonical_view_change_like_latents(
                            noisy_latents, device=device, dtype=noisy_latents.dtype
                        )
                    )
                data = self._apply_mosaic_interval_to_data(
                    data, noisy_latents, phase="zero_memory_fallback"
                )
                data = self._refresh_train_clean_history_rope_indices(
                    data,
                    noisy_latents,
                    include_mosaic=not bool(data.get("_no_mosaic_this_step")),
                )
            data["needs_vae_materialization"] = False
            return data

        # ---- Phase 2: fuse via FrustumHandler ---------------------------
        with timing.scope("materialize.get_frustum_handler"):
            handler = self._get_trainer_frustum_handler()
        # The new forward-warp + scatter fuse path runs torch ops directly
        # on whatever device the latents live on. Keep memory_latents on
        # the active device (typically CUDA) -- the previous .to("cpu")
        # round-trip dropped fuse throughput by ~10x once the implementation
        # switched from CPU grid_sample to GPU scatter_reduce_.
        memory_latents_arg = memory_latents_full.detach().float()
        # Per-frame K when available (video flow training emits both
        # memory_K=(M,3,3) and query_K=(Q,3,3); legacy paths still ship
        # only the single ``K`` field). The handler accepts either
        # signature.
        if "memory_K" in fused_inputs and "query_K" in fused_inputs:
            memory_K_arg = fused_inputs["memory_K"]
            query_K_arg = fused_inputs["query_K"]
            single_K_arg = None
        else:
            memory_K_arg = None
            query_K_arg = None
            single_K_arg = fused_inputs["K"]
        return_source_frame_ids = bool(
            getattr(self, "mosaic_return_source_frame_ids", False)
        )
        with timing.scope("materialize.fuse_candidates"):
            source_valid_masks_arg = fused_inputs.get("source_valid_masks")
            fused_out = handler.fuse_candidates(
                query_extrinsics=fused_inputs["query_extrinsics"],
                candidate_frame_ids=fused_inputs["candidate_frame_ids"],
                latents=memory_latents_arg,
                w2c=fused_inputs["w2c"],
                depths=fused_inputs["depths"],
                K=single_K_arg,
                memory_K=memory_K_arg,
                query_K=query_K_arg,
                H_lat=H_lat,
                W_lat=W_lat,
                memory_total_frames=memory_total_frames,
                fuse_mode=resolved_fuse_mode,
                interpolation_mode="nearest",
                return_revgrid=True,
                return_view_change=getattr(self, "mosaic_view_change_prope", False),
                latent_merge_4frames=False,
                query_reference_frame=fused_inputs.get(
                    "query_reference_frame", self.mosaic_query_reference_frame
                ),
                fill_ratio_threshold=0.95,
                memory_frames=data.get("recent_first_memory_frames_by_id"),
                recent_first_cfg=(
                    self.recent_first_cfg
                    if _is_recent_first_fuse_mode(resolved_fuse_mode)
                    else None
                ),
                fuse_block_size=int(getattr(self, "mosaic_fuse_block_size", 1)),
                return_source_frame_ids=return_source_frame_ids,
                source_valid_masks=source_valid_masks_arg,
            )
        queried_latent = fused_out[0]
        revgrid = fused_out[1]
        extra_idx = 2
        if getattr(self, "mosaic_view_change_prope", False):
            view_change = fused_out[extra_idx]
            extra_idx += 1
            data["ql_view_change_pef"] = view_change
        if return_source_frame_ids:
            source_fid = fused_out[extra_idx]
        else:
            source_fid = None
        with timing.scope("materialize.queried_to_dtype"):
            data["queried_latent"] = queried_latent.to(device=device, dtype=dtype)
        data["ql_rope_indice_pef"] = revgrid
        data["ql_source_frame_ids"] = source_fid
        data = self._apply_mosaic_interval_to_data(
            data, noisy_latents, phase="materialize"
        )
        data = self._refresh_train_clean_history_rope_indices(data, noisy_latents)
        # Mark consumed so a re-entrant call (e.g. preview that runs
        # before forward in the same step) is a cheap no-op.
        data["needs_vae_materialization"] = False
        return data
