from diffsynth.pipelines.wan_video import WAN_VIDEO_PROPE_CAMERA_KEYS

from .common import *
from .config import _is_recent_first_fuse_mode, _normalize_mosaic_fuse_mode
from .history import (
    _compute_mosaic_frame_indices,
    _mosaic_kept_frame_ranges,
)
from .prompting import _prompt_text_for_preview, _train_prompt_from_data
from .qwen import (
    _TrainQwenPromptGenerator,
    _train_qwen_frames_from_cn_frames,
    _write_train_qwen_prompt_cache,
)
from .timing import get_timing_reporter
from .video_io import (
    _annotate_frames_with_index,
    _blank_preview_frame,
    _build_camera_trajectory_panel,
    _build_preview_candidate_diagnostics,
    _build_train_clean_history_preview_meta,
    _build_train_clean_history_preview_panel,
    _draw_preview_label,
    _first_valid_candidate,
    _fit_preview_frame_to_slot,
    _preview_candidate_groups_from_fused,
    _preview_raw_frame_labels,
    _project_candidate_frame_to_query_preview,
    _read_video_gt_frames,
    _render_plan_text_banner,
    _render_prompt_panel,
    _resolve_K_for_index,
    _resolve_group_query_index,
    _write_json_atomic,
    _write_video,
)


class _ModelLoadingMixin:
    @staticmethod
    def _is_subject_ref_memory_key(key):
        return key.startswith("subject_ref_")

    def _load_dit_state_dict_allowing_subject_ref_init(self, state_dict):
        load_result = self.pipe.dit.load_state_dict(state_dict, strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        non_subject_unexpected = [
            key for key in unexpected if not self._is_subject_ref_memory_key(key)
        ]
        if non_subject_unexpected:
            raise RuntimeError(
                "Unexpected keys while loading trained_dit with subject ref memory: "
                f"{non_subject_unexpected[:20]}"
            )
        non_subject_missing = [
            key for key in missing if not self._is_subject_ref_memory_key(key)
        ]
        if non_subject_missing:
            raise RuntimeError(
                "Missing non-subject-ref keys while loading trained_dit: "
                f"{non_subject_missing[:20]}"
            )
        if missing:
            print(
                "[subject-ref-memory] initializing new DiT keys not found in "
                f"checkpoint: {missing}",
                flush=True,
            )
        if unexpected:
            print(
                "[subject-ref-memory] ignoring stale subject-ref keys not used "
                f"by this model: {unexpected}",
                flush=True,
            )
        return load_result

    def _load_trained_dit(self, trained_dit):
        if trained_dit is None:
            return
        if self.pipe.dit is None:
            raise ValueError(
                "Cannot load trained_dit because pipe.dit is not initialized."
            )
        print(f"Loading trained DIT weights from: {trained_dit}")
        from diffsynth.core import load_state_dict

        state_dict = load_state_dict(
            trained_dit, torch_dtype=torch.bfloat16, device="cpu"
        )
        load_result = self._load_dit_state_dict_with_prefix_fallback(state_dict)
        print(f"Trained DIT weights loaded successfully: {load_result}")
        if getattr(self.pipe, "dit2", None) is not None:
            print(
                "Disabling pipe.dit2 after loading --trained_dit so "
                "validation/inference always uses the trained DIT."
            )
            self.pipe.dit2 = None

    def _load_dit_state_dict_with_prefix_fallback(self, state_dict):
        allow_subject_init = bool(getattr(self, "subject_ref_memory", False))
        load_fn = (
            self._load_dit_state_dict_allowing_subject_ref_init
            if allow_subject_init
            else lambda sd: self.pipe.dit.load_state_dict(sd, strict=True)
        )
        try:
            return load_fn(state_dict)
        except RuntimeError as original_error:
            for prefix in ("pipe.dit.", "dit."):
                stripped = {
                    key[len(prefix) :]: value
                    for key, value in state_dict.items()
                    if key.startswith(prefix)
                }
                if not stripped:
                    continue
                try:
                    return load_fn(stripped)
                except RuntimeError:
                    pass
            raise original_error

    @staticmethod
    def _set_use_prope(
        pipe,
        use_prope,
        prope_attention_interval=1,
        prope_disable_native_rope=False,
        prope_disable_t_rope=False,
        prope_camera_layout="full",
    ):
        """Toggle PRoPE self-attention on DiT blocks.

        ``prope_attention_interval`` controls per-block granularity:

        * ``1`` (default): every block uses PRoPE (legacy behaviour).
        * ``N > 1``: only blocks whose 0-based index satisfies
          ``block_id % N == 0`` use PRoPE; the remaining blocks have
          ``self_attn.use_prope`` flipped back to ``False`` and fall
          through to the plain SDPA path in ``SelfAttention.forward``.

        ``prope_disable_native_rope`` drops the full 3D RoPE inside PRoPE
        blocks; ``prope_disable_t_rope`` drops only the temporal RoPE band
        and keeps x/y (intra-frame spatial structure). Both delegate
        cross-frame addressing to the camera matrices and therefore
        require ``prope_attention_interval > 1`` so the remaining blocks
        keep full positional information; they are mutually exclusive.

        ``prope_camera_layout`` selects the head_dim camera tiling: ``"full"``
        (legacy, tile over all dims) or ``"sf13"`` (contiguous 2-camera layout
        for sub-frames {1,3} written into the rope-free t-band). ``"sf13"``
        only makes sense when the t-band is rope-free, so it requires
        ``prope_disable_t_rope``.

        The container-level ``dit.use_prope`` flag stays ``True`` as long
        as at least one block has PRoPE enabled so downstream guards that
        only check the DiT-level flag continue to work.
        """
        if not use_prope:
            return
        interval = max(1, int(prope_attention_interval))
        prope_disable_native_rope = bool(prope_disable_native_rope)
        prope_disable_t_rope = bool(prope_disable_t_rope)
        prope_camera_layout = str(prope_camera_layout or "full")
        if prope_disable_native_rope and prope_disable_t_rope:
            raise ValueError(
                "prope_disable_native_rope and prope_disable_t_rope are "
                "mutually exclusive."
            )
        if prope_disable_t_rope and interval <= 1:
            raise ValueError(
                "prope_disable_t_rope requires prope_attention_interval > 1 "
                "so that non-PRoPE blocks retain full temporal RoPE."
            )
        if prope_camera_layout != "full" and not prope_disable_t_rope:
            raise ValueError(
                f"prope_camera_layout={prope_camera_layout!r} requires "
                "prope_disable_t_rope so the camera block lands in a "
                "rope-free subspace."
            )
        for dit_name in ("dit", "dit2"):
            dit = getattr(pipe, dit_name, None)
            if dit is None:
                continue
            dit.use_prope = True
            dit.prope_disable_native_rope = prope_disable_native_rope
            dit.prope_disable_t_rope = prope_disable_t_rope
            dit.prope_camera_layout = prope_camera_layout
            for block_id, block in enumerate(getattr(dit, "blocks", [])):
                block_use_prope = (block_id % interval) == 0
                block.use_prope = block_use_prope
                block.prope_disable_native_rope = prope_disable_native_rope
                block.prope_disable_t_rope = prope_disable_t_rope
                block.prope_camera_layout = prope_camera_layout
                if hasattr(block, "self_attn"):
                    block.self_attn.use_prope = block_use_prope
                    block.self_attn.prope_disable_native_rope = (
                        prope_disable_native_rope
                    )
                    block.self_attn.prope_disable_t_rope = prope_disable_t_rope
                    block.self_attn.prope_camera_layout = prope_camera_layout

    # ------------------------------------------------------------------
    # Reuse parent's parse_extra_inputs / get_pipeline_inputs
    # ------------------------------------------------------------------


class _ScheduleMixin:
    @staticmethod
    def _parse_mosaic_schedule(spec):
        """Parse ``"interval3:3,interval1:4,nomosaic:3"`` into (modes, weights).

        ``modes`` is a list of ``(kind, interval)`` tuples where ``kind`` is
        ``"interval"`` or ``"nomosaic"`` (``interval`` is 0 for nomosaic).
        ``weights`` is the parallel list of non-negative ints. Raises
        ``ValueError`` on a malformed spec or when every weight is zero.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(
                "mosaic_train_schedule must be a non-empty string like "
                "'interval3:3,interval1:4,nomosaic:3'."
            )
        modes = []
        weights = []
        for raw in spec.split(","):
            token = raw.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(
                    f"mosaic_train_schedule entry {token!r} must be "
                    "'<mode>:<weight>'."
                )
            name, weight_str = token.rsplit(":", 1)
            name = name.strip().lower()
            try:
                weight = int(weight_str.strip())
            except ValueError as exc:
                raise ValueError(
                    f"mosaic_train_schedule weight in {token!r} must be an int."
                ) from exc
            if weight < 0:
                raise ValueError(
                    f"mosaic_train_schedule weight in {token!r} must be >= 0."
                )
            if name == "nomosaic":
                modes.append(("nomosaic", 0))
            elif name.startswith("interval"):
                try:
                    interval = int(name[len("interval") :])
                except ValueError as exc:
                    raise ValueError(
                        f"mosaic_train_schedule mode {name!r} must be "
                        "'interval<N>' or 'nomosaic'."
                    ) from exc
                if interval < 1:
                    raise ValueError(
                        f"mosaic_train_schedule interval in {name!r} must be >= 1."
                    )
                modes.append(("interval", interval))
            else:
                raise ValueError(
                    f"mosaic_train_schedule mode {name!r} must be "
                    "'interval<N>' or 'nomosaic'."
                )
            weights.append(weight)
        if not modes:
            raise ValueError("mosaic_train_schedule produced no modes.")
        if sum(weights) <= 0:
            raise ValueError(
                "mosaic_train_schedule must have at least one positive weight."
            )
        return modes, weights

    def _draw_schedule_index(self, step_index):
        """Deterministic weighted draw of a schedule-mode index.

        Seeded purely by ``(base_seed, step_index)`` so that every rank that
        calls this with the same step index computes the same index with zero
        communication. ``base_seed`` is ``args.seed`` (shared across ranks).
        """
        rng = random.Random((int(self._schedule_base_seed) << 20) ^ int(step_index))
        idx = rng.choices(
            range(len(self._mosaic_schedule_modes)),
            weights=self._mosaic_schedule_weights,
            k=1,
        )[0]
        return int(idx)

    def _broadcast_schedule_index(self, idx):
        """Broadcast rank 0's schedule index to all ranks.

        Falls back to the local ``idx`` when torch.distributed is unavailable,
        not initialised, or single-process. Uses ``self.pipe.device`` so the
        collective lands on the right device for the active backend.
        """
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return int(idx)
        if int(torch.distributed.get_world_size()) <= 1:
            return int(idx)
        device = getattr(getattr(self, "pipe", None), "device", None) or "cpu"
        box = torch.tensor([int(idx)], dtype=torch.long, device=device)
        torch.distributed.broadcast(box, src=0)
        return int(box.item())

    def _sample_mosaic_step_mode(self, step_index):
        """Return the ``(kind, interval)`` mode for this training step.

        ``kind`` is ``"interval"`` (use ``interval`` frames) or ``"nomosaic"``.
        The choice is synchronised across DDP ranks: ``seed`` mode lets every
        rank derive the same index from ``(base_seed, step_index)``;
        ``rank0_broadcast`` additionally forces rank 0's draw onto every rank
        (a safety net should step indices ever diverge).
        """
        idx = self._draw_schedule_index(int(step_index))
        if self.mosaic_schedule_sync_mode == "rank0_broadcast":
            idx = self._broadcast_schedule_index(idx)
        return self._mosaic_schedule_modes[idx]


class _TrainQwenMixin:
    def _train_qwen_args_namespace(self):
        import types

        return types.SimpleNamespace(
            train_qwen_model_path=self.train_qwen_model_path,
            validation_qwen_model_path=self.train_qwen_model_path,
            train_qwen_provider=self.train_qwen_provider,
            train_qwen_device_map=self.train_qwen_device_map,
            train_qwen_torch_dtype=self.train_qwen_torch_dtype,
            train_qwen_max_new_tokens=self.train_qwen_max_new_tokens,
            train_qwen_word_limit=self.train_qwen_word_limit,
            train_qwen_trust_remote_code=self.train_qwen_trust_remote_code,
        )

    def _get_train_qwen_prompt_generator(self):
        if self._train_qwen_prompt_generator is None:
            rank = (
                int(torch.distributed.get_rank())
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            self._train_qwen_prompt_generator = _TrainQwenPromptGenerator(
                self._train_qwen_args_namespace(),
                rank=rank,
            )
        return self._train_qwen_prompt_generator

    def _train_qwen_prompt_debug_path(self):
        if not self.log_dir:
            return None
        rank = (
            int(torch.distributed.get_rank())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        prompt_dir = os.path.join(self.log_dir, "train_qwen_prompts")
        os.makedirs(prompt_dir, exist_ok=True)
        return os.path.join(prompt_dir, f"train_qwen_prompts_rank{rank:03d}.jsonl")

    def _write_train_qwen_prompt_debug(self, record):
        if not self.debug:
            return
        path = self._train_qwen_prompt_debug_path()
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _maybe_write_train_qwen_prompt_cache(self, data, clean_name, qwen_prompt):
        if (
            not self.train_qwen_prompt_cache
            or not self.train_qwen_prompt_missing_only
            or not self.train_qwen_prompt_cache_dir
        ):
            return None
        fused_inputs = data.get("fused_query_inputs") or {}
        frame_indices = fused_inputs.get("query_frame_indices")
        if frame_indices is None:
            return {
                "enabled": True,
                "skipped": True,
                "skip_reason": "missing_query_frame_indices",
            }
        return _write_train_qwen_prompt_cache(
            self.train_qwen_prompt_cache_dir,
            clean_name,
            frame_indices,
            qwen_prompt,
        )

    def release_train_qwen_prompt_generator(self):
        if self._train_qwen_prompt_generator is None:
            return
        if self.debug:
            print("[train][qwen] releasing prompt model", flush=True)
        self._train_qwen_prompt_generator.model = None
        self._train_qwen_prompt_generator.processor = None
        self._train_qwen_prompt_generator = None

    def _maybe_apply_train_qwen_prompt(self, data):
        if not self.train_qwen_prompt or not self.training:
            return data
        if not isinstance(data, dict):
            return data

        timing = get_timing_reporter(self)
        original_prompt = str(data.get("prompt") or "")
        info = data.get("info") or {}
        clean_name = str(info.get("clean_name", "unknown"))
        if self.train_qwen_prompt_missing_only and original_prompt.strip():
            debug_record = {
                "enabled": True,
                "skipped": True,
                "skip_reason": "dataset_prompt_present",
                "step": int(self.step_count),
                "clean_name": clean_name,
                "original_prompt": original_prompt,
                "original_prompt_len": int(len(original_prompt)),
            }
            data["train_qwen_prompt_debug"] = debug_record
            timing.set_metadata(train_qwen_prompt=debug_record)
            self._write_train_qwen_prompt_debug(debug_record)
            if self.debug:
                print(
                    f"[train][qwen] step={self.step_count} clean={clean_name!r} "
                    "skip generation because dataset prompt is already present",
                    flush=True,
                )
            return data
        if "cn_frames" not in data:
            raise ValueError(
                "--train_qwen_prompt currently requires video-flow training "
                "samples with data['cn_frames']."
            )
        with timing.scope(
            "forward.train_qwen_prompt.frames",
            metadata={
                "sample_frames": int(self.train_qwen_sample_frames),
                "frame_max_side": int(self.train_qwen_frame_max_side),
            },
        ):
            qwen_images, frame_indices = _train_qwen_frames_from_cn_frames(
                data["cn_frames"],
                sample_frames=self.train_qwen_sample_frames,
                max_side=self.train_qwen_frame_max_side,
            )
        if not qwen_images:
            raise ValueError("--train_qwen_prompt could not sample any GT frames.")
        image_sizes = [tuple(int(v) for v in image.size) for image in qwen_images]
        generator = self._get_train_qwen_prompt_generator()
        with timing.scope(
            "forward.train_qwen_prompt.generate",
            metadata={
                "clean_name": clean_name,
                "sampled_frame_indices": [int(i) for i in frame_indices],
                "image_sizes": image_sizes,
            },
        ):
            qwen_prompt = generator.generate_prompt(
                qwen_images,
                batch_idx=int(self.step_count),
                section_idx=0,
                clean_name=clean_name,
                dataset_prompt=original_prompt,
                timing_reporter=timing,
            )
        word_count = len(str(qwen_prompt).split())
        cache_record = None
        cache_error = None
        with timing.scope(
            "forward.train_qwen_prompt.cache_write",
            metadata={
                "cache_enabled": bool(self.train_qwen_prompt_cache),
                "cache_dir": self.train_qwen_prompt_cache_dir,
                "missing_only": bool(self.train_qwen_prompt_missing_only),
            },
        ):
            try:
                cache_record = self._maybe_write_train_qwen_prompt_cache(
                    data,
                    clean_name,
                    qwen_prompt,
                )
            except OSError as exc:
                cache_error = str(exc)
                print(
                    f"[train][qwen][cache] WARNING: failed to write prompt cache "
                    f"for {clean_name!r}: {exc}",
                    flush=True,
                )
        debug_record = {
            "enabled": True,
            "step": int(self.step_count),
            "clean_name": clean_name,
            "model_path": self.train_qwen_model_path,
            "sample_frames": int(self.train_qwen_sample_frames),
            "frame_max_side": int(self.train_qwen_frame_max_side),
            "missing_only": bool(self.train_qwen_prompt_missing_only),
            "sampled_frame_indices": [int(i) for i in frame_indices],
            "image_sizes": image_sizes,
            "original_prompt": original_prompt,
            "qwen_prompt": qwen_prompt,
            "qwen_prompt_words": int(word_count),
            "original_prompt_len": int(len(original_prompt)),
            "qwen_prompt_len": int(len(qwen_prompt)),
        }
        if cache_record is not None:
            debug_record["prompt_cache"] = cache_record
        if cache_error is not None:
            debug_record["prompt_cache_error"] = cache_error
        data["prompt"] = qwen_prompt
        data["train_qwen_prompt_debug"] = debug_record
        timing.set_metadata(train_qwen_prompt=debug_record)
        self._write_train_qwen_prompt_debug(debug_record)
        if self.debug:
            print(
                f"[train][qwen] step={self.step_count} clean={clean_name!r} "
                f"frames={frame_indices} words={word_count} prompt={qwen_prompt!r}",
                flush=True,
            )
        return data


class _PipelineInputMixin:
    def _mosaic_frame_indices_for_noisy(self, noisy_count, interval=None, device=None):
        if interval is None:
            interval = self.mosaic_interval
        return _compute_mosaic_frame_indices(noisy_count, int(interval), device=device)

    @staticmethod
    def _index_latent_time(latent, indices):
        if latent.ndim == 5:
            return latent.index_select(2, indices.to(device=latent.device))
        if latent.ndim == 4:
            return latent.index_select(1, indices.to(device=latent.device))
        raise ValueError(
            f"Expected latent with 4 or 5 dims, got shape {tuple(latent.shape)}."
        )

    @staticmethod
    def _index_revgrid_time(revgrid, indices):
        if torch.is_tensor(revgrid):
            return revgrid.index_select(0, indices.to(device=revgrid.device))
        return np.asarray(revgrid)[indices.detach().cpu().numpy()]

    @staticmethod
    def _index_view_change_time(view_change, indices):
        if torch.is_tensor(view_change):
            return view_change.index_select(0, indices.to(device=view_change.device))
        return np.asarray(view_change)[indices.detach().cpu().numpy()]

    @staticmethod
    def _canonical_view_change_like_latents(latents, *, device=None, dtype=None):
        if latents.ndim == 5:
            t, h, w = latents.shape[2], latents.shape[-2], latents.shape[-1]
        elif latents.ndim == 4:
            t, h, w = latents.shape[1], latents.shape[-2], latents.shape[-1]
        else:
            raise ValueError(
                f"Expected latent with 4 or 5 dims, got shape {tuple(latents.shape)}."
            )
        vc = torch.zeros(
            int(t),
            int(h),
            int(w),
            3,
            dtype=dtype if dtype is not None else latents.dtype,
            device=device if device is not None else latents.device,
        )
        vc[..., 0] = 1.0
        return vc

    def _apply_mosaic_interval_to_data(self, data, noisy_latents, *, phase):
        noisy_count = int(
            noisy_latents.shape[2]
            if noisy_latents.ndim == 5
            else noisy_latents.shape[1]
        )
        # Per-step interval (set by the synchronised schedule) takes priority
        # over the instance-level ``self.mosaic_interval`` fallback.
        interval = int(data.get("_mosaic_step_interval", self.mosaic_interval) or 1)
        indices = self._mosaic_frame_indices_for_noisy(
            noisy_count, interval=interval, device=noisy_latents.device
        )
        if data.get("mosaic_interval_applied"):
            return data
        if interval <= 1:
            # mosaic_interval=1 is the identity selection. Skip
            # ``index_select`` so we don't allocate an extra copy of the
            # already-correct ``queried_latent`` and ``ql_rope_indice_pef``
            # on every step. Still publish the metadata so downstream code
            # paths (pipeline, preview, debug) can treat the case uniformly.
            data["mosaic_frame_indices"] = indices.detach().cpu()
            data["mosaic_noisy_count"] = noisy_count
            data["mosaic_interval"] = 1
            data["mosaic_interval_applied"] = True
            if "ql_view_change_pef" in data:
                data["ql_view_change_pef_full_shape"] = tuple(
                    data["ql_view_change_pef"].shape
                )
            if self.debug:
                print(
                    f"[mosaic_interval:{phase}] interval=1 (no-op) "
                    f"N={noisy_count} M={int(indices.numel())}"
                )
            return data
        full_queried = data["queried_latent"]
        full_revgrid = data["ql_rope_indice_pef"]
        keep_full_for_preview = bool(
            getattr(self, "preview_dataset", False)
            and not getattr(self, "_preview_done", False)
            and int(getattr(self, "preview_num", 0))
            < int(getattr(self, "num_preview", 0))
        )
        # Stash the unsliced versions only for dataset preview. Keeping the
        # dense GPU mosaic in normal training defeats the interval sparsity
        # memory win because the full tensor stays alive until backward.
        if keep_full_for_preview:
            data["queried_latent_full"] = full_queried
            data["ql_rope_indice_pef_full"] = full_revgrid
        full_view_change = data.get("ql_view_change_pef")
        if full_view_change is not None and keep_full_for_preview:
            data["ql_view_change_pef_full"] = full_view_change
        data["mosaic_frame_indices"] = indices.detach().cpu()
        data["mosaic_noisy_count"] = noisy_count
        data["mosaic_interval"] = interval
        data["queried_latent_full_shape"] = tuple(full_queried.shape)
        data["ql_rope_indice_pef_full_shape"] = tuple(full_revgrid.shape)
        data["queried_latent"] = self._index_latent_time(full_queried, indices)
        data["ql_rope_indice_pef"] = self._index_revgrid_time(full_revgrid, indices)
        if full_view_change is not None:
            data["ql_view_change_pef_full_shape"] = tuple(full_view_change.shape)
            data["ql_view_change_pef"] = self._index_view_change_time(
                full_view_change, indices
            )
        data["mosaic_interval_applied"] = True
        if self.debug:
            print(
                f"[mosaic_interval:{phase}] interval={interval} "
                f"N={noisy_count} M={int(indices.numel())} "
                f"indices={indices.detach().cpu().tolist()} "
                f"queried {data['queried_latent_full_shape']} -> {tuple(data['queried_latent'].shape)} "
                f"revgrid {data['ql_rope_indice_pef_full_shape']} -> {tuple(data['ql_rope_indice_pef'].shape)}"
            )
        return data

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input in ("reference_image", "vace_reference_image"):
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared

    def get_pipeline_inputs(self, data):
        prompt = _train_prompt_from_data(
            data,
            phase="train",
            allow_empty=self.allow_no_prompt,
        )
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        if "latents" in data:
            input_latents = self._ensure_batched_latents(data["latents"])
            first_frame_latents = self._ensure_batched_latents(data["clean_latents"])
            inputs_shared = {
                "input_video": None,
                "input_latents": input_latents,
                "first_frame_latents": first_frame_latents,
                "height": input_latents.shape[-2] * 16,
                "width": input_latents.shape[-1] * 16,
                "num_frames": 4 * (input_latents.shape[2] - 1) + 1,
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "fuse_vae_embedding_in_latents": True,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }
            # Per-step schedule values take priority; fall back to the
            # instance-level config when the schedule is disabled.
            effective_interval = int(
                data.get("_mosaic_step_interval", self.mosaic_interval) or 1
            )
            effective_drop_holes = bool(
                data.get(
                    "_mosaic_drop_holes_effective",
                    getattr(self, "mosaic_drop_holes", False),
                )
            )
            if self.debug or get_timing_reporter(self).enabled:
                inputs_shared["mosaic_sequence_debug"] = {
                    "status": "pending",
                    "drop_holes_enabled": effective_drop_holes,
                    "mosaic_interval": effective_interval,
                    "sequence_balance_mode": str(
                        getattr(self, "mosaic_sequence_balance_mode", "profile_only")
                    ),
                }
            if (
                getattr(self, "enable_mosaic", True)
                and not getattr(self, "only_prope", False)
                and not bool(data.get("_no_mosaic_this_step"))
            ):
                if not data.get("mosaic_interval_applied"):
                    noisy_count = int(input_latents.shape[2])
                    indices = self._mosaic_frame_indices_for_noisy(
                        noisy_count,
                        interval=effective_interval,
                        device=input_latents.device,
                    )
                    queried_raw = data["queried_latent"]
                    queried_t = int(
                        queried_raw.shape[2]
                        if queried_raw.ndim == 5
                        else queried_raw.shape[1]
                    )
                    if queried_t == noisy_count:
                        data = self._apply_mosaic_interval_to_data(
                            data, input_latents, phase="get_pipeline_inputs"
                        )
                    elif queried_t == int(indices.numel()):
                        data["mosaic_frame_indices"] = indices.detach().cpu()
                        data["mosaic_interval_applied"] = True
                    else:
                        raise ValueError(
                            "queried_latent temporal length must be either full noisy "
                            f"length {noisy_count} or sparse mosaic length {int(indices.numel())}; "
                            f"got {queried_t}."
                        )
                data = self._refresh_train_clean_history_rope_indices(
                    data, input_latents, include_mosaic=True
                )
                mosaic_latent = self._ensure_batched_latents(data["queried_latent"])
                mosaic_revgrid = data["ql_rope_indice_pef"]
                if torch.is_tensor(mosaic_revgrid):
                    mosaic_revgrid = mosaic_revgrid.detach().cpu().numpy()
                inputs_shared.update(
                    {
                        "mosaic_latent": mosaic_latent,
                        "mosaic_timestep_zero": True,
                        "mosaic_revgrid": mosaic_revgrid,
                        "mosaic_use_revgrid_rope": getattr(
                            self, "mosaic_use_revgrid_rope", False
                        ),
                        "mosaic_frame_indices": data.get("mosaic_frame_indices"),
                        "mosaic_debug": self.debug,
                        "mosaic_drop_holes": effective_drop_holes,
                    }
                )
                if getattr(self, "mosaic_view_change_prope", False):
                    if "ql_view_change_pef" not in data:
                        raise ValueError(
                            "ql_view_change_pef is required when "
                            "mosaic_view_change_prope is enabled."
                        )
                    inputs_shared["mosaic_view_change"] = data["ql_view_change_pef"]
                    inputs_shared["mosaic_view_change_prope"] = True
            else:
                data = self._refresh_train_clean_history_rope_indices(
                    data, input_latents, include_mosaic=False
                )
            if "latent_rope_time_indices" in data:
                inputs_shared["latent_rope_time_indices"] = data[
                    "latent_rope_time_indices"
                ]
                if self.debug:
                    rope_time_indices = inputs_shared["latent_rope_time_indices"]
                    if torch.is_tensor(rope_time_indices):
                        rope_time_list = (
                            rope_time_indices.detach().cpu().reshape(-1).tolist()
                        )
                    else:
                        rope_time_list = (
                            np.asarray(rope_time_indices).reshape(-1).tolist()
                        )
                    print(
                        "[train][clean-history-rope-final] "
                        f"step={self.step_count} "
                        f"include_mosaic={not bool(data.get('_no_mosaic_this_step'))} "
                        f"latent_rope_time_indices={rope_time_list}",
                        flush=True,
                    )
            for prope_key in WAN_VIDEO_PROPE_CAMERA_KEYS:
                if prope_key in data:
                    inputs_shared[prope_key] = data[prope_key]
            if (
                getattr(self, "subject_ref_memory", False)
                and torch.is_tensor(data.get("subject_ref_latents"))
                and int(data["subject_ref_latents"].shape[0]) > 0
            ):
                inputs_shared["subject_ref_latents"] = data["subject_ref_latents"]
                inputs_shared["subject_ref_slot_ratio"] = float(
                    getattr(self, "subject_ref_canvas_slot_ratio", 0.5)
                )
                inputs_shared["subject_ref_time_gap"] = int(
                    getattr(self, "subject_ref_time_gap", 1)
                )
                inputs_shared["subject_ref_prope_mode"] = str(
                    getattr(self, "subject_ref_prope_mode", "identity") or "identity"
                )
            if (
                float(getattr(self, "subject_object_loss_alpha", 0.0)) > 0.0
                and torch.is_tensor(data.get("subject_object_loss_mask"))
            ):
                inputs_shared["subject_object_loss_mask"] = data[
                    "subject_object_loss_mask"
                ]
                inputs_shared["subject_object_loss_alpha"] = float(
                    getattr(self, "subject_object_loss_alpha", 0.0)
                )
                inputs_shared["subject_object_loss_min_mask_ratio"] = float(
                    getattr(self, "subject_object_loss_min_mask_ratio", 0.0)
                )
            return inputs_shared, inputs_posi, inputs_nega

        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def _ensure_batched_latents(latents):
        if latents.ndim == 4:
            return latents.unsqueeze(0)
        if latents.ndim == 5:
            return latents
        raise ValueError(
            f"Expected latent tensor with 4 or 5 dims, got shape {tuple(latents.shape)}."
        )

    def _preview_plan_banner_text(self, data):
        """Summarize the ACTUALLY-sampled per-step modes for the preview video.

        Confirms in-video what the per-step plan / dataset draws resolved
        to: clean-anchor form (single frame vs 4-frame block), history plan
        mode + sampler, dataset-side block draw, mosaic schedule, and the
        history RoPE times. This is the ground truth the model trains on,
        not the configured ratios.
        """
        hist = data.get("train_clean_latent_history_debug") or {}
        dataset_blocks = int(
            data.get("cn_drop_count", hist.get("dataset_blocks", 0)) or 0
        )
        info = data.get("info") or {}
        G = info.get("generating_first_frame_idx")

        if hist.get("enabled"):
            anchor_form = hist.get("anchor_form") or (
                "block" if hist.get("mode") in ("A", "B") else "single"
            )
            if hist.get("plan_sampler") == "anchor_context_v2":
                headline = (
                    f"CLEAN ANCHOR={str(anchor_form).upper()}"
                    + (
                        f" frames={hist.get('block_anchor_frames')}"
                        f" depth={hist.get('anchor_depth')}"
                        if anchor_form == "block"
                        else ""
                    )
                    + f" | ctx={hist.get('context_count_actual', hist.get('context_count'))}"
                    + f" | sampler=v2 p_block={float(hist.get('anchor_block_prob', 0.0)):.2f}"
                    + ("" if hist.get("applied", True) else " (NOT APPLIED)")
                )
            else:
                headline = (
                    f"CLEAN ANCHOR={str(anchor_form).upper()}"
                    f" | history plan: mode={hist.get('mode')}"
                    f" hc={hist.get('history_count')}"
                    f" sampler={hist.get('plan_sampler', 'legacy')}"
                )
                if hist.get("p_single") is not None:
                    headline += f" p_single={float(hist['p_single']):.2f}"
        else:
            anchor_form = "block" if dataset_blocks > 0 else "single"
            headline = (
                f"CLEAN ANCHOR={anchor_form.upper()}"
                " (dataset draw, history plan OFF)"
            )

        details = [f"dataset_blocks={dataset_blocks}", f"G={G}"]
        if hist.get("selected_block_starts"):
            details.append(f"hist_blocks={hist['selected_block_starts']}")
        if hist.get("actual_latent_times"):
            details.append(f"rope_clean_t={hist['actual_latent_times']}")
        if data.get("_no_mosaic_this_step"):
            details.append("mosaic=OFF(this step)")
        elif data.get("_mosaic_step_interval") is not None:
            details.append(f"mosaic=ON interval={int(data['_mosaic_step_interval'])}")
        else:
            details.append(
                f"mosaic={'ON' if getattr(self, 'enable_mosaic', True) else 'OFF'}"
                f" interval={int(getattr(self, 'mosaic_interval', 1) or 1)}"
            )
        fuse_mode = data.get("_resolved_train_fuse_mode")
        if fuse_mode:
            details.append(f"fuse={fuse_mode}")
        lines = [headline, " | ".join(str(item) for item in details)]

        # Context (clean-history pool) debug breakdown -- only when this step
        # actually drew context latents. Surfaces the selection mode, actual
        # vs planned count, the pool/fallback block split (real history vs
        # padding), the context-only RoPE times (anchor excluded) and the
        # source frame blocks the context was encoded from.
        # Gate on the ACTUAL drawn count (matches the camera-pose panel, which
        # also keys off context_count_actual); planned context_count is only
        # used below for the "(plan N)" annotation.
        ctx_actual = hist.get("context_count_actual")
        try:
            has_ctx = int(ctx_actual or 0) > 0
        except (TypeError, ValueError):
            has_ctx = False
        if has_ctx:
            ctx_items = [f"sel={getattr(self, 'train_context_selection', '?')}"]
            planned = hist.get("context_count")
            cnt = f"count={ctx_actual}"
            if planned is not None and planned != ctx_actual:
                cnt += f"(plan {planned})"
            ctx_items.append(cnt)
            pool = hist.get("pool_block_starts")
            fallback = hist.get("fallback_block_starts")
            if pool is not None:
                ctx_items.append(f"pool_blocks={list(pool)}")
            if fallback:
                ctx_items.append(f"fallback_blocks={list(fallback)}")
            alt = hist.get("actual_latent_times")
            if alt and len(alt) > 1:
                ctx_items.append(f"ctx_rope_t={list(alt[:-1])}")
            fblocks = hist.get("frame_blocks")
            if fblocks:
                s = str(fblocks)
                ctx_items.append("frames=" + (s if len(s) <= 80 else s[:77] + "..."))
            lines.append("CONTEXT " + " | ".join(str(x) for x in ctx_items))
        return "\n".join(lines)

    def maybe_preview_dataset(self, data):
        if self._preview_done or not self.preview_dataset:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = int(torch.distributed.get_rank())
        else:
            rank = 0
        self._preview_dataset(
            data,
            special_filename=f"step_{self.step_count:05d}_rank_{rank}_{time.time_ns()}",
        )
        self.preview_num += 1
        if self.preview_num >= self.num_preview:
            self._preview_done = True

    @torch.no_grad()
    def _preview_dataset(self, data, special_filename=None):
        if self.log_dir is None:
            return
        # DA3 video datasets ship raw frames + a fuse plan; materialization
        # fills the actual ``latents`` / ``queried_latent`` tensors before
        # the preview path can render anything.
        if data.get("needs_vae_materialization"):
            data = self._materialize_data(data)
        if "latents" not in data or "queried_latent" not in data:
            return

        preview_dir = os.path.join(self.log_dir, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        name = special_filename or f"preview_{self.preview_num:03d}"
        save_path = os.path.join(preview_dir, f"{name}_mosaic_dataset.mp4")
        subject_preview_path = os.path.join(preview_dir, f"{name}_subject_ref_mask.png")
        subject_vae_source_path = os.path.join(
            preview_dir, f"{name}_subject_context_vae_sources.jpg"
        )

        generating_latents = self._ensure_batched_latents(data["latents"]).to(
            device=self.pipe.device,
            dtype=self.pipe.torch_dtype,
        )
        # Visualization-only: prefer the unsliced ``queried_latent_full``
        # over the mosaic_interval-sparse ``queried_latent`` so the mosaic
        # preview video stays the same length as the generating one.
        # Dropped positions are repainted as black frames after decode.
        queried_source_key = (
            "queried_latent_full" if "queried_latent_full" in data else "queried_latent"
        )
        queried_latents = self._ensure_batched_latents(data[queried_source_key]).to(
            device=self.pipe.device,
            dtype=self.pipe.torch_dtype,
        )
        # Keep the main noisy/mosaic comparison on the usual timeline:
        # one visible clean anchor, then the noisy window. Extra train
        # clean-history latents are rendered as a separate right-side panel
        # below, matching validation's temp-history diagnostic layout.
        clean_latents = self._ensure_batched_latents(data["clean_latents"]).to(
            device=self.pipe.device,
            dtype=self.pipe.torch_dtype,
        )
        clean_anchor_latents = clean_latents[:, :, -1:].contiguous()
        generating_with_cl = torch.cat(
            [clean_anchor_latents, generating_latents], dim=2
        )
        queried_with_cl = torch.cat([clean_anchor_latents, queried_latents], dim=2)

        self.pipe.load_models_to_device(["vae"])
        # See note on _decode_latents_to_numpy_frames for why this is
        # off by default: legacy 30x52 tiles were a no-op at 832x448 but
        # cause 12 redundant decodes at 704x1280.
        decode_kwargs = {"device": self.pipe.device}
        if self.vae_decode_tiled:
            decode_kwargs.update(
                {"tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26)}
            )
        else:
            decode_kwargs["tiled"] = False
        generating_video = self.pipe.vae.decode(generating_with_cl, **decode_kwargs)
        queried_video = self.pipe.vae.decode(queried_with_cl, **decode_kwargs)
        train_clean_history_debug = data.get("train_clean_latent_history_debug") or {}
        train_clean_history_meta = _build_train_clean_history_preview_meta(
            train_clean_history_debug
        )
        generating_frames = self.pipe.vae_output_to_video(generating_video)
        queried_frames = self.pipe.vae_output_to_video(queried_video)
        self.pipe.load_models_to_device([])

        # Mosaic-side visualization-only black-frame padding.
        #
        # ``queried_frames`` was decoded from the unsliced N-length mosaic
        # latent so its frame count matches ``generating_frames`` (which
        # always comes from the full noisy sequence -- noisy is NEVER
        # sampled, anywhere in this codepath). We now repaint the temporal
        # slots that ``mosaic_interval`` DROPS ON THE MOSAIC SIDE with
        # all-black frames so the viewer can see which slots receive no
        # mosaic conditioning at training/inference time.
        #
        # ``mosaic_indices`` are positions on the shared noisy timeline at
        # which the mosaic *keeps real content*; the remaining positions
        # on the mosaic row become black. The noisy row stays untouched.
        mosaic_indices_tensor = data.get("mosaic_frame_indices")
        mosaic_interval = int(data.get("mosaic_interval", self.mosaic_interval) or 1)
        noisy_count_for_mask = int(
            data.get(
                "mosaic_noisy_count",
                (
                    generating_latents.shape[2]
                    if generating_latents.ndim == 5
                    else generating_latents.shape[1]
                ),
            )
        )
        mosaic_used_full = queried_source_key == "queried_latent_full"
        if (
            mosaic_used_full
            and mosaic_interval > 1
            and mosaic_indices_tensor is not None
            and len(mosaic_indices_tensor) > 0
        ):
            mosaic_indices_list = (
                mosaic_indices_tensor.tolist()
                if torch.is_tensor(mosaic_indices_tensor)
                else list(mosaic_indices_tensor)
            )
            mosaic_kept_ranges, mosaic_dropped_ranges = _mosaic_kept_frame_ranges(
                noisy_count_for_mask,
                mosaic_indices_list,
                total_frame_count=len(queried_frames),
            )
            if mosaic_dropped_ranges is None:
                if self.debug:
                    print(
                        "[preview_dataset:mosaic_interval] WARNING: VAE output "
                        f"mosaic_video_frames={len(queried_frames)} does not match "
                        f"expected 4*N+1={4 * noisy_count_for_mask + 1} -- "
                        "skipping mosaic black-frame padding to avoid mis-mapping."
                    )
            else:
                repainted_mosaic_frames = []
                for fi, qf in enumerate(queried_frames):
                    arr = np.asarray(qf)
                    in_kept = any(
                        start <= fi < end for start, end in mosaic_kept_ranges
                    )
                    if not in_kept:
                        arr = np.zeros_like(arr)
                    repainted_mosaic_frames.append(arr)
                queried_frames = repainted_mosaic_frames
                if self.debug:
                    print(
                        "[preview_dataset:mosaic_interval] mosaic-side "
                        "visualization-only black-frame padding applied "
                        "(noisy row is full N length and untouched). "
                        f"interval={mosaic_interval} "
                        f"N={noisy_count_for_mask} M={len(mosaic_indices_list)} "
                        f"mosaic_indices={mosaic_indices_list} "
                        f"mosaic_kept_frame_ranges={mosaic_kept_ranges} "
                        f"mosaic_dropped_frame_ranges={mosaic_dropped_ranges} "
                        f"mosaic_source={queried_source_key} "
                        f"noisy_video_frames={len(generating_frames)} "
                        f"mosaic_video_frames={len(queried_frames)}"
                    )
        elif self.debug:
            print(
                "[preview_dataset:mosaic_interval] no mosaic padding applied "
                f"(interval={mosaic_interval} mosaic_source={queried_source_key} "
                f"mosaic_indices_len="
                f"{None if mosaic_indices_tensor is None else len(mosaic_indices_tensor)} "
                f"noisy_video_frames={len(generating_frames)} "
                f"mosaic_video_frames={len(queried_frames)})"
            )

        # ``zip`` here is now safe because both videos are full N-length.
        # (Sanity check guards against a future VAE that changes the
        # latent->frame mapping.)
        if len(generating_frames) != len(queried_frames):
            print(
                "[preview_dataset] WARNING: noisy_video_frames "
                f"({len(generating_frames)}) and mosaic_video_frames "
                f"({len(queried_frames)}) have different lengths; "
                "preview will be truncated to the shorter one."
            )

        # Vertical stack: noisy (full N) on top, mosaic (full N w/ dropped
        # slots blacked-out for interval>1) on bottom.
        combined_frames = [
            np.concatenate([np.asarray(noisy_frame), np.asarray(mosaic_frame)], axis=0)
            for noisy_frame, mosaic_frame in zip(generating_frames, queried_frames)
        ]

        candidate_diagnostics = None
        if combined_frames:
            combined_h, combined_w = combined_frames[0].shape[:2]
            single_h = combined_h // 2
            single_w = combined_w
            prompt_text = _prompt_text_for_preview(data)
            fused_inputs = data.get("fused_query_inputs") or {}
            candidate_groups = _preview_candidate_groups_from_fused(
                fused_inputs, prefer_original=True
            )
            depth_candidate_groups = _preview_candidate_groups_from_fused(
                fused_inputs, prefer_original=False
            )
            candidate_diagnostics = _build_preview_candidate_diagnostics(
                data,
                preview_frame_count=len(combined_frames),
                candidate_frame_ids=candidate_groups,
            )

            candidate_frame_ids = sorted(
                {
                    int(record["first_candidate_frame_id"])
                    for record in candidate_diagnostics["frames"]
                    if int(record["first_candidate_frame_id"]) >= 0
                }
            )
            candidate_frames_by_id = {}
            info = data.get("info") or {}
            video_path = info.get("video_path")
            if video_path and candidate_frame_ids:
                try:
                    candidate_arr = _read_video_gt_frames(
                        video_path, candidate_frame_ids, single_h, single_w
                    )
                    candidate_frames_by_id = {
                        int(fid): np.asarray(candidate_arr[i])
                        for i, fid in enumerate(candidate_frame_ids)
                    }
                except Exception as exc:
                    print(
                        "[preview_dataset] failed to read candidate memory frames: "
                        f"{exc}"
                    )
            train_clean_history_frames_by_id = {}
            train_clean_history_frame_ids = sorted(
                {
                    int(record["preview_frame_id"])
                    for record in train_clean_history_meta.get("records", [])
                    if int(record.get("preview_frame_id", -1)) >= 0
                }
            )
            if video_path and train_clean_history_frame_ids:
                try:
                    train_clean_history_arr = _read_video_gt_frames(
                        video_path, train_clean_history_frame_ids, single_h, single_w
                    )
                    train_clean_history_frames_by_id = {
                        int(fid): np.asarray(train_clean_history_arr[i])
                        for i, fid in enumerate(train_clean_history_frame_ids)
                    }
                except Exception as exc:
                    print(
                        "[preview_dataset] failed to read train clean-history frames: "
                        f"{exc}"
                    )

            depths = fused_inputs.get("depths")
            w2c = fused_inputs.get("w2c")
            memory_K = fused_inputs.get("memory_K", fused_inputs.get("K"))
            query_K = fused_inputs.get("query_K", fused_inputs.get("K"))
            query_extrinsics = fused_inputs.get("query_extrinsics")
            query_reference_frame = int(
                fused_inputs.get(
                    "query_reference_frame",
                    getattr(self, "mosaic_query_reference_frame", 2),
                )
                or 2
            )
            candidate_rows = [
                _blank_preview_frame(
                    single_h, single_w, "clean anchor: no query candidate"
                )
            ]
            projected_rows = [
                _blank_preview_frame(single_h, single_w, "clean anchor: no projection")
            ]
            projection_cache = {}
            for record in candidate_diagnostics["frames"]:
                group_idx = int(record["query_group_idx"])
                candidate_abs_id = int(record["first_candidate_frame_id"])
                candidate_img = candidate_frames_by_id.get(candidate_abs_id)
                if candidate_img is None:
                    candidate_img = _blank_preview_frame(
                        single_h,
                        single_w,
                        f"candidate frame {candidate_abs_id}: unavailable",
                    )
                else:
                    candidate_img = _draw_preview_label(
                        candidate_img, f"candidate RGB fid={candidate_abs_id}"
                    )
                candidate_rows.append(candidate_img)

                depth_group = (
                    depth_candidate_groups[group_idx]
                    if group_idx < len(depth_candidate_groups)
                    else []
                )
                depth_candidate_idx = _first_valid_candidate(depth_group)
                q_idx = _resolve_group_query_index(
                    group_idx,
                    0 if query_extrinsics is None else len(query_extrinsics),
                    query_reference_frame,
                )
                projected = None
                if (
                    candidate_abs_id >= 0
                    and depth_candidate_idx >= 0
                    and candidate_abs_id in candidate_frames_by_id
                    and depths is not None
                    and w2c is not None
                    and query_extrinsics is not None
                    and q_idx >= 0
                ):
                    depths_arr = np.asarray(depths)
                    w2c_arr = np.asarray(w2c)
                    if depth_candidate_idx < len(
                        depths_arr
                    ) and depth_candidate_idx < len(w2c_arr):
                        try:
                            cache_key = (
                                int(candidate_abs_id),
                                int(depth_candidate_idx),
                                int(q_idx),
                            )
                            projected = projection_cache.get(cache_key)
                            if projected is None:
                                projected = _project_candidate_frame_to_query_preview(
                                    candidate_frames_by_id[candidate_abs_id],
                                    depths_arr[depth_candidate_idx],
                                    memory_K=_resolve_K_for_index(
                                        memory_K, depth_candidate_idx
                                    ),
                                    memory_w2c=w2c_arr[depth_candidate_idx],
                                    query_K=_resolve_K_for_index(query_K, q_idx),
                                    query_w2c=np.asarray(query_extrinsics)[q_idx],
                                    downsample=4,
                                )
                                projection_cache[cache_key] = projected
                        except Exception as exc:
                            print(
                                "[preview_dataset] failed to project candidate "
                                f"{candidate_abs_id}: {exc}"
                            )
                if projected is None:
                    projected = _blank_preview_frame(
                        single_h,
                        single_w,
                        f"projected candidate {candidate_abs_id}: unavailable",
                    )
                else:
                    projected = _draw_preview_label(
                        projected,
                        "candidate depth projection "
                        f"fid={candidate_abs_id} -> qref={record['query_reference_absolute_frame_id']}",
                    )
                projected_rows.append(projected)

            # Left stack: generated / queried preview, first candidate RGB, then
            # candidate depth reprojected into the query-reference camera.
            combined_frames = [
                np.concatenate([frame, candidate_rows[i], projected_rows[i]], axis=0)
                for i, frame in enumerate(combined_frames)
            ]

            # Render the prompt panel at the enlarged height and concatenate it
            # horizontally so the right side grows with the new diagnostic rows.
            combined_h, combined_w = combined_frames[0].shape[:2]
            train_clean_history_panel = (
                _build_train_clean_history_preview_panel(
                    panel_count=len(combined_frames),
                    panel_height=combined_h,
                    panel_width=single_w,
                    preview_frames_by_id=train_clean_history_frames_by_id,
                    meta=train_clean_history_meta,
                )
                if train_clean_history_meta.get("enabled")
                else None
            )
            prompt_panel = _render_prompt_panel(
                prompt_text, height=combined_h, width=combined_w
            )
            # Animated isometric camera-pose panel, split by frame role:
            # context / anchor (from clean prope extrinsics) + noisy (the
            # generating window). Clean = cat([context, anchor]) per video
            # frame (4 per latent block); context occupies the first
            # context_count_actual*4 frames, anchor the rest. Static roles are
            # subsampled to one camera per latent block (cameras within a block
            # are near-identical). None when no camera extrinsics are present.
            def _np_w2c(t):
                if t is None:
                    return None
                if torch.is_tensor(t):
                    t = t.detach().cpu().numpy()
                a = np.asarray(t)
                if a.ndim == 4:  # (B,N,4,4) -> first batch
                    a = a[0]
                if a.ndim != 3 or a.shape[-2:] != (4, 4):
                    return None
                return a

            clean_extr = _np_w2c(data.get("clean_latent_indices_prope_extrinsic"))
            noisy_extr = _np_w2c(data.get("noisy_latent_indices_prope_extrinsic"))
            if noisy_extr is None:
                noisy_extr = _np_w2c(query_extrinsics)
            hist_dbg = data.get("train_clean_latent_history_debug") or {}
            ctx_n = int(hist_dbg.get("context_count_actual", 0) or 0) * 4
            context_extr = anchor_extr = None
            if clean_extr is not None and clean_extr.shape[0] > 0:
                if ctx_n > 0:
                    ctx_n = min(ctx_n, clean_extr.shape[0])
                    context_extr, anchor_extr = clean_extr[:ctx_n], clean_extr[ctx_n:]
                elif clean_extr.shape[0] > 4:
                    context_extr, anchor_extr = clean_extr[:-4], clean_extr[-4:]
                else:
                    anchor_extr = clean_extr
            context_extr = context_extr[::4] if context_extr is not None else None
            anchor_extr = anchor_extr[::4] if anchor_extr is not None else None
            camera_panel = _build_camera_trajectory_panel(
                context_w2c=context_extr,
                anchor_w2c=anchor_extr,
                noisy_w2c=noisy_extr,
                panel_count=len(combined_frames),
                panel_height=combined_h,
                panel_width=single_w,
            )
            # Each side panel is a per-frame list (animated) or a single static
            # image (broadcast). Concatenate them all to the right of the main
            # stack: [main][train-clean-history?][camera-traj?][prompt].
            side_panels = []
            if train_clean_history_panel is not None:
                side_panels.append(train_clean_history_panel)
            if camera_panel is not None:
                side_panels.append(camera_panel)
            side_panels.append([prompt_panel] * len(combined_frames))
            combined_frames = [
                np.concatenate(
                    [combined_frames[i]] + [panel[i] for panel in side_panels],
                    axis=1,
                )
                for i in range(len(combined_frames))
            ]

        frame_labels = _preview_raw_frame_labels(data, len(combined_frames))
        annotated = _annotate_frames_with_index(combined_frames, labels=frame_labels)
        # Prepend the per-step plan banner AFTER index annotation so the
        # banner text never collides with the frame-id label drawn at the
        # video's top-left corner.
        if annotated:
            plan_banner = _render_plan_text_banner(
                self._preview_plan_banner_text(data),
                width=int(np.asarray(annotated[0]).shape[1]),
            )
            annotated = [
                np.concatenate([plan_banner, np.asarray(frame)], axis=0)
                for frame in annotated
            ]
        _write_video(save_path, annotated)
        subject_preview_written = self._write_subject_ref_mask_preview(
            data,
            subject_preview_path,
        )
        subject_vae_source_written = self._write_subject_context_vae_source_preview(
            data,
            subject_vae_source_path,
        )
        if candidate_diagnostics is not None:
            # Per-sample neighbour-window tally (how many of this training
            # sample's candidate memory frames went single-frame S1 vs the
            # 5-frame window W5). Provided by the dataset; only present when
            # --memory_neighbor_window_train is on.
            candidate_diagnostics["memory_neighbor_window_stats"] = data.get(
                "memory_neighbor_window_stats", {"enabled": False}
            )
            candidate_diagnostics["train_clean_latent_history"] = (
                train_clean_history_meta
            )
            candidate_diagnostics["subject_ref_info"] = data.get(
                "subject_ref_info", {}
            )
            candidate_diagnostics["subject_context_patch_mask_info"] = data.get(
                "subject_context_patch_mask_info", {}
            )
            candidate_diagnostics["subject_object_loss_mask_info"] = data.get(
                "subject_object_loss_mask_info", {}
            )
            candidate_diagnostics["subject_ref_latent_info"] = data.get(
                "subject_ref_latent_info", {}
            )
            candidate_diagnostics["subject_preview_path"] = (
                subject_preview_path if subject_preview_written else ""
            )
            candidate_diagnostics["subject_context_vae_source_preview_path"] = (
                subject_vae_source_path if subject_vae_source_written else ""
            )
            diagnostic_path = os.path.join(
                preview_dir, f"{name}_preview_diagnostics.json"
            )
            _write_json_atomic(diagnostic_path, candidate_diagnostics)
            # print(
            #     "[preview_dataset] candidate diagnostics: "
            #     + json.dumps(candidate_diagnostics, ensure_ascii=False)
            # ) 这个信息又多又没用，json里面看看得了
            print(f"Saved mosaic dataset preview diagnostics to {diagnostic_path}")
        print(f"Saved mosaic dataset preview to {save_path}")

    @staticmethod
    def _tensor_chw_to_uint8_hwc(frame):
        if not torch.is_tensor(frame):
            frame = torch.as_tensor(frame)
        arr = (
            ((frame.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        return arr

    @staticmethod
    def _mask_to_rgb(mask, *, height, width):
        if mask is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        if torch.is_tensor(mask):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)
        arr = np.asarray(arr, dtype=bool)
        if arr.ndim == 3:
            arr = arr.any(axis=0)
        if arr.shape != (height, width):
            import cv2

            arr = cv2.resize(
                arr.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        out = np.zeros((height, width, 3), dtype=np.uint8)
        out[arr] = np.array([255, 64, 64], dtype=np.uint8)
        return out

    def _decode_subject_masked_context_preview(self, data):
        latents = data.get("subject_context_masked_clean_latents")
        if not torch.is_tensor(latents) or latents.numel() <= 0:
            return None
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            return None
        latents = latents[:, :, :1].contiguous()
        self.pipe.load_models_to_device(["vae"])
        decode_kwargs = {"device": self.pipe.device}
        if self.vae_decode_tiled:
            decode_kwargs.update(
                {"tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26)}
            )
        else:
            decode_kwargs["tiled"] = False
        decoded = self.pipe.vae.decode(
            latents.to(device=self.pipe.device, dtype=self.pipe.torch_dtype),
            **decode_kwargs,
        )
        self.pipe.load_models_to_device([])
        frames = self.pipe.vae_output_to_video(decoded)
        if not frames:
            return None
        return np.asarray(frames[0])

    @staticmethod
    def _overlay_mask_on_preview_frame(frame, mask, *, alpha=0.45):
        arr = np.asarray(frame).astype(np.uint8, copy=True)
        if mask is None:
            return arr
        if torch.is_tensor(mask):
            mask_arr = mask.detach().cpu().numpy()
        else:
            mask_arr = np.asarray(mask)
        mask_arr = np.asarray(mask_arr, dtype=bool)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr.any(axis=0)
        if mask_arr.shape != arr.shape[:2]:
            import cv2

            mask_arr = cv2.resize(
                mask_arr.astype(np.uint8),
                (int(arr.shape[1]), int(arr.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        if mask_arr.any():
            overlay = arr.astype(np.float32)
            color = np.asarray([255.0, 32.0, 32.0], dtype=np.float32)
            overlay[mask_arr] = overlay[mask_arr] * (1.0 - float(alpha)) + color * float(
                alpha
            )
            arr = np.clip(overlay, 0, 255).astype(np.uint8)
        return arr

    def _write_subject_context_vae_source_preview(self, data, save_path):
        info = data.get("info") or {}
        video_path = info.get("video_path")
        if not video_path:
            return False
        mask_info = data.get("subject_context_patch_mask_info") or {}
        rows = []
        anchor_frames = [int(v) for v in mask_info.get("anchor_source_frame_ids") or []]
        anchor_mask = data.get("subject_context_dynamic_patch_mask")
        if anchor_frames:
            rows.append(
                {
                    "name": f"anchor {mask_info.get('anchor_form', '')}".strip(),
                    "frame_ids": anchor_frames,
                    "mask": anchor_mask[0] if torch.is_tensor(anchor_mask) else None,
                }
            )
        pool_windows = data.get("subject_pool_context_frame_windows") or []
        pool_masks = data.get("subject_pool_context_dynamic_patch_mask")
        for idx, window in enumerate(pool_windows):
            mask = None
            if torch.is_tensor(pool_masks) and idx < int(pool_masks.shape[0]):
                mask = pool_masks[idx]
            rows.append(
                {
                    "name": f"pool {idx}",
                    "frame_ids": [int(v) for v in window],
                    "mask": mask,
                }
            )
        if not rows:
            return False

        cn_frames = data.get("cn_frames")
        if torch.is_tensor(cn_frames) and cn_frames.ndim == 4:
            height = int(cn_frames.shape[-2])
            width = int(cn_frames.shape[-1])
        else:
            height = int(getattr(self, "height"))
            width = int(getattr(self, "width"))
        all_frame_ids = sorted({fid for row in rows for fid in row["frame_ids"]})
        if not all_frame_ids:
            return False
        try:
            frames_np = _read_video_gt_frames(video_path, all_frame_ids, height, width)
        except Exception as exc:
            print(
                "[preview_dataset] failed to read subject context VAE source frames: "
                f"{exc}"
            )
            return False
        frames_by_id = {
            int(fid): np.asarray(frames_np[i]) for i, fid in enumerate(all_frame_ids)
        }

        max_cols = max(len(row["frame_ids"]) for row in rows)
        tile_h = max(1, min(220, height // 2 if height >= 2 else height))
        tile_w = max(1, int(round(tile_h * width / max(1, height))))
        row_canvases = []
        for row in rows:
            tiles = []
            for col, fid in enumerate(row["frame_ids"]):
                src = frames_by_id.get(int(fid))
                if src is None:
                    tile = _blank_preview_frame(tile_h, tile_w, f"missing {fid}")
                else:
                    tile = self._overlay_mask_on_preview_frame(src, row.get("mask"))
                    tile = _fit_preview_frame_to_slot(tile, tile_h, tile_w)
                    tile = _draw_preview_label(
                        tile,
                        f"{row['name']} src{col} fid={fid}",
                    )
                tiles.append(tile)
            while len(tiles) < max_cols:
                tiles.append(_blank_preview_frame(tile_h, tile_w, ""))
            row_canvases.append(np.concatenate(tiles, axis=1))
        canvas = np.concatenate(row_canvases, axis=0)
        Image.fromarray(canvas).save(save_path)
        print(f"Saved subject context VAE source preview to {save_path}")
        return True

    def _write_subject_ref_mask_preview(self, data, save_path):
        ref_images = data.get("subject_ref_images")
        patch_mask = data.get("subject_context_patch_mask")
        dynamic_patch_mask = data.get("subject_context_dynamic_patch_mask")
        if not torch.is_tensor(ref_images) and patch_mask is None:
            return False
        info = data.get("info") or {}
        video_path = info.get("video_path")
        context_frame_ids = (
            data.get("subject_context_patch_mask_info") or {}
        ).get("context_frame_ids") or []
        tiles = []
        cn_frames = data.get("cn_frames")
        if torch.is_tensor(cn_frames) and cn_frames.ndim == 4:
            tile_h = int(cn_frames.shape[-2])
            tile_w = int(cn_frames.shape[-1])
        else:
            tile_h = 0
            tile_w = 0
        if torch.is_tensor(ref_images) and ref_images.numel() > 0:
            for idx in range(int(ref_images.shape[0])):
                tile = self._tensor_chw_to_uint8_hwc(ref_images[idx])
                tile = _draw_preview_label(tile, f"ref canvas {idx}")
                tiles.append(tile)
                tile_h, tile_w = tile.shape[:2]
        if video_path and context_frame_ids:
            try:
                context_arr = _read_video_gt_frames(
                    video_path,
                    [int(v) for v in context_frame_ids],
                    tile_h or int(data["cn_frames"].shape[-2]),
                    tile_w or int(data["cn_frames"].shape[-1]),
                )
                if len(context_arr) > 0:
                    tiles.append(
                        _draw_preview_label(
                            np.asarray(context_arr[-1]), "context latest RGB"
                        )
                    )
                    tile_h, tile_w = tiles[-1].shape[:2]
            except Exception as exc:
                print(f"[preview_dataset] failed to read context subject frame: {exc}")
        masked_tile = self._decode_subject_masked_context_preview(data)
        if masked_tile is not None:
            tiles.append(_draw_preview_label(masked_tile, "masked clean anchor"))
            tile_h, tile_w = tiles[-1].shape[:2]
        if patch_mask is not None:
            mask_rgb = self._mask_to_rgb(patch_mask, height=tile_h, width=tile_w)
            tiles.append(_draw_preview_label(mask_rgb, "sampled patch mask"))
        if dynamic_patch_mask is not None:
            dyn_patch_rgb = self._mask_to_rgb(
                dynamic_patch_mask,
                height=tile_h,
                width=tile_w,
            )
            tiles.append(_draw_preview_label(dyn_patch_rgb, "dynamic patch mask"))
        if not tiles:
            return False
        if len({tile.shape[:2] for tile in tiles}) > 1:
            tiles = [
                _fit_preview_frame_to_slot(tile, tile_h, tile_w) for tile in tiles
            ]
        canvas = np.concatenate(tiles, axis=1)
        Image.fromarray(canvas).save(save_path)
        print(f"Saved subject ref/mask preview to {save_path}")
        return True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Video-flow materialization: dataset returns raw frames + fuse plan;
    # we run all VAE encodes here on the GPU rank, then plug the resulting
    # latents into the same dict slots the latent-flow datasets produce.
    # ------------------------------------------------------------------
