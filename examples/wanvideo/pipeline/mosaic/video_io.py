from .common import *
from .history import _candidate_groups_to_lists, _mosaic_kept_frame_ranges


def _write_video(path, frames, fps=16, quality=5, fast_debug=False):
    """Write frames as an MP4 with broadly compatible H.264 encoding.

    Defaults aim for small file size + smooth remote playback:
      - libx264 + yuv420p plays in any browser / VLC / QuickTime.
      - quality=5 maps to roughly CRF 23 (ffmpeg default); quality=8 (the
        previous setting) is much larger with little perceptual gain on
        debug videos.
      - +faststart moves the moov atom to the head so the file streams
        progressively over the network instead of needing a full download.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "pixelformat": "yuv420p",
        "macro_block_size": 1,
        "quality": quality,
        "output_params": ["-movflags", "+faststart"],
    }
    if fast_debug:
        writer_kwargs.update(
            {
                "quality": None,
                "bitrate": "4M",
                "output_params": [
                    "-preset",
                    "ultrafast",
                    "-movflags",
                    "+faststart",
                ],
            }
        )
    writer = imageio.get_writer(
        path,
        **writer_kwargs,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


class _ValidationHistoryTempVideoWriter:
    """Compose readable validation history-temp previews after every section.

    The expensive path used to VAE-decode the full generated history every
    section before passing it into ``_history_verbose_frames``. This cache keeps
    only that generated-history strip as already-decoded frames; the original
    verbose compositor still adds mosaic, candidate, and pool-clean panels.
    """

    def __init__(
        self,
        path,
        fps=16,
        quality=5,
        video_write_fn=None,
        initial_history_frames=None,
        initial_mosaic_history_frames=None,
        initial_candidate_history_frames=None,
        initial_pool_clean_history_frames=None,
    ):
        self.path = str(path)
        self.fps = fps
        self.quality = quality
        self._video_write_fn = video_write_fn or _write_video
        self._history_frames = []
        self._verbose_frames = []
        self.frame_count = 0
        if initial_history_frames is not None:
            self.extend_verbose_frames(
                initial_history_frames,
                mosaic_history_frames=initial_mosaic_history_frames,
                candidate_history_frames=initial_candidate_history_frames,
                pool_clean_history_frames=initial_pool_clean_history_frames,
            )

    def extend_verbose_frames(
        self,
        history_frames,
        *,
        mosaic_history_frames=None,
        candidate_history_frames=None,
        pool_clean_history_frames=None,
    ):
        history_frames = [np.asarray(frame).copy() for frame in history_frames]
        if not history_frames:
            return 0
        if mosaic_history_frames is None:
            mosaic_history_frames = [np.zeros_like(frame) for frame in history_frames]
        verbose_frames = _history_verbose_frames(
            history_frames,
            mosaic_history_frames,
            candidate_history_frames=candidate_history_frames,
            pool_clean_history_frames=pool_clean_history_frames,
            start_index=self.frame_count,
        )
        self._history_frames.extend(history_frames)
        self._verbose_frames.extend(
            [np.asarray(frame).copy() for frame in verbose_frames]
        )
        self.frame_count = len(self._history_frames)
        return len(history_frames)

    def write(self):
        if not self._verbose_frames:
            return 0
        self._video_write_fn(
            self.path,
            self._verbose_frames,
            fps=self.fps,
            quality=self.quality,
            fast_debug=True,
        )
        return self.frame_count

    def append_section_frames_and_write(
        self,
        frames,
        *,
        mosaic_history_frames=None,
        candidate_history_frames=None,
        pool_clean_history_frames=None,
    ):
        appended = self.extend_verbose_frames(
            frames,
            mosaic_history_frames=mosaic_history_frames,
            candidate_history_frames=candidate_history_frames,
            pool_clean_history_frames=pool_clean_history_frames,
        )
        self.write()
        return appended

    @property
    def single_frame_shape(self):
        if not self._history_frames:
            raise RuntimeError("history temp writer has no history frames yet.")
        return np.asarray(self._history_frames[0]).shape

    def close(self):
        return None


def _save_validation_history_latents_pth(path, history_latents):
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(history_latents.detach().to("cpu"), path)
    return path


def _resize_frames_half_even(frames):
    resized = []
    for frame in frames:
        arr = np.asarray(frame)
        h, w = arr.shape[:2]
        target_w = max(2, (w // 2) // 2 * 2)
        target_h = max(2, (h // 2) // 2 * 2)
        resized.append(
            np.asarray(Image.fromarray(arr).resize((target_w, target_h), Image.BICUBIC))
        )
    return resized


def _load_frame_index_font(size=32):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size)
        )
    except Exception:
        return ImageFont.load_default()


def _draw_frame_index_label(frame, text, *, font=None):
    img = Image.fromarray(np.asarray(frame).astype(np.uint8, copy=False)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = font or _load_frame_index_font(32)
    x, y = 8, 6
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), str(text), fill=(0, 0, 0), font=font)
    draw.text((x, y), str(text), fill=(255, 255, 0), font=font)
    return np.asarray(img)


def _candidate_frame_to_uint8_hwc(frame):
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 4:
        if arr.shape[0] in (1, 3, 4):
            # C,T,H,W memory videos: use the second RGB frame when present.
            arr = arr[:, min(1, arr.shape[1] - 1)]
        elif arr.shape[-1] in (1, 3, 4):
            # T,H,W,C memory videos.
            arr = arr[min(1, arr.shape[0] - 1)]
        else:
            arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"candidate frame must be image-like, got shape={arr.shape}.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size and float(finite.min()) >= -1.01 and float(finite.max()) <= 1.01:
            if float(finite.min()) < 0.0:
                arr = (arr + 1.0) * 127.5
            else:
                arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _validation_snapshot_safe_name(value):
    text = str(value or "unknown").strip()
    if not text:
        text = "unknown"
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _cn_frame_to_uint8_hwc(cn_frame):
    if torch.is_tensor(cn_frame):
        cn_frame = cn_frame.detach().cpu()
        if cn_frame.dtype == torch.uint8:
            arr = cn_frame[:3].permute(1, 2, 0).numpy()
            return np.ascontiguousarray(arr)
        cn_frame = cn_frame.float().numpy()
    arr = np.asarray(cn_frame)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(
            "validation episode first-frame snapshot expects one cn frame with "
            f"shape (3,H,W), got {tuple(arr.shape)}."
        )
    arr = np.moveaxis(arr, 0, -1).astype(np.float32, copy=False)
    if arr.size and arr.max() > 2.0:
        arr = arr
    elif arr.size and arr.min() >= 0.0:
        arr = arr * 255.0
    else:
        arr = (arr + 1.0) * 127.5
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _save_validation_episode_first_frame_jpg(
    data,
    validation_temp_dir,
    *,
    epoch_id,
    batch_idx,
    proc_tag,
    clean_name,
    timestamp_ns=None,
):
    if not isinstance(data, dict) or "cn_frames" not in data:
        return None
    cn_frames = data["cn_frames"]
    if not torch.is_tensor(cn_frames):
        raise TypeError(
            "validation episode first-frame snapshot requires tensor cn_frames."
        )
    if cn_frames.ndim != 4:
        raise ValueError(
            "validation episode first-frame snapshot expects cn_frames with "
            f"shape (C,T,H,W), got {tuple(cn_frames.shape)}."
        )
    if int(cn_frames.shape[1]) <= 0:
        raise ValueError("validation episode first-frame snapshot got empty cn_frames.")
    timestamp_ns = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
    frame = _cn_frame_to_uint8_hwc(cn_frames[:, 0])
    safe_clean_name = _validation_snapshot_safe_name(clean_name)
    filename = (
        f"epoch_{int(epoch_id):04d}_{int(batch_idx):03d}_{proc_tag}_{safe_clean_name}"
        f"_first_frame_{timestamp_ns}.jpg"
    )
    os.makedirs(validation_temp_dir, exist_ok=True)
    save_path = os.path.join(validation_temp_dir, filename)
    Image.fromarray(frame).convert("RGB").save(save_path, quality=95)
    return save_path


def _build_validation_candidate_panel(
    candidate_frame_ids,
    *,
    single_frame_shape,
    candidate_frames_by_id,
):
    single_h, single_w = int(single_frame_shape[0]), int(single_frame_shape[1])
    panel_h, panel_w = single_h * 2, single_w
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    tile_h = max(1, single_h // 2)
    tile_w = max(1, single_w // 2)
    cols = max(1, panel_w // tile_w)
    font = _load_frame_index_font(32)

    for rank, raw_fid in enumerate(candidate_frame_ids or []):
        row = int(rank) // cols
        col = int(rank) % cols
        y0 = row * tile_h
        x0 = col * tile_w
        if y0 >= panel_h:
            break
        try:
            fid = int(raw_fid)
        except Exception:
            continue
        if fid < 0 or fid not in candidate_frames_by_id:
            continue
        try:
            frame = _candidate_frame_to_uint8_hwc(candidate_frames_by_id[fid])
        except Exception:
            continue
        tile = np.asarray(
            Image.fromarray(frame).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        )
        y1 = min(y0 + tile_h, panel_h)
        x1 = min(x0 + tile_w, panel_w)
        panel[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
        panel[y0:y1, x0:x1] = _draw_frame_index_label(
            panel[y0:y1, x0:x1], fid, font=font
        )
    return panel


def _build_validation_candidate_history_frames(
    *,
    history_frame_count,
    single_frame_shape,
    candidate_groups,
    candidate_frames_by_id,
    panel_cache=None,
    panel_builder=None,
):
    candidate_groups = _candidate_groups_to_lists(candidate_groups)
    panel_builder = panel_builder or _build_validation_candidate_panel
    panel_cache = {} if panel_cache is None else panel_cache
    frames = []
    blank = np.zeros(
        (int(single_frame_shape[0]) * 2, int(single_frame_shape[1]), 3),
        dtype=np.uint8,
    )
    for frame_idx in range(int(history_frame_count)):
        if frame_idx <= 0:
            frames.append(blank.copy())
            continue
        group_idx = (int(frame_idx) - 1) // 4
        if group_idx >= len(candidate_groups):
            frames.append(blank.copy())
            continue
        candidate_group = tuple(int(fid) for fid in candidate_groups[group_idx])
        if not candidate_group:
            frames.append(blank.copy())
            continue
        cache_key = (
            id(candidate_frames_by_id),
            tuple(int(dim) for dim in single_frame_shape[:2]),
            candidate_group,
        )
        panel = panel_cache.get(cache_key)
        if panel is None:
            panel = panel_builder(
                candidate_group,
                single_frame_shape=single_frame_shape,
                candidate_frames_by_id=candidate_frames_by_id,
            )
            panel_cache[cache_key] = np.asarray(panel).copy()
        frames.append(np.asarray(panel_cache[cache_key]).copy())
    return frames


def _record_validation_candidate_frames(
    candidate_frames_by_id,
    frames,
    *,
    start_index,
):
    if frames is None:
        return candidate_frames_by_id
    for offset, frame in enumerate(frames):
        candidate_frames_by_id[int(start_index) + int(offset)] = (
            _candidate_frame_to_uint8_hwc(frame).copy()
        )
    return candidate_frames_by_id


def _parse_query_hits_with_candidate_frame_ids(
    queried_latent_mosaic_revgrid,
    *,
    return_revgrid,
    return_view_change,
):
    if not isinstance(queried_latent_mosaic_revgrid, tuple):
        return queried_latent_mosaic_revgrid, [], None, None
    output = queried_latent_mosaic_revgrid[0]
    candidate_frame_ids = queried_latent_mosaic_revgrid[1]
    extra_idx = 3
    revgrid = None
    view_change = None
    if return_revgrid:
        revgrid = queried_latent_mosaic_revgrid[extra_idx]
        extra_idx += 1
    if return_view_change:
        view_change = queried_latent_mosaic_revgrid[extra_idx]
    return output, candidate_frame_ids, revgrid, view_change


def _history_verbose_frames(
    history_frames,
    mosaic_history_frames,
    *,
    candidate_history_frames=None,
    pool_clean_history_frames=None,
    start_index=0,
):
    history_frames = _annotate_frames_with_index(
        list(history_frames), start_index=start_index
    )
    mosaic_history_frames = _annotate_frames_with_index(
        list(mosaic_history_frames), start_index=start_index
    )
    combined = []
    for frame_idx, history_frame in enumerate(history_frames):
        history_arr = np.asarray(history_frame)
        if frame_idx < len(mosaic_history_frames):
            mosaic_arr = np.asarray(mosaic_history_frames[frame_idx])
            if mosaic_arr.shape[:2] != history_arr.shape[:2]:
                mosaic_arr = np.asarray(
                    Image.fromarray(mosaic_arr).resize(
                        (history_arr.shape[1], history_arr.shape[0]),
                        Image.Resampling.LANCZOS,
                    )
                )
        else:
            mosaic_arr = np.zeros_like(history_arr)
        left_arr = np.concatenate([history_arr, mosaic_arr], axis=0)
        side_panels = []
        if candidate_history_frames is not None:
            if frame_idx < len(candidate_history_frames):
                candidate_arr = np.asarray(candidate_history_frames[frame_idx])
                if candidate_arr.shape[:2] != left_arr.shape[:2]:
                    candidate_arr = np.asarray(
                        Image.fromarray(candidate_arr).resize(
                            (left_arr.shape[1], left_arr.shape[0]),
                            Image.Resampling.LANCZOS,
                        )
                    )
            else:
                candidate_arr = np.zeros_like(left_arr)
            side_panels.append(candidate_arr)
        if pool_clean_history_frames is not None:
            if frame_idx < len(pool_clean_history_frames):
                pool_clean_arr = np.asarray(pool_clean_history_frames[frame_idx])
                if pool_clean_arr.shape[:2] != left_arr.shape[:2]:
                    pool_clean_arr = np.asarray(
                        Image.fromarray(pool_clean_arr).resize(
                            (left_arr.shape[1], left_arr.shape[0]),
                            Image.Resampling.LANCZOS,
                        )
                    )
            else:
                pool_clean_arr = np.zeros_like(left_arr)
            side_panels.append(pool_clean_arr)
        if not side_panels:
            combined.append(left_arr)
            continue
        combined.append(np.concatenate([left_arr] + side_panels, axis=1))
    return _resize_frames_half_even(combined)


def _build_train_clean_history_preview_meta(debug):
    debug = dict(debug or {})
    block_starts = [int(v) for v in debug.get("selected_block_starts") or []]
    actual_times = [int(v) for v in debug.get("actual_latent_times") or []]
    frame_blocks = debug.get("frame_blocks") or []
    single_anchor = str(debug.get("anchor_form") or "") == "single"
    # per_3 only: which generating-latent group each CONTEXT block covers
    # (anchor excluded), aligned to selected_block_starts[:-1]. Empty/None for
    # other selection modes.
    context_group_indices = list(debug.get("context_group_indices") or [])
    # per_3 only: per CONTEXT slot, whether it was filled with a RECENT block
    # because its group found no coverage pick (still labelled with its group).
    context_is_recent = list(debug.get("context_is_recent") or [])
    records = []
    for idx, block_start in enumerate(block_starts):
        is_anchor_slot = idx == len(block_starts) - 1
        per3_group = (
            context_group_indices[idx]
            if (not is_anchor_slot and idx < len(context_group_indices))
            else None
        )
        per3_recent = (
            bool(context_is_recent[idx])
            if (not is_anchor_slot and idx < len(context_is_recent))
            else False
        )
        block = frame_blocks[idx] if idx < len(frame_blocks) else None
        if block:
            block = [int(v) for v in block]
        else:
            block = [block_start + offset for offset in range(4)]
        if is_anchor_slot and single_anchor:
            # v2 (single anchor + context): the anchor latent is the
            # dataset's SINGLE frame G-1, not the [G-4..G-1] block the
            # selection machinery lists for camera bookkeeping -- render
            # it as the one frame it actually encodes.
            block = [block[-1]]
            preview_frame_id = block[0]
        else:
            preview_frame_id = block[min(1, len(block) - 1)]
        latent_frame_id = int(actual_times[idx]) if idx < len(actual_times) else None
        records.append(
            {
                "slot": int(idx),
                "raw_frame_block_start": int(block[0]),
                "raw_frame_block": block,
                "preview_frame_id": int(preview_frame_id),
                "latent_frame_id": latent_frame_id,
                "selected_raw_frame_id": int(preview_frame_id),
                "selected_latent_frame_id": latent_frame_id,
                "is_anchor": bool(is_anchor_slot),
                "per3_group": None if per3_group is None else int(per3_group),
                "per3_recent": bool(per3_recent),
            }
        )
    context_count = debug.get("context_count")
    history_count = (
        int(context_count) + 1
        if context_count is not None
        else int(debug.get("history_count", len(records)) or 0)
    )
    return {
        "enabled": bool(records),
        "mode": debug.get("bucket") or debug.get("mode"),
        "history_count": history_count,
        "records": records,
        "selected_block_starts": block_starts,
        "actual_latent_times": actual_times,
    }


def _draw_train_clean_history_tile_label(frame, text):
    arr = np.asarray(frame).astype(np.uint8, copy=True)
    if arr.shape[0] < 24 or arr.shape[1] < 96:
        return arr
    try:
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        font_size = max(18, min(44, int(min(arr.shape[0], arr.shape[1]) * 0.08)))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                font_size,
            )
        except Exception:
            font = ImageFont.load_default()
        x = max(4, font_size // 4)
        y = max(4, font_size // 4)
        outline = max(1, font_size // 12)
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), str(text), fill=(0, 0, 0), font=font)
        draw.text((x, y), str(text), fill=(255, 255, 0), font=font)
        return np.asarray(img)
    except Exception:
        return arr


def _fit_preview_frame_to_slot(src, slot_h, slot_w):
    slot_h = max(1, int(slot_h))
    slot_w = max(1, int(slot_w))
    canvas = np.zeros((slot_h, slot_w, 3), dtype=np.uint8)
    arr = np.asarray(src).astype(np.uint8, copy=False)
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        return canvas
    src_h, src_w = arr.shape[:2]
    scale = min(slot_w / float(src_w), slot_h / float(src_h))
    resized_w = max(1, min(slot_w, int(round(src_w * scale))))
    resized_h = max(1, min(slot_h, int(round(src_h * scale))))
    resized = np.asarray(
        Image.fromarray(arr).resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    )
    y0 = (slot_h - resized_h) // 2
    x0 = (slot_w - resized_w) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def _build_train_clean_history_preview_panel(
    *,
    panel_count,
    panel_height,
    panel_width,
    preview_frames_by_id,
    meta,
):
    meta = meta or {}
    records = list(meta.get("records") or [])
    title = (
        f"clean pool mode={meta.get('mode')} "
        f"history_count={meta.get('history_count')}"
    )
    frames = [
        _blank_preview_frame(panel_height, panel_width, title)
        for _ in range(max(0, int(panel_count)))
    ]
    if not frames:
        return frames
    panel_height = int(panel_height)
    panel_width = int(panel_width)
    # Match validation temp-history style: the side panel has the height of
    # four vertically stacked preview rows. Each selected clean-history frame
    # is rendered at half source width/height, giving 2x2 slots per row and
    # 16 slots total without aspect-ratio distortion.
    slot_rows = 8
    slot_cols = 2
    slot_h = max(1, panel_height // slot_rows)
    slot_w = max(1, panel_width // slot_cols)
    preview_frames_by_id = preview_frames_by_id or {}
    for panel_idx in range(len(frames)):
        panel = np.asarray(
            _blank_preview_frame(panel_height, panel_width, title)
        ).copy()
        for rec in records[: slot_rows * slot_cols]:
            slot = int(rec.get("slot", 0))
            row_block = slot // 4
            inner_slot = slot % 4
            row = row_block * 2 + inner_slot // 2
            col = inner_slot % 2
            y0 = row * slot_h
            x0 = col * slot_w
            y1 = (
                panel_height if row == slot_rows - 1 else min(panel_height, y0 + slot_h)
            )
            x1 = panel_width if col == slot_cols - 1 else min(panel_width, x0 + slot_w)
            if y1 <= y0 or x1 <= x0:
                continue
            preview_frame_id = int(rec.get("preview_frame_id", -1))
            src = preview_frames_by_id.get(preview_frame_id)
            is_anchor = bool(rec.get("is_anchor"))
            per3_group = rec.get("per3_group")
            lat = rec.get("selected_latent_frame_id")
            if is_anchor:
                # The always-present bridge latent (latest clean block / single
                # G-1 frame); NOT one of the per_3 context picks.
                label = f"ANCHOR raw2={preview_frame_id} lat={lat}"
            elif per3_group is not None:
                # per_3: slot for generating-latent group g (== gen latents
                # [3g, 3g+2]). Either the coverage pick for that group, or a
                # RECENT-block fill when the group found no distinct pool latent.
                g = int(per3_group)
                if bool(rec.get("per3_recent")):
                    label = (
                        f"cg{g} RECENT genL{3 * g}-{3 * g + 2} "
                        f"raw2={preview_frame_id} lat={lat}"
                    )
                else:
                    label = (
                        f"cg{g} genL{3 * g}-{3 * g + 2} "
                        f"raw2={preview_frame_id} lat={lat}"
                    )
            else:
                label = (
                    f"hc={meta.get('history_count')} "
                    f"raw2={preview_frame_id} lat={lat}"
                )
            if src is None:
                # Build the blank tile WITHOUT its own caption and fold the
                # "MISSING <id>" prefix into the single label below; drawing both
                # would stamp two captions in the same top-left corner and overlap
                # into an unreadable smear.
                tile = _blank_preview_frame(y1 - y0, x1 - x0)
                label = f"MISSING {preview_frame_id} | {label}"
            else:
                tile = _fit_preview_frame_to_slot(src, y1 - y0, x1 - x0)
            tile = _draw_train_clean_history_tile_label(tile, label)
            panel[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
            if is_anchor:
                # Blue frame (RGB) so the anchor reads at a glance, distinct
                # from the context tiles.
                bt = max(2, min(y1 - y0, x1 - x0) // 20)
                panel[y0:y1, x0 : min(x1, x0 + bt)] = (0, 0, 255)
                panel[y0:y1, max(x0, x1 - bt) : x1] = (0, 0, 255)
                panel[y0 : min(y1, y0 + bt), x0:x1] = (0, 0, 255)
                panel[max(y0, y1 - bt) : y1, x0:x1] = (0, 0, 255)
        frames[panel_idx] = panel
    return frames


def _render_plan_text_banner(text, width, *, min_height=56):
    """Render a horizontal text banner strip for the dataset preview video.

    ``text`` may contain newlines; each line is drawn left-aligned in a
    bright color on black so the actually-sampled per-step modes (clean
    anchor form, history plan, mosaic schedule, ...) are readable straight
    from the saved mp4.
    """
    width = max(1, int(width))
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return np.zeros((int(min_height), width, 3), dtype=np.uint8)

    font_size = max(16, min(40, width // 48))
    line_step = int(font_size * 1.35)
    pad = max(6, font_size // 3)
    height = max(int(min_height), pad * 2 + line_step * len(lines))
    # Keep the strip height even so prepending it preserves the frame's
    # height parity for yuv420p video encoders.
    height += height % 2
    panel = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(panel)
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    for line_idx, line in enumerate(lines):
        y = pad + line_idx * line_step
        # First line (the headline with the sampled modes) pops in green;
        # detail lines in white.
        fill = (80, 255, 80) if line_idx == 0 else (235, 235, 235)
        draw.text((pad, y), line, fill=fill, font=font)
    return np.asarray(panel)


def _render_prompt_panel(prompt, height, width, font_size=None, padding=None):
    """Render ``prompt`` as a large, pixel-wrapped text panel for previews."""
    height = int(height)
    width = int(width)
    padding = int(padding) if padding is not None else max(24, min(width, height) // 28)
    panel = Image.new("RGB", (width, height), color=(0, 0, 0))
    if not prompt:
        return np.asarray(panel)

    draw = ImageDraw.Draw(panel)
    prompt = str(prompt).strip()
    usable_w = max(width - 2 * padding, 1)
    usable_h = max(height - 2 * padding, 1)
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    def _load_font(size):
        for path in font_paths:
            try:
                return ImageFont.truetype(path, int(size))
            except Exception:
                continue
        return ImageFont.load_default()

    def _text_width(text, font):
        if not text:
            return 0
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return int(bbox[2] - bbox[0])
        except Exception:
            return int(draw.textlength(text, font=font))

    def _font_line_height(font, size):
        try:
            bbox = font.getbbox("Ag")
            text_h = int(bbox[3] - bbox[1])
        except Exception:
            text_h = int(size)
        return max(int(size * 1.22), text_h + max(4, int(size * 0.18)))

    def _wrap_paragraph(paragraph, font):
        lines = []
        current = ""
        for char in paragraph:
            trial = current + char
            if not current or _text_width(trial.rstrip(), font) <= usable_w:
                current = trial
                continue

            break_at = max(current.rfind(" "), current.rfind("\t"))
            if break_at > 0:
                head = current[:break_at].rstrip()
                tail = current[break_at + 1 :].lstrip() + char
                if head:
                    lines.append(head)
                current = tail
            else:
                lines.append(current.rstrip())
                current = char.lstrip()
        if current.strip():
            lines.append(current.rstrip())
        return lines or [""]

    def _layout(size):
        font = _load_font(size)
        lines = []
        for paragraph in prompt.splitlines():
            if paragraph.strip():
                lines.extend(_wrap_paragraph(paragraph, font))
            else:
                lines.append("")
        line_h = _font_line_height(font, size)
        text_h = max(0, len(lines) * line_h - max(1, int(size * 0.12)))
        max_line_w = max((_text_width(line, font) for line in lines), default=0)
        return font, lines, line_h, text_h, max_line_w

    min_font_size = 20
    max_font_size = (
        int(font_size)
        if font_size is not None
        else min(220, max(48, min(width, height) // 3))
    )
    chosen = None
    for size in range(max_font_size, min_font_size - 1, -2):
        layout = _layout(size)
        if layout[3] <= usable_h and layout[4] <= usable_w:
            chosen = (size, *layout)
            break
    if chosen is None:
        size = min_font_size
        chosen = (size, *_layout(size))

    size, font, lines, line_h, text_h, _ = chosen
    max_lines = max(1, usable_h // max(1, line_h))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = (lines[-1].rstrip(" .") + " ...") if lines[-1] else "..."
        text_h = max(0, len(lines) * line_h - max(1, int(size * 0.12)))

    y = padding + max(0, (usable_h - text_h) // 2)
    for line in lines:
        line_w = _text_width(line, font)
        # Short prompts look less lost when centered; multi-line captions stay
        # left-aligned for easier reading.
        x = padding
        if len(lines) <= 2 and line_w < usable_w:
            x = padding + (usable_w - line_w) // 2
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    return np.asarray(panel)


def _write_json_atomic(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _camera_axes_from_w2c(w2c):
    """world-to-camera (N,4,4) -> (centers, right, down, forward), all (N,3)
    in world coords. Camera axis i in world = i-th row of R (OpenCV: +X right,
    +Y down, +Z forward)."""
    w = np.asarray(w2c, dtype=np.float64)
    if w.ndim == 2:
        w = w[None]
    if w.ndim != 3 or w.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4) w2c, got {tuple(w.shape)}")
    R = w[:, :3, :3]
    t = w[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)
    return centers, R[:, 0, :], R[:, 1, :], R[:, 2, :]


def _nice_tick_step(extent, target_ticks=8):
    """Round a span to a human-friendly grid step in {1,2,5}*10^k."""
    if not np.isfinite(extent) or extent <= 0:
        return 1.0
    raw = extent / max(1, target_ticks)
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


CAMERA_ROLE_COLORS = {
    "noisy": (90, 230, 120),    # green (the generating window; path is a gradient)
    "context": (0, 210, 255),   # cyan
    "anchor": (255, 90, 220),   # magenta
}


def _build_camera_trajectory_panel(
    context_w2c,
    anchor_w2c,
    noisy_w2c,
    panel_count,
    panel_height,
    panel_width,
):
    """Animated isometric ("StarCraft"-style oblique) camera-pose panel.

    Visualizes the camera poses of three frame roles in distinct colors on a
    ticked ground grid (see ``CAMERA_ROLE_COLORS``): NOISY (the generating
    window) as a green->yellow time-gradient path with a bright white current
    frustum; CONTEXT (cyan) and ANCHOR (magenta) as static colored frustums.
    Each camera is a small frustum (apex = optical center, cone = view
    direction). Returns a list of ``panel_count`` (H,W,3) uint8 frames, or
    ``None`` when no usable camera pose is available (caller skips the panel).
    """
    H, W = int(panel_height), int(panel_width)
    n = int(panel_count)
    if n <= 0:
        return None

    # Parse each role -> (name, centers, right, down, forward); skip empties.
    roles = []
    for name, w in (("noisy", noisy_w2c), ("context", context_w2c),
                    ("anchor", anchor_w2c)):
        if w is None:
            continue
        try:
            C, rgt, dwn, fwd = _camera_axes_from_w2c(w)
        except Exception:
            continue
        if C.shape[0] == 0 or not np.isfinite(C).all():
            continue
        roles.append((name, C, rgt, dwn, fwd))
    if not roles:
        return None
    role_idx = {r[0]: i for i, r in enumerate(roles)}
    # Animated trajectory = noisy if present, else the first available role.
    traj_name = "noisy" if "noisy" in role_idx else roles[0][0]
    Ctraj = roles[role_idx[traj_name]][1]
    allC = np.concatenate([r[1] for r in roles], 0)

    # World axis with least spread ≈ vertical -> the other two form the ground
    # plane. Keeps ticks in true world units while auto-picking a plane.
    spread = allC.max(0) - allC.min(0)
    up_axis = int(np.argmin(spread))
    g0, g1 = [a for a in (0, 1, 2) if a != up_axis]
    ground_w = float(allC[:, up_axis].min())

    # Isometric basis (screen y points down). Two INDEPENDENT display-convention
    # knobs reconcile the dataset's axes with this Y-up right-handed viewer:
    #   VERT_DOWN     : up-axis renders downward (dataset is Y-down / OpenCV).
    #   GROUND_MIRROR : horizontally mirror the ground plane (negates screen-x
    #                   only). NOT the same flip as VERT_DOWN.
    # Flip either knob if a dataset renders inverted/mirrored.
    VERT_DOWN = True
    GROUND_MIRROR = True
    EU = np.array([(-0.866 if GROUND_MIRROR else 0.866), 0.5])
    EV = np.array([(0.866 if GROUND_MIRROR else -0.866), 0.5])
    EW = np.array([0.0, (0.9 if VERT_DOWN else -0.9)])

    def proj(P):
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        u, v, w = P[:, g0], P[:, g1], P[:, up_axis] - ground_w
        return u[:, None] * EU + v[:, None] * EV + w[:, None] * EW  # (N,2)

    umin, umax = float(allC[:, g0].min()), float(allC[:, g0].max())
    vmin, vmax = float(allC[:, g1].min()), float(allC[:, g1].max())
    du, dv = max(umax - umin, 1e-3), max(vmax - vmin, 1e-3)
    umin -= 0.15 * du; umax += 0.15 * du
    vmin -= 0.15 * dv; vmax += 0.15 * dv
    step = _nice_tick_step(max(umax - umin, vmax - vmin), target_ticks=6)
    u0 = float(np.floor(umin / step)) * step; u1 = float(np.ceil(umax / step)) * step
    v0 = float(np.floor(vmin / step)) * step; v1 = float(np.ceil(vmax / step)) * step
    u_lines = np.arange(u0, u1 + 0.5 * step, step)
    v_lines = np.arange(v0, v1 + 0.5 * step, step)

    # Tick label decimals scale with the grid step so sub-unit grids do not
    # collapse to a run of "0,0,0" / "-1,-1,-1"; normalize "-0" -> "0".
    dec = 0 if step >= 1 else max(0, min(4, int(np.ceil(-np.log10(step) - 1e-9))))

    def fmt_tick(val):
        s = f"{val:.{dec}f}"
        return f"{0.0:.{dec}f}" if float(s) == 0.0 else s

    def world(uu, vv, ww=None):
        P = np.zeros(3)
        P[g0], P[g1] = uu, vv
        P[up_axis] = ground_w if ww is None else ww
        return P

    grid_corners = np.stack([world(a, b) for a in (u0, u1) for b in (v0, v1)])
    extent = np.concatenate([grid_corners, allC], 0)
    S = proj(extent)
    sx0, sx1 = float(S[:, 0].min()), float(S[:, 0].max())
    sy0, sy1 = float(S[:, 1].min()), float(S[:, 1].max())
    margin = 0.1
    scale = min(W * (1 - 2 * margin) / max(sx1 - sx0, 1e-3),
                H * (1 - 2 * margin) / max(sy1 - sy0, 1e-3))
    off_x = W * 0.5 - 0.5 * (sx0 + sx1) * scale
    off_y = H * 0.5 - 0.5 * (sy0 + sy1) * scale

    def to_px(P):
        s = proj(P)
        return np.stack([s[:, 0] * scale + off_x, s[:, 1] * scale + off_y], 1)

    def frustum_px(C, rgt, dwn, fwd, depth):
        hw, hh = 0.42 * depth, 0.30 * depth
        far = C + fwd * depth
        corners = np.stack([
            far - rgt * hw - dwn * hh, far + rgt * hw - dwn * hh,
            far + rgt * hw + dwn * hh, far - rgt * hw + dwn * hh,
        ], 1)  # (N,4,3)
        pts = np.concatenate([C[:, None, :], corners], 1)  # (N,5,3)
        return to_px(pts.reshape(-1, 3)).reshape(pts.shape[0], 5, 2)

    # Pre-project centers + frustums per role.
    role_px = {}
    for name, C, rgt, dwn, fwd in roles:
        depth = (0.6 if name == traj_name else 0.5) * step
        role_px[name] = (to_px(C), frustum_px(C, rgt, dwn, fwd, depth))
    Ctraj_px = role_px[traj_name][0]

    BG, GRID, TICK = (14, 18, 30), (44, 72, 92), (185, 205, 220)

    def grad(a):  # noisy path over time: green -> yellow
        a = float(np.clip(a, 0.0, 1.0))
        return (int(60 + a * 195), int(205 - a * 35), int(95 - a * 65))

    def dim(c, f=0.5):
        return tuple(int(x * f) for x in c)

    def _font(sz):
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(p, int(sz))
            except Exception:
                continue
        return ImageFont.load_default()

    tick_font = _font(max(20, W // 16))
    title_font = _font(max(17, W // 22))
    foot_font = _font(max(13, W // 30))

    def text_o(d, xy, s, font, fill, anchor=None):
        x, y = xy
        for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            d.text((x + ox, y + oy), s, fill=(0, 0, 0), font=font, anchor=anchor)
        d.text((x, y), s, fill=fill, font=font, anchor=anchor)

    def gp(uu, vv):
        return tuple(to_px(world(uu, vv))[0])

    def draw_frustum(d, pts5, color, width):
        apex = tuple(pts5[0])
        corners = [tuple(p) for p in pts5[1:]]
        for c in corners:
            d.line([apex, c], fill=color, width=width)
        d.line(corners + [corners[0]], fill=color, width=width)

    tlast = max(1, Ctraj.shape[0] - 1)
    sparse_idx = sorted(set(int(round(t)) for t in
                            np.linspace(0, tlast, min(Ctraj.shape[0], 8))))
    legend_y = 8 + int(getattr(title_font, "size", 24) * 1.35)

    frames = []
    for f in range(n):
        idx = 0 if Ctraj.shape[0] <= 1 else int(round(f / max(1, n - 1) * tlast))
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for uu in u_lines:
            d.line([gp(uu, v0), gp(uu, v1)], fill=GRID, width=1)
        for vv in v_lines:
            d.line([gp(u0, vv), gp(u1, vv)], fill=GRID, width=1)
        for uu in u_lines:
            px, py = gp(uu, v1)
            text_o(d, (px, py + 4), fmt_tick(uu), tick_font, TICK, anchor="ma")
        for vv in v_lines:
            px, py = gp(u1, vv)
            text_o(d, (px + 6, py), fmt_tick(vv), tick_font, TICK, anchor="lm")

        # static roles: solid colored frustums (context cyan, anchor magenta)
        for name in ("context", "anchor"):
            if name not in role_px:
                continue
            for fp in role_px[name][1]:
                draw_frustum(d, fp, CAMERA_ROLE_COLORS[name], 2)

        # animated trajectory (noisy): gradient polyline + sparse orientation
        # frustums + bright current frustum/marker
        traj_frus = role_px[traj_name][1]
        if Ctraj.shape[0] >= 2:
            d.line([tuple(p) for p in Ctraj_px], fill=(58, 78, 98), width=2)
            for i in range(1, idx + 1):
                d.line([tuple(Ctraj_px[i - 1]), tuple(Ctraj_px[i])],
                       fill=grad(i / tlast), width=3)
        for si in sparse_idx:
            if si != idx:
                draw_frustum(d, traj_frus[si], dim(grad(si / tlast)), 1)
        draw_frustum(d, traj_frus[idx], (255, 255, 255), 3)
        mx, my = Ctraj_px[idx]
        d.ellipse([mx - 5, my - 5, mx + 5, my + 5],
                  fill=(255, 64, 64), outline=(255, 255, 255), width=2)

        # title + color legend (one entry per available role) + footer
        text_o(d, (8, 6), "camera poses (iso)", title_font, (230, 230, 255))
        lx = 8
        for name in ("noisy", "context", "anchor"):
            if name not in role_px:
                continue
            label = f"{name}:{role_px[name][0].shape[0]}"
            text_o(d, (lx, legend_y), label, foot_font, CAMERA_ROLE_COLORS[name])
            try:
                lx += int(d.textlength(label, font=foot_font)) + 16
            except Exception:
                lx += len(label) * 9 + 16
        text_o(d, (8, H - 24), f"grid step={step:g}  up=axis{up_axis}",
               foot_font, TICK)
        frames.append(np.asarray(img))
    return frames


def _preview_query_reference_offset(query_reference_frame):
    ref = int(query_reference_frame or 2)
    if ref < 1 or ref > 4:
        ref = 2
    return ref - 1


def _first_valid_candidate(frame_ids):
    if frame_ids is None:
        return -1
    if torch.is_tensor(frame_ids):
        frame_ids = frame_ids.detach().cpu().tolist()
    elif isinstance(frame_ids, np.ndarray):
        frame_ids = frame_ids.tolist()
    for fid in frame_ids:
        fid = int(fid)
        if fid >= 0:
            return fid
    return -1


def _preview_candidate_groups_from_fused(fused, prefer_original=False):
    if prefer_original:
        candidate_groups = fused.get("candidate_original_frame_ids")
        if candidate_groups is not None:
            return _candidate_groups_to_lists(candidate_groups)
    return _candidate_groups_to_lists(fused.get("candidate_frame_ids"))


def _query_frame_indices_for_preview(data, preview_frame_count):
    fused = data.get("fused_query_inputs") or {}
    explicit = fused.get("query_frame_indices")
    if explicit is not None:
        return [int(x) for x in explicit]
    info = data.get("info") or {}
    start = int(info.get("generating_first_frame_idx", 0))
    return [start + i for i in range(max(0, int(preview_frame_count) - 1))]


def _build_preview_candidate_diagnostics(
    data,
    *,
    preview_frame_count,
    candidate_frame_ids=None,
):
    fused = data.get("fused_query_inputs") or {}
    candidate_groups = (
        _candidate_groups_to_lists(candidate_frame_ids)
        if candidate_frame_ids is not None
        else _preview_candidate_groups_from_fused(fused, prefer_original=True)
    )
    query_reference_frame = int(fused.get("query_reference_frame", 2) or 2)
    ref_offset = _preview_query_reference_offset(query_reference_frame)
    query_frame_indices = _query_frame_indices_for_preview(data, preview_frame_count)
    info = data.get("info") or {}
    frames = []
    for preview_idx in range(1, int(preview_frame_count)):
        query_local_idx = preview_idx - 1
        group_idx = query_local_idx // 4
        group = candidate_groups[group_idx] if group_idx < len(candidate_groups) else []
        ref_local_idx = min(group_idx * 4 + ref_offset, len(query_frame_indices) - 1)
        frames.append(
            {
                "preview_frame_index": int(preview_idx),
                "query_group_idx": int(group_idx),
                "query_local_frame_idx": int(query_local_idx),
                "absolute_query_frame_id": (
                    int(query_frame_indices[query_local_idx])
                    if query_local_idx < len(query_frame_indices)
                    else None
                ),
                "query_reference_frame": int(query_reference_frame),
                "query_reference_absolute_frame_id": (
                    int(query_frame_indices[ref_local_idx])
                    if ref_local_idx >= 0 and query_frame_indices
                    else None
                ),
                "candidate_frame_ids": [int(fid) for fid in group],
                "first_candidate_frame_id": int(_first_valid_candidate(group)),
            }
        )
    return {
        "clean_name": info.get("clean_name"),
        "video_path": info.get("video_path"),
        "prompt_path": info.get("prompt_path"),
        "generating_first_frame_idx": info.get("generating_first_frame_idx"),
        "query_reference_frame": int(query_reference_frame),
        "frames": frames,
    }


def _draw_preview_label(frame, text):
    arr = np.asarray(frame).astype(np.uint8, copy=True)
    try:
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except Exception:
            font = ImageFont.load_default()
        x, y = 8, 8
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + dx, y + dy), str(text), fill=(0, 0, 0), font=font)
        draw.text((x, y), str(text), fill=(255, 255, 0), font=font)
        return np.asarray(img)
    except Exception:
        return arr


def _blank_preview_frame(height, width, text=""):
    frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    return _draw_preview_label(frame, text) if text else frame


def _resolve_group_query_index(group_idx, query_count, query_reference_frame):
    if int(query_count) <= 0:
        return -1
    ref_offset = _preview_query_reference_offset(query_reference_frame)
    return min(int(group_idx) * 4 + ref_offset, int(query_count) - 1)


def _resolve_K_for_index(K, idx):
    if K is None:
        return None
    arr = np.asarray(K, dtype=np.float32)
    if arr.shape == (3, 3):
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (3, 3) and arr.shape[0] > 0:
        return arr[min(max(int(idx), 0), arr.shape[0] - 1)]
    return None


def _project_candidate_frame_to_query_preview(
    candidate_frame,
    candidate_depth,
    *,
    memory_K,
    memory_w2c,
    query_K,
    query_w2c,
    downsample=4,
    splat_radius=2,
):
    projected = _project_candidate_frame_to_query_metrics(
        candidate_frame,
        candidate_depth,
        target_frame=None,
        memory_K=memory_K,
        memory_w2c=memory_w2c,
        query_K=query_K,
        query_w2c=query_w2c,
        downsample=downsample,
        splat_radius=splat_radius,
    )
    return projected["projected"]


def _project_candidate_frame_to_query_metrics(
    candidate_frame,
    candidate_depth,
    *,
    target_frame=None,
    memory_K,
    memory_w2c,
    query_K,
    query_w2c,
    downsample=4,
    splat_radius=2,
):
    from frustum.frustum_handler import reproject_pixels_nd_to_nd

    frame = np.asarray(candidate_frame, dtype=np.uint8)
    H, W = frame.shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.bool_)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    def _empty_metrics():
        return {
            "projected": canvas,
            "mask": mask,
            "zbuf": zbuf,
            "valid_pixels": 0,
            "coverage": 0.0,
            "mse": float("nan"),
            "mae": float("nan"),
            "psnr": float("nan"),
        }

    if (
        candidate_depth is None
        or memory_K is None
        or memory_w2c is None
        or query_K is None
        or query_w2c is None
    ):
        return _empty_metrics()
    depth = np.asarray(candidate_depth, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        return _empty_metrics()
    step = max(1, int(downsample))
    yy, xx = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing="ij")
    dh, dw = depth.shape
    d_y = np.clip(
        np.rint((yy.astype(np.float64) / float(H)) * dh - 0.5), 0, dh - 1
    ).astype(np.int64)
    d_x = np.clip(
        np.rint((xx.astype(np.float64) / float(W)) * dw - 0.5), 0, dw - 1
    ).astype(np.int64)
    depth_z = depth[d_y, d_x]
    valid = np.isfinite(depth_z) & (depth_z > 1e-6)
    if not np.any(valid):
        return _empty_metrics()
    pix_hw = np.stack([yy.astype(np.float64), xx.astype(np.float64)], axis=-1)
    pix_new_hw, zc_new, _xw, _xc = reproject_pixels_nd_to_nd(
        pix_hw=pix_hw[valid],
        depth_z=depth_z[valid].astype(np.float64),
        K_old=np.asarray(memory_K, dtype=np.float64),
        T_old_w2c=np.asarray(memory_w2c, dtype=np.float64),
        K_new=np.asarray(query_K, dtype=np.float64),
        T_new_w2c=np.asarray(query_w2c, dtype=np.float64),
        eps=0.0,
    )
    pix_new_hw = np.asarray(pix_new_hw)
    zc_new = np.asarray(zc_new)
    qy = np.rint(pix_new_hw[:, 0]).astype(np.int64)
    qx = np.rint(pix_new_hw[:, 1]).astype(np.int64)
    ok = (
        np.isfinite(pix_new_hw[:, 0])
        & np.isfinite(pix_new_hw[:, 1])
        & np.isfinite(zc_new)
        & (zc_new > 0)
        & (qy >= 0)
        & (qy < H)
        & (qx >= 0)
        & (qx < W)
    )
    if not np.any(ok):
        return _empty_metrics()
    src_colors = frame[yy[valid], xx[valid]][ok]
    qy = qy[ok]
    qx = qx[ok]
    z = zc_new[ok]
    radius = max(0, int(splat_radius))
    for y, x, z_one, color in zip(qy, qx, z, src_colors):
        y0 = max(0, int(y) - radius)
        y1 = min(H, int(y) + radius + 1)
        x0 = max(0, int(x) - radius)
        x1 = min(W, int(x) + radius + 1)
        patch = zbuf[y0:y1, x0:x1]
        update = float(z_one) < patch
        if np.any(update):
            patch[update] = float(z_one)
            canvas_patch = canvas[y0:y1, x0:x1]
            canvas_patch[update] = color
            mask[y0:y1, x0:x1][update] = True

    valid_pixels = int(mask.sum())
    coverage = float(valid_pixels) / float(max(1, H * W))
    mse = float("nan")
    mae = float("nan")
    psnr = float("nan")
    if target_frame is not None and valid_pixels > 0:
        target = np.asarray(target_frame, dtype=np.uint8)
        if target.shape[:2] != (H, W):
            raise ValueError(
                "target_frame spatial shape must match candidate_frame, got "
                f"{target.shape[:2]} vs {(H, W)}."
            )
        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError(
                "target_frame must have shape (H,W,3), got " f"{target.shape}."
            )
        diff = canvas[mask].astype(np.float32) - target[mask].astype(np.float32)
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        psnr = (
            float("inf")
            if mse <= 0.0
            else float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))
        )
    return {
        "projected": canvas,
        "mask": mask,
        "zbuf": zbuf,
        "valid_pixels": valid_pixels,
        "coverage": coverage,
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
    }


def _preview_raw_frame_labels(data, preview_frame_count):
    info = data.get("info") or {}
    try:
        generating_first = int(info.get("generating_first_frame_idx", 0) or 0)
    except (TypeError, ValueError):
        generating_first = 0
    query_frame_indices = _query_frame_indices_for_preview(data, preview_frame_count)
    labels = [f"preview=0 raw={generating_first - 1}"]
    for preview_idx in range(1, int(preview_frame_count)):
        query_idx = preview_idx - 1
        raw = (
            int(query_frame_indices[query_idx])
            if query_idx < len(query_frame_indices)
            else generating_first + query_idx
        )
        labels.append(f"preview={preview_idx} raw={raw}")
    return labels


def _annotate_frames_with_index(frames, start_index=0, labels=None, font_size=52):
    annotated = []
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(font_size)
        )
    except Exception:
        font = ImageFont.load_default()
    for offset, frame in enumerate(frames):
        arr = np.asarray(frame)
        img = Image.fromarray(arr).convert("RGB")
        draw = ImageDraw.Draw(img)
        text = (
            str(labels[offset])
            if labels is not None and offset < len(labels)
            else str(start_index + offset)
        )
        x, y = 8, 6
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 0), font=font)
        annotated.append(np.asarray(img))
    return annotated


def _prope_camera_kwargs(camera_data):
    return {
        "clean_latent_indices_prope_intrinsic": camera_data[
            "clean_latent_indices_prope_intrinsic"
        ],
        "clean_latent_indices_prope_extrinsic": camera_data[
            "clean_latent_indices_prope_extrinsic"
        ],
        "noisy_latent_indices_prope_intrinsic": camera_data[
            "noisy_latent_indices_prope_intrinsic"
        ],
        "noisy_latent_indices_prope_extrinsic": camera_data[
            "noisy_latent_indices_prope_extrinsic"
        ],
    }


def _blank_like_frame(frame):
    return np.zeros_like(np.asarray(frame))


def _compare_videos(top_frames, bottom_frames, middle_frames=None):
    compared_frames = []
    for frame_idx, bottom_frame in enumerate(bottom_frames):
        bottom_arr = np.asarray(bottom_frame)
        rows = []
        if frame_idx < len(top_frames):
            top_arr = np.asarray(top_frames[frame_idx])
            if top_arr.shape[:2] != bottom_arr.shape[:2]:
                top_arr = np.asarray(
                    Image.fromarray(top_arr).resize(
                        (bottom_arr.shape[1], bottom_arr.shape[0]),
                        Image.Resampling.LANCZOS,
                    )
                )
            rows.append(top_arr)
        else:
            rows.append(_blank_like_frame(bottom_arr))
        if middle_frames is not None:
            if frame_idx < len(middle_frames):
                middle_arr = np.asarray(middle_frames[frame_idx])
                if middle_arr.shape[:2] != bottom_arr.shape[:2]:
                    middle_arr = np.asarray(
                        Image.fromarray(middle_arr).resize(
                            (bottom_arr.shape[1], bottom_arr.shape[0]),
                            Image.Resampling.LANCZOS,
                        )
                    )
                rows.append(middle_arr)
            else:
                rows.append(_blank_like_frame(bottom_arr))
        rows.append(bottom_arr)
        compared_frames.append(np.concatenate(rows, axis=0))
    return compared_frames


def _get_frustum_handler_cls():
    try:
        from frustum.frustum_handler import FrustumHandler
    except ModuleNotFoundError:
        import sys

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from frustum.frustum_handler import FrustumHandler
    return FrustumHandler


def _get_dynamic_object_masker_cls():
    from examples.wanvideo.pipeline.dynamic_object_filter import (
        YoloDynamicObjectMaskerSubprocess,
    )

    return YoloDynamicObjectMaskerSubprocess


def _get_dynamic_overlay_fn():
    from examples.wanvideo.pipeline.dynamic_object_filter import (
        overlay_masks_on_frames,
    )

    return overlay_masks_on_frames


_DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def _rank_local_cuda_device():
    if not torch.cuda.is_available():
        return None
    for env_name in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            local_rank = int(value)
        except ValueError:
            continue
        if local_rank >= 0:
            return torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    current = torch.cuda.current_device()
    return torch.device(f"cuda:{current}")


def _set_rank_local_cuda_device():
    device = _rank_local_cuda_device()
    if device is not None:
        torch.cuda.set_device(device)
    return device


def _init_da3_depth_estimator(device=None):
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    # The package lives under ``third_party/depth-anything-3/src`` (pyproject
    # ``packages = ["src/depth_anything_3"]``); add it plus the parent dir.
    for da3_path in (
        os.path.join(repo_root, "third_party", "depth-anything-3", "src"),
        os.path.join(repo_root, "third_party", "depth-anything-3"),
    ):
        if os.path.isdir(da3_path) and da3_path not in sys.path:
            sys.path.insert(0, da3_path)
    from depth_anything_3.api import DepthAnything3

    model_ref = (
        os.environ.get("DA3_MODEL_PATH")
        or os.environ.get("DA3_MODEL_ID")
        or "runjia.qian/models/depth-anything/DA3NESTED-GIANT-LARGE-1.1"
        or _DA3_MODEL_ID
    )
    if device is not None and str(device) == "cuda":
        device = _rank_local_cuda_device() or device
    print(f"Loading depth estimator: {model_ref}")
    estimator = DepthAnything3.from_pretrained(model_ref)
    estimator.eval()
    if device is not None:
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.set_device(device)
        estimator = estimator.to(device)
    return estimator


def _init_registration_estimator(args, device=None):
    """Build the validation registration estimator selected by config.

    Returns a ``DepthAnything3`` (``args.registration_estimator == "da3"``, the
    legacy default) or a ``VGGTOmegaMetricEstimator``
    (``args.registration_estimator == "vggt_omega_metric"``). Both satisfy the
    same FrustumHandler estimator contract: ``.to()/.eval()/.inference()``
    where ``.inference(frames, ...)`` returns a prediction object exposing
    ``.depth (N,H,W)``, ``.extrinsics (N,3,4) w2c``, ``.intrinsics (N,3,3)``.

    The vggt_omega_metric path reuses ``_init_da3_depth_estimator`` (injected as
    ``da3_init_fn``) to load DA3's metric branch, so DA3 weights/resolution match
    the standalone path exactly.
    """
    estimator_kind = str(getattr(args, "registration_estimator", "da3") or "da3")
    if estimator_kind == "da3":
        return _init_da3_depth_estimator(device=device)
    if estimator_kind == "vggt_omega_metric":
        from examples.wanvideo.pipeline.mosaic.vggt_omega_metric_estimator import (
            init_vggt_omega_metric_estimator,
        )

        # ``... or <default>`` guards against an omegaconf null/empty YAML value:
        # ``parser.set_defaults`` bypasses argparse ``type=int`` coercion, so a
        # ``null`` key arrives as None (present-but-None), which ``getattr``'s
        # default would not cover and ``int(None)`` would crash on.
        return init_vggt_omega_metric_estimator(
            device=device,
            vggt_ckpt=getattr(args, "vggt_omega_ckpt", None),
            image_resolution=int(
                getattr(args, "vggt_omega_image_resolution", None) or 512
            ),
            da3_process_res=int(
                getattr(args, "vggt_omega_da3_process_res", None) or 504
            ),
            da3_init_fn=_init_da3_depth_estimator,
        )
    raise ValueError(
        f"Unknown registration_estimator {estimator_kind!r}; "
        "expected 'da3' or 'vggt_omega_metric'."
    )


def _read_video_gt_frames(video_path, frame_indices, target_h, target_w):
    """Read GT frames from ``video_path`` for the validation comparison strip.

    Mirrors the DA3 dataset frame reader /
    ``batch_encode_videos.VideoLatentDataset.center_crop_resize`` so the
    returned frames sit in exactly the same crop / resize as everything
    the trainer encoded; difference is the output is uint8 ``(T,H,W,3)``
    so it slots straight into ``_compare_videos`` without going through
    the VAE.

    Out-of-range indices are clamped to the last available frame, which
    happens for the last validation section (predicting beyond the
    captured video duration).
    """
    from decord import VideoReader, cpu
    import torch.nn.functional as F

    reader = VideoReader(str(video_path), ctx=cpu())
    last_idx = len(reader) - 1
    safe_indices = [max(0, min(int(idx), last_idx)) for idx in frame_indices]
    if not safe_indices:
        return np.zeros((0, int(target_h), int(target_w), 3), dtype=np.uint8)
    frames = reader.get_batch(safe_indices).asnumpy()  # (T, H, W, 3) uint8
    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    _, _, src_h, src_w = frames_t.shape
    crop_h = min(src_h, int(target_h))
    crop_w = min(src_w, int(target_w))
    top = max((src_h - crop_h) // 2, 0)
    left = max((src_w - crop_w) // 2, 0)
    frames_t = frames_t[:, :, top : top + crop_h, left : left + crop_w]
    if frames_t.shape[-2:] != (int(target_h), int(target_w)):
        frames_t = F.interpolate(
            frames_t,
            size=(int(target_h), int(target_w)),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    arr = (frames_t * 255.0).clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).numpy()
    return arr


def _decode_latents_to_numpy_frames(pipe, latents, device, tiled=False):
    """Decode latents (shape (1,C,T,H,W) or (C,T,H,W)) into a (T,H,W,3) uint8
    numpy array suitable for FrustumHandler.register_source_sequence / DA3.

    ``tiled=True`` keeps the legacy 30x52 / 15x26 tile_size+stride which
    was a no-op at the historical 832x448 latent grid (28x52 fits in
    one tile) but causes 12 redundant overlapping decodes at
    video-flow's 704x1280 (44x80) latent grid. Default ``False`` does
    a single full-image decode -- much faster when VRAM is plentiful.
    """
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    pipe.load_models_to_device(["vae"])
    decode_kwargs = {"device": device}
    if tiled:
        decode_kwargs.update(
            {"tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26)}
        )
    else:
        decode_kwargs["tiled"] = False
    decoded = pipe.vae.decode(
        latents.to(device=device, dtype=pipe.torch_dtype),
        **decode_kwargs,
    )
    pipe.load_models_to_device([])
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if decoded.ndim == 5:
        decoded = decoded[0]
    decoded = decoded.float().permute(1, 2, 3, 0)
    arr = ((decoded + 1.0) * 127.5).clamp(0, 255).to("cpu").numpy().astype(np.uint8)
    return arr


def _encode_frames_per_frame(pipe, frames_np, device, tiled=False, single2four=False):
    """VAE-encode each RGB frame **independently** to a single-frame latent
    and stack along the time axis.

    This mirrors the per-frame memory-latent buffer that
    ``WanMosaicPipelineModule._materialize_data`` builds at training time
    (see ``mem_latents_batched = pipe.vae.encode([(C,1,H,W), ...])``).
    Validation needs the same 1-RGB-frame -> 1-latent layout so that
    ``FrustumHandler.query_hits_mode_new(..., latent_merge_4frames=False)``
    can index ``latents`` 1:1 with ``self.extrinsics`` / ``self.depths``.

    Args:
        frames_np: ``(N, H, W, 3)`` uint8 numpy array (or list of those).
        device:    torch device for the VAE pass.
        tiled:     forwarded to ``pipe.vae.encode`` (off by default).

    Returns:
        ``(1, C_lat, N, H_lat, W_lat)`` CPU float tensor, ready to concat
        onto a running buffer or to slice as ``latents[0]`` for the
        FrustumHandler.
    """
    frames_arr = np.asarray(frames_np)
    if frames_arr.ndim != 4 or frames_arr.shape[-1] != 3:
        raise ValueError(
            f"_encode_frames_per_frame: expected (N,H,W,3) frames, "
            f"got shape {tuple(frames_arr.shape)}."
        )
    pipe.load_models_to_device(["vae"])
    dtype = pipe.torch_dtype
    frame_videos = []
    for fr in frames_arr:
        t = torch.from_numpy(fr).to(device=device, dtype=dtype)
        # (H, W, 3) -> (3, 1, H, W) in [-1, 1] (matches cn_frames convention).
        t = t.permute(2, 0, 1).unsqueeze(1) / 127.5 - 1.0
        if single2four:
            frame_videos.append(t.repeat(1, 5, 1, 1))
        else:
            frame_videos.append(t)
    encoded = pipe.vae.encode(frame_videos, device=device, tiled=tiled)
    if single2four:
        if encoded.shape[0] == 0:
            encoded = encoded[:, :, :1]
        else:
            encoded = encoded[:, :, 1:]
    pipe.load_models_to_device([])
    # encode() returns (N, C, 1, H_lat, W_lat) for length-1 single-frame
    # videos; rearrange to (1, C, N, H_lat, W_lat) to match history_latents.
    if encoded.ndim != 5 or encoded.shape[2] != 1:
        raise RuntimeError(
            f"_encode_frames_per_frame: vae.encode returned unexpected "
            f"shape {tuple(encoded.shape)} (expected (N,C,1,H_lat,W_lat))."
        )
    out = encoded.squeeze(2).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return out.detach().to("cpu", dtype=torch.float32)


def _validation_window_motion(extrinsics):
    """Return max pairwise translation/rotation for a validation 4-frame group.

    ``extrinsics`` are w2c matrices with metric translation. The math matches
    the training neighbour-window judge: camera centers are ``C = -R^T t`` and
    rotation is geodesic angle from ``trace(R_a R_b^T)``.
    """
    w2c = np.asarray(extrinsics, dtype=np.float64)
    if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
        raise ValueError(
            f"_validation_window_motion expected (N,4,4), got {tuple(w2c.shape)}"
        )
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)
    diff = centers[:, None, :] - centers[None, :, :]
    max_trans = float(np.sqrt((diff * diff).sum(-1)).max())

    rel = np.einsum("aij,bkj->abik", R, R)
    trace = np.einsum("abii->ab", rel)
    cos_a = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    max_rot_deg = float(np.degrees(np.arccos(cos_a)).max())
    return max_trans, max_rot_deg


def _build_validation_neighbor_window_memory_latents(
    pipe,
    frames_np,
    dense_latents,
    extrinsics,
    device,
    tiled=False,
    max_trans_m=0.2,
    max_rot_deg=5.0,
):
    """Build validation memory latents with safe per-slot dense-latent reuse.

    Validation still registers every raw frame with the FrustumHandler, so the
    returned tensor keeps one latent slot per registered frame. For each
    generated dense latent (covering four decoded RGB frames):

    * low camera motion across those four frames -> copy the dense latent into
      all four per-frame slots (no decode->VAE round-trip);
    * large motion -> preserve the legacy path for those four frames by VAE
      encoding them independently as single-frame sparse latents.

    This preserves the existing ``source_latents.T == len(handler.extrinsics)``
    invariant while letting low-motion generated latent groups reproject from
    the model-produced dense latent.
    """
    frames_arr = np.asarray(frames_np)
    if frames_arr.ndim != 4 or frames_arr.shape[-1] != 3:
        raise ValueError(
            "_build_validation_neighbor_window_memory_latents expected "
            f"(N,H,W,3) frames, got {tuple(frames_arr.shape)}"
        )
    if dense_latents is None:
        raise ValueError(
            "dense_latents is required for validation neighbour-window reuse"
        )

    dense = dense_latents.detach().cpu().to(dtype=torch.float32)
    if dense.ndim == 4:
        dense = dense.unsqueeze(0)
    if dense.ndim != 5 or dense.shape[0] != 1:
        raise ValueError(
            "_build_validation_neighbor_window_memory_latents expected dense_latents "
            f"shape (1,C,T,H,W) or (C,T,H,W), got {tuple(dense_latents.shape)}"
        )

    n_frames = int(frames_arr.shape[0])
    dense_t = int(dense.shape[2])
    expected_frames = dense_t * 4
    if n_frames != expected_frames:
        raise ValueError(
            "validation neighbour-window reuse expects 4 decoded frames per dense "
            f"latent; got frames={n_frames}, dense_T={dense_t} "
            f"(expected {expected_frames})."
        )
    extr_arr = np.asarray(extrinsics)
    if extr_arr.shape[0] != n_frames:
        raise ValueError(
            f"extrinsics length {extr_arr.shape[0]} does not match frames {n_frames}."
        )

    out = dense.new_empty(1, dense.shape[1], n_frames, dense.shape[3], dense.shape[4])
    reencode_frame_ids = []
    group_records = []

    for group_idx in range(dense_t):
        start = group_idx * 4
        end = start + 4
        max_trans, max_rot = _validation_window_motion(extr_arr[start:end])
        use_s1 = max_trans > float(max_trans_m) or max_rot > float(max_rot_deg)
        group_records.append(
            {
                "group_idx": int(group_idx),
                "frame_start": int(start),
                "frame_end_exclusive": int(end),
                "max_trans_m": float(max_trans),
                "max_rot_deg": float(max_rot),
                "mode": "s1" if use_s1 else "w5",
            }
        )
        if use_s1:
            reencode_frame_ids.extend(range(start, end))
        else:
            out[:, :, start:end] = dense[:, :, group_idx : group_idx + 1].repeat(
                1, 1, 4, 1, 1
            )

    if reencode_frame_ids:
        encoded = _encode_frames_per_frame(
            pipe,
            frames_arr[reencode_frame_ids],
            device,
            tiled=tiled,
        )
        out[:, :, reencode_frame_ids] = encoded.to(dtype=out.dtype)

    stats = {
        "enabled": True,
        "max_trans_m": float(max_trans_m),
        "max_rot_deg": float(max_rot_deg),
        "num_groups": int(dense_t),
        "num_s1_groups": int(sum(1 for r in group_records if r["mode"] == "s1")),
        "num_w5_groups": int(sum(1 for r in group_records if r["mode"] == "w5")),
        "num_reencoded_frames": int(len(reencode_frame_ids)),
        "num_reused_frames": int(n_frames - len(reencode_frame_ids)),
        "groups": group_records,
    }
    return out.contiguous(), stats


def _decoded_first_image(pipe, latents, device, tiled=False):
    arr = _decode_latents_to_numpy_frames(pipe, latents, device, tiled=tiled)
    if arr.shape[0] == 0:
        raise RuntimeError("Decoded zero frames from clean latents.")
    return Image.fromarray(arr[-1])
