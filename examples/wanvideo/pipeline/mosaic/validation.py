from .common import *
from diffsynth.utils.data import save_video
from PIL import Image
from .cleanup import (
    _rank_local_cuda_device,
    _release_cached_memory,
    _set_rank_local_cuda_device,
)
from .config import (
    _intrinsics_mode_arg,
    _is_recent_first_fuse_mode,
    _normalize_mosaic_fuse_mode,
)
from .history import (
    _build_validation_pool_rope_time_indices,
    _candidate_groups_to_lists,
    _compute_mosaic_frame_indices,
    _is_validation_clean_latent_pool_history_enabled,
    _select_validation_clean_latents,
    _select_validation_clean_latents_from_candidate_pool,
    _validation_clean_prefix_decode_drop_count,
    _validation_clean_prefix_frame_count,
    _validation_pool_clean_frame_ids_from_selection,
    _validation_pool_raw_frame_to_latent_index,
    _validation_recent_frame_window,
)
from .inference import run_mosaic_segment_inference
from .prompting import _require_nonempty_prompt
from .qwen import (
    _ValidationQwenPromptGenerator,
    _load_validation_qwen_events,
    _normalize_validation_qwen_mode,
    _prepare_validation_qwen_full_rollout_prompt_frames,
    _prepare_validation_qwen_video_prompt_frames,
    _select_validation_qwen_event,
    _simulate_validation_qwen_prompt_mode,
    _summarize_validation_camera_motion,
    _validation_qwen_actual_camera_summaries,
    _validation_qwen_context_prompt_frames,
    _validation_qwen_extrinsics_to_numpy,
    _validation_qwen_full_rollout_camera_summary,
    _validation_qwen_full_rollout_frame_indices,
    _validation_qwen_generate_from_payload,
    _validation_qwen_history_prompt_frames,
    _validation_qwen_initial_context_latents,
    _validation_qwen_section_frame_indices,
)
from .timing import (
    _debug_timing_enabled,
    _format_timing_breakdown,
    _timing_record_event_count,
    get_timing_reporter,
)
from .val_image import (
    _load_and_preprocess_val_image,
    _resolve_val_image_pool,
    _select_val_image_for_sample,
)
from .video_io import (
    _ValidationHistoryTempVideoWriter,
    _annotate_frames_with_index,
    _blank_preview_frame,
    _build_validation_candidate_history_frames,
    _build_validation_candidate_panel,
    _candidate_frame_to_uint8_hwc,
    _cn_frame_to_uint8_hwc,
    _compare_videos,
    _decode_latents_to_numpy_frames,
    _decoded_first_image,
    _draw_preview_label,
    _get_dynamic_object_masker_cls,
    _get_dynamic_overlay_fn,
    _get_frustum_handler_cls,
    _fit_preview_frame_to_slot,
    _history_verbose_frames,
    _init_registration_estimator,
    _parse_query_hits_with_candidate_frame_ids,
    _prope_camera_kwargs,
    _read_video_gt_frames,
    _record_validation_candidate_frames,
    _resize_frames_half_even,
    _save_validation_episode_first_frame_jpg,
    _save_validation_history_latents_pth,
    _validation_snapshot_safe_name,
    _validation_window_motion,
    _write_json_atomic,
    _write_video,
    _encode_frames_per_frame,
    _build_validation_neighbor_window_memory_latents,
)


def _subject_ref_tensor_to_uint8_hwc(frame):
    if not torch.is_tensor(frame):
        frame = torch.as_tensor(frame)
    return (
        ((frame.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )


def _write_validation_subject_ref_preview(data, save_path, *, height, width):
    ref_images = data.get("subject_ref_images")
    ref_info = data.get("subject_ref_info") or {}
    latent_info = data.get("subject_ref_latent_info") or {}
    tiles = []
    if torch.is_tensor(ref_images) and ref_images.numel() > 0:
        for idx in range(int(ref_images.shape[0])):
            tile = _subject_ref_tensor_to_uint8_hwc(ref_images[idx])
            frame_idx = ""
            records = ref_info.get("ref_records") or []
            if idx < len(records):
                frame_idx = f" frame={records[idx].get('frame_idx', '')}"
            tile = _draw_preview_label(tile, f"ref {idx}{frame_idx}")
            tiles.append(tile)
    else:
        status = str(ref_info.get("status") or latent_info.get("status") or "no_ref")
        ref_dir = str(ref_info.get("ref_dir") or "")
        label = f"subject ref: {status}"
        if ref_dir:
            label += f"\n{os.path.basename(os.path.dirname(ref_dir))}/{os.path.basename(ref_dir)}"
        tiles.append(_blank_preview_frame(max(64, min(320, height)), max(128, min(480, width)), label))
    if not tiles:
        return ""
    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    tiles = [_fit_preview_frame_to_slot(tile, tile_h, tile_w) for tile in tiles]
    canvas = np.concatenate(tiles, axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(canvas).save(save_path)
    return save_path


class _ValidationResources:
    """Holder for the heavy validation-only resources so the teardown can
    reach them from a ``finally`` even if the work function raises midway."""

    __slots__ = (
        "da3_estimator",
        "dynamic_object_masker",
        "qwen_prompt_generator",
        "history_temp_writer",
    )

    def __init__(self):
        self.da3_estimator = None
        self.dynamic_object_masker = None
        self.qwen_prompt_generator = None
        self.history_temp_writer = None


def _snapshot_module_training_flags(module):
    """Record every submodule's ``.training`` flag before validation.

    ``freeze_except`` keeps the frozen components (UMT5 text encoder, VAE)
    in eval() and only the trainable models in train(). A blanket
    ``module.train()`` after validation flips the text encoder -- which has
    real ``nn.Dropout(p=0.1)`` layers -- back into train mode, so every
    later training step would condition on dropout-corrupted prompt
    embeddings. Restoring from this snapshot keeps the exact
    pre-validation split instead.
    """
    return {name: m.training for name, m in module.named_modules()}


def _restore_module_training_flags(module, snapshot):
    if not snapshot:
        # Defensive fallback (no snapshot captured): prefer train() so a
        # failed validation can never leave the trainable models in eval().
        module.train()
        return
    for name, m in module.named_modules():
        flag = snapshot.get(name)
        if flag is not None:
            m.training = bool(flag)


def _release_validation_resources(
    unwrapped_model, timing, resources, *, proc_rank=0, train_mode_snapshot=None
):
    """Best-effort teardown shared by the success and error paths.

    Releasing here (instead of only at the tail of the work function)
    guarantees a failed/raised validation never leaves training stuck in
    ``eval()`` with the multi-GB DA3/Qwen weights, the dynamic-object worker,
    or the val-time CUDA cache still resident -- the classic "validation
    errored and now training is wedged / very slow" failure mode. Every step
    is independently guarded so one failing release cannot skip the rest.
    """
    if resources.dynamic_object_masker is not None:
        print(
            f"[validation] closing dynamic object worker rank={proc_rank}",
            flush=True,
        )
        try:
            resources.dynamic_object_masker.close()
        except Exception:
            pass
        resources.dynamic_object_masker = None
    if resources.history_temp_writer is not None:
        try:
            resources.history_temp_writer.close()
        except Exception:
            pass
        resources.history_temp_writer = None
    if resources.qwen_prompt_generator is not None:
        print(
            f"[validation][qwen][rank {proc_rank}] releasing prompt model",
            flush=True,
        )
        resources.qwen_prompt_generator.model = None
        resources.qwen_prompt_generator.processor = None
        resources.qwen_prompt_generator = None
    # Drop the holder's DA3 reference (the work function's frame has already
    # unwound, so this is the last strong ref) before returning the CUDA cache
    # to the driver, otherwise fragmented val blocks linger into training.
    resources.da3_estimator = None
    _release_cached_memory()
    try:
        timing.finish_record()
    except Exception:
        pass
    _restore_module_training_flags(unwrapped_model, train_mode_snapshot)


def _apply_val_image_initial_frame(
    data, image_path, *, pipe, height, width, proc_rank, batch_idx
):
    """Replace segment 0's clean latent with a user-provided image.

    Called after ``_materialize_data`` and BEFORE the loop builds
    ``history_latents`` from ``data['clean_latents']``. The image is
    cover-resized + centre-cropped (no stretch) to ``width`` x ``height`` and
    VAE-encoded exactly like a dataset frame, then overwrites
    ``data['clean_latents']``. Because both the rollout's segment-0
    ``first_frame_latents`` and (for self_rollout / self_rollout_event) the
    Qwen segment-0 prompt frames read the tail of ``history_latents``, this one
    write makes the rollout AND the prompter start from the image. Later
    segments chain off generated frames and are untouched; camera extrinsics /
    intrinsics live in separate prope fields and are never modified here.
    """
    frame_np = _load_and_preprocess_val_image(image_path, height=height, width=width)
    # tiled=False to match the dataset's clean-latent encode
    # (module_materialize.py vae.encode(..., tiled=False)); the unrelated
    # --vae_decode_tiled flag must NOT leak into this encode or the anchor
    # latent would carry tile-seam artifacts the dataset latent never has.
    user_latent = _encode_frames_per_frame(
        pipe, frame_np[None], pipe.device, tiled=False
    )
    orig = data.get("clean_latents")
    if torch.is_tensor(orig):
        if tuple(user_latent.shape) != tuple(orig.shape):
            raise RuntimeError(
                "[validation][val_image] encoded latent shape "
                f"{tuple(user_latent.shape)} != dataset clean_latents shape "
                f"{tuple(orig.shape)} for {image_path!r}; check --height/--width."
            )
        data["clean_latents"] = user_latent.to(device=orig.device, dtype=orig.dtype)
    else:
        data["clean_latents"] = user_latent
    # Keep the first-frame snapshot honest: _save_validation_episode_first_frame_jpg
    # reads cn_frames[:, 0] (CTHW in [-1, 1]). This is diagnostic only -- the
    # authoritative substitution is data['clean_latents'] above, and
    # materialization already consumed cn_frames before this point.
    cn_frames = data.get("cn_frames")
    if torch.is_tensor(cn_frames) and cn_frames.ndim == 4 and int(cn_frames.shape[1]) > 0:
        cthw = (
            torch.from_numpy(frame_np).to(dtype=cn_frames.dtype).permute(2, 0, 1)
            / 127.5
            - 1.0
        )
        cn_frames[:, 0] = cthw.to(cn_frames.device)
    print(
        f"[validation][val_image][rank {proc_rank}] batch={batch_idx} "
        f"segment-0 initial frame <- {image_path!r} latent={tuple(user_latent.shape)}",
        flush=True,
    )
    return data


def run_mosaic_validation(
    accelerator,
    val_dataset,
    model,
    log_dir,
    args,
    epoch_id=-1,
):
    if val_dataset is None:
        return
    unwrapped_model = accelerator.unwrap_model(model)
    timing = get_timing_reporter(unwrapped_model)
    proc_rank = int(getattr(accelerator, "process_index", 0))
    resources = _ValidationResources()
    # Captured BEFORE eval() flips anything so teardown can restore the
    # exact freeze_except split (dit train / text encoder + vae eval).
    train_mode_snapshot = _snapshot_module_training_flags(unwrapped_model)
    original_subject_context_patch_mask_prob = getattr(
        unwrapped_model, "subject_context_patch_mask_prob", None
    )
    try:
        if original_subject_context_patch_mask_prob is not None:
            unwrapped_model.subject_context_patch_mask_prob = 0.0
        _run_mosaic_validation_impl(
            accelerator,
            val_dataset,
            model,
            log_dir,
            args,
            epoch_id,
            resources=resources,
        )
    finally:
        if original_subject_context_patch_mask_prob is not None:
            unwrapped_model.subject_context_patch_mask_prob = (
                original_subject_context_patch_mask_prob
            )
        _release_validation_resources(
            unwrapped_model,
            timing,
            resources,
            proc_rank=proc_rank,
            train_mode_snapshot=train_mode_snapshot,
        )
    accelerator.wait_for_everyone()


def _run_mosaic_validation_impl(
    accelerator,
    val_dataset,
    model,
    log_dir,
    args,
    epoch_id=-1,
    *,
    resources,
):
    FrustumHandler = _get_frustum_handler_cls()

    unwrapped_model = accelerator.unwrap_model(model)
    timing = get_timing_reporter(unwrapped_model)
    timing.start_record("validation", epoch_id=int(epoch_id))

    unwrapped_model.eval()
    proc_rank = int(getattr(accelerator, "process_index", 0))
    num_processes = max(1, int(getattr(accelerator, "num_processes", 1) or 1))
    proc_tag = f"proc_{proc_rank:03d}"
    t2v_no_ref = bool(getattr(args, "validation_t2v_no_ref", False))
    use_mosaic = unwrapped_model.enable_mosaic and not unwrapped_model.only_prope
    if t2v_no_ref:
        use_mosaic = False
        if proc_rank == 0:
            print(
                "[validation][t2v-no-ref] enabled: section 0 uses no "
                "input_image/first_frame_latents and no mosaic memory.",
                flush=True,
            )
    use_mosaic_view_change_prope = bool(
        use_mosaic and getattr(unwrapped_model, "mosaic_view_change_prope", False)
    )
    if use_mosaic_view_change_prope and _is_recent_first_fuse_mode(
        args.mosaic_fuse_mode
    ):
        raise ValueError(
            "mosaic_view_change_prope does not support validation "
            "mosaic_fuse_mode='recent_first_smooth' yet."
        )
    vae_decode_tiled = bool(getattr(unwrapped_model, "vae_decode_tiled", False))
    # ``--val_image_path``: optional segment-0 initial-frame override. Resolved
    # once here (single file -> shared across ranks; directory -> sorted pool
    # striped across global ranks). Only affects each episode's first frame.
    val_image_pool = _resolve_val_image_pool(getattr(args, "val_image_path", None))
    if val_image_pool and proc_rank == 0:
        print(
            f"[validation][val_image] enabled pool_size={len(val_image_pool)} "
            f"from {getattr(args, 'val_image_path', None)!r}; segment-0 initial "
            "frame replaced per global rank (camera params unchanged).",
            flush=True,
        )
    # ``--validation_camera_pingpong``: when an episode's camera trajectory is
    # shorter than the rollout, ping-pong (forward/backward/forward) through it
    # instead of clamping to the last pose. Threaded into read_camera_params via
    # the per-section recalculate_input dict.
    camera_pingpong = bool(getattr(args, "validation_camera_pingpong", False))
    if camera_pingpong and proc_rank == 0:
        print(
            "[validation][camera_pingpong] enabled: camera trajectory will "
            "ping-pong (forward/backward) past its end instead of clamping to "
            "the final pose.",
            flush=True,
        )
    validation_neighbor_window_enabled = bool(
        getattr(args, "memory_neighbor_window_val", False)
        and getattr(args, "force_using_input_extrinics", False)
    )
    if getattr(args, "memory_neighbor_window_val", False) and not getattr(
        args, "force_using_input_extrinics", False
    ):
        print(
            "[validation] --memory_neighbor_window_val requires "
            "--force_using_input_extrinics; falling back to per-frame VAE encode."
        )
    dynamic_filter_enabled = bool(
        getattr(args, "validation_dynamic_object_filter", False)
    )
    validation_clean_latent_history_count = max(
        1, int(getattr(args, "validation_clean_latent_history_count", 1) or 1)
    )
    validation_clean_latent_pool_history = (
        _is_validation_clean_latent_pool_history_enabled(args)
    )
    validation_clean_latent_pool_history_selection_mode = str(
        getattr(args, "validation_clean_latent_pool_history_selection_mode", "recent")
        or "recent"
    )
    if validation_clean_latent_pool_history_selection_mode not in {
        "recent",
        "oldest_spread",
        "segment_coverage",
        "per_3latentframe",
    }:
        raise ValueError(
            "--validation_clean_latent_pool_history_selection_mode must be "
            "'recent', 'oldest_spread', 'segment_coverage', or "
            "'per_3latentframe', got "
            f"{validation_clean_latent_pool_history_selection_mode!r}."
        )
    validation_per3_context = (
        validation_clean_latent_pool_history_selection_mode == "per_3latentframe"
    )
    if validation_per3_context:
        # per_3 derives its count from the val noisy-latent count // 3 and
        # requires the pool-history path. Keep the opt-in switch explicit (per
        # user request): error -- don't silently enable it -- so the operator
        # stays in control. The divisibility hard-error is NOT done here on
        # args.latent_window_size: the TRUE val generating count is
        # data["noisy_latent_indices"].shape[-1] (== train noisy_latents.shape[2];
        # equal across train/val), which is only known per-item below, so the
        # divisibility check fires there against the real n_gen.
        if not validation_clean_latent_pool_history:
            raise ValueError(
                "--validation_clean_latent_pool_history_selection_mode="
                "'per_3latentframe' requires --validation_clean_latent_pool_"
                "history to be enabled explicitly."
            )
        if validation_clean_latent_history_count > 1:
            print(
                "[validation] NOTE: per_3latentframe supersedes "
                "--validation_context_count / "
                "--validation_clean_latent_history_count; the emitted context "
                "count is forced to (val noisy-latent count)//3.",
                flush=True,
            )
    validation_clean_latent_pool_history_bias = int(
        getattr(args, "validation_clean_latent_pool_history_bias", 1) or 1
    )
    if validation_clean_latent_pool_history_bias < 1:
        raise ValueError(
            "--validation_clean_latent_pool_history_bias must be >= 1, got "
            f"{validation_clean_latent_pool_history_bias}."
        )
    if (
        validation_clean_latent_history_count > 1 or validation_per3_context
    ) and getattr(args, "validation_use_decoded_first_image", False):
        raise ValueError(
            "Multi-clean validation history (validation_clean_latent_history_"
            "count > 1, or selection_mode='per_3latentframe' which emits "
            "latent_window_size//3 context latents) requires "
            "--validation_use_decoded_first_image=False because the decoded "
            "PIL image path can only provide one clean anchor latent."
        )
    max_batches = max(1, int(args.num_val_batches))
    num_validation_blocks = max(1, int(args.num_validation_blocks))
    validation_offset = int(getattr(args, "validation_sample_offset", 0) or 0)
    if bool(getattr(args, "validation_rotate_samples", False)):
        validation_call_index = max(0, int(epoch_id))
        if int(epoch_id) >= 0:
            validation_call_index = int(epoch_id) // max(
                1, int(getattr(args, "validation_interval", 1) or 1)
            )
        validation_offset += (
            validation_call_index * max_batches * int(accelerator.num_processes)
        )
    if hasattr(val_dataset, "set_validation_sample_offset"):
        val_dataset.set_validation_sample_offset(validation_offset)
    with timing.scope("validation.dataloader_create"):
        _set_rank_local_cuda_device()
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=0,
        )
        val_dataloader = accelerator.prepare(val_dataloader)
    validation_dir = os.path.join(log_dir, "validation")
    validation_temp_dir = os.path.join(validation_dir, "_temp")
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(validation_temp_dir, exist_ok=True)
    timing.set_metadata(
        max_batches=int(max_batches),
        num_validation_blocks=int(num_validation_blocks),
        use_mosaic=bool(use_mosaic),
        validation_sample_offset=int(validation_offset),
    )
    if proc_rank == 0:
        print(
            f"[validation] sample_offset={validation_offset} "
            f"rotate_samples={bool(getattr(args, 'validation_rotate_samples', False))}",
            flush=True,
        )

    # DA3 (multi-GB) lives on GPU at start; ``register_source_sequence``
    # bounces it back to CPU after each call. The end-of-function cleanup
    # below relies on ``da3_estimator`` being deletable, so we keep the
    # name in this scope and set it to None in the ``finally``.
    da3_estimator = (
        _init_registration_estimator(
            args,
            device=_rank_local_cuda_device() or unwrapped_model.pipe.device,
        )
        if use_mosaic
        else None
    )
    resources.da3_estimator = da3_estimator
    dynamic_object_masker = None
    if use_mosaic and dynamic_filter_enabled:
        masker_cls = _get_dynamic_object_masker_cls()
        dynamic_yolo_device = (
            str(getattr(args, "validation_dynamic_object_device", "") or "").strip()
            or "gpu"
        )
        dynamic_object_masker = masker_cls(
            str(getattr(args, "validation_dynamic_object_yolo_model", "") or ""),
            class_names=getattr(args, "validation_dynamic_object_classes", None),
            conf=float(getattr(args, "validation_dynamic_object_conf", 0.25)),
            imgsz=int(getattr(args, "validation_dynamic_object_imgsz", 960)),
            device=dynamic_yolo_device,
            debug_device=bool(
                getattr(args, "validation_dynamic_object_debug_device", False)
            ),
            dilate_latent=int(
                getattr(args, "validation_dynamic_object_mask_dilate_latent", 0)
            ),
            temporal_dilate_radius=int(
                getattr(args, "validation_dynamic_object_temporal_dilate_radius", 0)
            ),
            track_gap_fill_max_gap=int(
                getattr(args, "validation_dynamic_object_track_gap_fill_max_gap", 0)
            ),
            tracker=str(
                getattr(args, "validation_dynamic_object_tracker", "bytetrack.yaml")
            ),
        )
        resources.dynamic_object_masker = dynamic_object_masker
        print(
            "[validation] dynamic object filter enabled "
            f"rank={getattr(accelerator, 'process_index', 0)} "
            f"local_cuda={_rank_local_cuda_device()} "
            f"pipe_device={getattr(unwrapped_model.pipe, 'device', None)} "
            f"classes={dynamic_object_masker.class_names} "
            f"class_ids={dynamic_object_masker.class_ids} "
            f"conf={dynamic_object_masker.conf} imgsz={dynamic_object_masker.imgsz} "
            f"device={dynamic_yolo_device} "
            f"worker_device={getattr(dynamic_object_masker, 'worker_device', None)} "
            f"worker_visible={getattr(dynamic_object_masker, 'worker_visible', None)} "
            f"dilate_latent={dynamic_object_masker.dilate_latent} "
            f"temporal_dilate_radius={dynamic_object_masker.temporal_dilate_radius} "
            f"track_gap_fill_max_gap={dynamic_object_masker.track_gap_fill_max_gap} "
            f"tracker={dynamic_object_masker.tracker}",
            flush=True,
        )

    # Each rank validates an independent sample to use the GPUs in parallel.
    # ``_MosaicDatasetBase._select_path`` already returns a different scene
    # per rank in distributed val mode; we just need to (a) drop the
    # main-process gate so every rank actually runs inference, and (b) tag
    # every file we write with the rank to avoid collisions when two ranks
    # happen to land on the same scene (e.g. when len(self.path) < world_size).
    validation_qwen_prompt = bool(getattr(args, "validation_qwen_prompt", False))
    validation_qwen_mode = _normalize_validation_qwen_mode(
        getattr(args, "validation_qwen_mode", "oracle_section_gt")
    )
    validation_qwen_events = (
        _load_validation_qwen_events(getattr(args, "validation_qwen_event_yaml", ""))
        if validation_qwen_mode == "self_rollout_event"
        else []
    )
    qwen_prompt_generator = (
        _ValidationQwenPromptGenerator(args, rank=proc_rank)
        if validation_qwen_prompt
        else None
    )
    resources.qwen_prompt_generator = qwen_prompt_generator
    if validation_qwen_prompt:
        print(
            f"[validation][qwen][rank {proc_rank}] enabled "
            f"model_path={getattr(args, 'validation_qwen_model_path', None)!r} "
            f"provider={getattr(qwen_prompt_generator, 'provider', None)!r} "
            f"device_map={getattr(args, 'validation_qwen_device_map', None)!r} "
            f"word_limit={getattr(args, 'validation_qwen_word_limit', None)} "
            f"mode={validation_qwen_mode} "
            f"sample_frames={getattr(args, 'validation_qwen_sample_frames', None)} "
            f"frame_max_side={getattr(args, 'validation_qwen_frame_max_side', None)} "
            "dataset prompts will be used as scene anchors.",
            flush=True,
        )

    val_wait_start = time.perf_counter()
    history_temp_writer = None
    for batch_idx, data in enumerate(val_dataloader):
        val_ready = time.perf_counter()
        timing.add_duration("validation.data_wait", val_ready - val_wait_start)
        if batch_idx >= max_batches:
            break

        # DA3 video datasets ship raw frames + a fuse plan; materialization
        # fills the latent slots expected by the validation rollout.
        if isinstance(data, dict) and data.get("needs_vae_materialization"):
            with timing.scope("validation.materialize"):
                data = unwrapped_model._materialize_data(data)

        # ``--val_image_path``: swap segment 0's clean latent for a user image
        # AFTER materialization (so data['clean_latents'] exists) and BEFORE
        # history_latents is built from it below, so the substitution flows into
        # both the rollout anchor and the Qwen segment-0 prompt frames.
        if val_image_pool is not None:
            chosen_val_image = _select_val_image_for_sample(
                val_image_pool,
                batch_idx=batch_idx,
                num_processes=num_processes,
                proc_rank=proc_rank,
            )
            try:
                with timing.scope("validation.val_image_initial_frame"):
                    data = _apply_val_image_initial_frame(
                        data,
                        chosen_val_image,
                        pipe=unwrapped_model.pipe,
                        height=int(args.height),
                        width=int(args.width),
                        proc_rank=proc_rank,
                        batch_idx=batch_idx,
                    )
            except Exception as exc:
                # The pool is striped across ranks, so a corrupt/unreadable
                # image fails on ONE rank only. Raising here would skip the
                # success-path accelerator.wait_for_everyone() and deadlock the
                # surviving ranks at the next collective. Fall back to the
                # dataset's segment-0 frame (data['clean_latents'] is untouched
                # because the encode happens before any mutation) and warn.
                print(
                    f"[validation][val_image][rank {proc_rank}] batch={batch_idx} "
                    f"FAILED to apply {chosen_val_image!r} "
                    f"({type(exc).__name__}: {exc}); falling back to the dataset "
                    "segment-0 frame so distributed validation does not deadlock.",
                    flush=True,
                )

        latent_window_size = int(data["noisy_latent_indices"].shape[-1])
        latent_stride = latent_window_size
        if validation_per3_context and latent_window_size % 3 != 0:
            # Authoritative per_3 divisibility hard-error against the REAL val
            # generating-latent count (one per noisy latent), which equals the
            # train n_gen. Per user request this is a hard error, not a floor.
            raise ValueError(
                "validation selection_mode='per_3latentframe' requires the val "
                f"noisy-latent count to be divisible by 3, got {latent_window_size} "
                "(this equals the train n_gen; set latent_window_size so this "
                "count is a multiple of 3)."
            )

        history_latents = (
            unwrapped_model._ensure_batched_latents(data["clean_latents"])
            .detach()
            .cpu()
        )
        mosaic_latent_history = history_latents.clone()
        # Per-frame VAE-encoded buffer that grows in lockstep with the
        # FrustumHandler source sequence (1 RGB frame -> 1 latent on T axis).
        # ``history_latents`` above is the WAN-VAE temporally-compressed
        # pipeline output (1 latent covers up to 4 RGB frames), which does
        # NOT match what ``query_hits_mode_new(..., latent_merge_4frames=
        # False)`` indexes -- the handler indexes ``latents`` 1:1 with
        # ``self.extrinsics``. We rebuild a parallel per-frame buffer below
        # so source_latents and the handler's frame ids stay aligned, the
        # same way training's ``memory_latents_full`` is per-frame encoded.
        mosaic_query_latents = None
        H_lat, W_lat = history_latents.shape[-2], history_latents.shape[-1]
        # WAN VAE downsamples 16x in space; keep this in lockstep with the
        # FrustumHandler(latent_stride=16) instance built below, otherwise
        # reproject sees pixel coords at the wrong scale.
        latent_stride_for_validation = 16
        H_img, W_img = (
            H_lat * latent_stride_for_validation,
            W_lat * latent_stride_for_validation,
        )
        clean_name = data.get("info", {}).get("clean_name", f"batch_{batch_idx}")
        subject_ref_preview_path = os.path.join(
            validation_dir,
            f"epoch_{epoch_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}"
            "_subject_ref_preview.jpg",
        )
        subject_ref_preview_path = _write_validation_subject_ref_preview(
            data,
            subject_ref_preview_path,
            height=H_img,
            width=W_img,
        )
        history_temp_path = os.path.join(
            validation_temp_dir,
            f"epoch_{epoch_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}"
            "_history_temp.mp4",
        )
        history_temp_latent_path = os.path.join(
            validation_temp_dir,
            f"epoch_{epoch_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}"
            "_history_temp.pth",
        )
        history_temp_writer = None
        first_frame_snapshot_path = _save_validation_episode_first_frame_jpg(
            data,
            validation_temp_dir,
            epoch_id=epoch_id,
            batch_idx=batch_idx,
            proc_tag=proc_tag,
            clean_name=clean_name,
        )
        if first_frame_snapshot_path:
            print(
                f"[validation] saved episode first frame snapshot to "
                f"{first_frame_snapshot_path}",
                flush=True,
            )
        validation_qwen_frames_by_section = None
        validation_qwen_full_payload = None
        validation_qwen_static_prompt = None
        validation_qwen_camera_summaries = []
        validation_qwen_event = None
        if validation_qwen_mode == "self_rollout_event":
            sample_index = int(batch_idx) * max(1, int(num_processes)) + int(proc_rank)
            validation_qwen_event = _select_validation_qwen_event(
                validation_qwen_events,
                sample_index=sample_index,
                seed=getattr(args, "validation_seed", 0),
            )
            print(
                f"[validation][qwen][rank {proc_rank}] batch={batch_idx} "
                f"event id={validation_qwen_event['id']} "
                f"text={validation_qwen_event['text']!r}",
                flush=True,
            )
        if validation_qwen_prompt:
            if validation_qwen_mode == "oracle_section_gt":
                with timing.scope(
                    "validation.qwen_prompt.frames",
                    metadata={
                        "batch_idx": int(batch_idx),
                        "clean_name": str(clean_name),
                        "mode": validation_qwen_mode,
                        "num_sections": int(num_validation_blocks),
                        "latent_window_size": int(latent_window_size),
                        "sample_frames": int(args.validation_qwen_sample_frames),
                        "frame_max_side": int(args.validation_qwen_frame_max_side),
                    },
                ):
                    validation_qwen_frames_by_section = (
                        _prepare_validation_qwen_video_prompt_frames(
                            data["info"],
                            num_sections=num_validation_blocks,
                            latent_window_size=latent_window_size,
                            sample_frames=args.validation_qwen_sample_frames,
                            frame_max_side=args.validation_qwen_frame_max_side,
                            target_h=H_img,
                            target_w=W_img,
                        )
                    )
                for sec_idx, payload in validation_qwen_frames_by_section.items():
                    print(
                        f"[validation][qwen][rank {proc_rank}] sec {sec_idx} "
                        "GT prompt frames "
                        f"indices={payload['frame_indices']} "
                        f"sizes={[getattr(image, 'size', None) for image in payload['images']]}",
                        flush=True,
                    )
            elif validation_qwen_mode == "oracle_full_gt_static":
                full_sample_frames = int(
                    getattr(args, "validation_qwen_full_video_sample_frames", 0) or 0
                )
                if full_sample_frames <= 0:
                    full_sample_frames = int(args.validation_qwen_sample_frames) * int(
                        num_validation_blocks
                    )
                with timing.scope(
                    "validation.qwen_prompt.full_frames",
                    metadata={
                        "batch_idx": int(batch_idx),
                        "clean_name": str(clean_name),
                        "mode": validation_qwen_mode,
                        "rollout_sections": int(num_validation_blocks),
                        "sample_frames": int(full_sample_frames),
                        "frame_max_side": int(args.validation_qwen_frame_max_side),
                    },
                ):
                    validation_qwen_full_payload = (
                        _prepare_validation_qwen_full_rollout_prompt_frames(
                            data["info"],
                            num_sections=num_validation_blocks,
                            latent_window_size=latent_window_size,
                            sample_frames=full_sample_frames,
                            frame_max_side=args.validation_qwen_frame_max_side,
                            target_h=H_img,
                            target_w=W_img,
                        )
                    )
                    validation_qwen_camera_summaries = (
                        _validation_qwen_actual_camera_summaries(
                            val_dataset,
                            data,
                            num_sections=num_validation_blocks,
                            latent_window_size=latent_window_size,
                            history_latents=history_latents,
                            validation_clean_latent_history_count=(
                                validation_clean_latent_history_count
                            ),
                            camera_pingpong=camera_pingpong,
                        )
                    )
                print(
                    f"[validation][qwen][rank {proc_rank}] full-rollout GT prompt "
                    f"frames indices={validation_qwen_full_payload['frame_indices']} "
                    f"sizes={[getattr(image, 'size', None) for image in validation_qwen_full_payload['images']]}",
                    flush=True,
                )

        running_clean_indices = data["clean_latent_indices"].clone()
        running_noisy_indices = data["noisy_latent_indices"].clone()

        handler = None
        if use_mosaic:
            validation_fixed_K = bool(
                _intrinsics_mode_arg(
                    args, "validation_mosaic_intrinsics_mode", "episode_mean"
                )
                == "first_frame"
            )
            init_K = data["clean_latent_indices_prope_intrinsic"][0].cpu().numpy()
            init_extrinsic = (
                data["clean_latent_indices_prope_extrinsic"][:1].cpu().numpy()
            )
            print(
                f"[validation] init handler  K={init_K.shape}  "
                f"init_extrinsic={init_extrinsic.shape}  "
                f"image_size={(H_img, W_img)}  grid_size={(H_lat, W_lat)}  "
                f"is_c2w=False  fixed_intrinsics={validation_fixed_K}  "
                f"(DA3 dataset outputs w2c camera matrices)"
            )
            handler = FrustumHandler(
                init_K,
                image_size=(H_img, W_img),
                grid_size=(H_lat, W_lat),
                depth_inf_thresh=1e9,
                depth_estimator=da3_estimator,
                is_c2w=False,
                use_gpu=True,
                # ``point_stride`` is dead code on the handler: it only
                # feeds ``self._stride``, which is never read. The active
                # spatial stride for reproject is ``latent_stride`` below.
                init_extrinsic=init_extrinsic,
                latent_stride=latent_stride_for_validation,
                fixed_intrinsics=validation_fixed_K,
            )
            # In --debug runs this routes handler-internal ``_timing_scope``
            # events (select_candidates / fuse_candidates / per-group splat)
            # into the active "validation" record so the breakdown printed
            # after ``query_hits_mode_new`` reflects real sub-step costs.
            # The reporter is a no-op when timing is disabled, so this is
            # safe to call unconditionally.
            if hasattr(handler, "set_timing_reporter"):
                handler.set_timing_reporter(timing)

        all_section_meta = {}
        last_predicting_frames = None
        clean_history_extr = data["clean_latent_indices_prope_extrinsic"].detach().cpu()
        clean_history_intr = data["clean_latent_indices_prope_intrinsic"].detach().cpu()
        # ``--force_using_input_extrinics`` opt-in: keep the per-frame
        # (extrinsics, intrinsics) that were fed to the model in the
        # PREVIOUS section so the next section's
        # ``register_source_sequence`` can replay them verbatim instead
        # of letting DA3 re-recover them from the rendered RGB. ``None``
        # means "either flag is off, or we're at section 0 and there's no
        # previous prediction yet".
        last_predicting_extr = None
        last_predicting_intr = None
        last_predicting_dense_latents = None
        validation_candidate_frames_by_id = {}
        validation_candidate_groups_history = []
        validation_pool_clean_groups_history = []
        mosaic_source_valid_masks = None

        def _append_validation_dynamic_masks(frames_np, *, start_index, label):
            nonlocal mosaic_source_valid_masks
            if dynamic_object_masker is None:
                return None
            with timing.scope(
                "validation.dynamic_object_masks",
                metadata={
                    "batch_idx": int(batch_idx),
                    "section_idx": int(section_idx),
                    "label": str(label),
                    "frame_count": int(len(frames_np)),
                },
            ):
                mask_result = dynamic_object_masker.masks_for_frames(
                    frames_np,
                    H_lat=int(H_lat),
                    W_lat=int(W_lat),
                    batch_idx=int(batch_idx),
                    section_idx=int(section_idx),
                    clean_name=str(clean_name),
                )
            valid_np = mask_result.source_valid_masks
            if valid_np.shape[0] != int(len(frames_np)):
                raise RuntimeError(
                    f"[validation] dynamic masks T={valid_np.shape[0]} does not "
                    f"match registered frames {len(frames_np)} for {label}."
                )
            valid_t = torch.from_numpy(valid_np.astype(np.bool_, copy=False))
            if mosaic_source_valid_masks is None:
                mosaic_source_valid_masks = valid_t
            else:
                mosaic_source_valid_masks = torch.cat(
                    [mosaic_source_valid_masks, valid_t], dim=0
                )
            if getattr(args, "validation_dynamic_object_save_preview", False):
                overlay_fn = _get_dynamic_overlay_fn()
                preview_frames = overlay_fn(frames_np, mask_result.dynamic_masks)
                preview_path = os.path.join(
                    validation_dir,
                    f"epoch{epoch_id}_batch{batch_idx:03d}_{proc_tag}_"
                    f"sec{section_idx:02d}_{label}_dynamic_masks.mp4",
                )
                save_video(preview_frames, preview_path, fps=15, quality=5)
            masked_fraction = float((~valid_t).float().mean().item())
            print(
                f"[validation] dynamic object masks {label} "
                f"start_index={start_index} frames={len(frames_np)} "
                f"valid_shape={tuple(valid_t.shape)} "
                f"masked_fraction={masked_fraction:.4f}",
                flush=True,
            )
            return valid_t

        for section_idx in range(num_validation_blocks):
            indice_drift = latent_stride if section_idx > 0 else 0
            running_clean_indices = running_clean_indices + indice_drift
            running_noisy_indices = running_noisy_indices + indice_drift
            # ``clean_latent_4x_indices`` / ``clean_latent_2x_indices`` are
            # not consumed by ``read_camera_params`` (verified); they were
            # tracked here as dead code from an earlier s2v lineage.

            current_clean_latents = _select_validation_clean_latents(
                history_latents,
                section_idx=section_idx,
                requested_history_count=validation_clean_latent_history_count,
            )
            current_clean_latent_count = int(current_clean_latents.shape[2])
            section_t2v_no_ref = bool(t2v_no_ref and section_idx == 0)
            model_clean_latent_count = 0 if section_t2v_no_ref else current_clean_latent_count
            current_clean_frame_count = _validation_clean_prefix_frame_count(
                model_clean_latent_count
            )

            recalculate_input = {
                "clean_latent_indices_start": data["clean_latent_indices_start"],
                "clean_latent_indices": running_clean_indices,
                "noisy_latent_indices": running_noisy_indices,
                "clean_latents": current_clean_latents,
                "info": data["info"],
                "lookup": data.get("lookup", {}),
                "is_starting": (section_idx == 0),
                "camera_pingpong": camera_pingpong,
            }
            camera_data = val_dataset.read_camera_params(
                recalculate_input,
                info=data["info"],
                lookup=data.get("lookup", {}),
            )
            prope_camera_kwargs = _prope_camera_kwargs(camera_data)
            if section_t2v_no_ref:
                prope_camera_kwargs["clean_latent_indices_prope_extrinsic"] = (
                    prope_camera_kwargs["clean_latent_indices_prope_extrinsic"][:0]
                )
                prope_camera_kwargs["clean_latent_indices_prope_intrinsic"] = (
                    prope_camera_kwargs["clean_latent_indices_prope_intrinsic"][:0]
                )
            if (not section_t2v_no_ref) and current_clean_latent_count > 1:
                prope_camera_kwargs["clean_latent_indices_prope_extrinsic"] = (
                    _validation_recent_frame_window(
                        clean_history_extr,
                        required_count=current_clean_latent_count * 4,
                    )
                )
                prope_camera_kwargs["clean_latent_indices_prope_intrinsic"] = (
                    _validation_recent_frame_window(
                        clean_history_intr,
                        required_count=current_clean_latent_count * 4,
                    )
                )
            print(
                f"[validation] sec {section_idx} clean prefix  "
                f"latents={tuple(current_clean_latents.shape)}  "
                f"clean_latent_count={model_clean_latent_count}  "
                f"clean_frames_for_decode_drop={current_clean_frame_count}  "
                f"prope_clean_frames="
                f"{int(prope_camera_kwargs['clean_latent_indices_prope_extrinsic'].shape[0])}"
            )
            mosaic_revgrid = None
            mosaic_latent = None
            mosaic_view_change = None
            latent_rope_time_indices = None
            validation_pool_clean_info = None
            validation_neighbor_stats = {"enabled": False}
            if use_mosaic:
                if section_idx == 0:
                    register_frames = _decode_latents_to_numpy_frames(
                        unwrapped_model.pipe,
                        current_clean_latents,
                        unwrapped_model.pipe.device,
                        tiled=vae_decode_tiled,
                    )
                    print(
                        f"[validation] sec 0 register start frame  "
                        f"frames.shape={register_frames.shape}  "
                        f"start_index=0  "
                        f"force_using_input_extrinics={args.force_using_input_extrinics}"
                    )
                    if args.force_using_input_extrinics:
                        # ``register_frames`` is decoded from
                        # ``current_clean_latents = history_latents[:, :, -1:]``,
                        # i.e. the latest clean latent (at sec 0 this is the
                        # bootstrap frame at clean_frame_indices[-1] == G-1).
                        # The matching dataset-supplied (intrinsics,
                        # extrinsics) live at the LAST entry of the per-frame
                        # clean_* slot. Slice to (1, ...) so the shapes line
                        # up with ``register_frames.shape[0] == 1``.
                        sec0_extr = (
                            data["clean_latent_indices_prope_extrinsic"][-1:]
                            .cpu()
                            .numpy()
                        )
                        sec0_intr = (
                            data["clean_latent_indices_prope_intrinsic"][-1:]
                            .cpu()
                            .numpy()
                        )
                        handler.register_source_sequence(
                            unwrapped_model.pipe.device,
                            register_frames,
                            start_index=0,
                            extrinsics=sec0_extr,
                            intrinsics=sec0_intr,
                            force_using_input_extrinics=True,
                            cache_frames=(
                                _is_recent_first_fuse_mode(args.mosaic_fuse_mode)
                                and args.recent_first_photo_refine
                            ),
                        )
                    else:
                        handler.register_source_sequence(
                            unwrapped_model.pipe.device,
                            register_frames,
                            start_index=0,
                            cache_frames=(
                                _is_recent_first_fuse_mode(args.mosaic_fuse_mode)
                                and args.recent_first_photo_refine
                            ),
                        )
                    _record_validation_candidate_frames(
                        validation_candidate_frames_by_id,
                        register_frames,
                        start_index=0,
                    )
                    _append_validation_dynamic_masks(
                        register_frames,
                        start_index=0,
                        label="anchor",
                    )
                    new_query_latents = _encode_frames_per_frame(
                        unwrapped_model.pipe,
                        register_frames,
                        unwrapped_model.pipe.device,
                        tiled=vae_decode_tiled,
                    )
                    mosaic_query_latents = new_query_latents
                    validation_neighbor_stats = {
                        "enabled": bool(validation_neighbor_window_enabled),
                        "active": False,
                        "reason": "section0_anchor_single_frame",
                    }
                else:
                    print(
                        f"[validation] sec {section_idx} register prev frames  "
                        f"frames.shape={last_predicting_frames.shape}  "
                        f"force_using_input_extrinics={args.force_using_input_extrinics}"
                    )
                    if args.force_using_input_extrinics:
                        if last_predicting_extr is None or last_predicting_intr is None:
                            raise RuntimeError(
                                "[validation] force_using_input_extrinics=True "
                                "but the previous section did not stash "
                                "predicted (extr, intr); did the section-end "
                                "stash get skipped?"
                            )
                        n_pred = int(last_predicting_frames.shape[0])
                        if (
                            last_predicting_extr.shape[0] != n_pred
                            or last_predicting_intr.shape[0] != n_pred
                        ):
                            raise RuntimeError(
                                f"[validation] stashed (extr, intr) length "
                                f"({last_predicting_extr.shape[0]}, "
                                f"{last_predicting_intr.shape[0]}) does not "
                                f"match last_predicting_frames length {n_pred}."
                            )
                        handler.register_source_sequence(
                            unwrapped_model.pipe.device,
                            last_predicting_frames,
                            extrinsics=last_predicting_extr,
                            intrinsics=last_predicting_intr,
                            force_using_input_extrinics=True,
                            cache_frames=(
                                _is_recent_first_fuse_mode(args.mosaic_fuse_mode)
                                and args.recent_first_photo_refine
                            ),
                        )
                    else:
                        handler.register_source_sequence(
                            unwrapped_model.pipe.device,
                            last_predicting_frames,
                            cache_frames=(
                                _is_recent_first_fuse_mode(args.mosaic_fuse_mode)
                                and args.recent_first_photo_refine
                            ),
                        )
                    register_start_index = len(validation_candidate_frames_by_id)
                    _record_validation_candidate_frames(
                        validation_candidate_frames_by_id,
                        last_predicting_frames,
                        start_index=register_start_index,
                    )
                    _append_validation_dynamic_masks(
                        last_predicting_frames,
                        start_index=register_start_index,
                        label="rollout",
                    )
                    if validation_neighbor_window_enabled:
                        if last_predicting_dense_latents is None:
                            raise RuntimeError(
                                "[validation] memory_neighbor_window_val=True but "
                                "the previous section did not stash dense latents."
                            )
                        new_query_latents, validation_neighbor_stats = (
                            _build_validation_neighbor_window_memory_latents(
                                pipe=unwrapped_model.pipe,
                                frames_np=last_predicting_frames,
                                dense_latents=last_predicting_dense_latents,
                                extrinsics=last_predicting_extr,
                                device=unwrapped_model.pipe.device,
                                tiled=vae_decode_tiled,
                                max_trans_m=getattr(
                                    args, "memory_neighbor_window_max_trans_m", 0.5
                                ),
                                max_rot_deg=getattr(
                                    args, "memory_neighbor_window_max_rot_deg", 10.0
                                ),
                            )
                        )
                        print(
                            f"[validation] sec {section_idx} "
                            "memory_neighbor_window_val "
                            f"S1_groups={validation_neighbor_stats['num_s1_groups']} "
                            f"W5_groups={validation_neighbor_stats['num_w5_groups']} "
                            f"reencoded_frames={validation_neighbor_stats['num_reencoded_frames']} "
                            f"reused_frames={validation_neighbor_stats['num_reused_frames']}"
                        )
                    else:
                        new_query_latents = _encode_frames_per_frame(
                            unwrapped_model.pipe,
                            last_predicting_frames,
                            unwrapped_model.pipe.device,
                            tiled=vae_decode_tiled,
                        )
                        validation_neighbor_stats = {
                            "enabled": bool(
                                getattr(args, "memory_neighbor_window_val", False)
                            ),
                            "active": False,
                            "reason": (
                                "force_using_input_extrinics_false"
                                if getattr(args, "memory_neighbor_window_val", False)
                                else "flag_disabled"
                            ),
                        }
                    mosaic_query_latents = torch.cat(
                        [mosaic_query_latents, new_query_latents], dim=2
                    )

                clean_extr = camera_data["clean_latent_indices_prope_extrinsic"]
                noisy_extr = camera_data["noisy_latent_indices_prope_extrinsic"]
                # Frustum query alignment still uses the current section's
                # latest clean latent camera. Multi-clean history is only
                # expanded in prope_camera_kwargs for the DiT clean prefix.
                query_extrinsic_input = torch.cat([clean_extr, noisy_extr], dim=0)
                query_extrinsic_input_raw_np = query_extrinsic_input.cpu().numpy()
                query_extrinsic_input_np = handler.align_w2c_trajectory(
                    query_extrinsic_input_raw_np[3:]
                )
                query_extrinsic_input_np = query_extrinsic_input_np[1:]
                # Bug-4 fix (per_3 / segment_coverage context SELECTION only): the
                # coverage selector ranks pool latents against the target segment
                # poses. The pool's extrinsics (``clean_history_extr``) are raw
                # dataset-GT poses -- ``read_camera_params`` reloads GT from disk
                # each section and never the DA3-registered poses -- so the target
                # must be compared in that SAME raw-GT frame. The aligned
                # ``query_extrinsic_input_np`` above re-anchors the target into the
                # handler's registered frame, which is the DA3-ESTIMATED frame when
                # --force_using_input_extrinics is off (the default) and drifts per
                # section; ranking a raw-GT pool against it gives cross-frame,
                # non-reproducible distances. Pass an UN-aligned copy with the
                # identical element selection (== raw[3:][1:]) so pool and target
                # share the dataset-GT frame -- which also matches training, where
                # both pool and target are raw w2c. The aligned array is still used
                # verbatim for the frustum query / generation below (line ~1209).
                query_extrinsic_target_np = query_extrinsic_input_raw_np[3:][1:]
                # ``latent_merge_4frames=False`` -> 1 latent / 1 RGB frame
                # along T, indexed 1:1 with ``handler.extrinsics``. We
                # therefore feed the per-frame buffer instead of
                # ``history_latents`` (which is WAN-VAE compressed).
                source_latents = mosaic_query_latents[0].clone()
                registered_count = len(handler.extrinsics)
                print(
                    f"[validation] sec {section_idx} query  "
                    f"query_extrinsic={query_extrinsic_input_np.shape}  "
                    f"source_latents={tuple(source_latents.shape)}  "
                    f"mosaic_query_latents={tuple(mosaic_query_latents.shape)}  "
                    f"history_latents={tuple(history_latents.shape)}  "
                    f"registered_frames={registered_count}"
                )
                if source_latents.shape[1] != registered_count:
                    raise RuntimeError(
                        f"[validation] source_latents T={source_latents.shape[1]} "
                        f"does not match handler registered frames "
                        f"{registered_count}; per-frame mosaic buffer drifted "
                        f"out of sync with FrustumHandler."
                    )
                if dynamic_filter_enabled and mosaic_source_valid_masks is None:
                    mosaic_source_valid_masks = torch.ones(
                        registered_count,
                        H_lat,
                        W_lat,
                        dtype=torch.bool,
                    )
                source_valid_shape = (
                    None
                    if mosaic_source_valid_masks is None
                    else tuple(mosaic_source_valid_masks.shape)
                )
                if (
                    mosaic_source_valid_masks is not None
                    and mosaic_source_valid_masks.shape[0] != registered_count
                ):
                    raise RuntimeError(
                        f"[validation] source_valid_masks T="
                        f"{mosaic_source_valid_masks.shape[0]} does not match "
                        f"handler registered frames {registered_count}."
                    )
                validation_candidate_budget = args.candidates_per_query_group_val
                validation_nms_mode = getattr(args, "mosaic_candidate_nms_mode", "none")
                validation_nms_mode = (
                    None
                    if validation_nms_mode in (None, "", "none", "None")
                    else str(validation_nms_mode)
                )
                # Debug-mode timing: when ``--debug`` is on we also dump a
                # sorted breakdown of the slowest inner ``_timing_scope``
                # events emitted by ``query_hits_mode_new`` (select_candidates
                # + fuse_candidates + per-group splat). Outside debug mode
                # ``timing.scope`` and ``_format_timing_breakdown`` are
                # no-ops, and only the wall-clock total is printed.
                debug_query_timing = _debug_timing_enabled(args)
                events_baseline = _timing_record_event_count(timing)
                validation_fuse_mode = _normalize_mosaic_fuse_mode(
                    args.mosaic_fuse_mode
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with timing.scope(
                    "validation.query_hits_mode_new",
                    metadata={
                        "section_idx": int(section_idx),
                        "fuse_mode": str(validation_fuse_mode),
                        "candidate_budget": int(validation_candidate_budget),
                        "query_count": int(query_extrinsic_input_np.shape[0]),
                        "source_T": int(source_latents.shape[1]),
                        "registered_frames": int(registered_count),
                        "source_valid_masks": str(source_valid_shape),
                    },
                ):
                    queried_latent_mosaic_revgrid = handler.query_hits_mode_new(
                        unwrapped_model.pipe.device,
                        query_extrinsic_input_np,
                        source_latents,
                        candidates_per_query_group=validation_candidate_budget,
                        angle_threshold=None,
                        distance_threshold=None,
                        temporal_threshold=None,
                        fuse_mode=validation_fuse_mode,
                        zbuffer_depth_preference="near",
                        interpolation_mode="nearest",
                        return_revgrid=args.mosaic_use_revgrid_rope,
                        return_candidate_frame_ids=True,
                        return_view_change=bool(use_mosaic_view_change_prope),
                        latent_merge_4frames=False,
                        query_reference_frame=args.mosaic_query_reference_frame,
                        selection_mode=getattr(
                            args, "mosaic_selection_mode", "projection_iou"
                        ),
                        candidate_nms_mode=validation_nms_mode,
                        candidate_nms_projection_iou_threshold=float(
                            args.mosaic_candidate_nms_projection_iou_threshold
                        ),
                        candidate_nms_min_temporal_gap=int(
                            args.mosaic_candidate_nms_min_temporal_gap
                        ),
                        candidate_nms_pose_distance_threshold=float(
                            args.mosaic_candidate_nms_pose_distance_threshold
                        ),
                        candidate_nms_pool_multiplier=float(
                            args.mosaic_candidate_nms_pool_multiplier
                        ),
                        coverage_grid_downsample=int(
                            getattr(args, "mosaic_coverage_grid_downsample", 4)
                        ),
                        coverage_pool_stride=int(
                            getattr(args, "mosaic_coverage_pool_stride", 2)
                        ),
                        source_valid_masks=mosaic_source_valid_masks,
                        recent_first_cfg=(
                            {
                                "photo_refine": args.recent_first_photo_refine,
                                "dedupe_tiles": args.recent_first_dedupe_tiles,
                                "warn_min_candidates": args.candidates_per_query_group_val,
                                "candidate_geometry_device": args.recent_first_geometry_device,
                                "backend": args.recent_first_backend,
                                "warp_dtype": args.recent_first_warp_dtype,
                                "enable_tf32": args.recent_first_enable_tf32,
                                "recent_first_zbuffer": args.recent_first_zbuffer,
                                "n_candidates_keyframe": args.recent_first_n_candidates_keyframe,
                                "keyframe_neighbour_window": args.recent_first_keyframe_neighbour_window,
                            }
                            if _is_recent_first_fuse_mode(args.mosaic_fuse_mode)
                            else None
                        ),
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                query_elapsed = time.perf_counter() - start
                if debug_query_timing:
                    # Exclude the outer wrapping scope from the breakdown so
                    # we see only the inner contributors (and their counts
                    # per group / per candidate).
                    breakdown = _format_timing_breakdown(
                        timing,
                        events_baseline,
                        top_k=15,
                        exclude_names={"validation.query_hits_mode_new"},
                    )
                    if breakdown:
                        print(
                            f"[validation] sec {section_idx} "
                            f"query_hits_mode_new time={query_elapsed:.3f}s "
                            f"breakdown (top events, cuda_sync=on):\n{breakdown}"
                        )
                    else:
                        print(
                            f"[validation] sec {section_idx} "
                            f"query_hits_mode_new time={query_elapsed:.3f}s "
                            f"(no inner timing events captured)"
                        )
                else:
                    print(
                        f"[validation] sec {section_idx} "
                        f"query_hits_mode_new time={query_elapsed:.3f}s"
                    )
                (
                    queried_latent,
                    validation_candidate_frame_ids,
                    mosaic_revgrid,
                    mosaic_view_change,
                ) = _parse_query_hits_with_candidate_frame_ids(
                    queried_latent_mosaic_revgrid,
                    return_revgrid=bool(args.mosaic_use_revgrid_rope),
                    return_view_change=bool(use_mosaic_view_change_prope),
                )
                validation_candidate_groups_history.extend(
                    _candidate_groups_to_lists(validation_candidate_frame_ids)
                )
                candidate_group_count = len(
                    _candidate_groups_to_lists(validation_candidate_frame_ids)
                )
                if (
                    int(section_idx) > 0
                    and validation_clean_latent_pool_history
                    and (
                        validation_clean_latent_history_count > 1
                        or validation_per3_context
                    )
                ):
                    # per_3 forces the requested count to latent_window_size//3
                    # context latents + the always-present anchor, ignoring the
                    # count knob. We pass it ONLY as this local request value so
                    # the qwen prompt path (which reads
                    # validation_clean_latent_history_count) stays decoupled; the
                    # rope-time / decode-drop / prope consumers below follow the
                    # actual selected tensor (current_clean_latent_count).
                    pool_requested_history_count = (
                        latent_window_size // 3 + 1
                        if validation_per3_context
                        else validation_clean_latent_history_count
                    )
                    validation_pool_clean_info = _select_validation_clean_latents_from_candidate_pool(
                        history_latents,
                        validation_candidate_frame_ids,
                        clean_history_extr=clean_history_extr,
                        clean_history_intr=clean_history_intr,
                        requested_history_count=pool_requested_history_count,
                        selection_mode=validation_clean_latent_pool_history_selection_mode,
                        target_extrinsics=query_extrinsic_target_np,
                    )
                    current_clean_latents = validation_pool_clean_info["clean_latents"]
                    current_clean_latent_count = int(current_clean_latents.shape[2])
                    current_clean_frame_count = _validation_clean_prefix_frame_count(
                        current_clean_latent_count
                    )
                    prope_camera_kwargs["clean_latent_indices_prope_extrinsic"] = (
                        validation_pool_clean_info["clean_extrinsic"]
                    )
                    prope_camera_kwargs["clean_latent_indices_prope_intrinsic"] = (
                        validation_pool_clean_info["clean_intrinsic"]
                    )
                    print(
                        f"[validation] sec {section_idx} pool clean prefix  "
                        f"selected_latents={validation_pool_clean_info['selected_latent_indices']}  "
                        f"pool_latents={validation_pool_clean_info['pool_latent_indices']}  "
                        f"fallback_latents={validation_pool_clean_info['fallback_latent_indices']}  "
                        f"selection_mode={validation_clean_latent_pool_history_selection_mode}  "
                        f"camera_frames={validation_pool_clean_info['camera_frame_indices']}  "
                        f"clean_latent_count={current_clean_latent_count}  "
                        f"prope_clean_frames="
                        f"{int(prope_camera_kwargs['clean_latent_indices_prope_extrinsic'].shape[0])}"
                    )
                    pool_clean_frame_ids = (
                        _validation_pool_clean_frame_ids_from_selection(
                            validation_pool_clean_info
                        )
                    )
                else:
                    pool_clean_frame_ids = []
                validation_pool_clean_groups_history.extend(
                    [list(pool_clean_frame_ids) for _ in range(candidate_group_count)]
                )
                # Validation is pinned to interval1 under the synchronised
                # training schedule (see --mosaic_train_random_schedule);
                # otherwise it honours the legacy --mosaic_interval.
                val_interval = (
                    1
                    if getattr(args, "mosaic_train_random_schedule", False)
                    else int(args.mosaic_interval)
                )
                validation_mosaic_indices = _compute_mosaic_frame_indices(
                    int(queried_latent.shape[1]),
                    val_interval,
                    device=queried_latent.device,
                )
                queried_latent_full_shape = tuple(queried_latent.shape)
                queried_latent = queried_latent.index_select(
                    1, validation_mosaic_indices
                )
                if mosaic_revgrid is not None:
                    mosaic_revgrid = np.asarray(mosaic_revgrid)[
                        validation_mosaic_indices.detach().cpu().numpy()
                    ]
                if mosaic_view_change is not None:
                    mosaic_view_change = np.asarray(mosaic_view_change)[
                        validation_mosaic_indices.detach().cpu().numpy()
                    ]
                mosaic_latent = queried_latent.unsqueeze(0)
                if validation_pool_clean_info is not None:
                    latent_rope_time_indices = _build_validation_pool_rope_time_indices(
                        clean_latent_count=current_clean_latent_count,
                        noisy_latent_count=latent_window_size,
                        mosaic_frame_indices=validation_mosaic_indices,
                        history_bias=validation_clean_latent_pool_history_bias,
                        selected_latent_indices=validation_pool_clean_info[
                            "selected_latent_indices"
                        ],
                        use_original_latent_indices=bool(
                            getattr(
                                args,
                                "validation_clean_latent_pool_history_rope_original_indices",
                                False,
                            )
                        ),
                    )
                    rope_time_indices_for_log = (
                        latent_rope_time_indices.detach().cpu().tolist()
                    )
                    mosaic_rope_start = current_clean_latent_count
                    mosaic_rope_end = mosaic_rope_start + int(
                        validation_mosaic_indices.numel()
                    )
                    clean_rope_times = rope_time_indices_for_log[
                        :current_clean_latent_count
                    ]
                    mosaic_rope_times = rope_time_indices_for_log[
                        mosaic_rope_start:mosaic_rope_end
                    ]
                    noisy_rope_times = rope_time_indices_for_log[mosaic_rope_end:]
                    print(
                        "[validation][pool-rope] "
                        f"sec {section_idx} "
                        "validation_clean_latent_pool_history_bias="
                        f"{validation_clean_latent_pool_history_bias} "
                        "rope_original_indices="
                        f"{bool(getattr(args, 'validation_clean_latent_pool_history_rope_original_indices', False))} "
                        f"clean_latent_count={current_clean_latent_count} "
                        f"noisy_latent_count={latent_window_size} "
                        f"selected_latents={validation_pool_clean_info['selected_latent_indices']} "
                        f"clean_rope={clean_rope_times} "
                        f"mosaic_rope={mosaic_rope_times} "
                        f"noisy_rope={noisy_rope_times} "
                        f"latent_rope_time_indices={rope_time_indices_for_log}",
                        flush=True,
                    )
                print(
                    f"[validation] sec {section_idx} queried_latent.shape="
                    f"{tuple(mosaic_latent.shape)} full={queried_latent_full_shape} "
                    f"mosaic_interval={val_interval} "
                    f"indices={validation_mosaic_indices.detach().cpu().tolist()}"
                )
            else:
                validation_mosaic_indices = None

            if section_t2v_no_ref:
                input_image_arg = None
                first_frame_latents_arg = None
            elif getattr(args, "validation_use_decoded_first_image", False):
                # Legacy path: VAE-decode the clean latent to a PIL image and
                # let the pipeline VAE-encode it back as `first_frame_latents`.
                input_image_arg = _decoded_first_image(
                    unwrapped_model.pipe,
                    current_clean_latents,
                    unwrapped_model.pipe.device,
                    tiled=vae_decode_tiled,
                )
                first_frame_latents_arg = None
            else:
                # Default path: feed the clean latent directly to the pipeline,
                # skipping the VAE decode/encode round-trip.
                input_image_arg = None
                first_frame_latents_arg = current_clean_latents.to(
                    device=unwrapped_model.pipe.device,
                    dtype=unwrapped_model.pipe.torch_dtype,
                )

            # Per-section prompt: ``read_camera_params`` was called above with
            # the drifted ``noisy_latent_indices`` for this section, so DA3
            # prompt refresh already reflects the caption segment(s) that the
            # section covers.
            # Fall back to ``data["prompt"]`` (section 0's prompt) when the
            # dataset doesn't expose a refreshed prompt.
            dataset_section_prompt = (
                camera_data.get("prompt") or data.get("prompt") or ""
            )
            if validation_qwen_prompt:
                qwen_camera_summary = _summarize_validation_camera_motion(
                    _validation_qwen_extrinsics_to_numpy(
                        camera_data["noisy_latent_indices_prope_extrinsic"]
                    )
                )
                if validation_qwen_mode == "oracle_full_gt_static":
                    if validation_qwen_static_prompt is None:
                        full_camera_summary = (
                            "; ".join(
                                summary
                                for summary in validation_qwen_camera_summaries
                                if summary
                            )
                            or qwen_camera_summary
                        )
                        section_qwen_payload = dict(validation_qwen_full_payload or {})
                        section_qwen_payload.update(
                            {
                                "mode": validation_qwen_mode,
                                "event": None,
                                "camera_summary": full_camera_summary,
                                "total_sections": int(num_validation_blocks),
                                "image_source": "gt_full_rollout",
                            }
                        )
                        with timing.scope(
                            "validation.qwen_prompt",
                            metadata={
                                "batch_idx": int(batch_idx),
                                "section_idx": int(section_idx),
                                "clean_name": str(clean_name),
                                "mode": validation_qwen_mode,
                                "camera_summary": full_camera_summary,
                            },
                        ):
                            validation_qwen_static_prompt = (
                                _validation_qwen_generate_from_payload(
                                    qwen_prompt_generator,
                                    section_qwen_payload,
                                    batch_idx=batch_idx,
                                    section_idx=section_idx,
                                    clean_name=clean_name,
                                    dataset_prompt=dataset_section_prompt,
                                    timing_reporter=timing,
                                    transient_cuda_device=unwrapped_model.pipe.device,
                                )
                            )
                    section_prompt = validation_qwen_static_prompt
                    prompt_source = "qwen"
                    print(
                        f"[validation][qwen][rank {proc_rank}] sec {section_idx} "
                        "using static full-rollout prompt",
                        flush=True,
                    )
                else:
                    if validation_qwen_mode == "oracle_section_gt":
                        section_qwen_payload = (
                            validation_qwen_frames_by_section.get(section_idx)
                            if validation_qwen_frames_by_section is not None
                            else None
                        )
                        image_source = "gt_section"
                    else:
                        if section_idx == 0:
                            context_latents = _validation_qwen_initial_context_latents(
                                history_latents,
                                requested_history_count=(
                                    validation_clean_latent_history_count
                                ),
                            )
                            context_frames = _decode_latents_to_numpy_frames(
                                unwrapped_model.pipe,
                                context_latents,
                                unwrapped_model.pipe.device,
                                tiled=vae_decode_tiled,
                            )
                        else:
                            if last_predicting_frames is None:
                                raise RuntimeError(
                                    "[validation][qwen][self_rollout] previous segment "
                                    f"frames are unavailable for section {section_idx}."
                                )
                            context_frames = last_predicting_frames
                        section_qwen_payload = _validation_qwen_context_prompt_frames(
                            context_frames,
                            frame_count=validation_clean_latent_history_count,
                            frame_max_side=args.validation_qwen_frame_max_side,
                        )
                        image_source = "rollout_context"
                        print(
                            f"[validation][qwen][rank {proc_rank}] sec {section_idx} "
                            f"{validation_qwen_mode} prompt frames "
                            f"indices={section_qwen_payload['frame_indices']} "
                            f"sizes={[getattr(image, 'size', None) for image in section_qwen_payload['images']]}",
                            flush=True,
                        )
                    section_qwen_payload = dict(section_qwen_payload or {})
                    section_qwen_payload.update(
                        {
                            "mode": validation_qwen_mode,
                            "event": (
                                validation_qwen_event
                                if validation_qwen_mode == "self_rollout_event"
                                else None
                            ),
                            "camera_summary": qwen_camera_summary,
                            "total_sections": int(num_validation_blocks),
                            "image_source": image_source,
                        }
                    )
                    print(
                        f"[validation][qwen][rank {proc_rank}] sec {section_idx} "
                        f"dataset prompt len={len(dataset_section_prompt)} "
                        f"camera={qwen_camera_summary!r}",
                        flush=True,
                    )
                    with timing.scope(
                        "validation.qwen_prompt",
                        metadata={
                            "batch_idx": int(batch_idx),
                            "section_idx": int(section_idx),
                            "clean_name": str(clean_name),
                            "mode": validation_qwen_mode,
                            "camera_summary": qwen_camera_summary,
                        },
                    ):
                        section_prompt = _validation_qwen_generate_from_payload(
                            qwen_prompt_generator,
                            section_qwen_payload,
                            batch_idx=batch_idx,
                            section_idx=section_idx,
                            clean_name=clean_name,
                            dataset_prompt=dataset_section_prompt,
                            timing_reporter=timing,
                            transient_cuda_device=unwrapped_model.pipe.device,
                        )
                    prompt_source = "qwen"
            else:
                section_prompt = dataset_section_prompt
                prompt_source = "dataset"
            section_prompt = _require_nonempty_prompt(
                section_prompt,
                phase=f"validation {prompt_source} sec={section_idx}",
                clean_name=clean_name,
            )
            print(
                f"[validation][prompt][rank {proc_rank}] "
                f"batch={batch_idx} sec={section_idx} clean_name={clean_name} "
                f"source={prompt_source} len={len(section_prompt)} START\n"
                f"{section_prompt}\n"
                f"[validation][prompt][rank {proc_rank}] "
                f"batch={batch_idx} sec={section_idx} clean_name={clean_name} END",
                flush=True,
            )
            section_output = unwrapped_model.inference_step(
                prompt=section_prompt,
                negative_prompt=args.validation_negative_prompt,
                input_image=input_image_arg,
                first_frame_latents=first_frame_latents_arg,
                height=H_img,
                width=W_img,
                num_frames=4 * (latent_window_size - 1) + 1,
                # Per-section term: without it every section of a rollout
                # denoises from the SAME noise tensor (training never sees
                # cross-window-correlated noise), causing repeated textures
                # and periodic seams.
                seed=int(args.validation_seed) + batch_idx * 1000 + section_idx,
                num_inference_steps=args.validation_num_inference_steps,
                cfg_scale=args.validation_cfg_scale,
                negative_no_prope=bool(
                    getattr(args, "validation_negative_no_prope", False)
                ),
                negative_no_context=bool(
                    getattr(args, "validation_negative_no_context", False)
                ),
                mosaic_latent=mosaic_latent,
                mosaic_revgrid=mosaic_revgrid,
                mosaic_use_revgrid_rope=args.mosaic_use_revgrid_rope,
                mosaic_view_change=mosaic_view_change,
                mosaic_view_change_prope=use_mosaic_view_change_prope,
                mosaic_frame_indices=validation_mosaic_indices,
                mosaic_drop_holes=bool(
                    getattr(args, "mosaic_train_random_schedule", False)
                    or getattr(args, "mosaic_drop_holes", False)
                ),
                tiled=vae_decode_tiled,
                prope_camera_kwargs=prope_camera_kwargs,
                latent_rope_time_indices=latent_rope_time_indices,
                subject_ref_latents=(
                    data.get("subject_ref_latents")
                    if (
                        getattr(unwrapped_model, "subject_ref_memory", False)
                        and torch.is_tensor(data.get("subject_ref_latents"))
                        and int(data["subject_ref_latents"].shape[0]) > 0
                    )
                    else None
                ),
                subject_ref_slot_ratio=float(
                    getattr(unwrapped_model, "subject_ref_canvas_slot_ratio", 0.5)
                ),
                subject_ref_time_gap=int(
                    getattr(unwrapped_model, "subject_ref_time_gap", 1)
                ),
                subject_ref_prope_mode=str(
                    getattr(unwrapped_model, "subject_ref_prope_mode", "identity")
                    or "identity"
                ),
            )
            if not isinstance(section_output, torch.Tensor):
                raise RuntimeError(
                    "Mosaic validation expects pipeline to return latent tensors "
                    "(return_latent=True)."
                )
            section_latents = section_output.detach().cpu()
            expected_section_latent_count = (
                model_clean_latent_count + latent_window_size
            )
            print(
                f"[validation] sec {section_idx} pipe latent shape="
                f"{tuple(section_latents.shape)}  "
                f"(expect (1,C,{expected_section_latent_count},H,W) = "
                f"{model_clean_latent_count} clean + {latent_window_size} noisy)"
            )
            if int(section_latents.shape[2]) != int(expected_section_latent_count):
                raise RuntimeError(
                    f"[validation] sec {section_idx} pipe returned "
                    f"{section_latents.shape[2]} latent steps, expected "
                    f"{expected_section_latent_count} "
                    f"({model_clean_latent_count} clean + "
                    f"{latent_window_size} noisy)."
                )
            decoded_full = _decode_latents_to_numpy_frames(
                unwrapped_model.pipe,
                section_latents,
                unwrapped_model.pipe.device,
                tiled=vae_decode_tiled,
            )
            clean_decode_drop_count = _validation_clean_prefix_decode_drop_count(
                model_clean_latent_count
            )
            if decoded_full.shape[0] <= clean_decode_drop_count:
                continue
            section_visualization = decoded_full[clean_decode_drop_count:]
            print(
                f"[validation] sec {section_idx} decoded {decoded_full.shape[0]} frames "
                f"({section_visualization.shape[0]} after dropping "
                f"{clean_decode_drop_count} clean-prefix frames)"
            )

            temp_verbose_path = os.path.join(
                validation_temp_dir,
                f"epoch_{epoch_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}"
                f"_part_{section_idx:03d}_temp_verbose.mp4",
            )
            section_start_frame = (section_idx * 4 * latent_window_size) + (0 if section_t2v_no_ref else 1)
            section_verbose_frames = _annotate_frames_with_index(
                section_visualization, start_index=section_start_frame
            )
            _write_video(temp_verbose_path, section_verbose_frames)
            history_temp_init_s = 0.0
            if history_temp_writer is None:
                step_start = time.perf_counter()
                with timing.scope(
                    "validation.history_temp.init_anchor",
                    metadata={
                        "section_idx": int(section_idx),
                        "batch_idx": int(batch_idx),
                        "reuse_register_frames": bool(
                            use_mosaic
                            and section_idx == 0
                            and register_frames is not None
                            and int(history_latents.shape[2])
                            == int(current_clean_latents.shape[2])
                        ),
                    },
                ):
                    if section_t2v_no_ref:
                        history_temp_anchor_frames = None
                        history_temp_anchor_count = 0
                        history_temp_mosaic_anchor_frames = None
                    elif (
                        use_mosaic
                        and section_idx == 0
                        and register_frames is not None
                        and int(history_latents.shape[2])
                        == int(current_clean_latents.shape[2])
                    ):
                        history_temp_anchor_frames = register_frames
                        history_temp_anchor_count = int(len(history_temp_anchor_frames))
                        history_temp_mosaic_anchor_frames = [
                            np.asarray(frame).copy() for frame in history_temp_anchor_frames
                        ]
                    else:
                        history_temp_anchor_frames = _decode_latents_to_numpy_frames(
                            unwrapped_model.pipe,
                            history_latents,
                            unwrapped_model.pipe.device,
                            tiled=vae_decode_tiled,
                        )
                        history_temp_anchor_count = int(len(history_temp_anchor_frames))
                        history_temp_mosaic_anchor_frames = [
                            np.asarray(frame).copy() for frame in history_temp_anchor_frames
                        ]
                    history_temp_candidate_anchor_frames = None
                    history_temp_pool_anchor_frames = None
                    if use_mosaic:
                        history_temp_candidate_anchor_frames = _build_validation_candidate_history_frames(
                            history_frame_count=history_temp_anchor_count,
                            single_frame_shape=history_temp_anchor_frames[0].shape,
                            candidate_groups=[],
                            candidate_frames_by_id=validation_candidate_frames_by_id,
                        )
                        if validation_clean_latent_pool_history:
                            history_temp_pool_anchor_frames = _build_validation_candidate_history_frames(
                                history_frame_count=history_temp_anchor_count,
                                single_frame_shape=history_temp_anchor_frames[0].shape,
                                candidate_groups=[],
                                candidate_frames_by_id=validation_candidate_frames_by_id,
                            )
                    history_temp_writer = _ValidationHistoryTempVideoWriter(
                        history_temp_path,
                        initial_history_frames=history_temp_anchor_frames,
                        initial_mosaic_history_frames=history_temp_mosaic_anchor_frames,
                        initial_candidate_history_frames=history_temp_candidate_anchor_frames,
                        initial_pool_clean_history_frames=history_temp_pool_anchor_frames,
                    )
                history_temp_init_s = time.perf_counter() - step_start

            last_predicting_frames = section_visualization
            if args.force_using_input_extrinics:
                # Stash the per-frame (extrinsics, intrinsics) that drove
                # THIS section's prediction so the NEXT section's
                # ``register_source_sequence`` can replay them.
                #
                # ``noisy_*`` is per-RGB-frame (length ``4*latent_window_size``;
                # see ``noisy_frame_indices`` in the dataset). The
                # pipeline emits ``4*(L-1)+1`` decoded frames and we drop
                # the first one (clean anchor) to obtain
                # ``section_visualization`` of length ``4*(L-1)``. Those
                # predicted frames cover the EARLIEST ``4*(L-1)`` slots
                # of the noisy window (the trailing 4 noisy slots have
                # no decoded counterpart because the pipeline only asked
                # for ``num_frames=4*(L-1)+1``). Take the first
                # ``len(last_predicting_frames)`` entries -- this is
                # robust to L and matches register's per-frame length
                # contract.
                noisy_extr_np = (
                    camera_data["noisy_latent_indices_prope_extrinsic"].cpu().numpy()
                )
                noisy_intr_np = (
                    camera_data["noisy_latent_indices_prope_intrinsic"].cpu().numpy()
                )
                n_pred = int(last_predicting_frames.shape[0])
                if noisy_extr_np.shape[0] < n_pred:
                    raise RuntimeError(
                        f"[validation] noisy_extr length "
                        f"{noisy_extr_np.shape[0]} < predicted frames "
                        f"{n_pred}; cannot stash for next section."
                    )
                last_predicting_extr = noisy_extr_np[:n_pred]
                last_predicting_intr = noisy_intr_np[:n_pred]
                print(
                    f"[validation] sec {section_idx} stash for next register  "
                    f"extr.shape={last_predicting_extr.shape}  "
                    f"intr.shape={last_predicting_intr.shape}"
                )
            n_pred = int(last_predicting_frames.shape[0])
            section_noisy_extr_history = (
                camera_data["noisy_latent_indices_prope_extrinsic"][:n_pred]
                .detach()
                .cpu()
            )
            section_noisy_intr_history = (
                camera_data["noisy_latent_indices_prope_intrinsic"][:n_pred]
                .detach()
                .cpu()
            )
            clean_history_extr = torch.cat(
                [clean_history_extr, section_noisy_extr_history], dim=0
            )
            clean_history_intr = torch.cat(
                [clean_history_intr, section_noisy_intr_history], dim=0
            )
            new_history_chunk = section_latents[:, :, model_clean_latent_count:]
            print(
                f"[validation] sec {section_idx} append history chunk shape="
                f"{tuple(new_history_chunk.shape)}  (expect (1,C,{latent_window_size},H,W))"
            )
            if section_t2v_no_ref:
                history_latents = new_history_chunk.detach().cpu()
                mosaic_latent_history = torch.zeros_like(history_latents)
            else:
                history_latents = torch.cat([history_latents, new_history_chunk], dim=2)
            last_predicting_dense_latents = new_history_chunk.detach().cpu()

            if mosaic_latent is not None:
                section_mosaic_chunk = torch.zeros_like(new_history_chunk).to(
                    dtype=mosaic_latent_history.dtype
                )
                section_mosaic_chunk.index_copy_(
                    2,
                    validation_mosaic_indices.detach().cpu(),
                    mosaic_latent.detach().cpu().to(dtype=mosaic_latent_history.dtype),
                )
                padding_note = "visualization-only black-frame padding"
            else:
                section_mosaic_chunk = torch.zeros_like(new_history_chunk).to(
                    dtype=mosaic_latent_history.dtype
                )
                padding_note = "no mosaic; black-frame padding"
            print(
                f"[validation] sec {section_idx} append mosaic chunk shape="
                f"{tuple(section_mosaic_chunk.shape)} actual_mosaic_M="
                f"{0 if mosaic_latent is None else int(mosaic_latent.shape[2])} "
                f"{padding_note}"
            )
            mosaic_latent_history = torch.cat(
                [mosaic_latent_history, section_mosaic_chunk], dim=2
            )
            history_temp_timing = {"init_anchor_s": float(history_temp_init_s)}
            history_temp_total_start = time.perf_counter()
            step_start = time.perf_counter()
            with timing.scope(
                "validation.history_temp.decode_mosaic",
                metadata={"section_idx": int(section_idx), "batch_idx": int(batch_idx)},
            ):
                section_mosaic_latent_count = int(section_mosaic_chunk.shape[2])
                if section_t2v_no_ref:
                    mosaic_decode_latents = section_mosaic_chunk
                else:
                    mosaic_context_latents = mosaic_latent_history[
                        :,
                        :,
                        -section_mosaic_latent_count - 1 : -section_mosaic_latent_count,
                    ]
                    mosaic_decode_latents = torch.cat(
                        [mosaic_context_latents, section_mosaic_chunk],
                        dim=2,
                    )
                mosaic_temp_full_frames = _decode_latents_to_numpy_frames(
                    unwrapped_model.pipe,
                    mosaic_decode_latents,
                    unwrapped_model.pipe.device,
                    tiled=vae_decode_tiled,
                )
                mosaic_temp_drop_count = 0 if section_t2v_no_ref else _validation_clean_prefix_decode_drop_count(1)
                mosaic_temp_frames = mosaic_temp_full_frames[mosaic_temp_drop_count:]
                expected_mosaic_temp_frames = int(len(section_visualization))
                if len(mosaic_temp_frames) != expected_mosaic_temp_frames:
                    mosaic_temp_frames = mosaic_temp_full_frames[
                        -expected_mosaic_temp_frames:
                    ]
            history_temp_timing["decode_mosaic_s"] = time.perf_counter() - step_start
            candidate_temp_frames = None
            pool_clean_temp_frames = None
            step_start = time.perf_counter()
            if use_mosaic:
                with timing.scope(
                    "validation.history_temp.build_candidate_panels",
                    metadata={
                        "section_idx": int(section_idx),
                        "batch_idx": int(batch_idx),
                    },
                ):
                    candidate_section_groups = _candidate_groups_to_lists(
                        validation_candidate_frame_ids
                    )
                    candidate_temp_frames = _build_validation_candidate_history_frames(
                        history_frame_count=len(section_visualization) + 1,
                        single_frame_shape=history_temp_writer.single_frame_shape,
                        candidate_groups=candidate_section_groups,
                        candidate_frames_by_id=validation_candidate_frames_by_id,
                    )[1:]
                if validation_clean_latent_pool_history:
                    with timing.scope(
                        "validation.history_temp.build_pool_panels",
                        metadata={
                            "section_idx": int(section_idx),
                            "batch_idx": int(batch_idx),
                        },
                    ):
                        pool_group_count = max(
                            0, (int(len(section_visualization)) + 3) // 4
                        )
                        pool_section_groups = [
                            list(pool_clean_frame_ids) for _ in range(pool_group_count)
                        ]
                        pool_clean_temp_frames = _build_validation_candidate_history_frames(
                            history_frame_count=len(section_visualization) + 1,
                            single_frame_shape=history_temp_writer.single_frame_shape,
                            candidate_groups=pool_section_groups,
                            candidate_frames_by_id=validation_candidate_frames_by_id,
                        )[
                            1:
                        ]
            history_temp_timing["build_panels_s"] = time.perf_counter() - step_start
            step_start = time.perf_counter()
            with timing.scope(
                "validation.history_temp.write_video",
                metadata={
                    "section_idx": int(section_idx),
                    "batch_idx": int(batch_idx),
                    "history_frames_before": int(history_temp_writer.frame_count),
                    "append_frames": int(len(section_visualization)),
                },
            ):
                appended_temp_frames = (
                    history_temp_writer.append_section_frames_and_write(
                        section_visualization,
                        mosaic_history_frames=mosaic_temp_frames,
                        candidate_history_frames=candidate_temp_frames,
                        pool_clean_history_frames=pool_clean_temp_frames,
                    )
                )
            history_temp_timing["write_video_s"] = time.perf_counter() - step_start
            step_start = time.perf_counter()
            with timing.scope(
                "validation.history_temp.save_latents_pth",
                metadata={"section_idx": int(section_idx), "batch_idx": int(batch_idx)},
            ):
                _save_validation_history_latents_pth(
                    history_temp_latent_path,
                    history_latents,
                )
            history_temp_timing["save_latents_s"] = time.perf_counter() - step_start
            history_temp_timing["total_s"] = (
                time.perf_counter()
                - history_temp_total_start
                + history_temp_timing["init_anchor_s"]
            )
            timing.add_duration(
                "validation.history_temp.total",
                history_temp_timing["total_s"],
                metadata={
                    "section_idx": int(section_idx),
                    "batch_idx": int(batch_idx),
                    "total_frames": int(history_temp_writer.frame_count),
                    "append_frames": int(appended_temp_frames),
                },
            )
            print(
                f"[validation] sec {section_idx} saved temp history verbose video "
                f"appended_frames={appended_temp_frames} "
                f"total_frames={history_temp_writer.frame_count} "
                f"time={history_temp_timing['total_s']:.3f}s "
                f"init_anchor={history_temp_timing['init_anchor_s']:.3f}s "
                f"decode_mosaic={history_temp_timing['decode_mosaic_s']:.3f}s "
                f"build_panels={history_temp_timing['build_panels_s']:.3f}s "
                f"write_video={history_temp_timing['write_video_s']:.3f}s "
                f"save_latents={history_temp_timing['save_latents_s']:.3f}s "
                f"path={history_temp_path} latents={history_temp_latent_path}",
                flush=True,
            )
            mosaic_temp_frames = None
            candidate_temp_frames = None
            pool_clean_temp_frames = None
            section_verbose_frames = None
            print(
                f"[validation] sec {section_idx} totals  "
                f"history_latents={tuple(history_latents.shape)}  "
                f"mosaic_latent_history={tuple(mosaic_latent_history.shape)}"
            )

            all_section_meta[str(section_idx)] = {
                "used_mosaic": mosaic_latent is not None,
                "t2v_no_ref": bool(section_t2v_no_ref),
                "used_prope": True,
                "validation_negative_no_prope": bool(
                    getattr(args, "validation_negative_no_prope", False)
                ),
                "validation_negative_no_context": bool(
                    getattr(args, "validation_negative_no_context", False)
                ),
                "indice_drift_total": int(section_idx * latent_stride),
                "prompt": section_prompt,
                "memory_neighbor_window_val": validation_neighbor_stats,
                "subject_ref_preview_path": subject_ref_preview_path,
                "subject_ref_info": data.get("subject_ref_info") or {},
                "subject_ref_latent_info": data.get("subject_ref_latent_info") or {},
            }

        if history_temp_writer is not None:
            history_temp_writer.close()
            print(
                f"[validation] finalized temp history verbose video "
                f"frames={history_temp_writer.frame_count} path={history_temp_path}",
                flush=True,
            )
            history_temp_writer = None

        history_frames = _decode_latents_to_numpy_frames(
            unwrapped_model.pipe,
            history_latents,
            unwrapped_model.pipe.device,
            tiled=vae_decode_tiled,
        )
        print(
            f"[validation] decoded history_frames.shape={history_frames.shape}  "
            f"(actual latent_count={int(history_latents.shape[2])}, "
            f"frames={1 + 4 * (int(history_latents.shape[2]) - 1)})"
        )

        history_path = os.path.join(
            validation_dir,
            f"epoch_{epoch_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}_history.mp4",
        )
        _write_video(
            history_path,
            _annotate_frames_with_index(list(history_frames), start_index=0),
        )
        # GT video for the comparison strip.
        # DA3 video flow: read the source mp4 directly and resize/crop with
        # the same center_crop_resize that the dataset ran on training frames,
        # so the GT strip lines up pixel-wise with the ``history_frames`` strip
        # from the predicted latents.
        info = data["info"]
        gt_latent_count = history_latents.shape[2]
        num_predicted_frames = 1 + 4 * (gt_latent_count - 1)
        if "latent_path" in info:
            # torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
            raw_latents = torch.load(
                info["latent_path"], weights_only=False, map_location="cpu"
            )["latents"]
            if raw_latents.ndim == 4:
                raw_latents = raw_latents.unsqueeze(0)
            gt_latent_count = min(history_latents.shape[2], raw_latents.shape[2])
            gt_frames = _decode_latents_to_numpy_frames(
                unwrapped_model.pipe,
                raw_latents[:, :, :gt_latent_count],
                unwrapped_model.pipe.device,
                tiled=vae_decode_tiled,
            )
        elif "video_path" in info:
            gt_first = int(info.get("generating_first_frame_idx", 0)) - 1
            gt_indices = list(range(gt_first, gt_first + num_predicted_frames))
            gt_frames = _read_video_gt_frames(
                info["video_path"], gt_indices, H_img, W_img
            )
        else:
            print(
                "[validation] WARNING: data['info'] has neither latent_path "
                "nor video_path; skipping GT strip."
            )
            gt_frames = np.zeros((0, H_img, W_img, 3), dtype=np.uint8)

        mosaic_history_frames = _decode_latents_to_numpy_frames(
            unwrapped_model.pipe,
            mosaic_latent_history,
            unwrapped_model.pipe.device,
            tiled=vae_decode_tiled,
        )
        print(
            f"[validation] decoded gt_frames.shape={gt_frames.shape}  "
            f"mosaic_history_frames.shape={mosaic_history_frames.shape}"
        )

        compared_frames = _compare_videos(
            list(gt_frames),
            list(history_frames),
            list(mosaic_history_frames),
        )
        filename = f"epoch_{epoch_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}.mp4"
        save_path = os.path.join(validation_dir, filename)
        _write_video(
            save_path,
            _annotate_frames_with_index(compared_frames, start_index=0),
        )

        rope_path = os.path.join(
            validation_dir,
            f"epoch_{epoch_id:04d}_{batch_idx:03d}_{proc_tag}_{clean_name}_sections.json",
        )
        import json

        with open(rope_path, "w") as f:
            json.dump(all_section_meta, f, indent=2)
        print(f"[rank {proc_rank}] Saved mosaic validation video to {save_path}")

        # Per-batch release of the big transient buffers built above.
        # Rebinding to None drops the refcount on the iteration's
        # objects so ``gc.collect`` can reclaim them now (for the LAST
        # iteration this is the only thing that frees them before the
        # function returns). Without this + ``empty_cache`` the GPU
        # caching allocator does not reclaim the val-time peak before the
        # next epoch's training starts -- the most common cause of
        # ``epoch 0 fine, epoch 1 OOM`` on this script. Assigning ``None``
        # to a name that may not have been bound (e.g. ``handler`` /
        # ``mosaic_query_latents`` when ``use_mosaic=False``) is safe in
        # Python: the assignment simply creates the local slot.
        handler = None
        mosaic_query_latents = None
        register_frames = None
        last_predicting_frames = None
        last_predicting_dense_latents = None
        decoded_full = None
        section_visualization = None
        section_latents = None
        history_latents = None
        mosaic_latent_history = None
        history_frames = None
        mosaic_history_frames = None
        gt_frames = None
        compared_frames = None
        raw_latents = None
        input_image_arg = None
        first_frame_latents_arg = None
        mosaic_latent = None
        mosaic_revgrid = None
        source_latents = None
        queried_latent = None
        new_history_chunk = None
        section_mosaic_chunk = None
        history_temp_writer = None
        section_verbose_frames = None
        with timing.scope("validation.batch_cleanup"):
            _release_cached_memory()
        val_wait_start = time.perf_counter()

    # Resource teardown (DA3/Qwen/dynamic-object worker release, CUDA-cache
    # reclaim, timing.finish_record, and restoring model.train()) now runs in
    # ``run_mosaic_validation``'s ``finally`` via
    # ``_release_validation_resources`` so it always executes even when a val
    # batch raises; ``accelerator.wait_for_everyone`` is the success-path
    # barrier there.
