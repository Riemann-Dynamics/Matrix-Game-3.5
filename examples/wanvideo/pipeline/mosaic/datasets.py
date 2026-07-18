from .common import *
from .history import _required_context_blocks_min
from .config import (
    RECENT_FIRST_ZBUFFER_MODE,
    _DATASET_TYPE_REGISTRY,
    _apply_intrinsics_mode_aliases,
    _intrinsics_mode_arg,
    _normalize_mosaic_fuse_mode,
)
from .timing import _debug_dataset_timing_enabled


def _resolve_dataset_cls_and_extra_kwargs(args):
    """Pick dataset class and class-specific kwargs from CLI args."""
    args = _apply_intrinsics_mode_aliases(args)
    dataset_type = getattr(args, "dataset_type", "da3_video")
    if dataset_type not in {"da3_video", "da3_video_subject_ref"}:
        raise ValueError(
            "Strict dataset mode supports only --dataset_type=da3_video or "
            "--dataset_type=da3_video_subject_ref; "
            f"got {dataset_type!r}."
        )
    if dataset_type not in _DATASET_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown --dataset_type={dataset_type!r}. "
            f"Available: {list(_DATASET_TYPE_REGISTRY)}"
        )
    dataset_cls = _DATASET_TYPE_REGISTRY[dataset_type]
    if getattr(args, "debug", False):
        from diffsynth.core.data import unified_dataset_timing as _timing_dataset

        dataset_cls = _timing_dataset.DA3MosaicVideoDataset

    extra_kwargs = {}
    extra_kwargs["vipe_prompt_type"] = int(getattr(args, "vipe_prompt_type", 1))
    extra_kwargs["vipe_prompt_segment_format"] = bool(
        getattr(args, "vipe_prompt_segment_format", False)
    )
    extra_kwargs["align_generating_to_prompt_segments"] = bool(
        getattr(args, "align_generating_to_prompt_segments", False)
    )
    extra_kwargs["mosaic_intrinsics_mode"] = _intrinsics_mode_arg(
        args, "train_mosaic_intrinsics_mode", "per_frame"
    )
    extra_kwargs["mosaic_query_reference_frame"] = int(
        getattr(args, "mosaic_query_reference_frame", 2)
    )
    extra_kwargs["mosaic_view_change_prope"] = bool(
        getattr(args, "mosaic_view_change_prope", False)
        and getattr(args, "enable_mosaic", True)
        and (getattr(args, "use_prope", False) or getattr(args, "only_prope", False))
        and not getattr(args, "only_prope", False)
    )
    # Strict DA3 flow reads prompt paths exclusively from sqlite records.
    # Legacy prompt roots/default prompts are intentionally ignored here.
    extra_kwargs["prompt"] = ""
    extra_kwargs["require_depth"] = True
    extra_kwargs["allow_no_prompt"] = bool(getattr(args, "allow_no_prompt", False))
    extra_kwargs["loose_prompt_match"] = bool(getattr(args, "loose_prompt_match", False))
    extra_kwargs["height"] = int(args.height)
    extra_kwargs["width"] = int(args.width)
    # v2 anchor/context plan: the dataset-side block draw is RETIRED for
    # training -- anchor form is decided per step by the trainer plan. The
    # dataset ships the single-frame-anchor window [G-1..G+79] plus, when
    # block anchors are possible, a 4*depth-frame warm-up prefix read in
    # the SAME decode (the trainer keeps it for block steps and slices it
    # off otherwise -- no second video read).
    extra_kwargs["vae_clean_context_blocks_max"] = 0
    extra_kwargs["vae_clean_context_mode_ratio"] = str(
        getattr(args, "vae_clean_context_mode_ratio", "3:7")
    )
    # G lower-bound reservation, shared single-source with the trainer's
    # runtime checks (history._required_context_blocks_min): block anchors
    # need G >= 4*depth+1 warm-up frames before the window; context blocks
    # need enough 4-aligned candidate starts in [4, G-8]. Returns 0 when
    # both axes are off, which restores the legacy free G sampling
    # (arbitrary start, no 4-alignment) for old-vs-new comparison configs.
    block_prob = float(args.train_anchor_block_prob)
    anchor_frames = int(args.train_block_anchor_frames)
    context_max = int(args.train_context_count_max)
    extra_kwargs["train_clean_history_context_blocks_min"] = (
        _required_context_blocks_min(
            anchor_block_prob=block_prob,
            block_anchor_frames=anchor_frames,
            context_count_max=context_max,
        )
    )
    extra_kwargs["train_anchor_warmup_prepend_frames"] = (
        (anchor_frames - 1) // 4 * 4 if block_prob > 0 else 0
    )
    extra_kwargs["memory_latent_cache_dir"] = str(
        getattr(args, "memory_latent_cache_dir", "") or ""
    )
    extra_kwargs["memory_latent_cache_version"] = str(
        getattr(args, "memory_latent_cache_version", "wan2.2_ti2v_5b_vae")
    )
    extra_kwargs["memory_neighbor_window"] = bool(
        getattr(args, "memory_neighbor_window_train", False)
    )
    extra_kwargs["memory_vae_encode_input_frames"] = int(
        getattr(args, "memory_vae_encode_input_frames", 1) or 1
    )
    extra_kwargs["memory_neighbor_window_max_trans_m"] = float(
        getattr(args, "memory_neighbor_window_max_trans_m", 0.2)
    )
    extra_kwargs["memory_neighbor_window_max_rot_deg"] = float(
        getattr(args, "memory_neighbor_window_max_rot_deg", 5.0)
    )
    extra_kwargs["dataset_compact_mode"] = bool(
        getattr(args, "dataset_compact_mode", False)
    )
    if getattr(args, "debug", False):
        extra_kwargs["dataset_timing_enabled"] = _debug_dataset_timing_enabled(args)
        extra_kwargs["dataset_timing_dir"] = os.path.join(
            getattr(args, "output_path", "."),
            getattr(args, "timing_dirname", "timing"),
            "dataset",
        )
        extra_kwargs["dataset_timing_rank"] = int(getattr(args, "rank", 0))
        extra_kwargs["dataset_timing_sample_every"] = int(
            getattr(args, "dataset_timing_sample_every", 1)
        )

    train_fuse_mode_raw = str(getattr(args, "mosaic_fuse_mode_train", "random"))
    extra_kwargs["mosaic_fuse_mode"] = _normalize_mosaic_fuse_mode(train_fuse_mode_raw)
    extra_kwargs["recent_first_photo_refine"] = bool(
        getattr(args, "recent_first_photo_refine", False)
    )
    extra_kwargs["candidates_per_query_group_train"] = int(
        getattr(args, "candidates_per_query_group_train", 20)
    )
    extra_kwargs["recent_first_backend"] = str(
        getattr(args, "recent_first_backend", "legacy")
    )
    extra_kwargs["recent_first_warp_dtype"] = str(
        getattr(args, "recent_first_warp_dtype", "none")
    )
    extra_kwargs["recent_first_enable_tf32"] = bool(
        getattr(args, "recent_first_enable_tf32", False)
    )
    extra_kwargs["recent_first_zbuffer"] = bool(
        getattr(args, "recent_first_zbuffer", False)
        or train_fuse_mode_raw == RECENT_FIRST_ZBUFFER_MODE
    )
    extra_kwargs["recent_first_n_candidates_keyframe"] = int(
        getattr(args, "recent_first_n_candidates_keyframe", 0)
    )
    extra_kwargs["recent_first_keyframe_neighbour_window"] = int(
        getattr(args, "recent_first_keyframe_neighbour_window", 2)
    )
    nms_mode = getattr(args, "mosaic_candidate_nms_mode", "none")
    extra_kwargs["mosaic_selection_mode"] = str(
        getattr(args, "mosaic_selection_mode", "projection_iou")
    )
    extra_kwargs["mosaic_candidate_nms_mode"] = (
        None if nms_mode in (None, "", "none", "None") else str(nms_mode)
    )
    extra_kwargs["mosaic_candidate_nms_projection_iou_threshold"] = float(
        getattr(args, "mosaic_candidate_nms_projection_iou_threshold", 0.7)
    )
    extra_kwargs["mosaic_candidate_nms_min_temporal_gap"] = int(
        getattr(args, "mosaic_candidate_nms_min_temporal_gap", 0)
    )
    extra_kwargs["mosaic_candidate_nms_pose_distance_threshold"] = float(
        getattr(args, "mosaic_candidate_nms_pose_distance_threshold", 0.0)
    )
    extra_kwargs["mosaic_candidate_nms_pool_multiplier"] = float(
        getattr(args, "mosaic_candidate_nms_pool_multiplier", 2.0)
    )
    extra_kwargs["mosaic_coverage_grid_downsample"] = int(
        getattr(args, "mosaic_coverage_grid_downsample", 4)
    )
    extra_kwargs["mosaic_coverage_pool_stride"] = int(
        getattr(args, "mosaic_coverage_pool_stride", 2)
    )
    extra_kwargs["dynamic_object_mask_filter"] = bool(
        getattr(args, "dynamic_object_mask_filter", False)
    )
    extra_kwargs["dynamic_object_mask_subdir"] = str(
        getattr(args, "dynamic_object_mask_subdir", "objects")
    )
    extra_kwargs["dynamic_object_mask_filename"] = str(
        getattr(args, "dynamic_object_mask_filename", "dynamic_masks.jsonl")
    )
    extra_kwargs["dynamic_object_mask_temporal_dilate_radius"] = int(
        getattr(args, "dynamic_object_mask_temporal_dilate_radius", 0)
    )
    extra_kwargs["dynamic_object_mask_dilate_latent"] = int(
        getattr(args, "dynamic_object_mask_dilate_latent", 0)
    )
    extra_kwargs["dynamic_object_mask_use_bbox_fallback"] = bool(
        getattr(args, "dynamic_object_mask_use_bbox_fallback", False)
    )
    if dataset_type == "da3_video_subject_ref":
        extra_kwargs["subject_ref_dir_name"] = str(
            getattr(args, "subject_ref_dir_name", "protagonist_refs")
        )
        extra_kwargs["subject_num_refs_max"] = int(
            getattr(args, "subject_num_refs_max", 2)
        )
        extra_kwargs["subject_dissimilar_top_k"] = int(
            getattr(args, "subject_dissimilar_top_k", 8)
        )
        extra_kwargs["subject_max_similarity"] = float(
            getattr(args, "subject_max_similarity", 0.94)
        )
        extra_kwargs["subject_shuffle_refs"] = bool(
            getattr(args, "subject_shuffle_refs", True)
        )
        extra_kwargs["subject_ref_dropout_prob"] = float(
            getattr(args, "subject_ref_dropout_prob", 0.0)
        )
        extra_kwargs["subject_ref_canvas_slot_ratio"] = float(
            getattr(args, "subject_ref_canvas_slot_ratio", 0.5)
        )
        extra_kwargs["subject_ref_background"] = str(
            getattr(args, "subject_ref_background", "imagenet_mean")
        )
        extra_kwargs["subject_context_patch_mask_prob"] = float(
            getattr(args, "subject_context_patch_mask_prob", 0.5)
        )
        extra_kwargs["subject_context_patch_mask_ratio_min"] = float(
            getattr(args, "subject_context_patch_mask_ratio_min", 0.25)
        )
        extra_kwargs["subject_context_patch_mask_ratio_max"] = float(
            getattr(args, "subject_context_patch_mask_ratio_max", 0.75)
        )
        extra_kwargs["subject_context_patch_mask_fill"] = str(
            getattr(args, "subject_context_patch_mask_fill", "background_patch")
        )
        extra_kwargs["subject_context_patch_mask_temporal_dilate_radius"] = int(
            getattr(args, "subject_context_patch_mask_temporal_dilate_radius", 0)
        )
        extra_kwargs["subject_object_loss_temporal_dilate_radius"] = int(
            getattr(args, "subject_object_loss_temporal_dilate_radius", 0)
        )
    return dataset_cls, extra_kwargs


def _resolve_dataset_cache_dir(args):
    """Honour --no_dataset_cache > --dataset_cache_dir=='' > default."""
    if getattr(args, "no_dataset_cache", False):
        return None
    raw = getattr(args, "dataset_cache_dir", "")
    if raw is None or raw == "":
        return None
    return raw


def _append_prompt_cache_dir_to_vipe_prompt_path(args):
    if not (
        getattr(args, "train_qwen_prompt", False)
        and getattr(args, "train_qwen_prompt_missing_only", False)
        and getattr(args, "train_qwen_prompt_cache", False)
    ):
        return None
    cache_dir = str(getattr(args, "train_qwen_prompt_cache_dir", "") or "").strip()
    if not cache_dir:
        raise ValueError(
            "--train_qwen_prompt_cache requires --train_qwen_prompt_cache_dir."
        )
    if getattr(args, "dataset_type", "da3_video") == "da3_video":
        return None
    existing = [
        part.strip()
        for part in str(getattr(args, "vipe_prompt_path", "") or "").split(",")
        if part.strip()
    ]
    if cache_dir not in existing:
        existing.append(cache_dir)
    args.vipe_prompt_path = ",".join(existing)
    if not getattr(args, "allow_no_prompt", False):
        args.allow_no_prompt = True
        rank = getattr(args, "rank", 0)
        if rank in (0, "0"):
            print(
                "[train][qwen][cache] enabling allow_no_prompt so missing-cache "
                "scenes stay trainable and can be filled by Qwen."
            )
    return cache_dir


def _dedupe_manifest_keys(keys):
    return list(dict.fromkeys(str(key) for key in (keys or []) if str(key)))


def _manifest_sample_keys(payload):
    split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
    for key in ("sample_keys", "dataset_keys"):
        values = payload.get(key)
        if isinstance(values, list) and values:
            return _dedupe_manifest_keys(values)
        values = split.get(key)
        if isinstance(values, list) and values:
            return _dedupe_manifest_keys(values)
    dataset_info = payload.get("dataset_info")
    if isinstance(dataset_info, dict):
        return _dedupe_manifest_keys(dataset_info.keys())
    return []


def _manifest_validation_keys(payload):
    split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
    for key in (
        "validation_keys",
        "val_raw_paths",
        "val_keys",
        "val_keys_appended",
    ):
        values = payload.get(key)
        if isinstance(values, list) and values:
            return _dedupe_manifest_keys(values)
        values = split.get(key)
        if isinstance(values, list) and values:
            return _dedupe_manifest_keys(values)
    return _manifest_sample_keys(payload)


def _read_manifest_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def _infer_sibling_train_manifest_path(manifest_path):
    parent = os.path.dirname(manifest_path)
    candidates = [
        os.path.join(parent, "dataset_manifest_train.json"),
        manifest_path.replace("_val.json", "_train.json"),
    ]
    for candidate in candidates:
        if candidate != manifest_path and os.path.exists(candidate):
            return candidate
    return None


def _load_validation_manifest_keys(manifest_path, exclude_manifest_path=None):
    """Load val sample keys from a saved manifest.

    For old ``dataset_manifest_val.json`` files whose ``dataset_info`` still
    contains both train and val samples, subtract the sibling train manifest
    so sanity-check evaluation cannot accidentally sample train scenes.
    """
    payload = _read_manifest_payload(manifest_path)
    keys = _manifest_validation_keys(payload)

    if exclude_manifest_path is None:
        exclude_manifest_path = _infer_sibling_train_manifest_path(manifest_path)
    if exclude_manifest_path:
        exclude_payload = _read_manifest_payload(exclude_manifest_path)
        exclude_keys = set(_manifest_sample_keys(exclude_payload))
        keys = [key for key in keys if key not in exclude_keys]
    return keys


def build_mosaic_latent_dataset(args, dataset_cls=None):
    cls_from_args, extra_kwargs = _resolve_dataset_cls_and_extra_kwargs(args)
    if dataset_cls is None:
        dataset_cls = cls_from_args
    if getattr(args, "dataset_type", "da3_video") == "da3_video" and not getattr(
        args, "dataset_index_path", None
    ):
        raise ValueError(
            "--dataset_index_path is required in strict da3_video sqlite mode."
        )
    latent_window_size = args.latent_window_size
    if latent_window_size is None:
        latent_window_size = (args.num_frames - 1) // 4 + 1
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = args.dataset_repeat
    return dataset_cls(
        base_path=(),
        camera_params_path=(),
        depth_path=(),
        steps_per_epoch=steps_per_epoch,
        latent_window_size=latent_window_size,
        val_assign_yaml=args.val_assign_yaml,
        val_ratio=args.val_ratio,
        filter_yaml=args.filter_yaml,
        include_yaml=args.include_yaml,
        random_start_latent_prob=args.random_start_latent_prob,
        seed=args.seed,
        rank=getattr(args, "rank", 0),
        max_data_items=args.max_data_items,
        max_scan_items=args.max_scan_items,
        dataset_cache_dir=_resolve_dataset_cache_dir(args),
        dataset_pass_id=int(getattr(args, "dataset_pass_id", 0) or 0),
        max_frames_per_scene=getattr(args, "max_frames_per_scene", None),
        force_override_extrinsic=None,
        dataset_index_path=getattr(args, "dataset_index_path", None),
        min_frame_count=getattr(args, "min_frame_count", None),
        dataset_balance_mode=getattr(args, "dataset_balance_mode", "global"),
        dataset_balance_root_probs=getattr(args, "dataset_balance_root_probs", ""),
        **extra_kwargs,
    )


def build_mosaic_validation_dataset(args, train_dataset, dataset_cls=None):
    cls_from_args, extra_kwargs = _resolve_dataset_cls_and_extra_kwargs(args)
    if dataset_cls is None:
        dataset_cls = cls_from_args
    extra_kwargs = dict(extra_kwargs)
    extra_kwargs["vae_clean_context_blocks_max"] = int(
        getattr(args, "vae_clean_context_blocks_max", 0)
    )
    validation_context_count = 0
    if bool(getattr(args, "validation_clean_latent_pool_history", False)):
        validation_context_count = max(
            0,
            int(getattr(args, "validation_clean_latent_history_count", 1) or 1) - 1,
        )
    extra_kwargs["train_clean_history_context_blocks_min"] = (
        _required_context_blocks_min(
            anchor_block_prob=0.0,
            block_anchor_frames=int(getattr(args, "train_block_anchor_frames", 13)),
            context_count_max=validation_context_count,
        )
    )
    extra_kwargs["train_anchor_warmup_prepend_frames"] = 0
    extra_kwargs["validation_min_anchor_frame_idx"] = int(
        getattr(args, "validation_min_anchor_frame_idx", 0) or 0
    )
    if getattr(args, "dataset_type", "da3_video") == "da3_video" and not (
        getattr(args, "val_dataset_index_path", None)
        or getattr(args, "dataset_index_path", None)
    ):
        raise ValueError(
            "--dataset_index_path or --val_dataset_index_path is required "
            "for strict da3_video validation."
        )
    extra_kwargs["mosaic_intrinsics_mode"] = _intrinsics_mode_arg(
        args, "validation_mosaic_intrinsics_mode", "episode_mean"
    )
    val_dataset_index_path = getattr(args, "val_dataset_index_path", None)
    if val_dataset_index_path:
        val_raw_paths = None
        val_assign_yaml = None
        val_ratio = 1.0
        dataset_index_path = val_dataset_index_path
        if getattr(args, "rank", 0) == 0:
            print(
                "[validation] using dedicated dataset index: "
                f"{val_dataset_index_path}"
            )
    else:
        val_raw_paths = (
            train_dataset.get_val_raw_paths()
            if hasattr(train_dataset, "get_val_raw_paths")
            else None
        )
        val_assign_yaml = args.val_assign_yaml
        val_ratio = args.val_ratio
        dataset_index_path = getattr(args, "dataset_index_path", None)
        validation_manifest_path = getattr(args, "validation_manifest_path", None)
        if validation_manifest_path:
            val_raw_paths = _load_validation_manifest_keys(
                validation_manifest_path,
                exclude_manifest_path=getattr(
                    args, "validation_manifest_exclude_path", None
                ),
            )
            if not val_raw_paths:
                raise ValueError(
                    f"--validation_manifest_path produced no validation samples: "
                    f"{validation_manifest_path}"
                )
            if getattr(args, "rank", 0) == 0:
                print(
                    f"[validation] loaded {len(val_raw_paths)} sample keys from "
                    f"{validation_manifest_path}"
                )
    return dataset_cls(
        base_path=(),
        camera_params_path=(),
        depth_path=(),
        # Validation __len__ returns the scanned path count. Keep
        # num_val_batches as a run_mosaic_validation loop cap only so
        # Accelerate can shard a full validation set across ranks first.
        steps_per_epoch=max(1, int(args.num_val_batches)),
        latent_window_size=(
            int(args.latent_window_size)
            if args.latent_window_size is not None
            else (args.num_frames - 1) // 4 + 1
        ),
        is_val_dataset=True,
        val_assign_yaml=val_assign_yaml,
        val_ratio=val_ratio,
        val_raw_paths=val_raw_paths,
        filter_yaml=args.filter_yaml,
        include_yaml=args.include_yaml,
        random_start_latent_prob=args.random_start_latent_prob,
        seed=args.seed,
        rank=getattr(args, "rank", 0),
        max_data_items=args.max_data_items,
        max_scan_items=args.max_scan_items,
        dataset_cache_dir=_resolve_dataset_cache_dir(args),
        # Validation should hit the SAME val set across resumes so val
        # metrics are comparable. Pin to 0 regardless of args.
        dataset_pass_id=0,
        max_frames_per_scene=getattr(args, "max_frames_per_scene", None),
        force_override_extrinsic=getattr(args, "force_override_extrinsic", None),
        dataset_index_path=dataset_index_path,
        min_frame_count=getattr(args, "min_frame_count", None),
        dataset_balance_mode=getattr(args, "dataset_balance_mode", "global"),
        dataset_balance_root_probs=getattr(args, "dataset_balance_root_probs", ""),
        val_dataset_path_shuffle_seed=getattr(
            args, "val_dataset_path_shuffle_seed", 42
        ),
        **extra_kwargs,
    )
