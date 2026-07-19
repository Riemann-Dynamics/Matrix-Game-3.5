"""Adapter from the public distilled config to the Matrix pipeline API."""

from __future__ import annotations

from pathlib import Path

from distilled_config import DistilledInferenceConfig, profile_runtime_settings


def build_runtime_args(
    config: DistilledInferenceConfig,
    *,
    checkpoint: str | Path,
    dataset_index: str | Path,
    workspace: str | Path,
    wan_dir: str | Path,
    tokenizer_dir: str | Path,
    memory_cache_dir: str | Path,
):
    """Adapt the small public config to the existing Matrix pipeline API."""
    from .config import parse_pipeline_args

    checkpoint = Path(checkpoint)
    dataset_index = Path(dataset_index)
    workspace = Path(workspace)
    wan_dir = Path(wan_dir)
    tokenizer_dir = Path(tokenizer_dir)
    model_spec = ",".join(
        (
            f"{wan_dir}:diffusion_pytorch_model*.safetensors",
            f"{wan_dir}:models_t5_umt5-xxl-enc-bf16.pth",
            f"{wan_dir}:Wan2.2_VAE.pth",
        )
    )
    argv = [
        "--dataset_type", "da3_video",
        "--dataset_index_path", str(dataset_index),
        "--val_dataset_index_path", str(dataset_index),
        "--dataset_base_path", str(workspace),
        "--camera_params_path", str(workspace / "scenes"),
        "--depth_path", str(workspace / "scenes"),
        "--vipe_prompt_path", str(workspace / "caption"),
        "--dataset_num_workers", "0",
        "--memory_latent_cache_dir", str(memory_cache_dir),
        "--tokenizer_path", str(tokenizer_dir),
        "--model_id_with_origin_paths", model_spec,
        "--trained_dit", str(checkpoint),
        "--height", str(config.height),
        "--width", str(config.width),
        "--latent_window_size", str(config.latent_window_size),
        "--vae_clean_context_blocks_max", str(config.vae_context_blocks),
        "--vae_clean_context_mode_ratio", "2:8",
        "--train_anchor_block_prob", "0.8",
        "--train_block_anchor_frames", "13",
        "--train_context_count_max", str(config.context_chunks),
        "--train_context_selection", config.context_selection,
        "--num_epochs", "0",
        "--num_val_batches", "1",
        "--num_validation_blocks", str(config.num_blocks),
        "--validation_num_inference_steps", str(len(config.schedule)),
        "--validation_cfg_scale", "1.0",
        "--validation_seed", str(config.seed),
        "--seed", str(config.seed),
        "--trans_scale", str(config.trans_scale),
        "--mosaic_selection_mode", config.mosaic_selection_mode,
        "--mosaic_candidate_nms_mode", config.mosaic_candidate_nms_mode,
        "--mosaic_candidate_nms_projection_iou_threshold",
        str(config.mosaic_candidate_projection_iou_threshold),
        "--mosaic_candidate_nms_pose_distance_threshold",
        str(config.mosaic_candidate_pose_distance_threshold),
        "--mosaic_candidate_nms_pool_multiplier",
        str(config.mosaic_candidate_pool_multiplier),
        "--mosaic_coverage_pool_stride", str(config.mosaic_coverage_pool_stride),
        "--mosaic_query_reference_frame", str(config.mosaic_query_reference_frame),
        "--validation_mosaic_intrinsics_mode", config.mosaic_intrinsics_mode,
        "--mosaic_fuse_mode", config.mosaic_fuse_mode,
        "--mosaic_fuse_mode_train", config.mosaic_fuse_mode,
        "--dataset_compact_mode",
        "--force_using_input_extrinics",
    ]
    if config.mosaic_drop_holes:
        argv.append("--mosaic_drop_holes")
    args = parse_pipeline_args(argv)
    profile = profile_runtime_settings(config)

    # Internal names are retained only at the rollout boundary so the causal
    # kernel does not duplicate the public Matrix pipeline.
    args.distilled_profile = config.profile
    args.causal_mode = True
    args.causal_chunk_size = config.chunk_size
    args.causal_context_chunks = config.context_chunks
    args.causal_memory_bootstrap_mode = "c0_only"
    args.causal_validation_memory_bootstrap_mode = "c0_only"
    args.causal_validation_memory_mode = config.memory_mode
    args.causal_generated_memory_publish_interval = config.memory_publish_interval
    args.causal_kv_dynamic_context = config.dynamic_context
    args.causal_kv_force_context_original_anchor = profile["force_original_anchor"]
    args.causal_validation_prefix_noise_mode = profile["prefix_noise_mode"]
    args.causal_validation_prefix_noise_dynamic_context = profile[
        "noise_dynamic_context"
    ]
    args.causal_validation_prefix_noise_scales_by_step = config.hiar_scales
    args.causal_dmd_num_steps = len(config.schedule)
    args.causal_dmd_denoising_step_list = config.schedule
    args.causal_dmd_validation_use_cfg = False
    args.causal_dmd_timestep_wrap = True
    args.causal_dmd_self_memory_da3_process_res = config.da3_process_res
    args.causal_dmd_self_memory_da3_autocast_dtype = config.da3_autocast_dtype

    defaults = {
        "no_context_on_nomosaic": True,
        "causal_validation_fixed_initial_anchor": False,
        "causal_validation_fixed_first_prompt": False,
        "causal_validation_prefix_chunks_by_step": "",
        "causal_validation_mosaic_guidance_scale": 1.0,
        "causal_validation_mosaic_token_repeat_by_step": "",
        "causal_validation_mosaic_context_frames": 0,
        "causal_validation_mosaic_context_topk": 0,
        "causal_validation_mosaic_context_guidance_scale": 1.0,
        "causal_validation_mosaic_context_nonlocal_min_age_chunks": 0,
        "causal_validation_mosaic_context_step_mask": "",
        "causal_use_prev_chunk_anchor": True,
        "causal_validation_camera_time_scale": 1.0,
        "causal_validation_t2v_no_ref": False,
        "causal_validation_register_rgb_source": "generated",
        "causal_validation_register_depth_source": "online",
    }
    for name, value in defaults.items():
        setattr(args, name, value)
    return args
