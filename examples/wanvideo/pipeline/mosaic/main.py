from .common import *
from .common import _derive_seed
from .cleanup import _set_rank_local_cuda_device
from .config import _dump_run_args_yaml, parse_pipeline_args
from .datasets import (
    build_mosaic_latent_dataset,
    build_mosaic_validation_dataset,
    _append_prompt_cache_dir_to_vipe_prompt_path,
)
from .runner import _maybe_resume_from_log_dir, run_mosaic_inference_task
from .pipeline_module import WanMosaicPipelineModule


def main():
    args = parse_pipeline_args()

    base_output_path = args.output_path

    # Resolve resume / dataset_pass_id BEFORE constructing the
    # accelerator so the printed startup banner reflects the choice
    # and ``trained_dit`` is finalized before model load.
    _maybe_resume_from_log_dir(args)

    _pre_accelerate_device = _set_rank_local_cuda_device()
    gradient_accumulation_plugin = accelerate.utils.GradientAccumulationPlugin(
        num_steps=args.gradient_accumulation_steps,
        sync_each_batch=True,
    )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters,
            )
        ],
    )
    _set_rank_local_cuda_device() or _pre_accelerate_device
    setattr(args, "rank", accelerator.process_index)

    # Create one shared log directory across ranks. Multi-node wrapper scripts
    # can pass --log_dir_name so every node uses the same run folder before any
    # Python-side timestamp/broadcast timing can diverge.
    log_dir_name = getattr(args, "log_dir_name", None)
    if log_dir_name:
        log_dir_name = str(log_dir_name).strip().strip("/\\")
        if not log_dir_name or os.path.basename(log_dir_name) != log_dir_name:
            raise ValueError(
                "--log_dir_name must be a single directory name, "
                f"got {getattr(args, 'log_dir_name', None)!r}."
            )
    else:
        log_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and accelerator.num_processes > 1
        ):
            name_box = [log_dir_name if accelerator.is_main_process else None]
            torch.distributed.broadcast_object_list(name_box, src=0)
            log_dir_name = name_box[0]
    log_dir = os.path.join(base_output_path, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    args.output_path = log_dir
    if accelerator.is_main_process:
        print(f"Log directory: {log_dir}")
        args_yaml_path = _dump_run_args_yaml(args, os.path.join(log_dir, "args.yaml"))
        print(f"[config] dumped run args to {args_yaml_path}", flush=True)
    accelerator.wait_for_everyone()

    # Per-rank, per-pass torch / numpy / python random seeding. Each
    # rank gets a DIFFERENT base seed so model init perturbations
    # (e.g. dropout RNG, init noise on lazy modules) don't all mirror
    # rank 0; same (seed, rank, dataset_pass_id) -> same draw stream.
    _master_seed = _derive_seed(
        args.seed, "trainer", accelerator.process_index, int(args.dataset_pass_id)
    )
    torch.manual_seed(_master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_master_seed)
    np.random.seed(_master_seed & 0xFFFFFFFF)
    random.seed(_master_seed)

    prompt_cache_dir = _append_prompt_cache_dir_to_vipe_prompt_path(args)
    if prompt_cache_dir and accelerator.is_main_process:
        print(
            "[train][qwen][cache] using prompt cache dir as prompt source: "
            f"{prompt_cache_dir}"
        )

    dataset = build_mosaic_latent_dataset(args)
    val_dataset = None
    if args.run_validation or args.run_sanity_check:
        val_dataset = build_mosaic_validation_dataset(args, dataset)

    # Save dataset manifest(s) into the log directory on the main process.
    # The content is identical to the on-disk cache under
    # ``--dataset_cache_dir``, so we always have two copies: one persistent
    # (for fast re-runs) and one per-run (for audit / debugging).
    if accelerator.is_main_process:
        manifest_train_path = os.path.join(
            args.output_path, "dataset_manifest_train.json"
        )
        saved_train = dataset.save_manifest(manifest_train_path)
        if saved_train:
            print(f"[manifest] train -> {saved_train}")
        if val_dataset is not None:
            manifest_val_path = os.path.join(
                args.output_path, "dataset_manifest_val.json"
            )
            saved_val = val_dataset.save_manifest(manifest_val_path)
            if saved_val:
                print(f"[manifest] val   -> {saved_val}")
    model = WanMosaicPipelineModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        trained_dit=args.trained_dit,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        enable_mosaic=args.enable_mosaic,
        preview_dataset=args.preview_dataset,
        num_preview=args.num_preview,
        use_prope=args.use_prope,
        prope_attention_interval=args.prope_attention_interval,
        prope_disable_native_rope=args.prope_disable_native_rope,
        prope_disable_t_rope=args.prope_disable_t_rope,
        prope_camera_layout=args.prope_camera_layout,
        only_prope=args.only_prope,
        vae_decode_tiled=args.vae_decode_tiled,
        mosaic_use_revgrid_rope=args.mosaic_use_revgrid_rope,
        mosaic_view_change_prope=args.mosaic_view_change_prope,
        mosaic_interval=args.mosaic_interval,
        mosaic_fuse_mode=args.mosaic_fuse_mode_train,
        mosaic_fuse_block_size=args.mosaic_fuse_block_size,
        mosaic_return_source_frame_ids=args.mosaic_return_source_frame_ids,
        mosaic_drop_holes=args.mosaic_drop_holes,
        mosaic_no_mosaic_sync_mode=args.mosaic_no_mosaic_sync_mode,
        mosaic_sequence_balance_mode=args.mosaic_sequence_balance_mode,
        recent_first_photo_refine=args.recent_first_photo_refine,
        recent_first_dedupe_tiles=args.recent_first_dedupe_tiles,
        candidates_per_query_group_train=args.candidates_per_query_group_train,
        recent_first_geometry_device=args.recent_first_geometry_device,
        recent_first_backend=args.recent_first_backend,
        recent_first_warp_dtype=args.recent_first_warp_dtype,
        recent_first_enable_tf32=args.recent_first_enable_tf32,
        recent_first_zbuffer=args.recent_first_zbuffer,
        recent_first_n_candidates_keyframe=args.recent_first_n_candidates_keyframe,
        recent_first_keyframe_neighbour_window=args.recent_first_keyframe_neighbour_window,
        mosaic_query_reference_frame=args.mosaic_query_reference_frame,
        memory_vae_encode_input_frames=args.memory_vae_encode_input_frames,
        memory_neighbor_window_train=args.memory_neighbor_window_train,
        no_mosaic_prob_train=args.no_mosaic_prob_train,
        mosaic_train_random_schedule=args.mosaic_train_random_schedule,
        mosaic_train_schedule=args.mosaic_train_schedule,
        mosaic_schedule_sync_mode=args.mosaic_schedule_sync_mode,
        schedule_base_seed=args.seed,
        train_anchor_block_prob=args.train_anchor_block_prob,
        train_block_anchor_frames=args.train_block_anchor_frames,
        train_context_count_max=args.train_context_count_max,
        train_context_selection=args.train_context_selection,
        loose_model_loading=args.loose_model_loading,
        trans_scale=args.trans_scale,
        clean_latent_noise_train=args.clean_latent_noise_train,
        clean_latent_noise_prob_train=args.clean_latent_noise_prob_train,
        clean_latent_noise_magnitude_train=args.clean_latent_noise_magnitude_train,
        mosaic_latent_noise_train=args.mosaic_latent_noise_train,
        mosaic_latent_noise_prob_train=args.mosaic_latent_noise_prob_train,
        mosaic_latent_noise_magnitude_train=args.mosaic_latent_noise_magnitude_train,
        context_latent_noise_train=args.context_latent_noise_train,
        context_latent_noise_prob_train=args.context_latent_noise_prob_train,
        context_latent_noise_magnitude_train=args.context_latent_noise_magnitude_train,
        subject_ref_memory=args.subject_ref_memory,
        subject_ref_time_gap=args.subject_ref_time_gap,
        subject_ref_prope_mode=args.subject_ref_prope_mode,
        subject_num_refs_max=args.subject_num_refs_max,
        subject_ref_canvas_slot_ratio=args.subject_ref_canvas_slot_ratio,
        subject_pool_context_dynamic_erase=args.subject_pool_context_dynamic_erase,
        subject_pool_context_dynamic_erase_prob=args.subject_pool_context_dynamic_erase_prob,
        subject_pool_context_dynamic_erase_ratio_min=args.subject_pool_context_dynamic_erase_ratio_min,
        subject_pool_context_dynamic_erase_ratio_max=args.subject_pool_context_dynamic_erase_ratio_max,
        subject_pool_context_dynamic_erase_fill=args.subject_pool_context_dynamic_erase_fill,
        subject_context_patch_mask_prob=args.subject_context_patch_mask_prob,
        subject_context_patch_mask_ratio_min=args.subject_context_patch_mask_ratio_min,
        subject_context_patch_mask_ratio_max=args.subject_context_patch_mask_ratio_max,
        subject_context_patch_mask_fill=args.subject_context_patch_mask_fill,
        subject_object_loss_alpha=args.subject_object_loss_alpha,
        subject_object_loss_min_mask_ratio=args.subject_object_loss_min_mask_ratio,
        train_qwen_prompt=args.train_qwen_prompt,
        train_qwen_provider=args.train_qwen_provider,
        train_qwen_model_path=(
            args.train_qwen_model_path or args.validation_qwen_model_path
        ),
        train_qwen_device_map=args.train_qwen_device_map,
        train_qwen_torch_dtype=args.train_qwen_torch_dtype,
        train_qwen_max_new_tokens=args.train_qwen_max_new_tokens,
        train_qwen_word_limit=args.train_qwen_word_limit,
        train_qwen_sample_frames=args.train_qwen_sample_frames,
        train_qwen_frame_max_side=args.train_qwen_frame_max_side,
        train_qwen_prompt_missing_only=args.train_qwen_prompt_missing_only,
        train_qwen_prompt_cache=args.train_qwen_prompt_cache,
        train_qwen_prompt_cache_dir=args.train_qwen_prompt_cache_dir,
        train_qwen_trust_remote_code=args.train_qwen_trust_remote_code,
        train_prompt_dropout_prob=args.train_prompt_dropout_prob,
        allow_no_prompt=args.allow_no_prompt,
        svi_num_grids=args.svi_num_grids,
        svi_error_buffer_k=args.svi_error_buffer_k,
        svi_buffer_replacement_strategy=args.svi_buffer_replacement_strategy,
        svi_save_error_buffer=args.svi_save_error_buffer,
        svi_inject_noise=args.svi_inject_noise,
        svi_noise_prob=args.svi_noise_prob,
        svi_noise_scale=args.svi_noise_scale,
        svi_buffer_warmup_steps=args.svi_buffer_warmup_steps,
        svi_inject_latent=args.svi_inject_latent,
        svi_latent_prob=args.svi_latent_prob,
        svi_latent_scale=args.svi_latent_scale,
        svi_inject_clean_context=args.svi_inject_clean_context,
        svi_clean_context_prob=args.svi_clean_context_prob,
        svi_clean_context_scale=args.svi_clean_context_scale,
        svi_clean_context_frames=args.svi_clean_context_frames,
        svi_clean_prob=args.svi_clean_prob,
        svi_clean_buffer_update_prob=args.svi_clean_buffer_update_prob,
        resume_from=args.resume_from,
        debug=args.debug,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    model.log_dir = log_dir
    # Resume continues the (seed, step_count)-keyed rng streams (anchor/
    # context plan, mosaic schedule) where the previous leg stopped instead
    # of replaying the same prefix every leg.
    resume_step_count = int(getattr(args, "_resume_step_count", 0) or 0)
    if resume_step_count > 0:
        model.step_count = resume_step_count
        if accelerator.is_main_process:
            print(f"[resume] model.step_count restored to {resume_step_count}")

    if args.task in ("sft:data_process", "direct_distill:data_process"):
        raise NotImplementedError(
            "Data-processing/training tasks are not included in this "
            "inference release."
        )
    else:
        run_mosaic_inference_task(
            accelerator,
            dataset,
            model,
            model_logger,
            log_dir=log_dir,
            args=args,
            val_dataset=val_dataset,
        )


if __name__ == "__main__":
    main()
