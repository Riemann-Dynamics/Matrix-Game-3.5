from .common import *
from .config import RECENT_FIRST_ZBUFFER_MODE, _normalize_mosaic_fuse_mode
from .history import (
    _validate_block_anchor_frames,
    _validate_train_context_selection,
)
from .prompting import _normalize_train_prompt_dropout_prob
from .svi import SVIState


class _MosaicModuleBase:
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        trained_dit=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        enable_mosaic=True,
        preview_dataset=False,
        num_preview=1,
        use_prope=False,
        prope_attention_interval=1,
        prope_disable_native_rope=False,
        prope_disable_t_rope=False,
        prope_camera_layout="full",
        only_prope=False,
        vae_decode_tiled=False,
        mosaic_use_revgrid_rope=False,
        mosaic_view_change_prope=False,
        mosaic_interval=1,
        mosaic_fuse_mode="fill_stop",
        mosaic_fuse_block_size=1,
        mosaic_return_source_frame_ids=False,
        mosaic_drop_holes=False,
        mosaic_no_mosaic_sync_mode="none",
        mosaic_sequence_balance_mode="profile_only",
        recent_first_photo_refine=False,
        recent_first_dedupe_tiles=True,
        candidates_per_query_group_train=20,
        recent_first_geometry_device="cpu",
        recent_first_backend="legacy",
        recent_first_warp_dtype="none",
        recent_first_enable_tf32=False,
        recent_first_zbuffer=False,
        recent_first_n_candidates_keyframe=0,
        recent_first_keyframe_neighbour_window=2,
        mosaic_query_reference_frame=2,
        memory_vae_encode_input_frames=1,
        memory_neighbor_window_train=False,
        no_mosaic_prob_train=0.0,
        mosaic_train_random_schedule=False,
        mosaic_train_schedule="interval3:3,interval1:4,nomosaic:3",
        mosaic_schedule_sync_mode="seed",
        schedule_base_seed=0,
        train_anchor_block_prob=0.8,
        train_block_anchor_frames=13,
        train_context_count_max=0,
        train_context_selection="oldest_spread",
        loose_model_loading=False,
        trans_scale=50.0,
        clean_latent_noise_train=False,
        clean_latent_noise_prob_train=0.2,
        clean_latent_noise_magnitude_train=0.03,
        mosaic_latent_noise_train=False,
        mosaic_latent_noise_prob_train=0.2,
        mosaic_latent_noise_magnitude_train=0.03,
        context_latent_noise_train=False,
        context_latent_noise_prob_train=0.2,
        context_latent_noise_magnitude_train=0.03,
        subject_ref_memory=False,
        subject_ref_time_gap=1,
        subject_ref_prope_mode="identity",
        subject_num_refs_max=2,
        subject_ref_canvas_slot_ratio=0.5,
        subject_pool_context_dynamic_erase=True,
        subject_pool_context_dynamic_erase_prob=1.0,
        subject_pool_context_dynamic_erase_ratio_min=1.0,
        subject_pool_context_dynamic_erase_ratio_max=1.0,
        subject_pool_context_dynamic_erase_fill="background_patch",
        subject_context_patch_mask_prob=0.5,
        subject_context_patch_mask_ratio_min=0.25,
        subject_context_patch_mask_ratio_max=0.75,
        subject_context_patch_mask_fill="background_patch",
        subject_object_loss_alpha=0.0,
        subject_object_loss_min_mask_ratio=0.0,
        train_qwen_prompt=False,
        train_qwen_provider="auto",
        train_qwen_model_path="",
        train_qwen_device_map="auto",
        train_qwen_torch_dtype="bfloat16",
        train_qwen_max_new_tokens=256,
        train_qwen_word_limit=200,
        train_qwen_sample_frames=8,
        train_qwen_frame_max_side=448,
        train_qwen_prompt_missing_only=False,
        train_qwen_prompt_cache=False,
        train_qwen_prompt_cache_dir="",
        train_qwen_trust_remote_code=False,
        train_prompt_dropout_prob=0.0,
        allow_no_prompt=False,
        svi_num_grids=50,
        svi_error_buffer_k=4,
        svi_buffer_replacement_strategy="random",
        svi_save_error_buffer=True,
        svi_inject_noise=True,
        svi_noise_prob=0.5,
        svi_noise_scale=1.0,
        svi_buffer_warmup_steps=50,
        svi_inject_latent=True,
        svi_latent_prob=0.5,
        svi_latent_scale=1.0,
        svi_inject_clean_context=True,
        svi_clean_context_prob=0.5,
        svi_clean_context_scale=1.0,
        svi_clean_context_frames=1,
        svi_clean_prob=0.5,
        svi_clean_buffer_update_prob=0.1,
        resume_from=None,
        debug=False,
    ):
        super().__init__()
        use_prope = use_prope or only_prope
        enable_mosaic = enable_mosaic and not only_prope
        # Revgrid RoPE only makes sense when there is a mosaic latent and
        # therefore a `data["ql_rope_indice_pef"]` to look up; silently
        # disable rather than ship a confusing pipeline kwarg.
        mosaic_use_revgrid_rope = bool(mosaic_use_revgrid_rope) and bool(enable_mosaic)
        mosaic_view_change_prope = (
            bool(mosaic_view_change_prope) and bool(enable_mosaic) and bool(use_prope)
        )
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. "
                "To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
            loose_loading=loose_model_loading,
        )
        tokenizer_config = (
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            )
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
            redirect_common_files=False,
        )
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model
        )
        self._set_use_prope(
            self.pipe,
            use_prope,
            prope_attention_interval=prope_attention_interval,
            prope_disable_native_rope=prope_disable_native_rope,
            prope_disable_t_rope=prope_disable_t_rope,
            prope_camera_layout=prope_camera_layout,
        )
        # PRoPE camera extrinsic translation is normalised by ``trans_scale``
        # in WanVideoUnit_PropeCamera: a number keeps the legacy linear
        # ``w2c[..., :3, 3] /= trans_scale``; the strings "log"/"logd4"/"tanh"
        # select direction-preserving radial compression of the magnitude.
        # The pipeline reads it via ``getattr(dit, "trans_scale", 50.0)``,
        # so we stamp the value on the DiT modules here. Applied even when
        # ``use_prope`` is off so saving/loading a checkpoint preserves the
        # attribute round-trip.
        self.trans_scale = self._parse_trans_scale(trans_scale)
        # Training-only clean-latent noise augmentation. Stamped on the DiT
        # modules so `model_fn_wan_video` can read it via getattr; it only
        # fires while `dit.training` is True, so validation (which switches
        # the model to eval mode) and inference are unaffected.
        self.clean_latent_noise_train = bool(clean_latent_noise_train)
        self.clean_latent_noise_prob_train = float(clean_latent_noise_prob_train)
        if not (0.0 <= self.clean_latent_noise_prob_train <= 1.0):
            raise ValueError(
                "clean_latent_noise_prob_train must be in [0, 1], "
                f"got {self.clean_latent_noise_prob_train}."
            )
        self.clean_latent_noise_magnitude_train = float(
            clean_latent_noise_magnitude_train
        )
        if self.clean_latent_noise_magnitude_train < 0.0:
            raise ValueError(
                "clean_latent_noise_magnitude_train must be >= 0, "
                f"got {self.clean_latent_noise_magnitude_train}."
            )
        # Training-only mosaic-latent noise augmentation, parallel to the
        # clean-latent noise above but targeting the mosaic/queried latent.
        self.mosaic_latent_noise_train = bool(mosaic_latent_noise_train)
        self.mosaic_latent_noise_prob_train = float(mosaic_latent_noise_prob_train)
        if not (0.0 <= self.mosaic_latent_noise_prob_train <= 1.0):
            raise ValueError(
                "mosaic_latent_noise_prob_train must be in [0, 1], "
                f"got {self.mosaic_latent_noise_prob_train}."
            )
        self.mosaic_latent_noise_magnitude_train = float(
            mosaic_latent_noise_magnitude_train
        )
        if self.mosaic_latent_noise_magnitude_train < 0.0:
            raise ValueError(
                "mosaic_latent_noise_magnitude_train must be >= 0, "
                f"got {self.mosaic_latent_noise_magnitude_train}."
            )
        # Training-only context-latent noise augmentation (parallel to the
        # clean-latent noise but targeting the context latents).
        self.context_latent_noise_train = bool(context_latent_noise_train)
        self.context_latent_noise_prob_train = float(context_latent_noise_prob_train)
        if not (0.0 <= self.context_latent_noise_prob_train <= 1.0):
            raise ValueError(
                "context_latent_noise_prob_train must be in [0, 1], "
                f"got {self.context_latent_noise_prob_train}."
            )
        self.context_latent_noise_magnitude_train = float(
            context_latent_noise_magnitude_train
        )
        if self.context_latent_noise_magnitude_train < 0.0:
            raise ValueError(
                "context_latent_noise_magnitude_train must be >= 0, "
                f"got {self.context_latent_noise_magnitude_train}."
            )
        self.subject_ref_memory = bool(subject_ref_memory)
        self.subject_ref_time_gap = max(1, int(subject_ref_time_gap))
        self.subject_ref_prope_mode = str(subject_ref_prope_mode or "identity")
        self.subject_num_refs_max = max(1, int(subject_num_refs_max))
        self.subject_ref_canvas_slot_ratio = float(subject_ref_canvas_slot_ratio)
        self.subject_pool_context_dynamic_erase = bool(subject_pool_context_dynamic_erase)
        self.subject_pool_context_dynamic_erase_prob = min(
            1.0, max(0.0, float(subject_pool_context_dynamic_erase_prob))
        )
        self.subject_pool_context_dynamic_erase_ratio_min = min(
            1.0, max(0.0, float(subject_pool_context_dynamic_erase_ratio_min))
        )
        self.subject_pool_context_dynamic_erase_ratio_max = min(
            1.0, max(0.0, float(subject_pool_context_dynamic_erase_ratio_max))
        )
        if (
            self.subject_pool_context_dynamic_erase_ratio_max
            < self.subject_pool_context_dynamic_erase_ratio_min
        ):
            self.subject_pool_context_dynamic_erase_ratio_min, self.subject_pool_context_dynamic_erase_ratio_max = (
                self.subject_pool_context_dynamic_erase_ratio_max,
                self.subject_pool_context_dynamic_erase_ratio_min,
            )
        self.subject_pool_context_dynamic_erase_fill = str(
            subject_pool_context_dynamic_erase_fill or "background_patch"
        ).strip().lower()
        self.subject_context_patch_mask_prob = min(
            1.0, max(0.0, float(subject_context_patch_mask_prob))
        )
        self.subject_context_patch_mask_ratio_min = min(
            1.0, max(0.0, float(subject_context_patch_mask_ratio_min))
        )
        self.subject_context_patch_mask_ratio_max = min(
            1.0, max(0.0, float(subject_context_patch_mask_ratio_max))
        )
        if (
            self.subject_context_patch_mask_ratio_max
            < self.subject_context_patch_mask_ratio_min
        ):
            self.subject_context_patch_mask_ratio_min, self.subject_context_patch_mask_ratio_max = (
                self.subject_context_patch_mask_ratio_max,
                self.subject_context_patch_mask_ratio_min,
            )
        self.subject_context_patch_mask_fill = str(
            subject_context_patch_mask_fill or "background_patch"
        ).strip().lower()
        self.subject_object_loss_alpha = max(0.0, float(subject_object_loss_alpha))
        self.subject_object_loss_min_mask_ratio = max(
            0.0, float(subject_object_loss_min_mask_ratio)
        )
        if self.subject_ref_memory:
            for dit_name in ("dit", "dit2"):
                dit = getattr(self.pipe, dit_name, None)
                if dit is not None:
                    dit.enable_subject_ref_memory(self.subject_num_refs_max)
        for dit_name in ("dit", "dit2"):
            dit = getattr(self.pipe, dit_name, None)
            if dit is not None:
                dit.trans_scale = self.trans_scale
                dit.clean_latent_noise_enabled = self.clean_latent_noise_train
                dit.clean_latent_noise_prob = self.clean_latent_noise_prob_train
                dit.clean_latent_noise_magnitude = (
                    self.clean_latent_noise_magnitude_train
                )
                dit.mosaic_latent_noise_enabled = self.mosaic_latent_noise_train
                dit.mosaic_latent_noise_prob = self.mosaic_latent_noise_prob_train
                dit.mosaic_latent_noise_magnitude = (
                    self.mosaic_latent_noise_magnitude_train
                )
                dit.context_latent_noise_enabled = self.context_latent_noise_train
                dit.context_latent_noise_prob = self.context_latent_noise_prob_train
                dit.context_latent_noise_magnitude = (
                    self.context_latent_noise_magnitude_train
                )
        self._load_trained_dit(trained_dit)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        # Store configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.fp8_models = fp8_models
        self.task = task
        self.svi_state = None
        if task == "sft:svi":
            enabled_gaussian_noise = []
            if self.clean_latent_noise_train:
                enabled_gaussian_noise.append("clean_latent_noise_train")
            if self.context_latent_noise_train:
                enabled_gaussian_noise.append("context_latent_noise_train")
            if enabled_gaussian_noise:
                print(
                    "[svi] WARNING: task=sft:svi is running with Gaussian "
                    "latent noise augmentation enabled: "
                    f"{', '.join(enabled_gaussian_noise)}. These augmentations "
                    "are applied inside model_fn_wan_video after SVI prepare, "
                    "so they will stack with SVI injections and can also "
                    "corrupt SVI clean-input steps. Prefer disabling them "
                    "unless this is an explicit ablation.",
                    flush=True,
                )
            svi_args = types.SimpleNamespace(
                svi_num_grids=svi_num_grids,
                svi_error_buffer_k=svi_error_buffer_k,
                svi_buffer_replacement_strategy=svi_buffer_replacement_strategy,
                svi_save_error_buffer=svi_save_error_buffer,
                svi_inject_noise=svi_inject_noise,
                svi_noise_prob=svi_noise_prob,
                svi_noise_scale=svi_noise_scale,
                svi_buffer_warmup_steps=svi_buffer_warmup_steps,
                svi_inject_latent=svi_inject_latent,
                svi_latent_prob=svi_latent_prob,
                svi_latent_scale=svi_latent_scale,
                svi_inject_clean_context=svi_inject_clean_context,
                svi_clean_context_prob=svi_clean_context_prob,
                svi_clean_context_scale=svi_clean_context_scale,
                svi_clean_context_frames=svi_clean_context_frames,
                svi_clean_prob=svi_clean_prob,
                svi_clean_buffer_update_prob=svi_clean_buffer_update_prob,
                debug=debug,
            )
            self.svi_state = SVIState(self, svi_args)
            self.svi_state.maybe_load_from_resume(resume_from)
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "sft:svi": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSVILoss(
                pipe, self.svi_state, **inputs_shared, **inputs_posi
            ),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # Ensure input_image is included (required for TI2V-5B fused VAE embedding)
        extra_inputs_list = extra_inputs.split(",") if extra_inputs is not None else []
        if "input_image" not in extra_inputs_list:
            extra_inputs_list.append("input_image")
        self.extra_inputs = extra_inputs_list

        self.enable_mosaic = enable_mosaic
        self.only_prope = only_prope
        self.mosaic_use_revgrid_rope = mosaic_use_revgrid_rope
        self.mosaic_view_change_prope = mosaic_view_change_prope
        self.mosaic_interval = max(1, int(mosaic_interval))
        self.debug = bool(debug)
        self.log_dir = None
        self.step_count = 0
        self.train_qwen_prompt = bool(train_qwen_prompt)
        self.train_qwen_provider = str(train_qwen_provider or "auto")
        self.train_qwen_model_path = str(train_qwen_model_path or "")
        self.train_qwen_device_map = str(train_qwen_device_map or "auto")
        self.train_qwen_torch_dtype = str(train_qwen_torch_dtype or "bfloat16")
        self.train_qwen_max_new_tokens = max(1, int(train_qwen_max_new_tokens or 256))
        self.train_qwen_word_limit = max(1, int(train_qwen_word_limit or 200))
        self.train_qwen_sample_frames = max(1, int(train_qwen_sample_frames or 1))
        self.train_qwen_frame_max_side = max(1, int(train_qwen_frame_max_side or 1))
        self.train_qwen_prompt_missing_only = bool(train_qwen_prompt_missing_only)
        self.train_qwen_prompt_cache = bool(train_qwen_prompt_cache)
        self.train_qwen_prompt_cache_dir = str(train_qwen_prompt_cache_dir or "")
        self.train_qwen_trust_remote_code = bool(train_qwen_trust_remote_code)
        self.allow_no_prompt = bool(allow_no_prompt)
        self._train_qwen_prompt_generator = None
        self.train_prompt_dropout_prob = _normalize_train_prompt_dropout_prob(
            train_prompt_dropout_prob
        )
        self.preview_dataset = preview_dataset
        self.num_preview = num_preview
        self.preview_num = 0
        self._preview_done = False
        self.vae_decode_tiled = bool(vae_decode_tiled)
        raw_mosaic_fuse_mode = str(mosaic_fuse_mode or "fill_stop")
        self.recent_first_zbuffer = bool(recent_first_zbuffer) or (
            raw_mosaic_fuse_mode == RECENT_FIRST_ZBUFFER_MODE
        )
        self.mosaic_fuse_mode = _normalize_mosaic_fuse_mode(raw_mosaic_fuse_mode)
        if self.mosaic_view_change_prope and self.mosaic_fuse_mode in {
            "recent_first_smooth",
            "random",
        }:
            raise ValueError(
                "mosaic_view_change_prope does not support "
                "mosaic_fuse_mode_train='recent_first_smooth' or 'random' yet. "
                "Use masked_mean, fill_stop, or fill_stop_zbuffer."
            )
        self.mosaic_fuse_block_size = int(mosaic_fuse_block_size)
        if self.mosaic_fuse_block_size not in (1, 2):
            raise ValueError(
                "mosaic_fuse_block_size must be 1 or 2; got "
                f"{self.mosaic_fuse_block_size}."
            )
        self.mosaic_return_source_frame_ids = bool(mosaic_return_source_frame_ids)
        self.mosaic_drop_holes = bool(mosaic_drop_holes)
        self.mosaic_no_mosaic_sync_mode = str(mosaic_no_mosaic_sync_mode or "none")
        if self.mosaic_no_mosaic_sync_mode not in {"none", "rank0_broadcast"}:
            raise ValueError(
                "mosaic_no_mosaic_sync_mode must be 'none' or 'rank0_broadcast'; "
                f"got {self.mosaic_no_mosaic_sync_mode!r}."
            )
        self.mosaic_sequence_balance_mode = str(
            mosaic_sequence_balance_mode or "profile_only"
        )
        if self.mosaic_sequence_balance_mode not in {
            "profile_only",
            "pad_to_world_max",
        }:
            raise ValueError(
                "mosaic_sequence_balance_mode must be 'profile_only' or "
                f"'pad_to_world_max'; got {self.mosaic_sequence_balance_mode!r}."
            )
        self.recent_first_photo_refine = bool(recent_first_photo_refine)
        self.recent_first_dedupe_tiles = bool(recent_first_dedupe_tiles)
        self.candidates_per_query_group_train = max(
            1, int(candidates_per_query_group_train)
        )
        self.recent_first_geometry_device = str(recent_first_geometry_device or "cpu")
        self.recent_first_backend = str(recent_first_backend or "legacy")
        self.recent_first_warp_dtype = str(recent_first_warp_dtype or "none")
        self.recent_first_enable_tf32 = bool(recent_first_enable_tf32)
        self.recent_first_n_candidates_keyframe = max(
            0, int(recent_first_n_candidates_keyframe)
        )
        self.recent_first_keyframe_neighbour_window = max(
            0, int(recent_first_keyframe_neighbour_window)
        )
        self.mosaic_query_reference_frame = int(mosaic_query_reference_frame)
        if (
            self.mosaic_query_reference_frame < 1
            or self.mosaic_query_reference_frame > 4
        ):
            raise ValueError(
                "mosaic_query_reference_frame must be in [1, 4], "
                f"got {self.mosaic_query_reference_frame}."
            )
        # Memory cache-miss VAE encode: how many frames to feed the VAE per
        # memory slot. Wan's VAE follows the 1+4N rule, so the value must be
        # of the form 1+4k (k>=0): 1, 5, 9, 13, ... The default (1) preserves
        # the legacy single-frame encode -> 1 latent frame behaviour. When
        # >1, the single source frame is replicated along the time axis and
        # the encoder's last latent step is kept, yielding the same
        # (N, C, 1, H_lat, W_lat) tensor downstream code expects.
        self.memory_vae_encode_input_frames = int(memory_vae_encode_input_frames)
        if self.memory_vae_encode_input_frames < 1 or (
            (self.memory_vae_encode_input_frames - 1) % 4 != 0
        ):
            raise ValueError(
                "memory_vae_encode_input_frames must be of the form 1+4k "
                f"(>=1), got {self.memory_vae_encode_input_frames}."
            )
        # Independent channel from ``memory_vae_encode_input_frames``: when
        # enabled, the dataset ships a real 5-frame temporal window (or a
        # single frame when camera motion is large) per candidate memory
        # slot, and the encode block below groups by the per-slot frame
        # count instead of replicating a single frame. See
        # ``--memory_neighbor_window_train``.
        self.memory_neighbor_window_train = bool(memory_neighbor_window_train)
        # Per-step probability that a TRAINING sample is run with no mosaic
        # conditioning at all (queried_latent set to zeros, mosaic kwargs
        # dropped from the pipeline call, memory VAE encode / frustum fuse
        # skipped). 0.0 keeps the legacy always-mosaic behaviour; only
        # consulted while ``self.training`` is True, so validation/inference
        # are unaffected.
        self.no_mosaic_prob_train = float(no_mosaic_prob_train)
        if not (0.0 <= self.no_mosaic_prob_train <= 1.0):
            raise ValueError(
                "no_mosaic_prob_train must be in [0, 1], "
                f"got {self.no_mosaic_prob_train}."
            )
        # Per-step mosaic schedule: when enabled, every training step samples
        # one mode from ``mosaic_train_schedule`` (default 3:4:3 over
        # interval3 / interval1 / nomosaic) in a way that is identical across
        # all DDP ranks, so a "nomosaic" step is a nomosaic step everywhere
        # (no fast-rank-waits-for-slow-rank skew). All three mosaic modes run
        # with drop_holes forced on. When disabled the legacy fixed
        # ``mosaic_interval`` + per-rank ``no_mosaic_prob_train`` path is kept.
        self.mosaic_train_random_schedule = bool(mosaic_train_random_schedule)
        self.mosaic_schedule_sync_mode = str(mosaic_schedule_sync_mode or "seed")
        if self.mosaic_schedule_sync_mode not in {"seed", "rank0_broadcast"}:
            raise ValueError(
                "mosaic_schedule_sync_mode must be 'seed' or 'rank0_broadcast'; "
                f"got {self.mosaic_schedule_sync_mode!r}."
            )
        # ``args.seed`` is shared across ranks (per-rank seeds are derived from
        # it downstream), so it is a safe base for a deterministic, comms-free
        # per-step draw under the 'seed' sync mode.
        self._schedule_base_seed = int(
            0 if schedule_base_seed is None else schedule_base_seed
        )
        self.mosaic_train_schedule_spec = str(mosaic_train_schedule)
        (
            self._mosaic_schedule_modes,
            self._mosaic_schedule_weights,
        ) = self._parse_mosaic_schedule(self.mosaic_train_schedule_spec)
        # v2 anchor/context plan (history._sample_train_anchor_context_plan):
        # anchor_form ~ Bernoulli(block_prob); block anchors are jointly
        # encoded with the targets at depth (frames-1)//4; context blocks
        # are discrete extra clean latents, separately encoded. Args arrive
        # canonicalized by config._resolve_train_anchor_context_args; the
        # validators here are the same shared functions, so direct (non-CLI)
        # construction gets identical checks.
        self.train_anchor_block_prob = min(
            1.0, max(0.0, float(train_anchor_block_prob))
        )
        self.train_block_anchor_frames = _validate_block_anchor_frames(
            13 if train_block_anchor_frames is None else train_block_anchor_frames
        )
        self.train_context_count_max = max(0, int(train_context_count_max or 0))
        self.train_context_selection = _validate_train_context_selection(
            train_context_selection
        )
        self.eval_clean_history_plan = None
        self.recent_first_cfg = {
            "photo_refine": self.recent_first_photo_refine,
            "dedupe_tiles": self.recent_first_dedupe_tiles,
            "warn_min_candidates": self.candidates_per_query_group_train,
            "candidate_geometry_device": self.recent_first_geometry_device,
            "backend": self.recent_first_backend,
            "warp_dtype": self.recent_first_warp_dtype,
            "enable_tf32": self.recent_first_enable_tf32,
            "recent_first_zbuffer": self.recent_first_zbuffer,
            "n_candidates_keyframe": self.recent_first_n_candidates_keyframe,
            "keyframe_neighbour_window": self.recent_first_keyframe_neighbour_window,
        }

    @staticmethod
    def _parse_trans_scale(trans_scale):
        """Normalise ``trans_scale`` to a float (legacy linear divide) or one
        of the compression-mode strings ``"log"``/``"logd4"``/``"tanh"``
        consumed by ``WanVideoUnit_PropeCamera._apply_trans_scale``. Accepts
        numbers, numeric strings (CLI/YAML round-trips) and the mode names."""
        if isinstance(trans_scale, str):
            mode = trans_scale.strip().lower()
            if mode in ("log", "logd4", "tanh"):
                return mode
            try:
                return float(mode)
            except ValueError:
                raise ValueError(
                    "trans_scale must be a number, 'log', 'logd4' or 'tanh', "
                    f"got {trans_scale!r}."
                )
        return float(trans_scale)
