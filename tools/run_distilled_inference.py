#!/usr/bin/env python3
"""Internal one-GPU runner used by the public ``infer_distilled.py`` CLI."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import accelerate
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core.data.unified_dataset import _derive_seed
from distilled_config import load_inference_config
from examples.wanvideo.pipeline.mosaic.causal_config import build_runtime_args
from examples.wanvideo.pipeline.mosaic.causal_inference import (
    run_distilled_inference,
)
from examples.wanvideo.pipeline.mosaic.datasets import (
    build_mosaic_validation_dataset,
)
from examples.wanvideo.pipeline.mosaic.pipeline_module import (
    WanMosaicPipelineModule,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--wan-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--memory-cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def validate_paths(args):
    for label, path, kind in (
        ("config", args.config, "file"),
        ("checkpoint", args.checkpoint, "file"),
        ("dataset index", args.dataset_index, "file"),
        ("workspace", args.workspace, "dir"),
        ("Wan model", args.wan_dir, "dir"),
        ("tokenizer", args.tokenizer_dir, "dir"),
    ):
        ok = path.is_file() if kind == "file" else path.is_dir()
        if not ok:
            raise FileNotFoundError(f"{label} {kind} does not exist: {path}")


def build_model(args, accelerator):
    """Construct the existing public Matrix model with inference settings."""
    return WanMosaicPipelineModule(
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
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        trained_dit=args.trained_dit,
        task="sft",
        device=accelerator.device,
        enable_mosaic=args.enable_mosaic,
        preview_dataset=False,
        num_preview=1,
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
        recent_first_keyframe_neighbour_window=(
            args.recent_first_keyframe_neighbour_window
        ),
        mosaic_query_reference_frame=args.mosaic_query_reference_frame,
        memory_vae_encode_input_frames=args.memory_vae_encode_input_frames,
        train_anchor_block_prob=0.0,
        train_context_count_max=args.train_context_count_max,
        train_context_selection=args.train_context_selection,
        loose_model_loading=args.loose_model_loading,
        trans_scale=args.trans_scale,
        allow_no_prompt=args.allow_no_prompt,
        debug=args.debug,
    )


def main(argv=None):
    cli = build_parser().parse_args(argv)
    validate_paths(cli)
    config = load_inference_config(cli.config)
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    args = build_runtime_args(
        config,
        checkpoint=cli.checkpoint,
        dataset_index=cli.dataset_index,
        workspace=cli.workspace,
        wan_dir=cli.wan_dir,
        tokenizer_dir=cli.tokenizer_dir,
        memory_cache_dir=cli.memory_cache_dir,
    )
    args.dataset_pass_id = 0
    args.start_epoch = 0
    args.start_global_step = 0

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    )
    args.rank = accelerator.process_index
    seed = _derive_seed(args.seed, "trainer", accelerator.process_index, 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)

    input_dataset = build_mosaic_validation_dataset(args, train_dataset=None)
    if len(input_dataset) != 1:
        raise RuntimeError(
            f"explicit input index must contain exactly one sample, got {len(input_dataset)}"
        )
    model = build_model(args, accelerator)
    model.eval()
    run_distilled_inference(
        accelerator,
        input_dataset,
        model,
        cli.output,
        args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
