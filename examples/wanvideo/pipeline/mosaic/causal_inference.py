"""Single-input distilled causal inference and ``result.mp4`` writer."""

from __future__ import annotations

import gc
import os

import torch

from .causal_rollout import (
    _normalize_causal_validation_memory_mode,
    _run_causal_kv_rollout,
    _validation_memory_mode_uses_online_depth,
    _validation_memory_mode_uses_runtime_pool,
)
from .cleanup import _rank_local_cuda_device, _release_cached_memory
from .config import _intrinsics_mode_arg
from .timing import get_timing_reporter
from .video_io import (
    _decode_latents_to_numpy_frames,
    _get_frustum_handler_cls,
    _init_registration_estimator,
    _write_video,
)


def _build_handler(data, args, *, height, width, latent_h, latent_w, estimator):
    handler_cls = _get_frustum_handler_cls()
    fixed_intrinsics = (
        _intrinsics_mode_arg(
            args, "validation_mosaic_intrinsics_mode", "episode_mean"
        )
        == "first_frame"
    )
    init_k = data["clean_latent_indices_prope_intrinsic"][0].cpu().numpy()
    init_extrinsic = data["clean_latent_indices_prope_extrinsic"][:1].cpu().numpy()
    return handler_cls(
        init_k,
        image_size=(int(height), int(width)),
        grid_size=(int(latent_h), int(latent_w)),
        depth_inf_thresh=1e9,
        depth_estimator=estimator,
        is_c2w=False,
        use_gpu=True,
        init_extrinsic=init_extrinsic,
        latent_stride=16,
        fixed_intrinsics=fixed_intrinsics,
        da3_process_res=int(args.causal_dmd_self_memory_da3_process_res),
        da3_autocast_dtype=str(args.causal_dmd_self_memory_da3_autocast_dtype),
    )


@torch.no_grad()
def run_distilled_inference(
    accelerator,
    input_dataset,
    model,
    output_file,
    args,
):
    """Run one explicit input through the causal rollout and save one video."""
    if input_dataset is None or len(input_dataset) != 1:
        raise ValueError("distilled inference requires exactly one input sample")
    output_file = os.path.abspath(os.fspath(output_file))
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()
    timing = get_timing_reporter(unwrapped_model)
    timing.start_record("distilled_inference", epoch_id=-1)
    dataloader = torch.utils.data.DataLoader(
        input_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda rows: rows[0],
        num_workers=0,
    )
    dataloader = accelerator.prepare(dataloader)

    memory_mode = _normalize_causal_validation_memory_mode(
        args.causal_validation_memory_mode
    )
    use_mosaic = bool(unwrapped_model.enable_mosaic and not unwrapped_model.only_prope)
    needs_pool = use_mosaic and _validation_memory_mode_uses_runtime_pool(memory_mode)
    needs_depth = needs_pool and _validation_memory_mode_uses_online_depth(memory_mode)
    estimator = (
        _init_registration_estimator(
            args,
            device=_rank_local_cuda_device() or unwrapped_model.pipe.device,
        )
        if needs_depth
        else None
    )

    try:
        data = next(iter(dataloader))
        if data.get("needs_vae_materialization") or data.get(
            "needs_first_latent_materialization"
        ):
            with timing.scope("distilled_inference.materialize"):
                data = unwrapped_model._materialize_data(data)

        history_latents = (
            unwrapped_model._ensure_batched_latents(data["clean_latents"])
            .detach()
            .cpu()
        )
        latent_window_size = int(data["noisy_latent_indices"].shape[-1])
        latent_h, latent_w = history_latents.shape[-2:]
        height, width = int(latent_h) * 16, int(latent_w) * 16
        handler = (
            _build_handler(
                data,
                args,
                height=height,
                width=width,
                latent_h=latent_h,
                latent_w=latent_w,
                estimator=estimator,
            )
            if needs_pool
            else None
        )
        if handler is not None and hasattr(handler, "set_timing_reporter"):
            handler.set_timing_reporter(timing)

        history_latents, _ = _run_causal_kv_rollout(
            unwrapped_model,
            args,
            data,
            input_dataset,
            handler,
            history_latents,
            latent_h,
            latent_w,
            height,
            width,
            bool(getattr(unwrapped_model, "vae_decode_tiled", False)),
            int(args.num_validation_blocks),
            latent_window_size,
            use_mosaic,
            0,
            accelerator=accelerator,
        )
        frames = _decode_latents_to_numpy_frames(
            unwrapped_model.pipe,
            history_latents,
            unwrapped_model.pipe.device,
            tiled=bool(getattr(unwrapped_model, "vae_decode_tiled", False)),
        )
        if accelerator.is_main_process:
            _write_video(output_file, list(frames))
            print(f"[distilled] saved {output_file}", flush=True)
    finally:
        estimator = None
        gc.collect()
        _release_cached_memory()
        try:
            timing.finish_record()
        except Exception:
            pass
    accelerator.wait_for_everyone()
    return output_file
