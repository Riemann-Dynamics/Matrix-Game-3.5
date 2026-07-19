"""Few-step scheduler helpers shared by distilled causal inference paths."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

import torch


def resolve_wrapped_timesteps(
    scheduler,
    denoising_step_list: Sequence[int],
    *,
    timestep_wrap: bool = True,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map CF++ integer denoising ids onto the active Wan scheduler grid."""
    if not hasattr(scheduler, "timesteps") or len(scheduler.timesteps) != 1000:
        scheduler.set_timesteps(1000, training=False)
    base_timesteps = scheduler.timesteps.detach().to(device=device)
    base_sigmas = scheduler.sigmas.detach().to(device=device)
    table_timesteps = torch.cat([base_timesteps, base_timesteps.new_zeros((1,))])
    table_sigmas = torch.cat([base_sigmas, base_sigmas.new_zeros((1,))])

    ids = torch.as_tensor(
        [int(value) for value in denoising_step_list],
        dtype=torch.long,
        device=table_timesteps.device,
    )
    if ids.numel() == 0:
        raise ValueError("denoising_step_list must not be empty")
    if timestep_wrap:
        indices = (1000 - ids).clamp(0, table_timesteps.numel() - 1)
    else:
        raw = ids.to(dtype=table_timesteps.dtype)
        indices = torch.argmin(
            (table_timesteps[:, None] - raw[None, :]).abs(), dim=0
        )
    return (
        table_timesteps.index_select(0, indices),
        table_sigmas.index_select(0, indices),
    )


def validate_causal_dmd_denoising_schedule(
    denoising_step_list: Sequence[int],
    *,
    num_steps: Optional[int] = None,
) -> tuple[int, ...]:
    """Validate the Stage3 few-step inference solver contract."""
    schedule = tuple(int(value) for value in denoising_step_list)
    resolved_steps = len(schedule) if num_steps is None else int(num_steps)
    if resolved_steps not in {2, 3, 4}:
        raise ValueError(
            "Stage3 DMD supports 2, 3, or 4 denoising steps; "
            f"got causal_dmd_num_steps={resolved_steps}."
        )
    if len(schedule) != resolved_steps:
        raise ValueError(
            "causal_dmd_denoising_step_list length must match "
            f"causal_dmd_num_steps={resolved_steps}; got {schedule!r}."
        )
    if not schedule or schedule[0] != 1000:
        raise ValueError(
            "causal_dmd_denoising_step_list must start at 1000; "
            f"got {schedule!r}."
        )
    if any(value <= 0 or value > 1000 for value in schedule):
        raise ValueError(
            "causal_dmd_denoising_step_list values must be in [1, 1000]; "
            f"got {schedule!r}."
        )
    if any(left <= right for left, right in zip(schedule, schedule[1:])):
        raise ValueError(
            "causal_dmd_denoising_step_list must be strictly descending; "
            f"got {schedule!r}."
        )
    return schedule


def _prepare_few_step_rollout_schedule(
    denoising_timesteps: torch.Tensor,
    denoising_sigmas: torch.Tensor,
    *,
    device,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the compact timestep and sigma tables to the DiT dtype together."""
    raw_timesteps = denoising_timesteps.reshape(-1)
    raw_sigmas = denoising_sigmas.reshape(-1)
    if raw_timesteps.numel() != raw_sigmas.numel():
        raise ValueError(
            "denoising timesteps/sigmas length mismatch: "
            f"{raw_timesteps.numel()} vs {raw_sigmas.numel()}."
        )
    model_timesteps = raw_timesteps.to(device=device, dtype=model_dtype)
    scheduler_timesteps = model_timesteps.to(dtype=torch.float32)
    scheduler_sigmas = raw_sigmas.to(device=device, dtype=model_dtype).to(
        dtype=torch.float32
    )
    return scheduler_timesteps, scheduler_sigmas, model_timesteps


def prepare_causal_dmd_eval_scheduler(
    scheduler,
    denoising_step_list: Sequence[int] = (1000, 750, 500, 250),
    *,
    timestep_wrap: bool = True,
    device=None,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the model-dtype-quantized Stage3 few-step evaluation schedule."""
    schedule = validate_causal_dmd_denoising_schedule(denoising_step_list)
    timesteps, sigmas = resolve_wrapped_timesteps(
        scheduler,
        schedule,
        timestep_wrap=timestep_wrap,
        device=device,
    )
    timesteps, sigmas, _ = _prepare_few_step_rollout_schedule(
        timesteps,
        sigmas,
        device=device,
        model_dtype=model_dtype,
    )
    return timesteps, sigmas


def hiar_sde_corrupt_clean_latents(
    scheduler,
    clean_latents: torch.Tensor,
    context_timestep,
    *,
    keep_first_clean: bool,
    corruption_scale: float = 1.0,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Re-noise detached clean condition latents to the next denoising level."""
    if torch.is_tensor(context_timestep):
        timestep_value = float(context_timestep.detach().float().item())
        scheduler_timestep = context_timestep.detach().cpu()
    else:
        timestep_value = float(context_timestep)
        scheduler_timestep = timestep_value
    corruption_scale = float(corruption_scale)
    if not math.isfinite(corruption_scale) or not 0.0 <= corruption_scale <= 1.0:
        raise ValueError(
            "HiAR corruption_scale must be finite and lie in [0, 1], got "
            f"{corruption_scale}."
        )
    clean_latents = clean_latents.detach()
    if timestep_value <= 0.0 or corruption_scale <= 0.0:
        return clean_latents.clone()
    if noise is None:
        noise = torch.randn_like(clean_latents)
    else:
        noise = noise.to(device=clean_latents.device, dtype=clean_latents.dtype)
        if noise.shape != clean_latents.shape:
            raise ValueError(
                "HiAR noise shape must match clean latents: "
                f"{tuple(noise.shape)} vs {tuple(clean_latents.shape)}."
            )
    scheduled = scheduler.add_noise(clean_latents, noise, scheduler_timestep)
    corrupted = (
        scheduled
        if corruption_scale >= 1.0
        else clean_latents + corruption_scale * (scheduled - clean_latents)
    )
    if keep_first_clean and corrupted.shape[2] > 0:
        corrupted[:, :, :1] = clean_latents[:, :, :1]
    return corrupted
