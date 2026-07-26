"""Lightweight public configuration for distilled causal inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


PROFILES = ("standard", "hiar-sde", "sink-anchor-context")


@dataclass(frozen=True)
class DistilledInferenceConfig:
    profile: str = "standard"
    height: int = 704
    width: int = 1280
    num_blocks: int = 1
    seed: int = 3407
    schedule: tuple[int, ...] = (1000, 667, 333)
    hiar_scales: tuple[float, ...] = ()
    latent_window_size: int = 21
    vae_context_blocks: int = 2
    chunk_size: int = 3
    context_chunks: int = 7
    context_selection: str = "per_3latentframe"
    memory_mode: str = "c0_plus_generated"
    memory_publish_interval: int = 1
    dynamic_context: bool = True
    dynamic_context_selection: str = "oldest"
    nonlocal_memory_context: bool = False
    context_pose_pool_size: int = 5
    student_cfg_scale: float = 3.0
    trans_scale: str | float = "logd4"
    mosaic_selection_mode: str = "projection_iou"
    mosaic_candidate_nms_mode: str = "coverage"
    mosaic_candidate_projection_iou_threshold: float = 0.7
    mosaic_candidate_pose_distance_threshold: float = 0.25
    mosaic_candidate_pool_multiplier: float = 2.5
    mosaic_coverage_pool_stride: int = 2
    mosaic_query_reference_frame: int = 4
    mosaic_intrinsics_mode: str = "per_frame"
    mosaic_fuse_mode: str = "fill_stop_zbuffer"
    mosaic_drop_holes: bool = False
    da3_process_res: int = 448
    da3_autocast_dtype: str = "bfloat16"


def _tuple(value, cast, name):
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a YAML list or comma-separated string")
    return tuple(cast(item) for item in value)


def _validate_schedule(schedule):
    if len(schedule) not in {2, 3, 4}:
        raise ValueError("schedule must contain 2, 3, or 4 denoising steps")
    if schedule[0] != 1000:
        raise ValueError("schedule must start at 1000")
    if any(value < 1 or value > 1000 for value in schedule):
        raise ValueError("schedule values must be in [1, 1000]")
    if any(left <= right for left, right in zip(schedule, schedule[1:])):
        raise ValueError("schedule must be strictly descending")


def validate_inference_config(config: DistilledInferenceConfig) -> None:
    if config.profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {config.profile!r}")
    if config.height <= 0 or config.width <= 0:
        raise ValueError("height and width must be positive")
    if config.num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if config.latent_window_size <= 0 or config.chunk_size <= 0:
        raise ValueError("latent_window_size and chunk_size must be positive")
    if config.latent_window_size % config.chunk_size:
        raise ValueError("latent_window_size must be divisible by chunk_size")
    if config.context_chunks < 0 or config.vae_context_blocks < 0:
        raise ValueError("context block counts must be non-negative")
    _validate_schedule(config.schedule)
    if config.profile != "hiar-sde" and config.hiar_scales:
        raise ValueError("hiar_scales is only valid for profile=hiar-sde")
    if config.hiar_scales:
        if len(config.hiar_scales) != len(config.schedule):
            raise ValueError("hiar_scales must have one value per schedule step")
        if any(scale < 0.0 or scale > 1.0 for scale in config.hiar_scales):
            raise ValueError("hiar_scales values must be in [0, 1]")
    if config.memory_publish_interval <= 0:
        raise ValueError("memory_publish_interval must be positive")
    if config.dynamic_context_selection not in {"latest", "oldest"}:
        raise ValueError(
            "dynamic_context_selection must be 'latest' or 'oldest'"
        )
    if config.context_pose_pool_size <= 0:
        raise ValueError("context_pose_pool_size must be positive")
    if config.student_cfg_scale < 1.0:
        raise ValueError("student_cfg_scale must be >= 1.0")


def load_inference_config(path: str | Path) -> DistilledInferenceConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"inference config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"inference config must be a YAML object: {path}")
    allowed = {field.name for field in fields(DistilledInferenceConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown inference config keys: {', '.join(unknown)}")
    if "schedule" in payload:
        payload["schedule"] = _tuple(payload["schedule"], int, "schedule")
    if "hiar_scales" in payload:
        payload["hiar_scales"] = _tuple(
            payload["hiar_scales"], float, "hiar_scales"
        )
    config = DistilledInferenceConfig(**payload)
    validate_inference_config(config)
    return config


def config_as_dict(config: DistilledInferenceConfig) -> dict:
    payload = asdict(config)
    payload["schedule"] = list(config.schedule)
    payload["hiar_scales"] = list(config.hiar_scales)
    return payload


def profile_runtime_settings(config: DistilledInferenceConfig) -> dict:
    """Translate one public profile into the three rollout policy switches."""
    return {
        "prefix_noise_mode": "hiar_sde" if config.profile == "hiar-sde" else "none",
        "noise_dynamic_context": config.profile == "hiar-sde" and config.dynamic_context,
        "force_original_anchor": config.profile == "sink-anchor-context",
    }
