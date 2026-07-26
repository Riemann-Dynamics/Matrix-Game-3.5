"""Inference-only memory and camera helpers for distilled causal rollout.

This module deliberately contains no loss, optimizer, teacher scorer, training
loop, or training checkpoint state. It is shared by the standalone Stage3
sampler and the public Matrix validation output writer.
"""

from __future__ import annotations

import gc
import contextlib
import json
import math
import os

import numpy as np
import torch

from diffsynth.core.data.unified_dataset import _load_depth_npz_sparse
from diffsynth.core.data.nonlocal_memory_context import (
    normalize_dynamic_context_selection_policy,
    normalize_memory_context_selection_policy,
    select_pose_near_oldest,
)
from .cleanup import _release_cached_memory, _trim_cpu_allocator
from .history import (
    _candidate_groups_to_lists,
    _select_per3_coverage_indices,
    _train_clean_history_actual_latent_time as _history_actual_latent_time,
)
from .video_io import (
    _decode_latents_to_numpy_frames,
    _encode_frames_per_frame,
    _read_video_gt_frames,
)

_DA3_INSUFFICIENT_NON_SKY_MESSAGE = "Insufficient non-sky pixels for alignment"


def _normalize_causal_memory_bootstrap_mode(value):
    mode = str(value or "c0_only").strip().lower().replace("-", "_")
    if mode in {"c0", "c0_only", "ref", "reference", "anchor", "anchor_only"}:
        return "c0_only"
    if mode in {"history", "history_plus_c0", "legacy", "old"}:
        return "history"
    if mode in {"mixed", "mix", "random", "rand", "sample"}:
        return "mixed"
    raise ValueError(
        "causal memory bootstrap mode must be c0_only, history, or mixed; "
        f"got {value!r}."
    )

def _normalize_optional_mode_value(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", "none", "None", "null"} else text

def _get_first_arg_value(args, *names, default=None):
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value
    return default

def _generated_memory_query_kwargs_from_args(args):
    return {
        "candidates_per_query_group": max(
            1,
            int(
                _get_first_arg_value(
                    args,
                    "candidates_per_query_group_val",
                    "candidates_per_query_group_train",
                    default=5,
                )
                or 5
            ),
        ),
        "angle_threshold": None,
        "distance_threshold": None,
        "temporal_threshold": None,
        "fuse_mode": str(
            _get_first_arg_value(
                args,
                "mosaic_fuse_mode",
                "mosaic_fuse_mode_train",
                default="fill_stop_zbuffer",
            )
            or "fill_stop_zbuffer"
        ),
        "zbuffer_depth_preference": "far",
        "interpolation_mode": "nearest",
        "return_revgrid": False,
        "latent_merge_4frames": False,
        "selection_mode": str(
            _get_first_arg_value(
                args, "mosaic_selection_mode", default="projection_iou"
            )
            or "pose_pool_temporal_earliest"
        ),
        "candidate_nms_mode": _normalize_optional_mode_value(
            _get_first_arg_value(args, "mosaic_candidate_nms_mode", default=None)
        ),
        "candidate_nms_projection_iou_threshold": float(
            _get_first_arg_value(
                args,
                "mosaic_candidate_nms_projection_iou_threshold",
                default=0.7,
            )
            or 0.7
        ),
        "candidate_nms_min_temporal_gap": int(
            _get_first_arg_value(
                args, "mosaic_candidate_nms_min_temporal_gap", default=0
            )
            or 0
        ),
        "candidate_nms_pose_distance_threshold": float(
            _get_first_arg_value(
                args,
                "mosaic_candidate_nms_pose_distance_threshold",
                default=0.25,
            )
            or 0.25
        ),
        "candidate_nms_pool_multiplier": float(
            _get_first_arg_value(
                args, "mosaic_candidate_nms_pool_multiplier", default=2.5
            )
            or 2.5
        ),
        "coverage_pool_stride": max(
            1,
            int(
                _get_first_arg_value(
                    args, "mosaic_coverage_pool_stride", default=2
                )
                or 2
            ),
        ),
        "query_reference_frame": int(
            _get_first_arg_value(args, "mosaic_query_reference_frame", default=4)
            or 4
        ),
    }

def _generated_memory_publish_interval_from_args(args):
    value = getattr(args, "causal_generated_memory_publish_interval", 1)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 1
    return max(1, value)

def _generated_memory_chunk_has_future_consumer(
    chunk_idx,
    total_chunks,
    publish_interval,
):
    """Return whether this chunk belongs to a K-group published before EOF.

    Generated memory becomes query-visible only after a complete K-chunk
    group.  A trailing incomplete group has no consumer, so decoding/encoding
    those chunks for memory registration is pure overhead.
    """

    chunk_idx = int(chunk_idx)
    total_chunks = int(total_chunks)
    publish_interval = max(1, int(publish_interval))
    if chunk_idx < 0 or chunk_idx >= total_chunks:
        return False
    group_end_exclusive = ((chunk_idx // publish_interval) + 1) * publish_interval
    return group_end_exclusive < total_chunks

def _stage_generated_context_entries_for_interval(
    *,
    entries,
    chunk_idx,
    publish_interval,
    generated_entries,
    pending_generated_entries,
    pending_generated_chunk_indices,
    published_generated_chunk_indices,
):
    """Publish generated visual context only after a complete K-chunk group."""

    chunk_idx = int(chunk_idx)
    publish_interval = max(1, int(publish_interval or 1))
    pending_generated_entries.extend(list(entries or []))
    pending_generated_chunk_indices.append(chunk_idx)
    if len(pending_generated_chunk_indices) < publish_interval:
        return {
            "published_context_chunks": [],
            "pending_context_chunks": list(pending_generated_chunk_indices),
            "generated_context_entries": len(generated_entries),
            "publish_interval": publish_interval,
        }

    published_chunks = list(pending_generated_chunk_indices)
    generated_entries.extend(pending_generated_entries)
    published_generated_chunk_indices.extend(published_chunks)
    pending_generated_entries.clear()
    pending_generated_chunk_indices.clear()
    return {
        "published_context_chunks": published_chunks,
        "pending_context_chunks": [],
        "generated_context_entries": len(generated_entries),
        "publish_interval": publish_interval,
    }

def _validation_dynamic_context_accepts_generated(validation_memory_mode):
    """Keep GT validation context provenance aligned with GT training.

    ``gt`` memory training only exposes static GT/history context selected for
    the current chunk. Generated rollout frames are a self-mode condition and
    must not silently enter the validation context pool when the requested
    memory mode is GT.
    """

    return str(validation_memory_mode or "").strip().lower() != "gt"

def _validation_dynamic_context_enabled_for_mode(
    args,
    validation_memory_mode,
    *,
    t2v_no_ref=False,
):
    if bool(t2v_no_ref) or not _causal_kv_dynamic_context_enabled(args):
        return False
    mode = str(validation_memory_mode or "").strip().lower()
    if (
        mode == "empty"
        and bool(getattr(args, "no_context_on_nomosaic", True))
        and not bool(getattr(args, "allow_validation_condition_mismatch", False))
    ):
        return False
    return True

def _apply_causal_validation_camera_time_scale(
    full_noisy_intr,
    full_noisy_extr,
    args,
    *,
    accelerator=None,
):
    """Validation-only camera time remap for controlled motion ablations.

    ``scale=0.5`` means output noisy RGB frame j uses the camera condition from
    original frame floor(j * 0.5), so the generated video keeps the same length
    while camera motion is slowed by 2x. Temporal/RoPE positions are left
    untouched; only the camera tensors consumed by PRoPE, dynamic context,
    memory query, and generated-memory registration are remapped.
    """
    scale = float(getattr(args, "causal_validation_camera_time_scale", 1.0) or 1.0)
    if abs(scale - 1.0) < 1e-8:
        return full_noisy_intr, full_noisy_extr
    if scale <= 0.0:
        raise ValueError(
            "causal_validation_camera_time_scale must be > 0, "
            f"got {scale}."
        )
    if full_noisy_intr is None or full_noisy_extr is None:
        return full_noisy_intr, full_noisy_extr
    if int(full_noisy_intr.shape[0]) != int(full_noisy_extr.shape[0]):
        raise RuntimeError(
            "camera time scale requires intr/extr lengths to match, got "
            f"intr={int(full_noisy_intr.shape[0])} extr={int(full_noisy_extr.shape[0])}."
        )
    n = int(full_noisy_extr.shape[0])
    if n <= 0:
        return full_noisy_intr, full_noisy_extr
    device = full_noisy_extr.device
    src = torch.floor(
        torch.arange(n, device=device, dtype=torch.float64) * scale
    ).to(dtype=torch.long)
    src = torch.clamp(src, 0, n - 1)
    intr_src = src.to(device=full_noisy_intr.device)
    scaled_intr = full_noisy_intr.index_select(0, intr_src)
    scaled_extr = full_noisy_extr.index_select(0, src)
    is_main = bool(getattr(accelerator, "is_main_process", True))
    if is_main:
        preview = src[: min(16, n)].detach().cpu().tolist()
        print(
            "[causal-kv] validation camera time scale "
            f"scale={scale:g}: output frame j reads source floor(j*scale); "
            f"frames={n}, first_src={preview}, last_src={int(src[-1].item())}.",
            flush=True,
        )
    return scaled_intr, scaled_extr

def _is_da3_insufficient_non_sky_error(exc):
    """Return True for the DA3 metric-depth alignment failure we can degrade from."""

    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _DA3_INSUFFICIENT_NON_SKY_MESSAGE in str(cur):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return _DA3_INSUFFICIENT_NON_SKY_MESSAGE in repr(exc)

def _validation_clean_frame_indices(info, frame_count):
    g = int(info.get("generating_first_frame_idx", 0) or 0)
    frame_count = int(frame_count)
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [max(0, g - 1)]
    return [max(0, g - frame_count + i) for i in range(frame_count)]

def _validation_noisy_frame_indices(info, rgb_start, frame_count):
    g = int(info.get("generating_first_frame_idx", 0) or 0)
    return [max(0, g + int(rgb_start) + i) for i in range(int(frame_count))]

def _validation_depth_metadata(info):
    metadata = info.get("depth_metadata") or info.get("depth_metadata_json")
    if isinstance(metadata, str) and metadata:
        try:
            return json.loads(metadata)
        except ValueError:
            return None
    return metadata if isinstance(metadata, dict) else None

def _read_validation_gt_depth_frames(info, frame_indices):
    depth_path = info.get("depth_path")
    if not depth_path:
        video_path = info.get("video_path")
        if video_path:
            candidate = os.path.join(
                os.path.dirname(os.path.dirname(os.fspath(video_path))),
                "depth",
                "video.npz",
            )
            if os.path.isfile(candidate):
                depth_path = candidate
    if not depth_path:
        raise RuntimeError(
            "validation dataset-depth registration requested, but data['info'] "
            "does not contain depth_path and no rgb/../depth/video.npz fallback exists."
        )
    frame_indices = [int(v) for v in frame_indices]
    if not frame_indices:
        return None
    frame_count = max(frame_indices) + 1
    depths = _load_depth_npz_sparse(
        depth_path,
        frame_count,
        frame_indices,
        depth_format=info.get("depth_format"),
        depth_metadata=_validation_depth_metadata(info),
        require_depth_format=False,
    )
    return np.stack([depths[int(i)] for i in frame_indices], axis=0).astype(np.float32)

def _validation_register_rgb_source(args):
    return str(
        getattr(args, "causal_validation_register_rgb_source", "generated") or "generated"
    ).strip().lower()

def _validation_register_depth_source(args):
    return str(
        getattr(args, "causal_validation_register_depth_source", "online") or "online"
    ).strip().lower()

def _decode_generated_chunk_from_prefix_latents(
    pipe,
    prefix_latents,
    *,
    noisy_start,
    noisy_count,
    device,
    tiled=False,
    context_latents=1,
    context="VAE prefix decode",
):
    """Decode one generated chunk with a bounded VAE latent context.

    Wan VAE maps ``T_lat`` latents to ``1 + 4 * (T_lat - 1)`` RGB frames.
    Latent slot 0 is the clean anchor frame. Noisy latent slot ``i`` maps to
    RGB frames ``[1 + 4*i : 1 + 4*(i+1)]``. For registration throughput we keep
    only a small number of raw latent slots before the current chunk, then slice
    out the current chunk frames. ``context_latents=None`` keeps the full prefix.
    """
    if prefix_latents.ndim == 4:
        prefix_t = int(prefix_latents.shape[1])
    elif prefix_latents.ndim == 5:
        prefix_t = int(prefix_latents.shape[2])
    else:
        raise RuntimeError(
            f"{context}: prefix_latents must have shape (C,T,H,W) or "
            f"(1,C,T,H,W), got {tuple(prefix_latents.shape)}."
        )
    noisy_start = int(noisy_start)
    noisy_count = int(noisy_count)
    expected_t = 1 + noisy_start + noisy_count
    if prefix_t < expected_t:
        raise RuntimeError(
            f"{context}: prefix latent length {prefix_t} is shorter than "
            f"required {expected_t} for noisy_start={noisy_start}, "
            f"noisy_count={noisy_count}."
        )
    current_latent_start = 1 + noisy_start
    if context_latents is None:
        window_start = 0
    else:
        context_latents = int(context_latents)
        if context_latents < 1:
            raise RuntimeError(f"{context}: context_latents must be >= 1 or None.")
        window_start = max(0, current_latent_start - context_latents)
    window_end = expected_t
    if prefix_latents.ndim == 4:
        decode_latents = prefix_latents[:, window_start:window_end, :, :]
    else:
        decode_latents = prefix_latents[:, :, window_start:window_end, :, :]
    local_noisy_start = current_latent_start - window_start - 1
    decoded = _decode_latents_to_numpy_frames(pipe, decode_latents, device, tiled=tiled)
    rgb_start = 1 + local_noisy_start * 4
    rgb_end = rgb_start + noisy_count * 4
    if int(decoded.shape[0]) < rgb_end:
        raise RuntimeError(
            f"{context}: decoded frame length {int(decoded.shape[0])} is shorter "
            f"than required rgb_end={rgb_end}."
        )
    frames = decoded[rgb_start:rgb_end]
    if int(frames.shape[0]) != noisy_count * 4:
        raise RuntimeError(
            f"{context}: decoded chunk has {int(frames.shape[0])} frames, "
            f"expected {noisy_count * 4}."
        )
    return np.ascontiguousarray(frames)

def _camera_array_for_frame_count(camera, frame_count, *, name, selection="head"):
    arr = camera.detach().float().cpu().numpy() if isinstance(camera, torch.Tensor) else np.asarray(camera)
    frame_count = int(frame_count)
    if frame_count <= 0:
        raise RuntimeError(f"{name} requested non-positive frame_count={frame_count}.")
    if arr.ndim != 3:
        raise RuntimeError(f"{name} camera must have shape (T,...), got {arr.shape}.")
    if arr.shape[0] == frame_count:
        return arr
    if arr.shape[0] > frame_count:
        selection = str(selection).lower().strip()
        if selection == "head":
            return arr[:frame_count]
        if selection == "tail":
            return arr[-frame_count:]
        raise RuntimeError(
            f"{name} camera selection must be 'head' or 'tail', got {selection!r}."
        )
    if arr.shape[0] <= 0:
        raise RuntimeError(f"{name} camera is empty; cannot match {frame_count} frames.")
    tail = np.repeat(arr[-1:], frame_count - arr.shape[0], axis=0)
    return np.concatenate([arr, tail], axis=0)

def _latent_memory_frame_count(source_latents, *, name):
    if not torch.is_tensor(source_latents):
        raise RuntimeError(f"{name} source_latents must be a tensor.")
    if source_latents.ndim == 4:
        return int(source_latents.shape[1])
    if source_latents.ndim == 5:
        return int(source_latents.shape[2])
    raise RuntimeError(
        f"{name} source_latents must have shape (C,T,H,W) or (B,C,T,H,W), "
        f"got {tuple(source_latents.shape)}."
    )

def _registered_memory_K_for_query(handler, source_latents, *, context):
    memory_K = np.asarray(getattr(handler, "_Kpixs", []), dtype=np.float32)
    source_T = _latent_memory_frame_count(source_latents, name=context)
    registered_count = len(getattr(handler, "extrinsics", []) or [])
    if registered_count != source_T or int(memory_K.shape[0]) != source_T:
        raise RuntimeError(
            f"{context} registered memory length mismatch: "
            f"source_latents_T={source_T}, handler_extrinsics={registered_count}, "
            f"handler_K={int(memory_K.shape[0])}."
        )
    if memory_K.ndim != 3 or memory_K.shape[1:] != (3, 3):
        raise RuntimeError(f"{context} handler_K must have shape (M,3,3), got {memory_K.shape}.")
    return memory_K

@contextlib.contextmanager
def _maybe_gather_zero3_module(accelerator, module, *, label):
    if _accelerator_zero_stage(accelerator) != 3:
        yield
        return
    try:
        import deepspeed
    except Exception as exc:
        raise RuntimeError(
            f"ZeRO-3 validation requires deepspeed.zero.GatheredParameters for {label}."
        ) from exc
    params = [p for p in module.parameters(recurse=True) if p is not None]
    if not params:
        yield
        return
    if getattr(accelerator, "is_main_process", False):
        print(f"[zero3][validation] gathering {label} parameters", flush=True)
    with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
        yield

def _encode_context_windows_to_latents(pipe, frame_windows_np, device, tiled=False):
    """Encode 5-RGB-frame context windows exactly like train clean-history C_i.

    Each input window is ``[prev_rgb, block_rgb_0..3]``. Wan VAE emits two
    latent slots for five frames; the last slot is the context latent that
    training appends into C0.
    """
    if not frame_windows_np:
        return None
    videos = []
    dtype = pipe.torch_dtype
    for frames in frame_windows_np:
        arr = np.asarray(frames)
        if arr.ndim != 4 or arr.shape[0] != 5 or arr.shape[-1] != 3:
            raise RuntimeError(
                "context window encode expects (5,H,W,3) uint8 frames, "
                f"got {tuple(arr.shape)}."
            )
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(
            device=device, dtype=dtype
        )
        videos.append(t.permute(3, 0, 1, 2).contiguous() / 127.5 - 1.0)
    pipe.load_models_to_device(["vae"])
    try:
        encoded = pipe.vae.encode(videos, device=device, tiled=tiled)
    finally:
        pipe.load_models_to_device([])
    if encoded.ndim != 5 or int(encoded.shape[2]) < 1:
        raise RuntimeError(
            "context window VAE encode returned unexpected shape "
            f"{tuple(encoded.shape)}."
        )
    encoded = encoded[:, :, -1:].contiguous()
    return (
        encoded.squeeze(2)
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
        .contiguous()
        .detach()
        .to("cpu", dtype=torch.float32)
    )

def _causal_kv_dynamic_context_enabled(args):
    if not bool(getattr(args, "causal_kv_dynamic_context", True)):
        return False
    modes = [
        getattr(args, "validation_context_selection", None),
        getattr(args, "validation_clean_latent_pool_history_selection_mode", None),
        getattr(args, "train_context_selection", None),
    ]
    return any(
        str(mode or "").strip().lower() == "per_3latentframe"
        for mode in modes
    )

def _w2c_camera_center(w2c):
    arr = np.asarray(w2c, dtype=np.float64)
    if arr.shape != (4, 4):
        return None
    return -(arr[:3, :3].T @ arr[:3, 3])

def _rotation_distance_rad(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        return 0.0
    rel = b[:3, :3] @ a[:3, :3].T
    cos_v = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_v))

def _causal_kv_context_pose_score(entry_extrinsic, target_extrinsics):
    target = np.asarray(target_extrinsics, dtype=np.float64)
    if target.ndim == 2 and target.shape == (4, 4):
        target = target[None, ...]
    if target.ndim != 3 or target.shape[1:] != (4, 4) or target.shape[0] <= 0:
        return 0.0
    c0 = _w2c_camera_center(entry_extrinsic)
    centers = np.asarray([_w2c_camera_center(x) for x in target], dtype=np.float64)
    if c0 is None or not np.isfinite(centers).all():
        trans = 0.0
    else:
        trans = float(np.linalg.norm(centers - c0[None, :], axis=-1).mean())
    rot = float(np.mean([_rotation_distance_rad(entry_extrinsic, x) for x in target]))
    return trans + rot

def _camera_block_or_none(values, start, count=4):
    if values is None:
        return None
    arr = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    start = int(start)
    end = start + int(count)
    if arr.ndim != 3 or start < 0 or end > int(arr.shape[0]):
        return None
    return np.asarray(arr[start:end], dtype=np.float32)

def _memory_intrinsic_block(fused_inputs, start, count=4):
    memory_K = fused_inputs.get("memory_K")
    block = _camera_block_or_none(memory_K, start, count=count)
    if block is not None:
        return block
    K = fused_inputs.get("K")
    if K is None:
        return None
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        return None
    return np.repeat(K[None, ...], int(count), axis=0)

def _candidate_history_block_start(frame_id):
    frame_id = int(frame_id)
    if frame_id < 0:
        return None
    return (frame_id // 4) * 4

class _CausalKVDynamicContextPool:
    """Chunk-level per3 visual context pool for KV validation/sampling.

    The pool has two sources:
      * original history: at most one pre-encoded C_i candidate per chunk;
      * generated prefix: after chunk i is decoded, each generated latent is
        re-encoded from a 5-frame window and becomes eligible for future chunks.

    Entries are filtered at query time so chunk i never sees generated chunks
    >= i. Initial history entries are tagged with their intended chunk, matching
    training's ``first_frame_context_chunk_indices`` contract.
    """

    def __init__(
        self,
        *,
        initial_entries=None,
        initial_context_count=0,
        anchor_position=0,
        anchor_camera_frame=0,
        noisy_position_offset=0,
        noisy_camera_offset=0,
        anchor_latent=None,
        generated_context_publish_interval=1,
        memory_context_selection_policy="legacy",
        dynamic_context_selection_policy="oldest",
        context_pose_pool_size=5,
    ):
        self.initial_entries = list(initial_entries or [])
        self.generated_entries = []
        self.pending_generated_entries = []
        self.pending_generated_chunk_indices = []
        self.published_generated_chunk_indices = []
        self.initial_context_count = int(initial_context_count)
        self.anchor_position = int(anchor_position)
        self.anchor_camera_frame = int(anchor_camera_frame)
        self.noisy_position_offset = int(noisy_position_offset)
        self.noisy_camera_offset = int(noisy_camera_offset)
        self.anchor_latent = anchor_latent
        self.generated_context_publish_interval = max(
            1, int(generated_context_publish_interval or 1)
        )
        self.memory_context_selection_policy = (
            normalize_memory_context_selection_policy(
                memory_context_selection_policy
            )
        )
        self.dynamic_context_selection_policy = (
            normalize_dynamic_context_selection_policy(
                dynamic_context_selection_policy
            )
        )
        self.context_pose_pool_size = max(1, int(context_pose_pool_size))
        self.last_rgb_frame = None

    def position_for_noisy_index(self, noisy_index):
        return self.noisy_position_offset + int(noisy_index)

    def camera_frame_for_noisy_index(self, noisy_index):
        return self.noisy_camera_offset + int(noisy_index)

    def select_for_chunk(
        self,
        chunk_idx,
        target_extrinsics,
        *,
        exclude_positions=None,
        exclude_camera_frames=None,
        max_source_position_exclusive=None,
    ):
        chunk_idx = int(chunk_idx)
        exclude_positions = {
            int(v) for v in (exclude_positions or []) if v is not None
        }
        exclude_camera_frames = {
            int(v) for v in (exclude_camera_frames or []) if v is not None
        }
        if bool(getattr(self, "force_context_original_anchor", False)):
            if chunk_idx == 0 or self.anchor_latent is None:
                return []
            # Keep chunk 0 context-free like training. Later chunks deliberately
            # duplicate the original anchor as chunk-local visual context C_i.
            return [
                {
                    "source": "forced_original_anchor",
                    "chunk_idx": int(chunk_idx),
                    "latent": self.anchor_latent,
                    "position": int(self.anchor_position),
                    "camera_frame": int(self.anchor_camera_frame),
                    "rep_extrinsic": None,
                }
            ]
        if self.memory_context_selection_policy == "nonlocal_oldest":
            candidates = list(self.initial_entries)
        else:
            candidates = [
                entry
                for entry in self.initial_entries
                if int(entry.get("chunk_idx", -999999)) == chunk_idx
            ]
        candidates.extend(self.generated_entries)
        if exclude_positions or exclude_camera_frames:
            candidates = [
                entry
                for entry in candidates
                if int(entry.get("position", -999999999)) not in exclude_positions
                and int(entry.get("camera_frame", -999999999)) not in exclude_camera_frames
            ]
        if self.memory_context_selection_policy == "nonlocal_oldest":
            if max_source_position_exclusive is None:
                raise RuntimeError(
                    "nonlocal_oldest context selection requires the current "
                    "rolling-anchor source position."
                )
            cutoff = int(max_source_position_exclusive)
            candidates = [
                entry
                for entry in candidates
                if int(entry.get("source_timeline_position", 2**62)) < cutoff
            ]
        if not candidates:
            return []
        if self.dynamic_context_selection_policy == "oldest":
            selected = select_pose_near_oldest(
                candidates,
                target_extrinsics=target_extrinsics,
                pose_score_fn=_causal_kv_context_pose_score,
                pose_pool_size=self.context_pose_pool_size,
            )
            return [] if selected is None else [selected]
        ranked = sorted(
            candidates,
            key=lambda entry: (
                _causal_kv_context_pose_score(entry["rep_extrinsic"], target_extrinsics),
                0 if entry.get("source") in {"generated", "gt_prefix"} else 1,
                -int(entry.get("position", 0)),
            ),
        )
        return [ranked[0]]

    def _stage_generated_context_entries(self, entries, *, chunk_idx):
        return _stage_generated_context_entries_for_interval(
            entries=entries,
            chunk_idx=chunk_idx,
            publish_interval=self.generated_context_publish_interval,
            generated_entries=self.generated_entries,
            pending_generated_entries=self.pending_generated_entries,
            pending_generated_chunk_indices=self.pending_generated_chunk_indices,
            published_generated_chunk_indices=(
                self.published_generated_chunk_indices
            ),
        )

    def add_generated_chunk(
        self,
        *,
        pipe,
        generated_frames,
        chunk_idx,
        frames_per_chunk,
        rgb_start,
        full_noisy_extr,
        full_noisy_intr,
        device,
        tiled=False,
        source="generated",
    ):
        frames = np.asarray(generated_frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise RuntimeError(
                "generated context registration expects (T,H,W,3) frames, "
                f"got {tuple(frames.shape)}."
            )
        cs = int(frames_per_chunk)
        if int(frames.shape[0]) != cs * 4:
            raise RuntimeError(
                "generated context registration frame count mismatch: "
                f"{int(frames.shape[0])} vs expected {cs * 4}."
            )
        if self.last_rgb_frame is None:
            raise RuntimeError(
                "dynamic context pool has no previous RGB frame for 5-frame "
                "generated context encode."
            )
        source = str(source or "generated").strip().lower()
        if source not in {"generated", "gt_prefix"}:
            raise ValueError(
                "dynamic context registration source must be generated or "
                f"gt_prefix, got {source!r}."
            )
        windows = []
        for local_i in range(cs):
            s = int(local_i) * 4
            prev = self.last_rgb_frame if local_i == 0 else frames[s - 1]
            windows.append(np.concatenate([prev[None, ...], frames[s : s + 4]], axis=0))
        latents = _encode_context_windows_to_latents(
            pipe, windows, device=device, tiled=tiled
        )
        if latents is None:
            self.last_rgb_frame = np.ascontiguousarray(frames[-1])
            return self._stage_generated_context_entries([], chunk_idx=chunk_idx)
        extr = np.asarray(full_noisy_extr, dtype=np.float32)
        intr = np.asarray(full_noisy_intr, dtype=np.float32)
        entries = []
        for local_i in range(cs):
            latent = latents[:, :, local_i : local_i + 1].contiguous()
            global_noisy_index = 1 + int(chunk_idx) * cs + int(local_i)
            block_s = int(rgb_start) + int(local_i) * 4
            block_e = block_s + 4
            if block_e > int(extr.shape[0]) or block_e > int(intr.shape[0]):
                raise RuntimeError(
                    "generated context camera slice exceeds full noisy camera "
                    f"length: [{block_s}:{block_e}] extr={int(extr.shape[0])} "
                    f"intr={int(intr.shape[0])}."
                )
            entries.append(
                {
                    "source": source,
                    "source_kind": source,
                    "chunk_idx": int(chunk_idx),
                    "latent": latent,
                    "position": self.position_for_noisy_index(global_noisy_index),
                    "source_timeline_position": int(global_noisy_index),
                    "camera_frame": self.camera_frame_for_noisy_index(global_noisy_index),
                    "rep_extrinsic": extr[block_s + min(1, block_e - block_s - 1)],
                    "extrinsic_block": extr[block_s:block_e],
                    "intrinsic_block": intr[block_s:block_e],
                }
            )
        self.last_rgb_frame = np.ascontiguousarray(frames[-1])
        return self._stage_generated_context_entries(
            entries,
            chunk_idx=chunk_idx,
        )

def _causal_kv_first_noisy_actual_time(generating_first_frame_idx, anchor_actual_time):
    return max(
        _history_actual_latent_time(int(generating_first_frame_idx)),
        int(anchor_actual_time) + 1,
    )

def _build_initial_causal_kv_dynamic_context_pool(
    *,
    args,
    data,
    pipe,
    device,
    H_img,
    W_img,
    total_chunks,
    frames_per_chunk,
    full_noisy_extr,
    initial_c0_frames,
    initial_anchor_latent=None,
    generated_context_publish_interval=1,
    tiled=False,
):
    if not _causal_kv_dynamic_context_enabled(args):
        return None, None, None
    source_bootstrap_mode = _normalize_causal_memory_bootstrap_mode(
        getattr(args, "causal_memory_bootstrap_mode", "c0_only")
    )
    bootstrap_mode = _normalize_causal_memory_bootstrap_mode(
        getattr(args, "causal_validation_memory_bootstrap_mode", "c0_only")
    )
    force_context_original_anchor = bool(
        getattr(args, "causal_kv_force_context_original_anchor", False)
    )
    memory_context_selection_policy = normalize_memory_context_selection_policy(
        getattr(args, "causal_memory_context_selection_policy", "legacy")
    )
    dynamic_context_selection_policy = (
        normalize_dynamic_context_selection_policy(
            getattr(args, "causal_dynamic_context_selection_policy", "oldest")
        )
    )
    context_pose_pool_size = max(
        1,
        int(
            getattr(args, "causal_dynamic_context_pose_pool_size", 5)
            or 5
        ),
    )
    if force_context_original_anchor:
        policy_note = (
            " Mosaic memory still uses nonlocal_oldest; the context channel "
            "intentionally overrides retrieval with the C0 sink."
            if memory_context_selection_policy == "nonlocal_oldest"
            else ""
        )
        print(
            "[causal-kv][context][DEBUG] forcing context to the original anchor "
            "latent after a context-free chunk 0. This is a diagnostic-only "
            f"sampling mode.{policy_note}",
            flush=True,
        )
    if bootstrap_mode == "history":
        print(
            "[causal-kv][context][WARNING] validation bootstrap=history: "
            "pre-C0 history frames are visible during sampling. This is only "
            "for explicit ablations and is not the default inference contract.",
            flush=True,
        )
    elif source_bootstrap_mode == "history":
        print(
            "[causal-kv][context] validation bootstrap=c0_only overrides "
            "training bootstrap=history; pre-C0 history is not visible during "
            "validation/sampling.",
            flush=True,
        )
    fused = data.get("fused_query_inputs") or {}
    info = data.get("info") or {}
    video_path = info.get("video_path")
    G = int(info.get("generating_first_frame_idx", 0) or 0)
    if not video_path or G <= 0:
        print(
            "[causal-kv][context] dynamic per3 disabled for this sample: "
            "missing video_path or generating_first_frame_idx.",
            flush=True,
        )
        return None, None, None
    w2c = fused.get("w2c")
    if w2c is None:
        print(
            "[causal-kv][context] dynamic per3 disabled for this sample: "
            "fused_query_inputs has no w2c history poses.",
            flush=True,
        )
        return None, None, None
    w2c = np.asarray(w2c, dtype=np.float32)
    base_total = int(
        fused.get(
            "causal_gt_prefix_base_memory_total",
            fused.get("memory_total_frames", int(w2c.shape[0])),
        )
    )
    base_total = min(base_total, int(w2c.shape[0]))
    candidate_groups = _candidate_groups_to_lists(fused.get("candidate_frame_ids") or [])
    selected = []
    used_blocks = set()
    latest_block_start = max(0, G - 4)
    cs = int(frames_per_chunk)
    total_chunks = int(total_chunks)
    anchor_actual_time = _history_actual_latent_time(latest_block_start)
    # G=1..3 maps the anchor and first noisy latent to the same coarse time.
    # Keep the causal KV order strict; for G>=4 this max is a no-op.
    noisy_actual_start = _causal_kv_first_noisy_actual_time(G, anchor_actual_time)
    if bootstrap_mode == "c0_only":
        rope_position_bias = 1 - int(anchor_actual_time)
        pool = _CausalKVDynamicContextPool(
            anchor_position=int(anchor_actual_time) + int(rope_position_bias),
            anchor_camera_frame=0,
            noisy_position_offset=int(noisy_actual_start) + int(rope_position_bias) - 1,
            noisy_camera_offset=0,
            anchor_latent=initial_anchor_latent,
            generated_context_publish_interval=generated_context_publish_interval,
            memory_context_selection_policy=memory_context_selection_policy,
            dynamic_context_selection_policy=dynamic_context_selection_policy,
            context_pose_pool_size=context_pose_pool_size,
        )
        pool.force_context_original_anchor = bool(force_context_original_anchor)
        pool.last_rgb_frame = np.ascontiguousarray(np.asarray(initial_c0_frames)[-1])
        print(
            "[causal-kv][context] bootstrap=c0_only: no pre-C0 history "
            "context is seeded; generated-prefix context remains enabled.",
            flush=True,
        )
        return pool, None, None
    for chunk_idx in range(total_chunks):
        qg_start = int(chunk_idx) * cs
        qg_end = min(len(candidate_groups), qg_start + cs)
        pool_blocks = set()
        for group in candidate_groups[qg_start:qg_end]:
            for frame_id in group:
                block_start = _candidate_history_block_start(frame_id)
                if block_start is None:
                    continue
                if block_start < 4 or block_start + 4 > base_total:
                    continue
                if block_start == latest_block_start or block_start in used_blocks:
                    continue
                pool_blocks.add(int(block_start))
        rgb_start = int(chunk_idx) * cs * 4
        rgb_end = rgb_start + cs * 4
        target = np.asarray(full_noisy_extr, dtype=np.float32)[rgb_start:rgb_end]
        block_start = None
        if pool_blocks:
            picked = _select_per3_coverage_indices(
                sorted(int(v) // 4 for v in pool_blocks),
                1,
                latest_latent_idx=base_total // 4,
                clean_history_extr=w2c[:base_total],
                target_extrinsics=target,
                recent_exclusion_latents=0,
            )
            if picked:
                block_start = int(picked[0]) * 4
        if block_start is None:
            for cand in range(min(latest_block_start - 4, base_total - 4), 3, -4):
                cand = int(cand)
                if cand not in used_blocks and cand != latest_block_start:
                    block_start = cand
                    break
        if block_start is None:
            continue
        used_blocks.add(int(block_start))
        selected.append((int(chunk_idx), int(block_start)))

    if not selected:
        rope_position_bias = 1 - int(anchor_actual_time)
        pool = _CausalKVDynamicContextPool(
            anchor_position=int(anchor_actual_time) + int(rope_position_bias),
            anchor_camera_frame=0,
            noisy_position_offset=int(noisy_actual_start) + int(rope_position_bias) - 1,
            noisy_camera_offset=0,
            anchor_latent=initial_anchor_latent,
            generated_context_publish_interval=generated_context_publish_interval,
            memory_context_selection_policy=memory_context_selection_policy,
            dynamic_context_selection_policy=dynamic_context_selection_policy,
            context_pose_pool_size=context_pose_pool_size,
        )
        pool.force_context_original_anchor = bool(force_context_original_anchor)
        pool.last_rgb_frame = np.ascontiguousarray(np.asarray(initial_c0_frames)[-1])
        return pool, None, None

    windows = []
    for _chunk_idx, block_start in selected:
        frame_ids = [max(0, int(block_start) - 1 + off) for off in range(5)]
        frames = _read_video_gt_frames(video_path, frame_ids, H_img, W_img)
        windows.append(frames)
    latents = _encode_context_windows_to_latents(
        pipe, windows, device=device, tiled=tiled
    )
    if latents is None:
        return None, None, None

    # Match training clean-history RoPE contract. PRoPE camera frame ids below
    # still index the compact validation camera_info layout; only temporal RoPE
    # positions carry the original history/target time gaps.
    context_actual_times = [
        _history_actual_latent_time(block_start)
        for _chunk_idx, block_start in selected
    ]
    clean_actual_times = [int(v) for v in context_actual_times] + [
        int(anchor_actual_time)
    ]
    rope_position_bias = 1 - min(clean_actual_times)
    context_rope_positions = [
        int(t) + int(rope_position_bias) for t in context_actual_times
    ]

    entries = []
    clean_extr_blocks = []
    clean_intr_blocks = []
    for idx, (chunk_idx, block_start) in enumerate(selected):
        extr_block = _camera_block_or_none(w2c, block_start, count=4)
        intr_block = _memory_intrinsic_block(fused, block_start, count=4)
        if extr_block is None or intr_block is None:
            raise RuntimeError(
                "[causal-kv][context] selected history context lacks camera "
                f"block at frame {block_start}."
            )
        clean_extr_blocks.append(extr_block)
        clean_intr_blocks.append(intr_block)
        entries.append(
            {
                "source": "history",
                "chunk_idx": int(chunk_idx),
                "latent": latents[:, :, idx : idx + 1].contiguous(),
                "position": int(context_rope_positions[idx]),
                "rope_actual_time": int(context_actual_times[idx]),
                "source_timeline_position": (
                    int(context_actual_times[idx]) - int(anchor_actual_time)
                ),
                "camera_frame": int(idx),
                "source_frame_start": int(block_start),
                "rep_extrinsic": extr_block[min(1, int(extr_block.shape[0]) - 1)],
                "extrinsic_block": extr_block,
                "intrinsic_block": intr_block,
            }
        )

    pool = _CausalKVDynamicContextPool(
        initial_entries=entries,
        initial_context_count=len(entries),
        anchor_position=int(anchor_actual_time) + int(rope_position_bias),
        anchor_camera_frame=len(entries),
        noisy_position_offset=int(noisy_actual_start) + int(rope_position_bias) - 1,
        noisy_camera_offset=len(entries),
        anchor_latent=initial_anchor_latent,
        generated_context_publish_interval=generated_context_publish_interval,
        memory_context_selection_policy=memory_context_selection_policy,
        dynamic_context_selection_policy=dynamic_context_selection_policy,
        context_pose_pool_size=context_pose_pool_size,
    )
    pool.force_context_original_anchor = bool(force_context_original_anchor)
    pool.last_rgb_frame = np.ascontiguousarray(np.asarray(initial_c0_frames)[-1])
    clean_context_extr = np.concatenate(clean_extr_blocks, axis=0)
    clean_context_intr = np.concatenate(clean_intr_blocks, axis=0)
    print(
        "[causal-kv][context] dynamic per3 enabled: "
        f"history_contexts={len(entries)} "
        f"chunk_to_frame={[(e['chunk_idx'], e['source_frame_start']) for e in entries]}",
        flush=True,
    )
    return pool, clean_context_intr, clean_context_extr

def _accelerator_zero_stage(accelerator):
    try:
        from accelerate.utils import DistributedType
    except Exception:
        return None
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        return None
    plugin = getattr(accelerator, "deepspeed_plugin", None)
    if plugin is None:
        state = getattr(accelerator, "state", None)
        plugin = getattr(state, "deepspeed_plugin", None)
    if plugin is None:
        return None
    zero_stage = getattr(plugin, "zero_stage", None)
    if zero_stage is None:
        ds_config = getattr(plugin, "hf_ds_config", None)
        cfg = getattr(ds_config, "config", None) if ds_config is not None else None
        if isinstance(cfg, dict):
            zero_stage = cfg.get("zero_optimization", {}).get("stage")
    try:
        return int(zero_stage) if zero_stage is not None else None
    except (TypeError, ValueError):
        return None
