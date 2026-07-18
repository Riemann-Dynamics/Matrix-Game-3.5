from __future__ import annotations

import contextlib
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from frustum.frustum_handler import make_hw_grid, reproject_pixels_nd_to_nd


CFG_DEFAULTS: Dict[str, Any] = {
    "backend": "legacy",
    "depth_res": "x4",
    "depth_resize": "area",
    "depth_filter": "edge_aware",
    "consensus_k": 2,
    "consensus_z_tol": 0.12,
    "stitch_min_density": 2,
    "stitch_smooth_iters": 3,
    "stitch_pinhole_close": True,
    "dedupe_tiles": True,
    "photo_refine": False,
    "photo_refine_iters": 1,
    "cross_frame_refine": False,
    "warn_min_candidates": 20,
    "candidate_geometry_device": "cpu",
    "device": None,
    "dtype": "float32",
    "warp_dtype": None,
    "enable_tf32": False,
    "recent_first_zbuffer": False,
    "n_candidates_keyframe": 0,
    "keyframe_rot_thresh_deg": 45.0,
    "keyframe_trans_thresh": 1.0,
    "keyframe_max_interval": 100,
    "keyframe_neighbour_window": 2,
}


@contextlib.contextmanager
def _null_scope(_name: str, metadata: Optional[Dict[str, Any]] = None):
    del metadata
    yield


def _resolve_timing_scope(timing_scope):
    return timing_scope if timing_scope is not None else _null_scope


def _final_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(CFG_DEFAULTS)
    if cfg:
        out.update(cfg)
    return out


def _resolve_backend(cfg: Dict[str, Any]) -> str:
    backend = str(cfg.get("backend", "legacy") or "legacy").lower().strip()
    if backend not in {"legacy", "gpu_fast"}:
        raise ValueError(
            "recent_first_smooth backend must be one of 'legacy' or 'gpu_fast'; "
            f"got {backend!r}."
        )
    return backend


def _resolve_torch_dtype(
    value: Any, *, default: torch.dtype = torch.float32
) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    key = str(value).lower().replace("torch.", "").strip()
    if key in {"float32", "fp32"}:
        return torch.float32
    if key in {"float64", "double", "fp64"}:
        return torch.float64
    if key in {"float16", "half", "fp16"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype {value!r}.")


def _resolve_optional_torch_dtype(value: Any) -> Optional[torch.dtype]:
    if value is None:
        return None
    if isinstance(value, str) and value.lower().strip() in {"", "none", "null"}:
        return None
    return _resolve_torch_dtype(value)


def _resolve_candidate_geometry_device(cfg: Dict[str, Any]) -> torch.device:
    device_opt = str(cfg.get("candidate_geometry_device", "cpu") or "cpu").lower()
    if device_opt == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_opt in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn(
            "candidate_geometry_device='cuda' requested but CUDA is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    if device_opt == "cpu":
        return torch.device("cpu")
    raise ValueError(
        "candidate_geometry_device must be one of 'cpu', 'cuda', 'gpu', or 'auto'; "
        f"got {device_opt!r}."
    )


def _resolve_fast_device(cfg: Dict[str, Any]) -> torch.device:
    if cfg.get("device") is not None:
        return torch.device(cfg["device"])
    return _resolve_candidate_geometry_device(cfg)


@contextlib.contextmanager
def _tf32_scope(enabled: bool):
    if not torch.cuda.is_available():
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    if enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _image_hw(H_lat: int, W_lat: int, latent_stride: int) -> Tuple[int, int]:
    return int(H_lat) * int(latent_stride), int(W_lat) * int(latent_stride)


def _scale_K_np(
    K: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]
) -> np.ndarray:
    sy = float(dst_hw[0]) / float(src_hw[0])
    sx = float(dst_hw[1]) / float(src_hw[1])
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= sx
    K2[0, 1] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def _scale_K_torch(
    K: torch.Tensor, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]
) -> torch.Tensor:
    sy = float(dst_hw[0]) / float(src_hw[0])
    sx = float(dst_hw[1]) / float(src_hw[1])
    K2 = K.clone()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[0, 1] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def _inv3x3(K: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse for one or a batch of 3x3 matrices."""
    a, b, c = K[..., 0, 0], K[..., 0, 1], K[..., 0, 2]
    d, e, f = K[..., 1, 0], K[..., 1, 1], K[..., 1, 2]
    g, h, i = K[..., 2, 0], K[..., 2, 1], K[..., 2, 2]
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    Fv = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I = a * e - b * d
    det = a * A + b * B + c * C
    adj = torch.stack(
        [
            torch.stack([A, D, G], dim=-1),
            torch.stack([B, E, H], dim=-1),
            torch.stack([C, Fv, I], dim=-1),
        ],
        dim=-2,
    )
    return adj / det[..., None, None].clamp_min(torch.finfo(K.dtype).eps)


def _scale_K_batch(
    K: torch.Tensor, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]
) -> torch.Tensor:
    sy = float(dst_hw[0]) / float(src_hw[0])
    sx = float(dst_hw[1]) / float(src_hw[1])
    K2 = K.clone()
    K2[..., 0, 0] *= sx
    K2[..., 0, 1] *= sx
    K2[..., 0, 2] *= sx
    K2[..., 1, 1] *= sy
    K2[..., 1, 2] *= sy
    return K2


def _edge_aware_smooth(d: torch.Tensor) -> torch.Tensor:
    dd = d[None, None]
    med = -F.max_pool2d(-F.max_pool2d(dd, 3, 1, 1), 3, 1, 1)[0, 0]
    grad = torch.abs(d - med)
    thr = (
        torch.quantile(grad[torch.isfinite(grad)], 0.9)
        if torch.isfinite(grad).any()
        else 0.0
    )
    return torch.where(grad > thr, med, d)


def _bilateral(d: torch.Tensor) -> torch.Tensor:
    sm = F.avg_pool2d(d[None, None], 3, 1, 1)[0, 0]
    return 0.5 * d + 0.5 * sm


def _block_min(d: torch.Tensor, target: Tuple[int, int]) -> torch.Tensor:
    h, w = d.shape
    th, tw = target
    he = torch.linspace(0, h, th + 1, device=d.device).long()
    we = torch.linspace(0, w, tw + 1, device=d.device).long()
    out = torch.full(target, -1.0, dtype=d.dtype, device=d.device)
    for i in range(th):
        for j in range(tw):
            blk = d[he[i] : he[i + 1], we[j] : we[j + 1]]
            v = blk[(blk > 1e-3) & torch.isfinite(blk)]
            if v.numel():
                out[i, j] = v.min()
    return out


def _process_depth(
    depth: np.ndarray,
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    d = torch.as_tensor(np.asarray(depth, dtype=np.float32))
    if device is not None:
        d = d.to(device=device)
    res = cfg["depth_res"]
    target = {
        "latent": (H_lat, W_lat),
        "x2": (H_lat * 2, W_lat * 2),
        "x4": (H_lat * 4, W_lat * 4),
        "x8": (H_lat * 8, W_lat * 8),
        "full": tuple(d.shape),
    }.get(res, (H_lat, W_lat))
    if cfg["depth_filter"] == "edge_aware" and min(d.shape) > 4:
        d = _edge_aware_smooth(d)
    if tuple(d.shape) != tuple(target):
        if cfg["depth_resize"] == "area":
            d = F.interpolate(d[None, None], size=target, mode="area")[0, 0]
        else:
            d = _block_min(d, target)
    if cfg["depth_filter"] == "bilateral":
        d = _bilateral(d)
    return d


def _depth_target_shape(
    depth_shape: Tuple[int, int], cfg: Dict[str, Any], *, H_lat: int, W_lat: int
) -> Tuple[int, int]:
    return {
        "latent": (H_lat, W_lat),
        "x2": (H_lat * 2, W_lat * 2),
        "x4": (H_lat * 4, W_lat * 4),
        "x8": (H_lat * 8, W_lat * 8),
        "full": tuple(depth_shape),
    }.get(cfg["depth_res"], (H_lat, W_lat))


def _edge_aware_smooth_batch(depths: torch.Tensor) -> torch.Tensor:
    dd = depths[:, None]
    med = -F.max_pool2d(-F.max_pool2d(dd, 3, 1, 1), 3, 1, 1)[:, 0]
    grad = torch.abs(depths - med)
    flat = grad.reshape(grad.shape[0], -1)
    finite = torch.isfinite(flat)
    thresholds = []
    for row, mask in zip(flat, finite):
        thresholds.append(
            torch.quantile(row[mask], 0.9) if mask.any() else row.new_tensor(0.0)
        )
    thr = torch.stack(thresholds).view(-1, 1, 1)
    return torch.where(grad > thr, med, depths)


def _process_depth_batch_fast(
    depths_list: Sequence[np.ndarray],
    cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    *,
    H_lat: int,
    W_lat: int,
) -> torch.Tensor:
    """Batch depth preprocessing for the gpu_fast backend.

    It preserves v40 config semantics. The common area-resize path is processed
    per shape batch; block-min keeps the legacy loop because it is both rare and
    intentionally exact.
    """
    if not depths_list:
        return torch.empty((0, H_lat, W_lat), dtype=dtype, device=device)

    outputs: List[Optional[torch.Tensor]] = [None] * len(depths_list)
    by_shape: Dict[Tuple[int, int], List[int]] = {}
    for idx, depth in enumerate(depths_list):
        shape = tuple(np.asarray(depth).shape)
        by_shape.setdefault(shape, []).append(idx)

    for shape, indices in by_shape.items():
        batch_np = np.stack(
            [np.asarray(depths_list[idx], dtype=np.float32) for idx in indices], axis=0
        )
        d = torch.as_tensor(batch_np, dtype=dtype, device=device)
        target = _depth_target_shape(shape, cfg, H_lat=H_lat, W_lat=W_lat)
        if cfg["depth_filter"] == "edge_aware" and min(shape) > 4:
            d = _edge_aware_smooth_batch(d)
        if tuple(shape) != tuple(target):
            if cfg["depth_resize"] == "area":
                d = F.interpolate(d[:, None], size=target, mode="area")[:, 0]
            else:
                for out_idx, src_idx in enumerate(indices):
                    outputs[src_idx] = _block_min(d[out_idx], target)
                continue
        if cfg["depth_filter"] == "bilateral":
            d = 0.5 * d + 0.5 * F.avg_pool2d(d[:, None], 3, 1, 1)[:, 0]
        for out_idx, src_idx in enumerate(indices):
            outputs[src_idx] = d[out_idx]

    return torch.stack([out for out in outputs if out is not None], dim=0)


def _forward_warp(
    depth: torch.Tensor,
    src_K: np.ndarray | torch.Tensor,
    src_w2c: np.ndarray | torch.Tensor,
    dst_K: np.ndarray | torch.Tensor,
    dst_w2c: np.ndarray | torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float64, device=device),
        torch.arange(w, dtype=torch.float64, device=device),
        indexing="ij",
    )
    src_K_t = torch.as_tensor(src_K, dtype=torch.float64).to(device=device)
    dst_K_t = torch.as_tensor(dst_K, dtype=torch.float64).to(device=device)
    src_w2c_t = torch.as_tensor(src_w2c, dtype=torch.float64).to(device=device)
    dst_w2c_t = torch.as_tensor(dst_w2c, dtype=torch.float64).to(device=device)
    depth_t = depth.to(dtype=torch.float64)
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)
    rays = torch.einsum("ij,hwj->hwi", torch.linalg.inv(src_K_t), pix)
    Xc = rays * depth_t[..., None]
    Rs, ts = src_w2c_t[:3, :3], src_w2c_t[:3, 3]
    Xw = torch.einsum("ij,hwj->hwi", Rs.T, Xc - ts)
    Rd, td = dst_w2c_t[:3, :3], dst_w2c_t[:3, 3]
    Xd = torch.einsum("ij,hwj->hwi", Rd, Xw) + td
    zd = Xd[..., 2]
    valid = (
        torch.isfinite(zd) & (zd > 1e-3) & torch.isfinite(depth_t) & (depth_t > 1e-3)
    )
    uv = torch.einsum("ij,hwj->hwi", dst_K_t, Xd)
    denom = torch.where(valid, zd, torch.ones_like(zd))
    x = uv[..., 0] / denom
    y = uv[..., 1] / denom
    src_xy = torch.stack([xs + 0.5, ys + 0.5], dim=-1)
    return torch.stack([x, y], dim=-1), zd, valid, src_xy


def _forward_warp_batch_fast(
    depths: torch.Tensor,
    Ks_d: torch.Tensor,
    w2cs: torch.Tensor,
    qK: torch.Tensor,
    qw2c: torch.Tensor,
    *,
    warp_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, dh, dw = depths.shape
    device = depths.device
    dtype = depths.dtype
    ys, xs = torch.meshgrid(
        torch.arange(dh, dtype=dtype, device=device),
        torch.arange(dw, dtype=dtype, device=device),
        indexing="ij",
    )
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)

    mat_dtype = warp_dtype or dtype
    depths_m = depths.to(mat_dtype)
    Ks_m = Ks_d.to(mat_dtype)
    w2cs_m = w2cs.to(mat_dtype)
    qK_m = qK.to(mat_dtype)
    qw2c_m = qw2c.to(mat_dtype)
    pix_m = pix.to(mat_dtype)

    rays = torch.einsum("nij,hwj->nhwi", _inv3x3(Ks_m), pix_m)
    Xc = rays * depths_m[..., None]
    Rs, ts = w2cs_m[:, :3, :3], w2cs_m[:, :3, 3]
    Xw = torch.einsum("nji,nhwj->nhwi", Rs, Xc - ts[:, None, None, :])
    Rq, tq = qw2c_m[:3, :3], qw2c_m[:3, 3]
    Xd = torch.einsum("ij,nhwj->nhwi", Rq, Xw) + tq
    zd = Xd[..., 2].to(dtype)
    valid = torch.isfinite(zd) & (zd > 1e-3) & torch.isfinite(depths) & (depths > 1e-3)
    uv = torch.einsum("ij,nhwj->nhwi", qK_m, Xd).to(dtype)
    denom = torch.where(valid, zd, torch.ones_like(zd))
    qx = uv[..., 0] / denom
    qy = uv[..., 1] / denom
    return qx, qy, zd, valid


def _coerce_source_valid_masks_torch(
    source_valid_masks: Optional[torch.Tensor | np.ndarray],
    *,
    H_lat: int,
    W_lat: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if source_valid_masks is None:
        return None
    if isinstance(source_valid_masks, torch.Tensor):
        masks = source_valid_masks.detach().to(device=device, dtype=torch.bool)
    else:
        masks = torch.as_tensor(
            np.asarray(source_valid_masks), dtype=torch.bool, device=device
        )
    if masks.ndim != 3 or tuple(masks.shape[-2:]) != (int(H_lat), int(W_lat)):
        raise ValueError(
            "source_valid_masks must have shape (T, H_lat, W_lat); "
            f"got {tuple(masks.shape)} for latent grid {(int(H_lat), int(W_lat))}."
        )
    return masks.contiguous()


def _source_mask_lookup_torch(
    src_lat: torch.Tensor,
    src_h: torch.Tensor,
    src_w: torch.Tensor,
    source_valid_masks: Optional[torch.Tensor],
    *,
    H_lat: int,
    W_lat: int,
) -> torch.Tensor:
    valid = (src_lat >= 0) & torch.isfinite(src_h) & torch.isfinite(src_w)
    if source_valid_masks is None:
        return valid
    T = int(source_valid_masks.shape[0])
    valid = valid & (src_lat < T)
    if not valid.any():
        return valid
    safe_lat = torch.where(valid, src_lat, torch.zeros_like(src_lat)).long()
    safe_h = torch.where(valid, src_h, torch.zeros_like(src_h))
    safe_w = torch.where(valid, src_w, torch.zeros_like(src_w))
    safe_h = torch.round(safe_h).long().clamp(0, int(H_lat) - 1)
    safe_w = torch.round(safe_w).long().clamp(0, int(W_lat) - 1)
    mask_ok = source_valid_masks[safe_lat, safe_h, safe_w]
    return valid & mask_ok


def _invalidate_masked_sources_torch(
    src_lat: torch.Tensor,
    src_h: torch.Tensor,
    src_w: torch.Tensor,
    src_z: Optional[torch.Tensor],
    source_valid_masks: Optional[torch.Tensor],
    *,
    H_lat: int,
    W_lat: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if source_valid_masks is None:
        return src_lat, src_h, src_w, src_z
    keep = _source_mask_lookup_torch(
        src_lat,
        src_h,
        src_w,
        source_valid_masks,
        H_lat=H_lat,
        W_lat=W_lat,
    )
    src_lat = torch.where(keep, src_lat, torch.full_like(src_lat, -1))
    src_h = torch.where(keep, src_h, torch.full_like(src_h, -1.0))
    src_w = torch.where(keep, src_w, torch.full_like(src_w, -1.0))
    if src_z is not None:
        src_z = torch.where(keep, src_z, torch.full_like(src_z, float("inf")))
    return src_lat, src_h, src_w, src_z


def _invalidate_masked_source_revgrid_np(
    source_revgrid: np.ndarray,
    source_valid_masks: Optional[torch.Tensor | np.ndarray],
    *,
    H_lat: int,
    W_lat: int,
) -> np.ndarray:
    if source_valid_masks is None:
        return source_revgrid
    rev = np.asarray(source_revgrid).copy()
    masks = (
        source_valid_masks.detach().cpu().numpy()
        if isinstance(source_valid_masks, torch.Tensor)
        else np.asarray(source_valid_masks)
    ).astype(bool, copy=False)
    if masks.ndim != 3 or tuple(masks.shape[-2:]) != (int(H_lat), int(W_lat)):
        raise ValueError(
            "source_valid_masks must have shape (T, H_lat, W_lat); "
            f"got {tuple(masks.shape)} for latent grid {(int(H_lat), int(W_lat))}."
        )
    flat = rev.reshape(-1, 3)
    lat = flat[:, 0].astype(np.int64, copy=False)
    valid = (
        (lat >= 0)
        & (lat < int(masks.shape[0]))
        & np.isfinite(flat[:, 1])
        & np.isfinite(flat[:, 2])
    )
    if np.any(valid):
        sh = np.clip(np.rint(flat[:, 1]).astype(np.int64), 0, int(H_lat) - 1)
        sw = np.clip(np.rint(flat[:, 2]).astype(np.int64), 0, int(W_lat) - 1)
        keep = np.zeros(flat.shape[0], dtype=bool)
        keep[valid] = masks[lat[valid], sh[valid], sw[valid]]
        flat[~keep] = -1.0
    else:
        flat[:] = -1.0
    return rev


def _grid_per_cand_batch(
    qx: torch.Tensor,
    qy: torch.Tensor,
    zd: torch.Tensor,
    valid: torch.Tensor,
    cand_latents: torch.Tensor,
    source_valid_masks: Optional[torch.Tensor] = None,
    *,
    H_lat: int,
    W_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, dh, dw = qx.shape
    size = H_lat * W_lat
    inb = valid & (qx >= 0) & (qx < W_lat) & (qy >= 0) & (qy < H_lat)
    frame_idx = torch.arange(n, device=device).view(n, 1, 1).expand(n, dh, dw)
    cell = qy.long().clamp(0, H_lat - 1) * W_lat + qx.long().clamp(0, W_lat - 1)
    flat_idx = (frame_idx * size + cell).reshape(-1)
    flat_mask = inb.reshape(-1)
    ys, xs = torch.meshgrid(
        torch.arange(dh, dtype=dtype, device=device),
        torch.arange(dw, dtype=dtype, device=device),
        indexing="ij",
    )
    src_h = ((ys + 0.5) / float(dh) * float(H_lat)).expand(n, dh, dw).reshape(-1)
    src_w = ((xs + 0.5) / float(dw) * float(W_lat)).expand(n, dh, dw).reshape(-1)
    frame_flat_all = frame_idx.reshape(-1)
    if source_valid_masks is not None and flat_mask.any():
        src_ok = _source_mask_lookup_torch(
            cand_latents[frame_flat_all[flat_mask]],
            src_h[flat_mask],
            src_w[flat_mask],
            source_valid_masks,
            H_lat=H_lat,
            W_lat=W_lat,
        )
        filtered_flat_mask = torch.zeros_like(flat_mask)
        keep_idx = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)[src_ok]
        filtered_flat_mask[keep_idx] = True
        flat_mask = filtered_flat_mask

    GL = torch.full((n, size), -1, dtype=torch.long, device=device)
    GH = torch.full((n, size), -1.0, dtype=dtype, device=device)
    GW = torch.full((n, size), -1.0, dtype=dtype, device=device)
    GZ = torch.full((n, size), float("inf"), dtype=dtype, device=device)
    GV = torch.zeros((n, size), dtype=torch.long, device=device)
    if not flat_mask.any():
        return GL, GH, GW, GZ, GV

    hit_idx = flat_idx[flat_mask]
    z_flat = zd.reshape(-1)[flat_mask]
    GZ.reshape(-1).scatter_reduce_(0, hit_idx, z_flat, reduce="amin", include_self=True)
    one = torch.ones_like(hit_idx, dtype=torch.long)
    GV.reshape(-1).scatter_add_(0, hit_idx, one)

    z_winners = GZ.reshape(-1)[hit_idx]
    win = torch.isclose(z_flat, z_winners, rtol=1e-5, atol=1e-6)
    win_idx = hit_idx[win]
    src_h_flat = src_h[flat_mask][win]
    src_w_flat = src_w[flat_mask][win]
    frame_flat = frame_flat_all[flat_mask][win]
    src_lat_flat = cand_latents[frame_flat]
    GH.reshape(-1).scatter_(0, win_idx, src_h_flat)
    GW.reshape(-1).scatter_(0, win_idx, src_w_flat)
    GL.reshape(-1).scatter_(0, win_idx, src_lat_flat)
    return GL, GH, GW, GZ, GV


def _candidate_geometry_cache_key(
    rec: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
) -> Tuple[Any, ...]:
    depth = np.asarray(rec["depth"])
    device = _resolve_candidate_geometry_device(cfg)
    return (
        int(rec.get("frame_id", rec.get("latent_idx", -1))),
        tuple(depth.shape),
        str(depth.dtype),
        int(H_lat),
        int(W_lat),
        int(latent_stride),
        str(cfg.get("depth_res")),
        str(cfg.get("depth_resize")),
        str(cfg.get("depth_filter")),
        str(device),
    )


def _prepare_candidate_geometry(
    rec: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
    candidate_geometry_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
    timing_scope=None,
    timing_prefix: str = "recent_first.candidate_geometry",
) -> Dict[str, Any]:
    scope = _resolve_timing_scope(timing_scope)
    key = _candidate_geometry_cache_key(
        rec,
        cfg,
        H_lat=H_lat,
        W_lat=W_lat,
        latent_stride=latent_stride,
    )
    if candidate_geometry_cache is not None and key in candidate_geometry_cache:
        with scope(
            f"{timing_prefix}.cache_hit",
            metadata={"frame_id": int(rec.get("frame_id", -1))},
        ):
            return candidate_geometry_cache[key]

    imgh, imgw = _image_hw(H_lat, W_lat, latent_stride)
    geometry_device = _resolve_candidate_geometry_device(cfg)
    with scope(
        f"{timing_prefix}.process_depth",
        metadata={
            "frame_id": int(rec.get("frame_id", -1)),
            "device": str(geometry_device),
        },
    ):
        depth = _process_depth(
            rec["depth"],
            cfg,
            H_lat=H_lat,
            W_lat=W_lat,
            device=geometry_device,
        )
    dh, dw = depth.shape
    with scope(
        f"{timing_prefix}.source_unproject",
        metadata={
            "frame_id": int(rec.get("frame_id", -1)),
            "dh": int(dh),
            "dw": int(dw),
        },
    ):
        cK = _scale_K_torch(
            torch.as_tensor(rec["K"], dtype=torch.float64).to(device=geometry_device),
            (imgh, imgw),
            (dh, dw),
        )
        ys, xs = torch.meshgrid(
            torch.arange(dh, dtype=torch.float64, device=geometry_device),
            torch.arange(dw, dtype=torch.float64, device=geometry_device),
            indexing="ij",
        )
        pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)
        rays = torch.einsum("ij,hwj->hwi", torch.linalg.inv(cK), pix)
        depth_t = depth.to(dtype=torch.float64)
        Xc = rays * depth_t[..., None]
        src_w2c_t = torch.as_tensor(rec["w2c"], dtype=torch.float64).to(
            device=geometry_device
        )
        Rs, ts = src_w2c_t[:3, :3], src_w2c_t[:3, 3]
        Xw = torch.einsum("ij,hwj->hwi", Rs.T, Xc - ts)
        valid_depth = torch.isfinite(depth_t) & (depth_t > 1e-3)
        src_xy = torch.stack([xs + 0.5, ys + 0.5], dim=-1)
    geom = {
        "dh": int(dh),
        "dw": int(dw),
        "Xw": Xw,
        "valid_depth": valid_depth,
        "src_xy": src_xy,
    }
    if candidate_geometry_cache is not None:
        candidate_geometry_cache[key] = geom
    return geom


def _project_candidate_geometry_to_query(
    geom: Dict[str, Any],
    *,
    query_K: np.ndarray | torch.Tensor,
    query_w2c: np.ndarray | torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Xw = geom["Xw"]
    dst_K_t = torch.as_tensor(query_K, dtype=torch.float64).to(device=Xw.device)
    dst_w2c_t = torch.as_tensor(query_w2c, dtype=torch.float64).to(device=Xw.device)
    Rd, td = dst_w2c_t[:3, :3], dst_w2c_t[:3, 3]
    Xd = torch.einsum("ij,hwj->hwi", Rd, Xw) + td
    zd = Xd[..., 2]
    valid = geom["valid_depth"] & torch.isfinite(zd) & (zd > 1e-3)
    uv = torch.einsum("ij,hwj->hwi", dst_K_t, Xd)
    denom = torch.where(valid, zd, torch.ones_like(zd))
    x = uv[..., 0] / denom
    y = uv[..., 1] / denom
    return torch.stack([x, y], dim=-1), zd, valid, geom["src_xy"]


def _majority_label(
    label2d: torch.Tensor,
    valid3d: torch.Tensor,
    smooth_iters: int,
    *,
    H_lat: int,
    W_lat: int,
) -> torch.Tensor:
    Cn = valid3d.shape[0]
    lab = label2d
    for _ in range(smooth_iters):
        onehot = torch.zeros(Cn, H_lat, W_lat, dtype=torch.float64)
        for c in range(Cn):
            onehot[c] = (lab == c).double()
        cnt = F.avg_pool2d(onehot.unsqueeze(1), 3, 1, 1)[:, 0] * 9.0
        cnt = torch.where(valid3d, cnt, torch.full_like(cnt, -1.0))
        newlab = cnt.argmax(0)
        newmax = cnt.max(0).values
        lab = torch.where((lab >= 0) & (newmax > 0), newlab, lab)
    return lab


def _pinhole_close(
    sl: torch.Tensor,
    sh: torch.Tensor,
    sw: torch.Tensor,
    *,
    H_lat: int,
    W_lat: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sl = sl.reshape(H_lat, W_lat).clone()
    sh = sh.reshape(H_lat, W_lat).clone()
    sw = sw.reshape(H_lat, W_lat).clone()
    filled = (sl >= 0).double()
    nfill = F.avg_pool2d(filled[None, None], 3, 1, 1)[0, 0] * 9.0 - filled
    fillable = (sl < 0) & (nfill >= 5)
    for di, dj in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        rl = torch.roll(sl, (di, dj), (0, 1))
        rh = torch.roll(sh, (di, dj), (0, 1))
        rw = torch.roll(sw, (di, dj), (0, 1))
        take = fillable & (sl < 0) & (rl >= 0)
        sl = torch.where(take, rl, sl)
        sh = torch.where(take, rh, sh)
        sw = torch.where(take, rw, sw)
    return sl.reshape(-1), sh.reshape(-1), sw.reshape(-1)


def _majority_label_gpu(
    label2d: torch.Tensor,
    valid3d: torch.Tensor,
    smooth_iters: int,
    *,
    H_lat: int,
    W_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cn = valid3d.shape[0]
    lab = label2d
    for _ in range(smooth_iters):
        onehot = torch.zeros(cn, H_lat, W_lat, dtype=dtype, device=device)
        idx_safe = lab.clamp(0)
        onehot.scatter_(0, idx_safe[None].expand(cn, -1, -1), 1.0)
        onehot = torch.where(
            (lab >= 0).unsqueeze(0).expand_as(onehot), onehot, torch.zeros_like(onehot)
        )
        cnt = F.avg_pool2d(onehot.unsqueeze(1), 3, 1, 1)[:, 0] * 9.0
        cnt = torch.where(valid3d, cnt, torch.full_like(cnt, -1.0))
        newlab = cnt.argmax(0)
        newmax = cnt.max(0).values
        lab = torch.where((lab >= 0) & (newmax > 0), newlab, lab)
    return lab


def _pinhole_close_gpu_fast(
    sl: torch.Tensor,
    sh: torch.Tensor,
    sw: torch.Tensor,
    *,
    H_lat: int,
    W_lat: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sl = sl.reshape(H_lat, W_lat).clone()
    sh = sh.reshape(H_lat, W_lat).clone()
    sw = sw.reshape(H_lat, W_lat).clone()
    filled = (sl >= 0).to(sh.dtype)
    nfill = F.avg_pool2d(filled[None, None], 3, 1, 1)[0, 0] * 9.0 - filled
    fillable = (sl < 0) & (nfill >= 5)
    if not fillable.any():
        return sl.reshape(-1), sh.reshape(-1), sw.reshape(-1)
    for di, dj in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]:
        rl = torch.roll(sl, (di, dj), (0, 1))
        rh = torch.roll(sh, (di, dj), (0, 1))
        rw = torch.roll(sw, (di, dj), (0, 1))
        take = fillable & (sl < 0) & (rl >= 0)
        sl = torch.where(take, rl, sl)
        sh = torch.where(take, rh, sh)
        sw = torch.where(take, rw, sw)
    return sl.reshape(-1), sh.reshape(-1), sw.reshape(-1)


def _collapse_per_latent_fast(
    GL: torch.Tensor,
    GH: torch.Tensor,
    GW: torch.Tensor,
    GZ: torch.Tensor,
    GV: torch.Tensor,
    lats: torch.Tensor,
    *,
    H_lat: int,
    W_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = GL.shape[0]
    size = H_lat * W_lat
    uniq_lats, inv = torch.unique_consecutive(lats, return_inverse=True)
    nu = int(uniq_lats.shape[0])
    cells = torch.arange(size, device=device)
    flat_idx = (inv.view(n, 1) * size + cells.view(1, size)).reshape(-1)

    UZ_buf = torch.full((nu * size,), float("inf"), dtype=dtype, device=device)
    UZ_buf.scatter_reduce_(
        0, flat_idx, GZ.contiguous().view(-1), reduce="amin", include_self=True
    )
    UZ = UZ_buf.view(nu, size)

    UV_buf = torch.zeros(nu * size, dtype=torch.long, device=device)
    UV_buf.scatter_add_(0, flat_idx, GV.contiguous().view(-1))
    UV = UV_buf.view(nu, size)

    UZ_per_frame = UZ[inv]
    is_win = (GZ == UZ_per_frame) & (GL >= 0)
    win_mask = is_win.reshape(-1)
    win_idx = flat_idx[win_mask]

    UH = torch.full((nu * size,), -1.0, dtype=dtype, device=device)
    UW = torch.full((nu * size,), -1.0, dtype=dtype, device=device)
    UL = torch.full((nu * size,), -1, dtype=torch.long, device=device)
    UH.scatter_(0, win_idx, GH.contiguous().view(-1)[win_mask])
    UW.scatter_(0, win_idx, GW.contiguous().view(-1)[win_mask])
    UL.scatter_(0, win_idx, GL.contiguous().view(-1)[win_mask])
    return UL.view(nu, size), UH.view(nu, size), UW.view(nu, size), UZ, UV


def _stitch_gpu(
    GL: torch.Tensor,
    GH: torch.Tensor,
    GW: torch.Tensor,
    GZ: torch.Tensor,
    GV: torch.Tensor,
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cn = GL.shape[0]
    size = H_lat * W_lat
    if cn == 0:
        return (
            torch.full((size,), -1, dtype=torch.long, device=device),
            torch.full((size,), -1.0, dtype=dtype, device=device),
            torch.full((size,), -1.0, dtype=dtype, device=device),
            torch.full((size,), float("inf"), dtype=dtype, device=device),
        )
    filled = GL >= 0
    Zf = torch.where(filled, GZ, torch.full_like(GZ, float("nan")))
    zmed = Zf.nanmedian(dim=0).values
    agree = filled & (
        torch.abs(GZ - zmed.unsqueeze(0))
        <= cfg["consensus_z_tol"] * torch.abs(zmed.unsqueeze(0))
    )
    trustworthy = agree.sum(0) >= int(cfg.get("consensus_k") or 2)
    rankw = (
        torch.linspace(1.0, 0.6, cn, device=device, dtype=dtype)
        if cn > 1
        else torch.ones(1, device=device, dtype=dtype)
    ).unsqueeze(1)
    pick = agree & (GV >= float(cfg["stitch_min_density"]))
    qual = torch.where(
        pick, GV.to(dtype) * rankw, torch.full_like(GV, -1.0, dtype=dtype)
    )
    best_q, best_c = qual.max(dim=0)
    has = trustworthy & (best_q > 0)
    label = torch.where(has, best_c, torch.full_like(best_c, -1)).reshape(H_lat, W_lat)
    label = _majority_label_gpu(
        label,
        pick.reshape(cn, H_lat, W_lat),
        cfg["stitch_smooth_iters"],
        H_lat=H_lat,
        W_lat=W_lat,
        device=device,
        dtype=dtype,
    )
    lf = label.reshape(-1)
    has = lf >= 0
    idx = lf.clamp(0).unsqueeze(0)
    src_lat = torch.gather(GL, 0, idx)[0].clone()
    src_h = torch.gather(GH, 0, idx)[0].clone()
    src_w = torch.gather(GW, 0, idx)[0].clone()
    src_z = torch.gather(GZ, 0, idx)[0].clone()
    src_lat[~has] = -1
    src_h[~has] = -1.0
    src_w[~has] = -1.0
    src_z[~has] = float("inf")
    if cfg["stitch_pinhole_close"]:
        src_lat, src_h, src_w = _pinhole_close_gpu_fast(
            src_lat, src_h, src_w, H_lat=H_lat, W_lat=W_lat
        )
    return src_lat, src_h, src_w, src_z


def _dedupe_recent_tiles_gpu(
    src_lat: torch.Tensor,
    src_h: torch.Tensor,
    src_w: torch.Tensor,
    recent_lat: int,
    *,
    H_lat: int,
    W_lat: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = src_lat.device
    dtype = src_h.dtype
    sl_cpu, sh_cpu, sw_cpu = _dedupe_recent_tiles(
        src_lat.detach().cpu(),
        src_h.detach().cpu(),
        src_w.detach().cpu(),
        recent_lat,
        H_lat=H_lat,
        W_lat=W_lat,
    )
    return (
        sl_cpu.to(device=device),
        sh_cpu.to(device=device, dtype=dtype),
        sw_cpu.to(device=device, dtype=dtype),
    )


def _cand_grid(
    latent_idx: int,
    cand_frames: Sequence[Dict[str, Any]],
    qK: np.ndarray,
    qw2c: np.ndarray,
    cfg: Dict[str, Any],
    source_valid_masks: Optional[torch.Tensor | np.ndarray] = None,
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
    candidate_geometry_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
    timing_scope=None,
    timing_prefix: str = "recent_first.phase2_candidate_grid",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scope = _resolve_timing_scope(timing_scope)
    cell_l, zs_l, sh_l, sw_l = [], [], [], []
    masks_t: Optional[torch.Tensor] = None
    for fr in cand_frames:
        geom = _prepare_candidate_geometry(
            fr,
            cfg,
            H_lat=H_lat,
            W_lat=W_lat,
            latent_stride=latent_stride,
            candidate_geometry_cache=candidate_geometry_cache,
            timing_scope=timing_scope,
            timing_prefix=f"{timing_prefix}.prepare_geometry",
        )
        with scope(
            f"{timing_prefix}.project_geometry",
            metadata={"frame_id": int(fr.get("frame_id", -1))},
        ):
            xy, zd, valid, sxy = _project_candidate_geometry_to_query(
                geom,
                query_K=qK,
                query_w2c=qw2c,
            )
        with scope(
            f"{timing_prefix}.bin_projected_pixels",
            metadata={"frame_id": int(fr.get("frame_id", -1))},
        ):
            qx = xy[..., 0] / latent_stride
            qy = xy[..., 1] / latent_stride
            inb = valid & (qx >= 0) & (qx < W_lat) & (qy >= 0) & (qy < H_lat)
            sh_all = (sxy[..., 1] / int(geom["dh"])) * H_lat
            sw_all = (sxy[..., 0] / int(geom["dw"])) * W_lat
            if source_valid_masks is not None and masks_t is None:
                masks_t = _coerce_source_valid_masks_torch(
                    source_valid_masks,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    device=qx.device,
                )
            if masks_t is not None and inb.any():
                src_lat_all = torch.full_like(qx, int(latent_idx), dtype=torch.long)
                inb &= _source_mask_lookup_torch(
                    src_lat_all,
                    sh_all,
                    sw_all,
                    masks_t,
                    H_lat=H_lat,
                    W_lat=W_lat,
                )
        if not inb.any():
            continue
        with scope(
            f"{timing_prefix}.collect_hits",
            metadata={
                "frame_id": int(fr.get("frame_id", -1)),
                "hit_count": int(inb.sum().item()),
            },
        ):
            cell_l.append(
                qy[inb].long().clamp(0, H_lat - 1) * W_lat
                + qx[inb].long().clamp(0, W_lat - 1)
            )
            zs_l.append(zd[inb])
            sh_l.append(sh_all[inb])
            sw_l.append(sw_all[inb])
    size = H_lat * W_lat
    with scope(
        f"{timing_prefix}.assemble_grid",
        metadata={"latent_idx": int(latent_idx), "has_hits": bool(cell_l)},
    ):
        grid_device = cell_l[0].device if cell_l else torch.device("cpu")
        gl = torch.full((size,), -1, dtype=torch.long, device=grid_device)
        gh = torch.full((size,), -1.0, dtype=torch.float64, device=grid_device)
        gw = torch.full((size,), -1.0, dtype=torch.float64, device=grid_device)
        gz = torch.full((size,), float("inf"), dtype=torch.float64, device=grid_device)
        gv = torch.zeros(size, dtype=torch.long, device=grid_device)
        if cell_l:
            cell = torch.cat(cell_l)
            zs = torch.cat(zs_l)
            sh = torch.cat(sh_l)
            sw = torch.cat(sw_l)
            gv = torch.bincount(cell, minlength=size)
            order = torch.argsort(zs, descending=True)
            c = cell[order]
            gl[c] = int(latent_idx)
            gh[c] = sh[order]
            gw[c] = sw[order]
            gz[c] = zs[order]
    return gl.cpu(), gh.cpu(), gw.cpu(), gz.cpu(), gv.cpu()


def _stitch(
    grids: Sequence[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not grids:
        return (
            np.full((H_lat, W_lat, 3), -1.0, dtype=np.float64),
            np.full((H_lat, W_lat), float("inf"), dtype=np.float64),
        )
    Cn = len(grids)
    GL = torch.stack([g[0] for g in grids])
    GH = torch.stack([g[1] for g in grids])
    GW = torch.stack([g[2] for g in grids])
    GZ = torch.stack([g[3] for g in grids]).double()
    GV = torch.stack([g[4] for g in grids]).double()
    filled = GL >= 0
    Zf = GZ.clone()
    Zf[~filled] = float("nan")
    zmed = Zf.nanmedian(dim=0).values
    agree = filled & (
        torch.abs(GZ - zmed.unsqueeze(0))
        <= cfg["consensus_z_tol"] * torch.abs(zmed.unsqueeze(0))
    )
    trustworthy = agree.sum(0) >= int(cfg.get("consensus_k") or 2)
    rankw = (torch.linspace(1.0, 0.6, Cn) if Cn > 1 else torch.ones(1)).unsqueeze(1)
    pick = agree & (GV >= float(cfg["stitch_min_density"]))
    qual = torch.where(pick, GV * rankw, torch.full_like(GV, -1.0))
    best_q, best_c = qual.max(dim=0)
    has = trustworthy & (best_q > 0)
    label = torch.where(has, best_c, torch.full_like(best_c, -1)).reshape(H_lat, W_lat)
    label = _majority_label(
        label,
        pick.reshape(Cn, H_lat, W_lat),
        cfg["stitch_smooth_iters"],
        H_lat=H_lat,
        W_lat=W_lat,
    )
    lf = label.reshape(-1)
    has = lf >= 0
    idx = lf.clamp(0).unsqueeze(0)
    src_lat = torch.gather(GL, 0, idx)[0].clone()
    src_h = torch.gather(GH, 0, idx)[0].clone()
    src_w = torch.gather(GW, 0, idx)[0].clone()
    src_z = torch.gather(GZ, 0, idx)[0].clone()
    src_lat[~has] = -1
    src_h[~has] = -1.0
    src_w[~has] = -1.0
    src_z[~has] = float("inf")
    if cfg["stitch_pinhole_close"]:
        src_lat, src_h, src_w = _pinhole_close(
            src_lat, src_h, src_w, H_lat=H_lat, W_lat=W_lat
        )
    source_revgrid = (
        torch.stack([src_lat.double(), src_h, src_w], dim=-1)
        .reshape(H_lat, W_lat, 3)
        .numpy()
    )
    return source_revgrid, src_z.reshape(H_lat, W_lat).numpy()


def _recent_first_smooth(
    recent_lat: int,
    recent_frame: Dict[str, Any],
    qK: np.ndarray,
    qw2c: np.ndarray,
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgh, imgw = _image_hw(H_lat, W_lat, latent_stride)
    geometry_device = _resolve_candidate_geometry_device(cfg)
    depth = _process_depth(
        recent_frame["depth"],
        cfg,
        H_lat=H_lat,
        W_lat=W_lat,
        device=geometry_device,
    )
    dh, dw = depth.shape
    fK_t = torch.as_tensor(recent_frame["K"], dtype=torch.float64).to(
        device=geometry_device
    )
    fw2c_t = torch.as_tensor(recent_frame["w2c"], dtype=torch.float64).to(
        device=geometry_device
    )
    qK_t = torch.as_tensor(qK, dtype=torch.float64).to(device=geometry_device)
    qw2c_t = torch.as_tensor(qw2c, dtype=torch.float64).to(device=geometry_device)
    fK_d = _scale_K_torch(fK_t, (imgh, imgw), (dh, dw))
    xy, zd, valid, _sxy = _forward_warp(depth, fK_d, fw2c_t, qK_t, qw2c_t)
    qx = xy[..., 0] / latent_stride
    qy = xy[..., 1] / latent_stride
    inb = valid & (qx >= 0) & (qx < W_lat) & (qy >= 0) & (qy < H_lat)
    Dq = torch.full(
        (H_lat * W_lat,), float("nan"), dtype=torch.float64, device=geometry_device
    )
    if inb.any():
        cell = qy[inb].long().clamp(0, H_lat - 1) * W_lat + qx[inb].long().clamp(
            0, W_lat - 1
        )
        zb = zd[inb].double()
        order = torch.argsort(zb, descending=True)
        Dq[cell[order]] = zb[order]
    Dq = Dq.reshape(H_lat, W_lat)
    valid_dq = torch.isfinite(Dq)
    if valid_dq.any():
        vm = valid_dq.double()
        ncount = F.avg_pool2d(vm[None, None], 3, 1, 1)[0, 0] * 9.0 - vm
        empty = ~valid_dq & (ncount >= 5)
        if empty.any():
            d0 = torch.where(valid_dq, Dq, torch.zeros_like(Dq))
            sumn = F.avg_pool2d(d0[None, None], 3, 1, 1)[0, 0] * 9.0 - d0
            avgn = sumn / ncount.clamp(min=1)
            Dq = torch.where(empty, avgn, Dq)
            valid_dq = torch.isfinite(Dq)

    ys, xs = torch.meshgrid(
        torch.arange(H_lat, dtype=torch.float64, device=geometry_device),
        torch.arange(W_lat, dtype=torch.float64, device=geometry_device),
        indexing="ij",
    )
    pix = torch.stack(
        [(xs + 0.5) * latent_stride, (ys + 0.5) * latent_stride, torch.ones_like(xs)],
        dim=-1,
    )
    rays = torch.einsum("ij,hwj->hwi", torch.linalg.inv(qK_t), pix)
    Rq, tq = qw2c_t[:3, :3], qw2c_t[:3, 3]
    Dq_safe = torch.where(valid_dq, Dq, torch.ones_like(Dq))
    Xc = rays * Dq_safe[..., None]
    Xw = torch.einsum("ij,hwj->hwi", Rq.T, Xc - tq)
    Rf, tf = fw2c_t[:3, :3], fw2c_t[:3, 3]
    Xd = torch.einsum("ij,hwj->hwi", Rf, Xw) + tf
    zc_t = Xd[..., 2]
    safe_z = torch.where(zc_t > 1e-3, zc_t, torch.ones_like(zc_t))
    uv = torch.einsum("ij,hwj->hwi", fK_t, Xd)
    cu = uv[..., 0] / safe_z
    cv = uv[..., 1] / safe_z
    in_frame = (
        valid_dq & (zc_t > 1e-3) & (cu >= 0) & (cu < imgw) & (cv >= 0) & (cv < imgh)
    )
    src_lat = torch.full((H_lat, W_lat), -1, dtype=torch.long, device=geometry_device)
    src_h = torch.full(
        (H_lat, W_lat), -1.0, dtype=torch.float64, device=geometry_device
    )
    src_w = torch.full(
        (H_lat, W_lat), -1.0, dtype=torch.float64, device=geometry_device
    )
    src_z = torch.full(
        (H_lat, W_lat), float("inf"), dtype=torch.float64, device=geometry_device
    )
    src_lat[in_frame] = int(recent_lat)
    src_h[in_frame] = cv[in_frame] / latent_stride
    src_w[in_frame] = cu[in_frame] / latent_stride
    src_z[in_frame] = Dq[in_frame]
    return (
        src_lat.reshape(-1).cpu(),
        src_h.reshape(-1).cpu(),
        src_w.reshape(-1).cpu(),
        src_z.reshape(-1).cpu(),
    )


def _recent_first_smooth_gpu_pre(
    recent_lat: int,
    depth: torch.Tensor,
    fK_d: torch.Tensor,
    fK_img: torch.Tensor,
    fw2c: torch.Tensor,
    qK: torch.Tensor,
    qw2c: torch.Tensor,
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dh, dw = depth.shape
    imgh, imgw = _image_hw(H_lat, W_lat, latent_stride)
    ys, xs = torch.meshgrid(
        torch.arange(dh, dtype=dtype, device=device),
        torch.arange(dw, dtype=dtype, device=device),
        indexing="ij",
    )
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)
    rays = torch.einsum("ij,hwj->hwi", _inv3x3(fK_d), pix)
    Xc = rays * depth[..., None]
    Rs, ts = fw2c[:3, :3], fw2c[:3, 3]
    Xw = torch.einsum("ji,hwj->hwi", Rs, Xc - ts)
    Rq, tq = qw2c[:3, :3], qw2c[:3, 3]
    Xd = torch.einsum("ij,hwj->hwi", Rq, Xw) + tq
    zd = Xd[..., 2]
    valid = torch.isfinite(zd) & (zd > 1e-3) & torch.isfinite(depth) & (depth > 1e-3)
    uv = torch.einsum("ij,hwj->hwi", qK, Xd)
    denom = torch.where(valid, zd, torch.ones_like(zd))
    qx = uv[..., 0] / denom / float(latent_stride)
    qy = uv[..., 1] / denom / float(latent_stride)
    inb = valid & (qx >= 0) & (qx < W_lat) & (qy >= 0) & (qy < H_lat)

    size = H_lat * W_lat
    Dq = torch.full((size,), float("inf"), dtype=dtype, device=device)
    if inb.any():
        cell = qy.long().clamp(0, H_lat - 1) * W_lat + qx.long().clamp(0, W_lat - 1)
        Dq.scatter_reduce_(
            0,
            cell[inb].reshape(-1),
            zd[inb].reshape(-1),
            reduce="amin",
            include_self=True,
        )
    Dq[Dq == float("inf")] = float("nan")
    Dq = Dq.reshape(H_lat, W_lat)
    valid_dq = torch.isfinite(Dq)
    if valid_dq.any():
        vm = valid_dq.to(dtype)
        ncount = F.avg_pool2d(vm[None, None], 3, 1, 1)[0, 0] * 9.0 - vm
        empty = ~valid_dq & (ncount >= 5)
        if empty.any():
            d0 = torch.where(valid_dq, Dq, torch.zeros_like(Dq))
            sumn = F.avg_pool2d(d0[None, None], 3, 1, 1)[0, 0] * 9.0 - d0
            Dq = torch.where(empty, sumn / ncount.clamp(min=1), Dq)
            valid_dq = torch.isfinite(Dq)

    ys2, xs2 = torch.meshgrid(
        torch.arange(H_lat, dtype=dtype, device=device),
        torch.arange(W_lat, dtype=dtype, device=device),
        indexing="ij",
    )
    pix2 = torch.stack(
        [
            (xs2 + 0.5) * float(latent_stride),
            (ys2 + 0.5) * float(latent_stride),
            torch.ones_like(xs2),
        ],
        dim=-1,
    )
    rays2 = torch.einsum("ij,hwj->hwi", _inv3x3(qK), pix2)
    Dq_safe = torch.where(valid_dq, Dq, torch.ones_like(Dq))
    Xc2 = rays2 * Dq_safe[..., None]
    Xw2 = torch.einsum("ji,hwj->hwi", Rq, Xc2 - tq)
    Xd2 = torch.einsum("ij,hwj->hwi", Rs, Xw2) + ts
    zc = Xd2[..., 2]
    safe_z = torch.where(zc > 1e-3, zc, torch.ones_like(zc))
    uv2 = torch.einsum("ij,hwj->hwi", fK_img, Xd2)
    cu = uv2[..., 0] / safe_z
    cv = uv2[..., 1] / safe_z
    in_frame = (
        valid_dq & (zc > 1e-3) & (cu >= 0) & (cu < imgw) & (cv >= 0) & (cv < imgh)
    )
    src_lat = torch.full((H_lat, W_lat), -1, dtype=torch.long, device=device)
    src_h = torch.full((H_lat, W_lat), -1.0, dtype=dtype, device=device)
    src_w = torch.full((H_lat, W_lat), -1.0, dtype=dtype, device=device)
    src_z = torch.full((H_lat, W_lat), float("inf"), dtype=dtype, device=device)
    src_lat[in_frame] = int(recent_lat)
    src_h[in_frame] = cv[in_frame] / float(latent_stride)
    src_w[in_frame] = cu[in_frame] / float(latent_stride)
    src_z[in_frame] = Dq[in_frame]
    return src_lat.reshape(-1), src_h.reshape(-1), src_w.reshape(-1), src_z.reshape(-1)


def _dedupe_recent_tiles(
    src_lat: torch.Tensor,
    src_h: torch.Tensor,
    src_w: torch.Tensor,
    recent_lat: int,
    *,
    H_lat: int,
    W_lat: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sl_np = src_lat.numpy() if isinstance(src_lat, torch.Tensor) else src_lat
    sh_np = src_h.numpy() if isinstance(src_h, torch.Tensor) else src_h
    sw_np = src_w.numpy() if isinstance(src_w, torch.Tensor) else src_w
    mask = sl_np == int(recent_lat)
    if mask.sum() < 2:
        return src_lat, src_h, src_w
    idx = np.where(mask)[0]
    fh = sh_np[idx].astype(np.float64)
    fw = sw_np[idx].astype(np.float64)
    nh = np.clip(np.round(fh).astype(np.int64), 0, H_lat - 1)
    nw = np.clip(np.round(fw).astype(np.int64), 0, W_lat - 1)
    flat = nh * W_lat + nw
    tile_used = np.zeros(H_lat * W_lat, dtype=bool)
    final_h = nh.copy()
    final_w = nw.copy()
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    contested = counts[inv] > 1
    np.add.at(tile_used, flat[~contested], True)
    cidx = np.where(contested)[0]
    if cidx.size:
        d2 = (fh[cidx] - nh[cidx]) ** 2 + (fw[cidx] - nw[cidx]) ** 2
        order = cidx[np.argsort(d2)]
        for k in order:
            hf, wf = fh[k], fw[k]
            cands = []
            for hh in (int(np.floor(hf)), int(np.ceil(hf))):
                for ww in (int(np.floor(wf)), int(np.ceil(wf))):
                    hh = max(0, min(H_lat - 1, hh))
                    ww = max(0, min(W_lat - 1, ww))
                    cands.append(((hf - hh) ** 2 + (wf - ww) ** 2, hh, ww))
            cands.sort()
            for _dist, hh, ww in cands:
                ft = hh * W_lat + ww
                if not tile_used[ft]:
                    tile_used[ft] = True
                    final_h[k] = hh
                    final_w[k] = ww
                    break
    sh_out = sh_np.copy()
    sw_out = sw_np.copy()
    sh_out[idx] = final_h.astype(sh_np.dtype)
    sw_out[idx] = final_w.astype(sw_np.dtype)
    return src_lat, torch.from_numpy(sh_out), torch.from_numpy(sw_out)


def _photometric_refine(
    rev: np.ndarray,
    frame_by_lat: Dict[int, np.ndarray],
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
) -> np.ndarray:
    iters = int(cfg.get("photo_refine_iters", 1))
    if iters <= 0:
        return rev
    cross_frame = bool(cfg.get("cross_frame_refine", False))
    rev = rev.copy()
    used_lats = np.unique(rev[..., 0].astype(int))
    used_lats = used_lats[used_lats >= 0]
    src_tiles: Dict[int, np.ndarray] = {}
    for sl in used_lats:
        sl = int(sl)
        if sl not in frame_by_lat:
            continue
        sf = np.asarray(frame_by_lat[sl])
        if sf.shape[:2] != (H_lat * latent_stride, W_lat * latent_stride):
            continue
        src_tiles[sl] = (
            sf.reshape(H_lat, latent_stride, W_lat, latent_stride, 3)
            .transpose(0, 2, 1, 3, 4)
            .astype(np.int16)
        )
    if not src_tiles:
        return rev
    for _ in range(iters):
        cur = np.zeros((H_lat, W_lat, latent_stride, latent_stride, 3), dtype=np.int16)
        valid = np.zeros((H_lat, W_lat), dtype=bool)
        for h in range(H_lat):
            for w in range(W_lat):
                sl = int(rev[h, w, 0])
                if sl < 0 or sl not in src_tiles:
                    continue
                sh = int(np.clip(round(float(rev[h, w, 1])), 0, H_lat - 1))
                sw = int(np.clip(round(float(rev[h, w, 2])), 0, W_lat - 1))
                cur[h, w] = src_tiles[sl][sh, sw]
                valid[h, w] = True
        new_rev = rev.copy()
        for h in range(H_lat):
            for w in range(W_lat):
                if not valid[h, w]:
                    continue
                sl = int(rev[h, w, 0])
                sh_c = int(np.clip(round(float(rev[h, w, 1])), 0, H_lat - 1))
                sw_c = int(np.clip(round(float(rev[h, w, 2])), 0, W_lat - 1))
                refs = []
                cross_nb = []
                for (di, dj), code in (
                    ((-1, 0), 0),
                    ((1, 0), 1),
                    ((0, -1), 2),
                    ((0, 1), 3),
                ):
                    hh, ww = h + di, w + dj
                    if 0 <= hh < H_lat and 0 <= ww < W_lat and valid[hh, ww]:
                        nb_lat = int(rev[hh, ww, 0])
                        nb_h = int(np.clip(round(float(rev[hh, ww, 1])), 0, H_lat - 1))
                        nb_w = int(np.clip(round(float(rev[hh, ww, 2])), 0, W_lat - 1))
                        if code == 0:
                            refs.append((code, cur[hh, ww, -1, :, :]))
                        elif code == 1:
                            refs.append((code, cur[hh, ww, 0, :, :]))
                        elif code == 2:
                            refs.append((code, cur[hh, ww, :, -1, :]))
                        else:
                            refs.append((code, cur[hh, ww, :, 0, :]))
                        if cross_frame and nb_lat != sl and nb_lat in src_tiles:
                            cross_nb.append((code, nb_lat, nb_h, nb_w))
                if not refs:
                    continue
                cands = []
                for hh in range(max(0, sh_c - 1), min(H_lat - 1, sh_c + 1) + 1):
                    for ww in range(max(0, sw_c - 1), min(W_lat - 1, sw_c + 1) + 1):
                        cands.append((sl, hh, ww))
                for code, nb_lat, nb_h, nb_w in cross_nb:
                    if code == 0:
                        th, tw = nb_h + 1, nb_w
                    elif code == 1:
                        th, tw = nb_h - 1, nb_w
                    elif code == 2:
                        th, tw = nb_h, nb_w + 1
                    else:
                        th, tw = nb_h, nb_w - 1
                    if 0 <= th < H_lat and 0 <= tw < W_lat:
                        cands.append((nb_lat, th, tw))
                patches = np.stack([src_tiles[lat][hh, ww] for lat, hh, ww in cands])
                bord = (
                    patches[:, 0, :, :],
                    patches[:, -1, :, :],
                    patches[:, :, 0, :],
                    patches[:, :, -1, :],
                )
                costs = np.zeros(len(cands), dtype=np.int64)
                for d, nb in refs:
                    costs += np.abs(bord[d] - nb[None]).reshape(len(cands), -1).sum(1)
                bi = int(np.argmin(costs))
                bl, bh, bw = cands[bi]
                if (bl, bh, bw) != (sl, sh_c, sw_c):
                    new_rev[h, w, 0] = bl
                    new_rev[h, w, 1] = bh
                    new_rev[h, w, 2] = bw
        rev = new_rev
    return rev


def _ordered_latent_groups(candidate_records: Sequence[Dict[str, Any]]):
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    order: List[int] = []
    for rec in candidate_records:
        lat = int(rec["latent_idx"])
        if lat not in grouped:
            grouped[lat] = []
            order.append(lat)
        grouped[lat].append(rec)
    if not order:
        return -1, []
    recent_lat = max(order)
    other = [(lat, grouped[lat]) for lat in order if lat != recent_lat]
    return recent_lat, other


def _source_to_query_revgrid(
    source_revgrid: np.ndarray,
    grouped_by_lat: Dict[int, List[Dict[str, Any]]],
    *,
    query_K: np.ndarray,
    query_w2c: np.ndarray,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
) -> np.ndarray:
    out = np.full((H_lat, W_lat, 2), -1.0, dtype=np.float32)
    valid = source_revgrid[..., 0] >= 0
    if not np.any(valid):
        return out

    flat_valid = valid.reshape(-1)
    source_flat = source_revgrid.reshape(-1, 3)
    out_flat = out.reshape(-1, 2)
    imgh, imgw = _image_hw(H_lat, W_lat, latent_stride)
    stride_f = float(latent_stride)

    for latent_idx, frames in grouped_by_lat.items():
        rec = frames[0]
        lat_mask = flat_valid & (source_flat[:, 0].astype(np.int64) == int(latent_idx))
        if not np.any(lat_mask):
            continue

        flat_idx = np.flatnonzero(lat_mask)
        sh = np.clip(np.rint(source_flat[flat_idx, 1]).astype(np.int64), 0, H_lat - 1)
        sw = np.clip(np.rint(source_flat[flat_idx, 2]).astype(np.int64), 0, W_lat - 1)
        src_h_pix = (sh.astype(np.float64) + 0.5) * stride_f
        src_w_pix = (sw.astype(np.float64) + 0.5) * stride_f

        depth = np.asarray(rec["depth"], dtype=np.float32)
        dh, dw = depth.shape
        d_h = np.clip(
            np.rint((src_h_pix / float(imgh)) * dh - 0.5).astype(np.int64), 0, dh - 1
        )
        d_w = np.clip(
            np.rint((src_w_pix / float(imgw)) * dw - 0.5).astype(np.int64), 0, dw - 1
        )
        pix_new_hw, zc, _xw, _xc = reproject_pixels_nd_to_nd(
            pix_hw=np.stack([src_h_pix, src_w_pix], axis=-1),
            depth_z=depth[d_h, d_w].astype(np.float64),
            K_old=np.asarray(rec["K"], dtype=np.float64),
            T_old_w2c=np.asarray(rec["w2c"], dtype=np.float64),
            K_new=np.asarray(query_K, dtype=np.float64),
            T_new_w2c=np.asarray(query_w2c, dtype=np.float64),
            eps=0.0,
        )
        zc_arr = np.asarray(zc)
        ok = np.isfinite(zc_arr) & (zc_arr > 0)
        if np.any(ok):
            dst_idx = flat_idx[ok]
            out_flat[dst_idx, 0] = (np.asarray(pix_new_hw)[ok, 1] / stride_f).astype(
                np.float32
            )
            out_flat[dst_idx, 1] = (np.asarray(pix_new_hw)[ok, 0] / stride_f).astype(
                np.float32
            )
    return out


def _fast_state_key(
    valid_records: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, ...]:
    record_fp = [_fast_record_key(rec) for rec in valid_records]
    return (
        "v405_gpu_fast_state",
        tuple(record_fp),
        int(H_lat),
        int(W_lat),
        int(latent_stride),
        str(device),
        str(dtype),
        str(cfg.get("depth_res")),
        str(cfg.get("depth_resize")),
        str(cfg.get("depth_filter")),
    )


def _fast_record_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    depth = np.asarray(rec["depth"])
    return (
        int(rec.get("frame_id", -1)),
        int(rec.get("latent_idx", -1)),
        int(depth.ctypes.data),
        tuple(depth.shape),
        str(depth.dtype),
    )


def _get_fast_per_call_state(
    valid_records: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
    device: torch.device,
    dtype: torch.dtype,
    candidate_geometry_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]],
) -> Dict[str, Any]:
    key = _fast_state_key(
        valid_records,
        cfg,
        H_lat=H_lat,
        W_lat=W_lat,
        latent_stride=latent_stride,
        device=device,
        dtype=dtype,
    )
    if candidate_geometry_cache is not None and key in candidate_geometry_cache:
        return candidate_geometry_cache[key]

    depths_t = _process_depth_batch_fast(
        [rec["depth"] for rec in valid_records],
        cfg,
        device,
        dtype,
        H_lat=H_lat,
        W_lat=W_lat,
    )
    if depths_t.ndim != 3:
        raise ValueError(
            "gpu_fast depth preprocessing produced an invalid depth batch."
        )
    dh, dw = int(depths_t.shape[1]), int(depths_t.shape[2])
    imgh, imgw = _image_hw(H_lat, W_lat, latent_stride)
    Ks_t = torch.as_tensor(
        np.stack([np.asarray(rec["K"], dtype=np.float32) for rec in valid_records]),
        dtype=dtype,
        device=device,
    )
    w2cs_t = torch.as_tensor(
        np.stack([np.asarray(rec["w2c"], dtype=np.float32) for rec in valid_records]),
        dtype=dtype,
        device=device,
    )
    state = {
        "depths_t": depths_t,
        "Ks_t": Ks_t,
        "Ks_d_t": _scale_K_batch(Ks_t, (imgh, imgw), (dh, dw)),
        "w2cs_t": w2cs_t,
        "lats_t": torch.as_tensor(
            [int(rec["latent_idx"]) for rec in valid_records],
            dtype=torch.long,
            device=device,
        ),
        "record_to_index": {
            _fast_record_key(rec): idx for idx, rec in enumerate(valid_records)
        },
        "dH": dh,
        "dW": dw,
    }
    if candidate_geometry_cache is not None:
        candidate_geometry_cache[key] = state
    return state


def _photo_refine_gpu_pre(
    rev: np.ndarray,
    frame_by_lat: Dict[int, np.ndarray],
    cfg: Dict[str, Any],
    *,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
) -> np.ndarray:
    # Keep the v40 semantics first; this wrapper is the fast backend hook for
    # replacing the CPU refine with a fully vectorized version later.
    return _photometric_refine(
        rev,
        frame_by_lat,
        cfg,
        H_lat=H_lat,
        W_lat=W_lat,
        latent_stride=latent_stride,
    )


def _gather_query_latent(
    source_revgrid: np.ndarray,
    memory_latents: torch.Tensor,
    *,
    H_lat: int,
    W_lat: int,
) -> torch.Tensor:
    C = int(memory_latents.shape[0])
    query_latent = torch.zeros(
        C,
        1,
        H_lat,
        W_lat,
        device=memory_latents.device,
        dtype=memory_latents.dtype,
    )
    valid = source_revgrid[..., 0] >= 0
    if not np.any(valid):
        return query_latent
    flat_idx_np = np.flatnonzero(valid.reshape(-1))
    src = source_revgrid.reshape(-1, 3)[flat_idx_np]
    lat_np = src[:, 0].astype(np.int64)
    sh_np = np.clip(np.rint(src[:, 1]).astype(np.int64), 0, H_lat - 1)
    sw_np = np.clip(np.rint(src[:, 2]).astype(np.int64), 0, W_lat - 1)
    flat_idx = torch.from_numpy(flat_idx_np).to(
        device=memory_latents.device, dtype=torch.long
    )
    lat_t = torch.from_numpy(lat_np).to(device=memory_latents.device, dtype=torch.long)
    sh_t = torch.from_numpy(sh_np).to(device=memory_latents.device, dtype=torch.long)
    sw_t = torch.from_numpy(sw_np).to(device=memory_latents.device, dtype=torch.long)
    query_latent[:, 0].reshape(C, -1)[:, flat_idx] = memory_latents[
        :, lat_t, sh_t, sw_t
    ]
    return query_latent


def _run_recent_first_smooth_gpu_fast(
    *,
    qK: np.ndarray,
    qw2c: np.ndarray,
    valid_records: List[Dict[str, Any]],
    grouped_by_lat: Dict[int, List[Dict[str, Any]]],
    recent_lat: int,
    other_groups: List[Tuple[int, List[Dict[str, Any]]]],
    recent_frame: Dict[str, Any],
    memory_latents: torch.Tensor,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
    final_cfg: Dict[str, Any],
    return_debug: bool,
    timing_scope=None,
    timing_prefix: str = "materialize.fuse_candidates.group.recent_first.engine",
    candidate_geometry_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
    source_valid_masks: Optional[torch.Tensor | np.ndarray] = None,
):
    scope = _resolve_timing_scope(timing_scope)
    device = _resolve_fast_device(final_cfg)
    dtype = _resolve_torch_dtype(final_cfg.get("dtype"), default=torch.float32)
    warp_dtype = _resolve_optional_torch_dtype(final_cfg.get("warp_dtype"))
    with _tf32_scope(bool(final_cfg.get("enable_tf32", False))):
        with scope(
            f"{timing_prefix}.gpu_fast.prepare_state",
            metadata={"record_count": len(valid_records), "device": str(device)},
        ):
            state = _get_fast_per_call_state(
                valid_records,
                final_cfg,
                H_lat=H_lat,
                W_lat=W_lat,
                latent_stride=latent_stride,
                device=device,
                dtype=dtype,
                candidate_geometry_cache=candidate_geometry_cache,
            )
            qK_t = torch.as_tensor(qK, dtype=dtype, device=device)
            qw2c_t = torch.as_tensor(qw2c, dtype=dtype, device=device)
            recent_idx = state["record_to_index"][_fast_record_key(recent_frame)]
            source_valid_masks_t = _coerce_source_valid_masks_torch(
                source_valid_masks,
                H_lat=H_lat,
                W_lat=W_lat,
                device=device,
            )

        with scope(
            f"{timing_prefix}.gpu_fast.phase1_recent_inverse_warp",
            metadata={"recent_lat": int(recent_lat)},
        ):
            src_lat, src_h, src_w, src_z = _recent_first_smooth_gpu_pre(
                int(recent_lat),
                state["depths_t"][recent_idx],
                state["Ks_d_t"][recent_idx],
                state["Ks_t"][recent_idx],
                state["w2cs_t"][recent_idx],
                qK_t,
                qw2c_t,
                H_lat=H_lat,
                W_lat=W_lat,
                latent_stride=latent_stride,
                device=device,
                dtype=dtype,
            )
            src_lat, src_h, src_w, src_z = _invalidate_masked_sources_torch(
                src_lat,
                src_h,
                src_w,
                src_z,
                source_valid_masks_t,
                H_lat=H_lat,
                W_lat=W_lat,
            )

        if final_cfg["dedupe_tiles"]:
            with scope(f"{timing_prefix}.gpu_fast.dedupe_recent_tiles"):
                src_lat, src_h, src_w = _dedupe_recent_tiles_gpu(
                    src_lat,
                    src_h,
                    src_w,
                    int(recent_lat),
                    H_lat=H_lat,
                    W_lat=W_lat,
                )
                src_lat, src_h, src_w, src_z = _invalidate_masked_sources_torch(
                    src_lat,
                    src_h,
                    src_w,
                    src_z,
                    source_valid_masks_t,
                    H_lat=H_lat,
                    W_lat=W_lat,
                )

        holes_after_recent = int((src_lat < 0).sum().item())
        recent_first_zbuffer = bool(final_cfg.get("recent_first_zbuffer", False))
        phase2_skipped = holes_after_recent == 0 and not recent_first_zbuffer
        if not phase2_skipped and other_groups:
            sel_indices = [
                state["record_to_index"][_fast_record_key(rec)]
                for _lat, frames in other_groups
                for rec in frames
                if _fast_record_key(rec) in state["record_to_index"]
            ]
            if sel_indices:
                with scope(
                    f"{timing_prefix}.gpu_fast.phase2_batch",
                    metadata={"frame_count": len(sel_indices)},
                ):
                    sel_idx = torch.as_tensor(
                        sel_indices, dtype=torch.long, device=device
                    )
                    depths = state["depths_t"].index_select(0, sel_idx)
                    Ks_d = state["Ks_d_t"].index_select(0, sel_idx)
                    w2cs = state["w2cs_t"].index_select(0, sel_idx)
                    lats = state["lats_t"].index_select(0, sel_idx)
                    qx, qy, zd, valid = _forward_warp_batch_fast(
                        depths,
                        Ks_d,
                        w2cs,
                        qK_t,
                        qw2c_t,
                        warp_dtype=warp_dtype,
                    )
                    GL, GH, GW, GZ, GV = _grid_per_cand_batch(
                        qx / float(latent_stride),
                        qy / float(latent_stride),
                        zd,
                        valid,
                        lats,
                        source_valid_masks_t,
                        H_lat=H_lat,
                        W_lat=W_lat,
                        device=device,
                        dtype=dtype,
                    )
                    UL, UH, UW, UZ, UV = _collapse_per_latent_fast(
                        GL,
                        GH,
                        GW,
                        GZ,
                        GV,
                        lats,
                        H_lat=H_lat,
                        W_lat=W_lat,
                        device=device,
                        dtype=dtype,
                    )
                    src_lat2, src_h2, src_w2, src_z2 = _stitch_gpu(
                        UL,
                        UH,
                        UW,
                        UZ,
                        UV,
                        final_cfg,
                        H_lat=H_lat,
                        W_lat=W_lat,
                        device=device,
                        dtype=dtype,
                    )
                    take = (src_lat2 >= 0) & (
                        (src_lat < 0)
                        | (
                            recent_first_zbuffer
                            & torch.isfinite(src_z2)
                            & (src_z2 < src_z)
                        )
                    )
                    src_lat = torch.where(take, src_lat2, src_lat)
                    src_h = torch.where(take, src_h2, src_h)
                    src_w = torch.where(take, src_w2, src_w)
                    src_z = torch.where(take, src_z2, src_z)

        if final_cfg["stitch_pinhole_close"] and not phase2_skipped:
            with scope(f"{timing_prefix}.gpu_fast.pinhole_close"):
                src_lat, src_h, src_w = _pinhole_close_gpu_fast(
                    src_lat, src_h, src_w, H_lat=H_lat, W_lat=W_lat
                )
                src_lat, src_h, src_w, src_z = _invalidate_masked_sources_torch(
                    src_lat,
                    src_h,
                    src_w,
                    src_z,
                    source_valid_masks_t,
                    H_lat=H_lat,
                    W_lat=W_lat,
                )

        with scope(f"{timing_prefix}.gpu_fast.build_source_revgrid"):
            source_revgrid = (
                torch.stack([src_lat.to(dtype), src_h, src_w], dim=-1)
                .reshape(H_lat, W_lat, 3)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

    if final_cfg["photo_refine"]:
        frame_by_lat: Dict[int, np.ndarray] = {}
        for lat, frames in grouped_by_lat.items():
            for rec in frames:
                frame = rec.get("frame")
                if frame is not None:
                    frame_by_lat[int(lat)] = np.asarray(frame)
                    break
        if frame_by_lat:
            with scope(
                f"{timing_prefix}.gpu_fast.photo_refine",
                metadata={"frame_count": len(frame_by_lat)},
            ):
                source_revgrid = _photo_refine_gpu_pre(
                    source_revgrid,
                    frame_by_lat,
                    final_cfg,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    latent_stride=latent_stride,
                )
                source_revgrid = _invalidate_masked_source_revgrid_np(
                    source_revgrid,
                    source_valid_masks,
                    H_lat=H_lat,
                    W_lat=W_lat,
                )
        else:
            warnings.warn(
                "recent_first_smooth photo_refine=True but no RGB frames were provided; skipping photo_refine.",
                RuntimeWarning,
                stacklevel=2,
            )

    with scope(f"{timing_prefix}.gpu_fast.latent_gather"):
        query_latent = _gather_query_latent(
            source_revgrid, memory_latents, H_lat=H_lat, W_lat=W_lat
        )
    with scope(
        f"{timing_prefix}.gpu_fast.source_to_query_revgrid",
        metadata={"return_debug": bool(return_debug)},
    ):
        revgrid = _source_to_query_revgrid(
            source_revgrid,
            grouped_by_lat,
            query_K=qK,
            query_w2c=qw2c,
            H_lat=H_lat,
            W_lat=W_lat,
            latent_stride=latent_stride,
        )
    if return_debug:
        return (
            query_latent,
            revgrid,
            {"source_revgrid": source_revgrid, "backend": "gpu_fast"},
        )
    return query_latent, revgrid


def run_recent_first_smooth(
    *,
    query_K: np.ndarray,
    query_w2c: np.ndarray,
    candidate_records: List[Dict[str, Any]],
    memory_latents: torch.Tensor,
    H_lat: int,
    W_lat: int,
    latent_stride: int,
    cfg: Optional[Dict[str, Any]] = None,
    return_debug: bool = False,
    timing_scope=None,
    timing_prefix: str = "materialize.fuse_candidates.group.recent_first.engine",
    candidate_geometry_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
    source_valid_masks: Optional[torch.Tensor | np.ndarray] = None,
):
    scope = _resolve_timing_scope(timing_scope)
    final_cfg = _final_cfg(cfg)
    backend = _resolve_backend(final_cfg)
    with scope(
        f"{timing_prefix}.filter_records",
        metadata={"candidate_count": len(candidate_records)},
    ):
        valid_records = [
            rec
            for rec in candidate_records
            if int(rec.get("frame_id", -1)) >= 0
            and 0 <= int(rec.get("latent_idx", -1)) < int(memory_latents.shape[1])
        ]
    warn_min = int(final_cfg.get("warn_min_candidates") or 0)
    if warn_min > 0 and len(valid_records) < warn_min:
        warnings.warn(
            f"recent_first_smooth received {len(valid_records)} candidates; "
            f"{warn_min} or more is recommended for consensus stitch.",
            RuntimeWarning,
            stacklevel=2,
        )

    C = int(memory_latents.shape[0])
    query_latent = torch.zeros(
        C,
        1,
        H_lat,
        W_lat,
        device=memory_latents.device,
        dtype=memory_latents.dtype,
    )
    empty_revgrid = np.full((H_lat, W_lat, 2), -1.0, dtype=np.float32)
    empty_source = np.full((H_lat, W_lat, 3), -1.0, dtype=np.float64)
    if not valid_records:
        if return_debug:
            return (
                query_latent,
                empty_revgrid,
                {
                    "source_revgrid": empty_source,
                    "backend": backend,
                },
            )
        return query_latent, empty_revgrid

    with scope(f"{timing_prefix}.group_records"):
        grouped_by_lat: Dict[int, List[Dict[str, Any]]] = {}
        for rec in valid_records:
            grouped_by_lat.setdefault(int(rec["latent_idx"]), []).append(rec)
        recent_lat, other_groups = _ordered_latent_groups(valid_records)
        recent_frame = max(
            grouped_by_lat[int(recent_lat)], key=lambda rec: int(rec["frame_id"])
        )
        qK = np.asarray(query_K, dtype=np.float64)
        qw2c = np.asarray(query_w2c, dtype=np.float64)

    if backend == "gpu_fast":
        return _run_recent_first_smooth_gpu_fast(
            qK=qK,
            qw2c=qw2c,
            valid_records=valid_records,
            grouped_by_lat=grouped_by_lat,
            recent_lat=int(recent_lat),
            other_groups=other_groups,
            recent_frame=recent_frame,
            memory_latents=memory_latents,
            H_lat=H_lat,
            W_lat=W_lat,
            latent_stride=latent_stride,
            final_cfg=final_cfg,
            return_debug=return_debug,
            timing_scope=timing_scope,
            timing_prefix=timing_prefix,
            candidate_geometry_cache=candidate_geometry_cache,
            source_valid_masks=source_valid_masks,
        )

    source_valid_masks_cpu = _coerce_source_valid_masks_torch(
        source_valid_masks,
        H_lat=H_lat,
        W_lat=W_lat,
        device=torch.device("cpu"),
    )
    with scope(
        f"{timing_prefix}.phase1_recent_inverse_warp",
        metadata={"recent_lat": int(recent_lat)},
    ):
        src_lat, src_h, src_w, src_z = _recent_first_smooth(
            int(recent_lat),
            recent_frame,
            qK,
            qw2c,
            final_cfg,
            H_lat=H_lat,
            W_lat=W_lat,
            latent_stride=latent_stride,
        )
        src_lat, src_h, src_w, src_z = _invalidate_masked_sources_torch(
            src_lat,
            src_h,
            src_w,
            src_z,
            source_valid_masks_cpu,
            H_lat=H_lat,
            W_lat=W_lat,
        )
    if final_cfg["dedupe_tiles"]:
        with scope(
            f"{timing_prefix}.dedupe_recent_tiles",
            metadata={"recent_lat": int(recent_lat)},
        ):
            src_lat, src_h, src_w = _dedupe_recent_tiles(
                src_lat,
                src_h,
                src_w,
                int(recent_lat),
                H_lat=H_lat,
                W_lat=W_lat,
            )
            src_lat, src_h, src_w, src_z = _invalidate_masked_sources_torch(
                src_lat,
                src_h,
                src_w,
                src_z,
                source_valid_masks_cpu,
                H_lat=H_lat,
                W_lat=W_lat,
            )

    with scope(f"{timing_prefix}.phase2_recent_fill_stats"):
        holes_after_recent = int((src_lat < 0).sum().item())
        fill_ratio_after_recent = 1.0 - (
            float(holes_after_recent) / float(max(H_lat * W_lat, 1))
        )

    recent_first_zbuffer = bool(final_cfg.get("recent_first_zbuffer", False))
    phase2_skipped = holes_after_recent == 0 and not recent_first_zbuffer
    if phase2_skipped:
        with scope(
            f"{timing_prefix}.phase2_skip",
            metadata={
                "holes_after_recent_first": holes_after_recent,
                "fill_ratio_after_recent_first": fill_ratio_after_recent,
                "other_group_count": len(other_groups),
                "reason": "recent_first_full",
            },
        ):
            pass
    else:
        other_grids = []
        with scope(
            f"{timing_prefix}.phase2_candidate_grids",
            metadata={
                "other_group_count": len(other_groups),
                "holes_after_recent_first": holes_after_recent,
                "fill_ratio_after_recent_first": fill_ratio_after_recent,
            },
        ):
            for lat, frames in other_groups:
                with scope(
                    f"{timing_prefix}.phase2_candidate_grid",
                    metadata={"latent_idx": int(lat), "frame_count": len(frames)},
                ):
                    other_grids.append(
                        _cand_grid(
                            lat,
                            frames,
                            qK,
                            qw2c,
                            final_cfg,
                            source_valid_masks_cpu,
                            H_lat=H_lat,
                            W_lat=W_lat,
                            latent_stride=latent_stride,
                            candidate_geometry_cache=candidate_geometry_cache,
                            timing_scope=timing_scope,
                            timing_prefix=f"{timing_prefix}.phase2_candidate_grid",
                        )
                    )
        if other_grids:
            with scope(
                f"{timing_prefix}.phase2_stitch",
                metadata={
                    "grid_count": len(other_grids),
                    "holes_after_recent_first": holes_after_recent,
                },
            ):
                rev2_np, z2_np = _stitch(
                    other_grids, final_cfg, H_lat=H_lat, W_lat=W_lat
                )
                rev2 = torch.from_numpy(rev2_np).reshape(-1, 3)
                z2 = torch.from_numpy(z2_np).reshape(-1)
            with scope(f"{timing_prefix}.phase2_fill_holes"):
                sl2 = rev2[:, 0].long()
                take = (sl2 >= 0) & (
                    (src_lat < 0)
                    | (
                        recent_first_zbuffer
                        & torch.isfinite(z2)
                        & (z2 < src_z)
                    )
                )
                src_lat[take] = sl2[take]
                src_h[take] = rev2[:, 1][take]
                src_w[take] = rev2[:, 2][take]
                src_z[take] = z2[take]

    if final_cfg["stitch_pinhole_close"] and not phase2_skipped:
        with scope(f"{timing_prefix}.pinhole_close"):
            src_lat, src_h, src_w = _pinhole_close(
                src_lat, src_h, src_w, H_lat=H_lat, W_lat=W_lat
            )
            src_lat, src_h, src_w, src_z = _invalidate_masked_sources_torch(
                src_lat,
                src_h,
                src_w,
                src_z,
                source_valid_masks_cpu,
                H_lat=H_lat,
                W_lat=W_lat,
            )

    with scope(f"{timing_prefix}.build_source_revgrid"):
        source_revgrid = (
            torch.stack([src_lat.double(), src_h, src_w], dim=-1)
            .reshape(H_lat, W_lat, 3)
            .numpy()
        )

    if final_cfg["photo_refine"]:
        with scope(f"{timing_prefix}.photo_refine_prepare"):
            frame_by_lat: Dict[int, np.ndarray] = {}
            for lat, frames in grouped_by_lat.items():
                for rec in frames:
                    frame = rec.get("frame")
                    if frame is not None:
                        frame_by_lat[int(lat)] = np.asarray(frame)
                        break
        if frame_by_lat:
            with scope(
                f"{timing_prefix}.photo_refine",
                metadata={"frame_count": len(frame_by_lat)},
            ):
                source_revgrid = _photometric_refine(
                    source_revgrid,
                    frame_by_lat,
                    final_cfg,
                    H_lat=H_lat,
                    W_lat=W_lat,
                    latent_stride=latent_stride,
                )
                source_revgrid = _invalidate_masked_source_revgrid_np(
                    source_revgrid,
                    source_valid_masks_cpu,
                    H_lat=H_lat,
                    W_lat=W_lat,
                )
        else:
            warnings.warn(
                "recent_first_smooth photo_refine=True but no RGB frames were provided; skipping photo_refine.",
                RuntimeWarning,
                stacklevel=2,
            )

    with scope(f"{timing_prefix}.latent_gather"):
        valid = source_revgrid[..., 0] >= 0
        if np.any(valid):
            flat_idx_np = np.flatnonzero(valid.reshape(-1))
            src = source_revgrid.reshape(-1, 3)[flat_idx_np]
            lat_np = src[:, 0].astype(np.int64)
            sh_np = np.clip(np.rint(src[:, 1]).astype(np.int64), 0, H_lat - 1)
            sw_np = np.clip(np.rint(src[:, 2]).astype(np.int64), 0, W_lat - 1)
            flat_idx = torch.from_numpy(flat_idx_np).to(
                device=memory_latents.device, dtype=torch.long
            )
            lat_t = torch.from_numpy(lat_np).to(
                device=memory_latents.device, dtype=torch.long
            )
            sh_t = torch.from_numpy(sh_np).to(
                device=memory_latents.device, dtype=torch.long
            )
            sw_t = torch.from_numpy(sw_np).to(
                device=memory_latents.device, dtype=torch.long
            )
            query_latent[:, 0].reshape(C, -1)[:, flat_idx] = memory_latents[
                :, lat_t, sh_t, sw_t
            ]

    with scope(
        f"{timing_prefix}.source_to_query_revgrid",
        metadata={"return_debug": bool(return_debug)},
    ):
        revgrid = _source_to_query_revgrid(
            source_revgrid,
            grouped_by_lat,
            query_K=qK,
            query_w2c=qw2c,
            H_lat=H_lat,
            W_lat=W_lat,
            latent_stride=latent_stride,
        )
    if return_debug:
        return (
            query_latent,
            revgrid,
            {
                "source_revgrid": source_revgrid,
                "backend": "legacy",
            },
        )
    return query_latent, revgrid
