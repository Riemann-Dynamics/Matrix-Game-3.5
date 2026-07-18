"""Postprocess dynamic-object mask records.

This module is intentionally independent of Ultralytics and training classes.
It operates on plain dict records with fields like ``frame_idx``, ``track_id``,
``x1/y1/x2/y2``, ``class_id``, ``class_name`` and either an in-memory ``mask``
array or a COCO ``mask_rle``.

The same functions can be used by offline dataset labeling and online
validation/inference so temporal smoothing stays train/infer consistent.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _bbox_from_record(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    vals = [_optional_float(row.get(k)) for k in ("x1", "y1", "x2", "y2")]
    if any(v is None for v in vals):
        return None
    x1, y1, x2, y2 = [float(v) for v in vals]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def encode_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("pycocotools is required to encode mask RLE.") from exc

    encoded = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in encoded["size"]], "counts": counts}


def decode_mask_rle(mask_rle: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(mask_rle, dict):
        return None
    if "counts" not in mask_rle or "size" not in mask_rle:
        return None
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("pycocotools is required to decode mask RLE.") from exc

    rle = dict(mask_rle)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("ascii")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.asarray(mask, dtype=bool)


def bbox_to_mask(
    bbox_xyxy: Sequence[float],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x0_i = max(0, min(int(np.floor(x1)), int(width) - 1))
    y0_i = max(0, min(int(np.floor(y1)), int(height) - 1))
    x1_i = max(x0_i + 1, min(int(np.ceil(x2)), int(width)))
    y1_i = max(y0_i + 1, min(int(np.ceil(y2)), int(height)))
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[y0_i:y1_i, x0_i:x1_i] = True
    return mask


def mask_from_record(
    row: dict[str, Any],
    *,
    height: int,
    width: int,
    use_bbox_fallback: bool = True,
) -> np.ndarray | None:
    mask = row.get("mask")
    if mask is not None:
        arr = np.asarray(mask, dtype=bool)
    else:
        arr = decode_mask_rle(row.get("mask_rle"))
    if arr is not None:
        if arr.shape[:2] != (int(height), int(width)):
            import cv2

            arr = cv2.resize(
                arr.astype(np.uint8),
                (int(width), int(height)),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        return arr.astype(bool, copy=False)

    if not use_bbox_fallback:
        return None
    bbox = _bbox_from_record(row)
    if bbox is None:
        return None
    return bbox_to_mask(bbox, height=int(height), width=int(width))


def apply_temporal_dilation_to_frame_masks(
    masks: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """OR each frame mask into neighboring frames within ``radius``."""
    radius = int(radius)
    masks = np.asarray(masks, dtype=bool)
    if radius <= 0 or masks.shape[0] == 0 or not masks.any():
        return masks.copy()

    out = masks.copy()
    T = int(masks.shape[0])
    for offset in range(1, radius + 1):
        out[offset:] |= masks[: T - offset]
        out[: T - offset] |= masks[offset:]
    return out


def records_to_frame_masks(
    records: Iterable[dict[str, Any]],
    *,
    frame_count: int,
    height: int,
    width: int,
    temporal_dilate_radius: int = 0,
    use_bbox_fallback: bool = True,
) -> np.ndarray:
    """Union detection records into ``(T,H,W)`` frame masks."""
    frame_count = int(frame_count)
    masks = np.zeros((frame_count, int(height), int(width)), dtype=bool)
    if frame_count <= 0:
        return masks

    for row in records:
        frame_idx = _optional_int(row.get("frame_idx"))
        if frame_idx is None or frame_idx < 0 or frame_idx >= frame_count:
            continue
        one = mask_from_record(
            row,
            height=int(height),
            width=int(width),
            use_bbox_fallback=use_bbox_fallback,
        )
        if one is not None:
            masks[int(frame_idx)] |= one

    return apply_temporal_dilation_to_frame_masks(
        masks,
        radius=int(temporal_dilate_radius),
    )


def records_to_window_frame_masks(
    records: Iterable[dict[str, Any]],
    *,
    frame_start: int,
    frame_end: int,
    height: int,
    width: int,
    temporal_dilate_radius: int = 0,
    use_bbox_fallback: bool = True,
) -> np.ndarray:
    """Union records into masks for ``[frame_start, frame_end)`` only.

    This is equivalent to ``records_to_frame_masks(...)[frame_start:frame_end]``
    but avoids allocating a full-video ``(T,H,W)`` bool array.  Records outside
    the requested window are still considered when temporal dilation would copy
    them into the window.
    """
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    if frame_end <= frame_start:
        return np.zeros((0, int(height), int(width)), dtype=bool)

    radius = max(0, int(temporal_dilate_radius))
    padded_start = max(0, frame_start - radius)
    padded_end = frame_end + radius
    local_count = int(padded_end - padded_start)
    local_masks = np.zeros((local_count, int(height), int(width)), dtype=bool)

    for row in records:
        frame_idx = _optional_int(row.get("frame_idx"))
        if frame_idx is None or frame_idx < padded_start or frame_idx >= padded_end:
            continue
        one = mask_from_record(
            row,
            height=int(height),
            width=int(width),
            use_bbox_fallback=use_bbox_fallback,
        )
        if one is not None:
            local_masks[int(frame_idx) - padded_start] |= one

    local_masks = apply_temporal_dilation_to_frame_masks(local_masks, radius=radius)
    out_start = frame_start - padded_start
    out_end = out_start + (frame_end - frame_start)
    return local_masks[out_start:out_end]


def image_mask_to_latent_mask(mask: np.ndarray, H_lat: int, W_lat: int) -> np.ndarray:
    """Downsample an image-space bool mask with any-hit semantics.

    A latent cell is marked dynamic if *any* covered training-resolution pixel is
    dynamic. This is intentionally conservative for mosaic filtering: touching
    a dynamic object invalidates the corresponding source latent cell.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}.")
    H, W = mask.shape
    H_lat = int(H_lat)
    W_lat = int(W_lat)
    if H_lat <= 0 or W_lat <= 0:
        raise ValueError(f"Bad latent size: H_lat={H_lat}, W_lat={W_lat}.")
    if H == H_lat and W == W_lat:
        return mask.copy()

    y_edges = np.linspace(0, H, H_lat + 1)
    x_edges = np.linspace(0, W, W_lat + 1)
    out = np.zeros((H_lat, W_lat), dtype=bool)
    for yi in range(H_lat):
        y0 = int(np.floor(y_edges[yi]))
        y1 = int(np.ceil(y_edges[yi + 1]))
        y0 = max(0, min(y0, H))
        y1 = max(y0 + 1, min(y1, H))
        for xi in range(W_lat):
            x0 = int(np.floor(x_edges[xi]))
            x1 = int(np.ceil(x_edges[xi + 1]))
            x0 = max(0, min(x0, W))
            x1 = max(x0 + 1, min(x1, W))
            out[yi, xi] = bool(mask[y0:y1, x0:x1].any())
    return out


def dilate_bool_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a bool mask by ``radius`` cells in the mask's own grid."""
    radius = int(radius)
    arr = np.asarray(mask, dtype=bool)
    if radius <= 0 or not arr.any():
        return arr.copy()
    import cv2

    k = 2 * radius + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(arr.astype(np.uint8), kernel, iterations=1) > 0


def image_masks_to_latent_masks(
    masks: np.ndarray,
    *,
    H_lat: int,
    W_lat: int,
    dilate_latent: int = 0,
) -> np.ndarray:
    """Convert ``(T,H,W)`` image masks to latent-grid dynamic masks."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks as (T,H,W), got {masks.shape}.")
    out = [
        dilate_bool_mask(
            image_mask_to_latent_mask(mask, int(H_lat), int(W_lat)),
            int(dilate_latent),
        )
        for mask in masks
    ]
    if not out:
        return np.zeros((0, int(H_lat), int(W_lat)), dtype=bool)
    return np.stack(out, axis=0).astype(bool, copy=False)


def _record_rank(row: dict[str, Any]) -> tuple[float, float]:
    return (
        float(row.get("conf", 0.0) or 0.0),
        float(row.get("mask_area", row.get("box_area", 0.0)) or 0.0),
    )


def _best_record_per_track_frame(
    rows: Iterable[dict[str, Any]],
) -> dict[int, dict[int, dict[str, Any]]]:
    by_track_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        track_id = _optional_int(row.get("track_id"))
        frame_idx = _optional_int(row.get("frame_idx"))
        if track_id is None or frame_idx is None:
            continue
        if _bbox_from_record(row) is None:
            continue
        existing = by_track_frame[track_id].get(frame_idx)
        if existing is None or _record_rank(row) > _record_rank(existing):
            by_track_frame[track_id][frame_idx] = row
    return by_track_frame


def _interpolate_bbox(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    return tuple((1.0 - alpha) * a + alpha * b for a, b in zip(left, right))  # type: ignore[return-value]


def make_track_gap_fill_records(
    records: Sequence[dict[str, Any]],
    *,
    frame_count: int | None,
    height: int,
    width: int,
    max_gap: int,
    encode_rle: bool = True,
) -> list[dict[str, Any]]:
    """Fill short same-track gaps with interpolated bbox masks.

    If a track is observed at frames ``t0`` and ``t1`` and the missing span is
    ``<= max_gap``, each missing frame receives a conservative rectangular mask
    from the linearly interpolated bbox. Long detector failures are left
    untouched so train/inference still see real YOLO failure modes.
    """
    max_gap = int(max_gap)
    if max_gap <= 0:
        return []

    by_track_frame = _best_record_per_track_frame(records)
    filled: list[dict[str, Any]] = []
    for track_id, frame_rows in sorted(by_track_frame.items()):
        frames = sorted(frame_rows)
        for left_frame, right_frame in zip(frames, frames[1:]):
            missing = int(right_frame) - int(left_frame) - 1
            if missing <= 0 or missing > max_gap:
                continue
            left_row = frame_rows[left_frame]
            right_row = frame_rows[right_frame]
            left_bbox = _bbox_from_record(left_row)
            right_bbox = _bbox_from_record(right_row)
            if left_bbox is None or right_bbox is None:
                continue

            for frame_idx in range(int(left_frame) + 1, int(right_frame)):
                if frame_count is not None and not (0 <= frame_idx < int(frame_count)):
                    continue
                alpha = (frame_idx - int(left_frame)) / float(int(right_frame) - int(left_frame))
                x1, y1, x2, y2 = _interpolate_bbox(left_bbox, right_bbox, alpha)
                mask = bbox_to_mask((x1, y1, x2, y2), height=int(height), width=int(width))
                template = left_row if alpha <= 0.5 else right_row
                row = copy.deepcopy(template)
                row.pop("mask", None)
                row["frame_idx"] = int(frame_idx)
                row["det_idx"] = -1
                row["track_id"] = int(track_id)
                row["x1"] = float(x1)
                row["y1"] = float(y1)
                row["x2"] = float(x2)
                row["y2"] = float(y2)
                row["box_area"] = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                row["mask_area"] = int(mask.sum())
                row["has_mask"] = True
                row["postprocess_source"] = "track_gap_fill_bbox"
                row["source_left_frame_idx"] = int(left_frame)
                row["source_right_frame_idx"] = int(right_frame)
                row["source_gap"] = int(missing)
                if encode_rle:
                    row["mask_rle"] = encode_mask_rle(mask)
                else:
                    row["mask"] = mask
                    row.pop("mask_rle", None)
                filled.append(row)
    return filled


def postprocess_dynamic_mask_records(
    records: Sequence[dict[str, Any]],
    *,
    frame_count: int | None,
    height: int,
    width: int,
    track_gap_fill_max_gap: int = 0,
    encode_gap_fill_rle: bool = True,
) -> list[dict[str, Any]]:
    """Return original records plus short-gap fill records."""
    out = []
    for row in records:
        copied = copy.deepcopy(row)
        copied.setdefault("postprocess_source", "yolo")
        out.append(copied)
    out.extend(
        make_track_gap_fill_records(
            out,
            frame_count=frame_count,
            height=int(height),
            width=int(width),
            max_gap=int(track_gap_fill_max_gap),
            encode_rle=bool(encode_gap_fill_rle),
        )
    )
    out.sort(
        key=lambda row: (
            _optional_int(row.get("frame_idx")) if _optional_int(row.get("frame_idx")) is not None else -1,
            _optional_int(row.get("track_id")) if _optional_int(row.get("track_id")) is not None else 10**12,
            int(row.get("det_idx", 0) or 0),
            str(row.get("postprocess_source", "")),
        )
    )
    return out
