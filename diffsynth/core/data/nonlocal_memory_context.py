"""Temporal selection helpers for causal runtime memory and visual context."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


MEMORY_CONTEXT_SELECTION_POLICIES = ("legacy", "nonlocal_oldest")
DYNAMIC_CONTEXT_SELECTION_POLICIES = ("latest", "oldest")


def normalize_memory_context_selection_policy(value: Any) -> str:
    policy = str(value or "legacy").strip().lower().replace("-", "_")
    if policy not in MEMORY_CONTEXT_SELECTION_POLICIES:
        raise ValueError(
            "causal_memory_context_selection_policy must be one of "
            f"{MEMORY_CONTEXT_SELECTION_POLICIES}, got {value!r}."
        )
    return policy


def normalize_dynamic_context_selection_policy(value: Any) -> str:
    policy = str(value or "oldest").strip().lower().replace("-", "_")
    if policy not in DYNAMIC_CONTEXT_SELECTION_POLICIES:
        raise ValueError(
            "dynamic_context_selection must be one of "
            f"{DYNAMIC_CONTEXT_SELECTION_POLICIES}, got {value!r}."
        )
    return policy


def rolling_anchor_source_position(
    *,
    local_chunk_idx: int,
    frames_per_chunk: int,
    window_chunks: int,
    initial_anchor_source_position: int = 0,
) -> int:
    """Return the source-timeline position of the rolling boundary anchor."""

    local_chunk_idx = int(local_chunk_idx)
    frames_per_chunk = int(frames_per_chunk)
    window_chunks = int(window_chunks)
    initial_anchor_source_position = int(initial_anchor_source_position)
    if local_chunk_idx < 0:
        raise ValueError(f"local_chunk_idx must be >= 0, got {local_chunk_idx}.")
    if frames_per_chunk <= 0:
        raise ValueError(
            f"frames_per_chunk must be > 0, got {frames_per_chunk}."
        )
    if window_chunks <= 0:
        raise ValueError(f"window_chunks must be > 0, got {window_chunks}.")
    generated_before = local_chunk_idx * frames_per_chunk
    recent_generated_capacity = max(0, window_chunks - 1) * frames_per_chunk
    return initial_anchor_source_position + max(
        0, generated_before - recent_generated_capacity
    )


def strictly_nonlocal_prefix_count(
    source_positions: Sequence[int],
    *,
    max_source_position_exclusive: int,
    expected_count: int | None = None,
) -> int:
    """Count the chronological source prefix strictly before the local anchor."""

    positions = [int(value) for value in source_positions]
    if expected_count is not None and len(positions) != int(expected_count):
        raise RuntimeError(
            "runtime memory provenance length mismatch: "
            f"positions={len(positions)}, expected={int(expected_count)}."
        )
    if any(right < left for left, right in zip(positions, positions[1:])):
        raise RuntimeError(
            "runtime memory source positions must be monotonic nondecreasing, "
            f"got {positions}."
        )
    cutoff = int(max_source_position_exclusive)
    count = 0
    for position in positions:
        if position >= cutoff:
            break
        count += 1
    return count


def select_pose_near_oldest(
    candidates: Sequence[Mapping[str, Any]],
    *,
    target_extrinsics: Any,
    pose_score_fn: Callable[[Any, Any], float],
    pose_pool_size: int,
) -> Mapping[str, Any] | None:
    """Build a pose-nearest pool, then choose its oldest source-time entry."""

    if not candidates:
        return None
    if target_extrinsics is None:
        raise RuntimeError("oldest context selection requires target extrinsics.")
    pose_pool_size = max(1, int(pose_pool_size))
    scored = []
    for stable_index, entry in enumerate(candidates):
        if "source_timeline_position" not in entry:
            raise RuntimeError(
                "oldest context candidate lacks source_timeline_position provenance."
            )
        rep = entry.get("rep_extrinsic")
        if rep is None:
            raise RuntimeError(
                "oldest context candidate lacks representative extrinsics."
            )
        pose_score = float(pose_score_fn(rep, target_extrinsics))
        source_priority = (
            0 if entry.get("source") in {"generated", "gt_prefix"} else 1
        )
        scored.append(
            (
                pose_score,
                source_priority,
                -int(entry.get("position", 0)),
                stable_index,
                entry,
            )
        )
    near_pool = sorted(scored, key=lambda item: item[:4])[:pose_pool_size]
    return min(
        near_pool,
        key=lambda item: (
            int(item[4]["source_timeline_position"]),
            item[0],
            item[1],
            int(item[4].get("position", 0)),
            item[3],
        ),
    )[4]
