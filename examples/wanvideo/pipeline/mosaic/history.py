from .common import *


def _candidate_groups_to_lists(candidate_groups):
    if candidate_groups is None:
        return []
    if torch.is_tensor(candidate_groups):
        candidate_groups = candidate_groups.detach().cpu().tolist()
    elif isinstance(candidate_groups, np.ndarray):
        candidate_groups = candidate_groups.tolist()
    groups = []
    for group in candidate_groups:
        if group is None:
            groups.append([])
        elif torch.is_tensor(group):
            groups.append([int(fid) for fid in group.detach().cpu().tolist()])
        elif isinstance(group, np.ndarray):
            groups.append([int(fid) for fid in group.tolist()])
        elif isinstance(group, (list, tuple)):
            groups.append([int(fid) for fid in group])
        else:
            groups.append([int(group)])
    return groups


def _compute_mosaic_frame_indices(noisy_count, interval, device=None):
    noisy_count = int(noisy_count)
    interval = int(interval)
    if interval < 1:
        raise ValueError(f"mosaic_interval must be >= 1, got {interval}.")
    assert noisy_count % interval == 0, (
        f"noisy latent length {noisy_count} must be divisible by "
        f"mosaic_interval {interval}."
    )
    return torch.arange(
        interval - 1, noisy_count, interval, dtype=torch.long, device=device
    )


def _mosaic_kept_frame_ranges(noisy_count, mosaic_indices, *, total_frame_count):
    """Return (mosaic_kept_ranges, mosaic_dropped_ranges) over the
    ``[clean + noisy]`` VAE-decoded video timeline -- i.e. which frame
    positions in the *mosaic* preview video carry real content vs. which
    ones should be painted black because ``mosaic_interval`` dropped the
    corresponding mosaic slot.

    The noisy sequence itself is always full length; the timeline here is
    only the canvas on which we visualize the mosaic side-by-side with
    noisy so the two videos can align.

    The WAN 3D VAE maps ``T_lat`` latents (``T_lat = 1 + N``) to
    ``4 * (T_lat - 1) + 1 = 4 * N + 1`` frames: ``frame[0]`` decodes the
    clean anchor and the i-th noisy latent decodes to
    ``frame[1 + 4*i : 1 + 4*(i+1)]``. ``mosaic_indices`` lists which of
    those i-positions mosaic_interval *keeps mosaic content for*; for the
    remaining positions, the mosaic was dropped and the visualization
    should show black frames. The clean anchor frame is always kept on
    the mosaic side too (it's the shared anchor between both rows).

    Returns ``None, None`` if the supplied ``total_frame_count`` doesn't
    match the expected ``4 * noisy_count + 1`` mapping (e.g. a future VAE
    with a different stride): callers fall back to skipping mask painting
    rather than mis-blacking-out frames.
    """
    expected = 4 * int(noisy_count) + 1
    if int(total_frame_count) != expected:
        return None, None
    mosaic_kept_ranges = [(0, 1)]
    for raw_i in mosaic_indices:
        i = int(raw_i)
        if i < 0 or i >= int(noisy_count):
            continue
        start = 1 + 4 * i
        end = start + 4
        mosaic_kept_ranges.append((start, min(end, expected)))
    mosaic_kept_ranges.sort()
    mosaic_dropped_ranges = []
    cursor = 0
    for start, end in mosaic_kept_ranges:
        if start > cursor:
            mosaic_dropped_ranges.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < expected:
        mosaic_dropped_ranges.append((cursor, expected))
    return mosaic_kept_ranges, mosaic_dropped_ranges


def _select_validation_clean_latents(
    history_latents,
    *,
    section_idx,
    requested_history_count,
):
    requested_history_count = max(1, int(requested_history_count or 1))
    clean_count = 1 if int(section_idx) == 0 else requested_history_count
    clean_count = min(clean_count, int(history_latents.shape[2]))
    return history_latents[:, :, -clean_count:]


def _validation_pool_raw_frame_to_latent_index(frame_id):
    frame_id = int(frame_id)
    if frame_id < 0:
        raise ValueError(f"frame_id must be >= 0, got {frame_id}.")
    if frame_id == 0:
        return 0
    return (frame_id + 3) // 4


def _validation_index_frame_values(frame_values, indices):
    if frame_values is None:
        raise ValueError("frame_values is required.")
    indices = [int(idx) for idx in indices]
    if torch.is_tensor(frame_values):
        if not indices:
            return frame_values[:0].clone()
        index_tensor = torch.as_tensor(
            indices, dtype=torch.long, device=frame_values.device
        )
        return frame_values.index_select(0, index_tensor).clone()

    arr = np.asarray(frame_values)
    if not indices:
        return arr[:0].copy()
    return arr[np.asarray(indices, dtype=np.int64)].copy()


def _select_validation_clean_latents_from_candidate_pool(
    history_latents,
    candidate_frame_ids,
    *,
    clean_history_extr,
    clean_history_intr,
    requested_history_count,
    selection_mode="recent",
    target_extrinsics=None,
):
    requested_history_count = max(1, int(requested_history_count or 1))
    if history_latents.ndim != 5:
        raise ValueError(
            "history_latents must have shape (B,C,T,H,W), got "
            f"{tuple(history_latents.shape)}."
        )
    history_count = int(history_latents.shape[2])
    if history_count <= 0:
        raise ValueError("history_latents must contain at least one latent.")

    clean_count = min(requested_history_count, history_count)
    latest_latent_idx = history_count - 1
    required_pool_count = max(0, clean_count - 1)
    required_camera_frames = history_count * 4
    for name, values in (
        ("clean_history_extr", clean_history_extr),
        ("clean_history_intr", clean_history_intr),
    ):
        if values is None:
            raise ValueError(f"{name} is required.")
        value_count = int(values.shape[0])
        if value_count < required_camera_frames:
            raise ValueError(
                f"{name} has {value_count} frames but history_latents has "
                f"{history_count} latent frames, requiring at least "
                f"{required_camera_frames} camera frames."
            )

    pool_latent_indices = set()
    for group in _candidate_groups_to_lists(candidate_frame_ids):
        for raw_frame_id in group:
            raw_frame_id = int(raw_frame_id)
            if raw_frame_id < 0:
                continue
            latent_idx = _validation_pool_raw_frame_to_latent_index(raw_frame_id)
            if 0 <= latent_idx < history_count and latent_idx != latest_latent_idx:
                pool_latent_indices.add(int(latent_idx))

    pool_sorted = sorted(pool_latent_indices)
    selector_pool = pool_sorted
    if str(selection_mode or "").strip().lower() in _FULL_HISTORY_POOL_MODES:
        selector_pool = list(range(0, latest_latent_idx))
    selected_history = _select_context_pool_indices(
        selector_pool,
        required_pool_count,
        mode=selection_mode,
        latest_latent_idx=latest_latent_idx,
        clean_history_extr=clean_history_extr,
        target_extrinsics=target_extrinsics,
        preferred_latent_indices=pool_sorted,
    )
    selected_set = set(selected_history)
    fallback_desc = []
    if len(selected_history) < required_pool_count:
        for latent_idx in range(latest_latent_idx - 1, -1, -1):
            if latent_idx in selected_set:
                continue
            fallback_desc.append(int(latent_idx))
            selected_history.append(int(latent_idx))
            selected_set.add(int(latent_idx))
            if len(selected_history) >= required_pool_count:
                break
    selected_history = sorted(selected_history)
    selected_latent_indices = selected_history + [latest_latent_idx]
    latent_index_tensor = torch.as_tensor(
        selected_latent_indices, dtype=torch.long, device=history_latents.device
    )
    clean_latents = history_latents.index_select(2, latent_index_tensor).clone()

    camera_frame_indices = []
    for latent_idx in selected_latent_indices:
        frame_start = int(latent_idx) * 4
        camera_frame_indices.extend(range(frame_start, frame_start + 4))
    clean_extrinsic = _validation_index_frame_values(
        clean_history_extr, camera_frame_indices
    )
    clean_intrinsic = _validation_index_frame_values(
        clean_history_intr, camera_frame_indices
    )
    return {
        "clean_latents": clean_latents,
        "clean_extrinsic": clean_extrinsic,
        "clean_intrinsic": clean_intrinsic,
        "selected_latent_indices": [int(idx) for idx in selected_latent_indices],
        "pool_latent_indices": pool_sorted,
        "fallback_latent_indices": fallback_desc,
        "camera_frame_indices": [int(idx) for idx in camera_frame_indices],
    }


def _validation_w2c_camera_centers(w2c_values):
    if w2c_values is None:
        return None
    if torch.is_tensor(w2c_values):
        arr = w2c_values.detach().cpu().numpy()
    else:
        arr = np.asarray(w2c_values)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (4, 4):
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        return None
    R = arr[:, :3, :3]
    t = arr[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)


def _validation_latent_pose_representatives(clean_history_extr, latent_indices):
    if clean_history_extr is None:
        return None, None
    if torch.is_tensor(clean_history_extr):
        arr = clean_history_extr.detach().cpu().numpy()
    else:
        arr = np.asarray(clean_history_extr)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        return None, None
    frame_indices = []
    for latent_idx in latent_indices:
        frame_idx = int(latent_idx) * 4 + 1
        frame_idx = min(max(frame_idx, 0), arr.shape[0] - 1)
        frame_indices.append(frame_idx)
    reps = arr[np.asarray(frame_indices, dtype=np.int64)]
    return reps, _validation_w2c_camera_centers(reps)


def _validation_rotation_distance_matrix(candidate_reps, target_extrinsics):
    if candidate_reps is None or target_extrinsics is None:
        return None
    cand = np.asarray(candidate_reps, dtype=np.float32)
    targ = np.asarray(target_extrinsics, dtype=np.float32)
    if targ.ndim == 2 and targ.shape == (4, 4):
        targ = targ[None, ...]
    if cand.ndim != 3 or targ.ndim != 3:
        return None
    Rc = cand[:, :3, :3]
    Rt = targ[:, :3, :3]
    out = np.zeros((cand.shape[0], targ.shape[0]), dtype=np.float32)
    for target_i, Rq in enumerate(Rt):
        rel = np.einsum("ij,njk->nik", Rq, np.transpose(Rc, (0, 2, 1)))
        trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
        cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        out[:, target_i] = np.arccos(cos).astype(np.float32)
    return out


def _select_validation_segment_coverage_indices(
    pool,
    count,
    *,
    latest_latent_idx,
    clean_history_extr=None,
    target_extrinsics=None,
    preferred_latent_indices=None,
    recent_exclusion_latents=1,
):
    if count <= 0 or not pool:
        return []

    count = int(count)
    latest_latent_idx = None if latest_latent_idx is None else int(latest_latent_idx)
    recent_exclusion_latents = max(0, int(recent_exclusion_latents))
    pool = sorted(
        {
            int(idx)
            for idx in pool
            if latest_latent_idx is None
            or int(idx) < latest_latent_idx - recent_exclusion_latents
        }
    )
    if not pool:
        return []
    if len(pool) <= count:
        return list(pool)

    preferred = {int(idx) for idx in (preferred_latent_indices or [])}
    target_pose_reps = None
    if target_extrinsics is not None:
        target_pose_reps = np.asarray(target_extrinsics, dtype=np.float32)
        if target_pose_reps.ndim == 2 and target_pose_reps.shape == (4, 4):
            target_pose_reps = target_pose_reps[None, ...]
        if target_pose_reps.ndim != 3 or target_pose_reps.shape[1:] != (4, 4):
            target_pose_reps = None
    target_centers = _validation_w2c_camera_centers(target_pose_reps)
    if target_centers is None and latest_latent_idx is not None:
        target_pose_reps, latest_centers = _validation_latent_pose_representatives(
            clean_history_extr, [int(latest_latent_idx)]
        )
        target_centers = latest_centers
    pool_reps, pool_centers = _validation_latent_pose_representatives(
        clean_history_extr, pool
    )
    if target_centers is None or pool_centers is None:
        return _select_context_pool_indices(
            pool,
            count,
            mode="oldest_spread",
            latest_latent_idx=latest_latent_idx,
        )
    if target_centers.shape[0] > count:
        positions = np.linspace(0, target_centers.shape[0] - 1, count)
        sampled = np.asarray([int(round(pos)) for pos in positions], dtype=np.int64)
        target_centers = target_centers[sampled]
        if target_pose_reps is not None and target_pose_reps.shape[0] > int(
            sampled.max()
        ):
            target_pose_reps = target_pose_reps[sampled]

    selected = []
    selected_set = set()
    trans_dists = np.linalg.norm(
        pool_centers[:, None, :] - target_centers[None, :, :], axis=-1
    )
    rot_dists = _validation_rotation_distance_matrix(pool_reps, target_pose_reps)
    if rot_dists is None or rot_dists.shape != trans_dists.shape:
        rot_dists = np.zeros_like(trans_dists, dtype=np.float32)
    trans_scale = max(float(np.nanmedian(trans_dists)), 1e-6)
    dists = (trans_dists / trans_scale) + rot_dists
    per_target_order = []
    min_gap = min(3, max(1, len(pool) // max(1, count)))
    for target_i in range(dists.shape[1]):
        ranked = sorted(
            range(len(pool)),
            key=lambda i: (
                float(dists[i, target_i])
                - (0.02 if int(pool[i]) in preferred else 0.0),
                abs(int(pool[i]) - int(latest_latent_idx or pool[-1])),
            ),
        )
        per_target_order.append(ranked)

    def _far_enough(idx, gap):
        if gap <= 1:
            return True
        return all(abs(int(idx) - int(old)) >= int(gap) for old in selected)

    def _try_append_from_ranked(ranked, gap):
        for pool_i in ranked:
            idx = int(pool[pool_i])
            if idx in selected_set:
                continue
            if not _far_enough(idx, gap):
                continue
            selected.append(idx)
            selected_set.add(idx)
            return True
        return False

    target_cycle = list(range(len(per_target_order)))
    while len(selected) < count and target_cycle:
        progressed = False
        for target_i in target_cycle:
            progressed = (
                _try_append_from_ranked(per_target_order[target_i], min_gap)
                or progressed
            )
            if len(selected) >= count:
                break
        if not progressed:
            break

    if len(selected) < count:
        best_dist = dists.min(axis=1)
        ranked_fill = sorted(
            range(len(pool)),
            key=lambda i: (
                int(pool[i]) in selected_set,
                float(best_dist[i]) - (0.02 if int(pool[i]) in preferred else 0.0),
                abs(int(pool[i]) - int(latest_latent_idx or pool[-1])),
            ),
        )
        gap = min_gap
        gap_levels = []
        while gap > 1:
            gap_levels.append(int(gap))
            next_gap = max(1, int(gap) // 2)
            if next_gap == gap:
                break
            gap = next_gap
        gap_levels.append(1)
        for gap in gap_levels:
            while len(selected) < count and _try_append_from_ranked(ranked_fill, gap):
                pass
            if len(selected) >= count:
                break
    return sorted(selected[:count])


# Canonical name + accepted aliases for the "one covering context latent per
# group of 3 generating latents" selector. The count is NOT taken from
# train_context_count_max / validation_context_count -- it is forced to
# n_gen // 3 by the caller (see module_materialize / validation), and this
# selector picks exactly that many context latents, one per consecutive group.
_PER3_CONTEXT_MODES = {"per_3latentframe", "per-3latentframe", "per3latentframe"}
# Coverage-style modes that need per-frame extrinsics and rank the WHOLE history
# (not just the candidate pool) by camera-pose distance to the generating
# segment. segment_coverage spreads picks across the segment; per_3latentframe
# pins one pick per group-of-3. Both share the latent-index pose lookup path.
_FULL_HISTORY_POOL_MODES = {
    "segment_coverage",
    "segment-coverage",
} | _PER3_CONTEXT_MODES


def _select_per3_coverage_indices(
    pool,
    count,
    *,
    latest_latent_idx,
    clean_history_extr=None,
    target_extrinsics=None,
    preferred_latent_indices=None,
    recent_exclusion_latents=1,
    return_groups=False,
):
    """Pick (up to) ``count`` context latents, one per consecutive group of 3
    generating latents, each chosen to best COVER its group's target camera
    poses -- a localized ``segment_coverage`` run per group.

    ``count`` is authoritative (callers derive it as ``n_gen // 3``). The target
    trajectory in ``target_extrinsics`` is split into ``count`` contiguous
    chunks (one chunk per group), so the routine is robust whether the targets
    carry one pose per generating latent or a coarser/finer sampling. Picks are
    DISTINCT; when the pool runs out the routine returns fewer than ``count`` and
    the caller's recent-block fallback fills the remainder (matching the
    ``segment_coverage`` degradation contract). Returns sorted latent indices.

    When ``return_groups`` is True, returns ``(sorted_indices, groups)`` where
    ``groups[i]`` is the generating-latent GROUP index (0-based, group g covers
    generating latents [3g, 3g+3)) that selected ``sorted_indices[i]`` -- used by
    the dataset preview to label which gen-latents each context covers. ``None``
    marks a pose-less / non-coverage pick.
    """

    def _ret(idx_list, grp_list):
        return (list(idx_list), list(grp_list)) if return_groups else list(idx_list)

    if count <= 0 or not pool:
        return _ret([], [])
    count = int(count)
    latest_latent_idx = None if latest_latent_idx is None else int(latest_latent_idx)
    recent_exclusion_latents = max(0, int(recent_exclusion_latents))
    pool = sorted(
        {
            int(idx)
            for idx in pool
            if latest_latent_idx is None
            or int(idx) < latest_latent_idx - recent_exclusion_latents
        }
    )
    if not pool:
        return _ret([], [])

    preferred = {int(idx) for idx in (preferred_latent_indices or [])}

    target_pose_reps = None
    if target_extrinsics is not None:
        target_pose_reps = np.asarray(target_extrinsics, dtype=np.float32)
        if target_pose_reps.ndim == 2 and target_pose_reps.shape == (4, 4):
            target_pose_reps = target_pose_reps[None, ...]
        if target_pose_reps.ndim != 3 or target_pose_reps.shape[1:] != (4, 4):
            target_pose_reps = None
    target_centers = _validation_w2c_camera_centers(target_pose_reps)
    pool_reps, pool_centers = _validation_latent_pose_representatives(
        clean_history_extr, pool
    )
    if (
        target_centers is None
        or pool_centers is None
        or target_centers.shape[0] <= 0
    ):
        # No usable poses -> behave like oldest_spread for the same count so the
        # mode never silently drops to recent-only. No per-group coverage was
        # computed, so the group labels are unknown (None).
        fallback_idx = _select_context_pool_indices(
            pool,
            count,
            mode="oldest_spread",
            latest_latent_idx=latest_latent_idx,
        )
        return _ret(fallback_idx, [None] * len(fallback_idx))

    # Combined pool->target distance: translation normalised by its median plus
    # the rotation geodesic, identical weighting to segment_coverage.
    trans_dists = np.linalg.norm(
        pool_centers[:, None, :] - target_centers[None, :, :], axis=-1
    )
    rot_dists = _validation_rotation_distance_matrix(pool_reps, target_pose_reps)
    if rot_dists is None or rot_dists.shape != trans_dists.shape:
        rot_dists = np.zeros_like(trans_dists, dtype=np.float32)
    trans_scale = max(float(np.nanmedian(trans_dists)), 1e-6)
    dists = (trans_dists / trans_scale) + rot_dists  # (len(pool), n_target)
    # A single degenerate/missing w2c pose (NaN translation or rotation) would
    # otherwise poison a whole group column: the group mean becomes NaN for
    # every candidate and the NaN-keyed sort returns pool order (oldest-first),
    # silently defeating coverage. Replace non-finite distances with a sentinel
    # ABOVE every finite distance so bad poses are de-selected, not preferred,
    # while coverage for the good poses is preserved.
    if not np.isfinite(dists).all():
        finite = dists[np.isfinite(dists)]
        sentinel = (float(finite.max()) + 1.0) if finite.size else 1.0
        dists = np.where(np.isfinite(dists), dists, sentinel)

    n_target = int(target_centers.shape[0])
    bounds = np.linspace(0, n_target, count + 1).astype(np.int64)
    anchor_ref = int(latest_latent_idx if latest_latent_idx is not None else pool[-1])

    selected = []
    selected_groups = []
    selected_set = set()
    for group_idx in range(count):
        if len(selected_set) >= len(pool):
            break
        lo = int(bounds[group_idx])
        hi = max(lo + 1, int(bounds[group_idx + 1]))
        group_cost = dists[:, lo:hi].mean(axis=1)
        ranked = sorted(
            range(len(pool)),
            key=lambda i: (
                float(group_cost[i]) - (0.02 if int(pool[i]) in preferred else 0.0),
                abs(int(pool[i]) - anchor_ref),
            ),
        )
        for i in ranked:
            idx = int(pool[i])
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_groups.append(int(group_idx))
            selected_set.add(idx)
            break
    order = sorted(range(len(selected)), key=lambda i: selected[i])
    return _ret(
        [selected[i] for i in order],
        [selected_groups[i] for i in order],
    )


# Shared context-pool selector used by BOTH validation pool-history and the
# training clean-history block picker (mode in recent / oldest_spread /
# segment_coverage / per_3latentframe). The ``validation`` lineage is kept only
# in the docstring; train passes block-start indices converted to latent-index
# space.
def _select_context_pool_indices(
    pool_latent_indices,
    count,
    *,
    mode="recent",
    latest_latent_idx=None,
    clean_history_extr=None,
    target_extrinsics=None,
    preferred_latent_indices=None,
):
    pool = sorted({int(idx) for idx in pool_latent_indices})
    count = max(0, int(count))
    if count <= 0 or not pool:
        return []
    mode = str(mode or "recent").strip().lower()
    if mode in {"recent", "recent_window"}:
        return pool[-count:]
    if mode in {"segment_coverage", "segment-coverage"}:
        return _select_validation_segment_coverage_indices(
            pool,
            count,
            latest_latent_idx=latest_latent_idx,
            clean_history_extr=clean_history_extr,
            target_extrinsics=target_extrinsics,
            preferred_latent_indices=preferred_latent_indices,
        )
    if mode in _PER3_CONTEXT_MODES:
        return _select_per3_coverage_indices(
            pool,
            count,
            latest_latent_idx=latest_latent_idx,
            clean_history_extr=clean_history_extr,
            target_extrinsics=target_extrinsics,
            preferred_latent_indices=preferred_latent_indices,
        )
    if mode not in {"oldest_spread", "oldest-spread"}:
        raise ValueError(
            "validation_clean_latent_pool_history_selection_mode must be "
            "'recent', 'oldest_spread', 'segment_coverage', or "
            f"'per_3latentframe', got {mode!r}."
        )
    if len(pool) <= count:
        return pool

    oldest_anchor = pool[0]
    if latest_latent_idx is None:
        right_anchor = pool[-1]
        divisor = max(1, count - 1)
    else:
        right_anchor = max(int(latest_latent_idx), oldest_anchor + 1)
        divisor = count
    step = (right_anchor - oldest_anchor) / float(divisor)
    targets = [oldest_anchor + step * idx for idx in range(count)]

    selected = []
    selected_set = set()
    for target in targets:
        candidate = min(
            (idx for idx in pool if idx not in selected_set),
            key=lambda idx: (abs(idx - target), idx),
        )
        selected.append(int(candidate))
        selected_set.add(int(candidate))
    return sorted(selected)


def _validation_latent_index_to_raw_frame_ids(latent_idx):
    latent_idx = int(latent_idx)
    if latent_idx < 0:
        raise ValueError(f"latent_idx must be >= 0, got {latent_idx}.")
    if latent_idx == 0:
        return [0]
    start = 1 + 4 * (latent_idx - 1)
    return list(range(start, start + 4))


def _validation_pool_clean_frame_ids_from_selection(pool_clean_info):
    if not pool_clean_info:
        return []
    selected = [int(idx) for idx in pool_clean_info.get("selected_latent_indices", [])]
    if len(selected) <= 1:
        return []
    frame_ids = []
    for latent_idx in selected[:-1]:
        latent_frame_ids = _validation_latent_index_to_raw_frame_ids(latent_idx)
        frame_ids.append(latent_frame_ids[min(1, len(latent_frame_ids) - 1)])
    return frame_ids


def _is_validation_clean_latent_pool_history_enabled(args):
    return bool(getattr(args, "validation_clean_latent_pool_history", False))


def _validation_clean_prefix_frame_count(clean_latent_count):
    clean_latent_count = max(1, int(clean_latent_count))
    return 1 + 4 * (clean_latent_count - 1)


def _validation_recent_frame_window(frame_values, *, required_count):
    if frame_values is None:
        raise ValueError("frame_values is required.")
    required_count = max(1, int(required_count))
    if torch.is_tensor(frame_values):
        if frame_values.shape[0] <= 0:
            raise ValueError("frame_values must contain at least one frame.")
        if frame_values.shape[0] >= required_count:
            return frame_values[-required_count:].clone()
        pad = frame_values[:1].expand(
            required_count - frame_values.shape[0],
            *frame_values.shape[1:],
        )
        return torch.cat([pad, frame_values], dim=0).clone()

    arr = np.asarray(frame_values)
    if arr.shape[0] <= 0:
        raise ValueError("frame_values must contain at least one frame.")
    if arr.shape[0] >= required_count:
        return arr[-required_count:].copy()
    pad = np.repeat(arr[:1], required_count - arr.shape[0], axis=0)
    return np.concatenate([pad, arr], axis=0).copy()


def _validation_clean_prefix_decode_drop_count(clean_latent_count):
    return _validation_clean_prefix_frame_count(clean_latent_count)


def _build_validation_pool_rope_time_indices(
    *,
    clean_latent_count,
    noisy_latent_count,
    mosaic_frame_indices=None,
    history_bias=1,
    selected_latent_indices=None,
    use_original_latent_indices=False,
):
    clean_latent_count = max(0, int(clean_latent_count))
    noisy_latent_count = max(0, int(noisy_latent_count))
    if use_original_latent_indices:
        if selected_latent_indices is None:
            raise ValueError(
                "selected_latent_indices is required when "
                "use_original_latent_indices=True."
            )
        clean_times = [int(idx) for idx in selected_latent_indices]
        if len(clean_times) != clean_latent_count:
            raise ValueError(
                "selected_latent_indices length must match clean_latent_count, got "
                f"{len(clean_times)} vs {clean_latent_count}."
            )
        if any(idx < 0 for idx in clean_times):
            raise ValueError("selected_latent_indices must be non-negative.")
        latest_clean_time = max(clean_times) if clean_times else -1
    else:
        history_bias = int(history_bias)
        if history_bias < 1:
            raise ValueError(
                "validation_clean_latent_pool_history_bias must be >= 1, got "
                f"{history_bias}."
            )

        if clean_latent_count <= 0:
            clean_times = []
            latest_clean_time = -1
        elif clean_latent_count == 1:
            clean_times = [0]
            latest_clean_time = 0
        else:
            clean_times = list(range(clean_latent_count - 1))
            latest_clean_time = clean_times[-1] + history_bias
            clean_times.append(latest_clean_time)

    noisy_start = latest_clean_time + 1
    noisy_times = list(range(noisy_start, noisy_start + noisy_latent_count))
    if mosaic_frame_indices is None:
        mosaic_times = []
    else:
        if torch.is_tensor(mosaic_frame_indices):
            mosaic_offsets = mosaic_frame_indices.detach().cpu().reshape(-1).tolist()
        else:
            mosaic_offsets = np.asarray(mosaic_frame_indices).reshape(-1).tolist()
        mosaic_times = [noisy_start + int(offset) for offset in mosaic_offsets]
    return torch.as_tensor(clean_times + mosaic_times + noisy_times, dtype=torch.long)


def _parse_train_clean_history_ac_ratio(spec):
    """Parse ``"<single>:<block>"`` into the probability of the FIRST side.

    The ratio string is shared with the dataset's
    ``--vae_clean_context_mode_ratio`` and keeps the dataset semantics:
    first weight = single-frame clean anchor (dataset mode A), second
    weight = four-frame block anchor (dataset mode B).
    """
    try:
        a_str, c_str = str(spec).split(":")
        a_weight = float(a_str.strip())
        c_weight = float(c_str.strip())
    except ValueError as exc:
        raise ValueError(
            "vae_clean_context_mode_ratio must be '<single>:<block>' when "
            "train_clean_latent_history_count_max is enabled."
        ) from exc
    if a_weight < 0 or c_weight < 0:
        raise ValueError(
            "vae_clean_context_mode_ratio weights must be non-negative, got "
            f"{spec!r}."
        )
    total = a_weight + c_weight
    if total <= 0:
        raise ValueError(
            "vae_clean_context_mode_ratio must contain at least one positive "
            f"weight, got {spec!r}."
        )
    return a_weight / total


TRAIN_CONTEXT_SELECTION_CHOICES = (
    "oldest_spread",
    "recent",
    "segment_coverage",
    "per_3latentframe",
)


def _validate_block_anchor_frames(frames):
    """Validate the block-anchor encode-group size (shared single source).

    Used by config parse-time resolution AND module construction so the
    two cannot drift. ``None`` is NOT accepted here -- callers decide the
    default; an explicit 0 must fail loudly, not fall back.
    """
    frames = int(frames)
    if frames < 5 or (frames - 1) % 4 != 0:
        raise ValueError(
            f"train_block_anchor_frames must be 4k+1 and >= 5, got {frames}."
        )
    return frames


def _validate_train_context_selection(value):
    value = str(value or "oldest_spread")
    if value not in TRAIN_CONTEXT_SELECTION_CHOICES:
        raise ValueError(
            "train_context_selection must be one of "
            f"{TRAIN_CONTEXT_SELECTION_CHOICES}, got {value!r}."
        )
    return value


def _required_context_blocks_min(*, anchor_block_prob, block_anchor_frames, context_count_max):
    """G lower-bound reservation shared by datasets.py and the trainer.

    Block anchors need ``G >= 4*depth + 1`` frames of warm-up before the
    window; ``context_count_max`` context blocks need that many distinct
    4-aligned starts in ``[4, G-8]`` (hence ``+1``). Returns 0 when both
    axes are off -- the dataset then keeps the legacy free G sampling
    (arbitrary start, no 4-alignment), which is what "everything off"
    comparison configs rely on.
    """
    frames = _validate_block_anchor_frames(block_anchor_frames)
    depth = (frames - 1) // 4
    depth_req = depth if float(anchor_block_prob) > 0 else 0
    ctx_req = int(context_count_max) + 1 if int(context_count_max) > 0 else 0
    return max(depth_req, ctx_req)


def _sample_train_anchor_context_plan(
    *,
    anchor_block_prob,
    block_anchor_frames,
    context_count_max,
    base_seed,
    step_index,
):
    """Per-step anchor+context plan shared by every DDP rank (v2 sampler).

    Replaces the old A/B/C modes with two ORTHOGONAL axes:

    * ``anchor_form`` -- "block" with prob ``anchor_block_prob``, else
      "single". The anchor is ALWAYS part of the joint clean+noisy encode
      (single -> dataset slot 0; block -> the ``[G-4..G-1]`` latent at
      chain depth ``(block_anchor_frames-1)//4``, warm-up frames prepended
      by the trainer). Isolated anchor re-encodes are banned.
    * ``context_count`` -- uniform over ``[0, context_count_max]`` discrete
      past blocks appended as extra clean latents with their own relative
      RoPE times and cameras; ALWAYS encoded separately (5-frame windows),
      since they have no contiguous pixel bridge to the generating window.

    Old-mode mapping: C=(single,0), A=(block,0), B=(block,n>0). The new
    combination (single, n>0) -- which mirrors the inference pool-history
    contract (generated single-latent anchor + N context latents) -- was
    previously unexpressible.

    ``bucket`` is the metrics/loss-bucket label:
    single / block / single_ctx / block_ctx. v2 plans carry ONLY the new
    vocabulary -- readers keep a legacy C/A/B fallback solely for debug
    dicts recorded by pre-v2 runs.
    """
    frames = _validate_block_anchor_frames(block_anchor_frames)
    anchor_block_prob = min(1.0, max(0.0, float(anchor_block_prob)))
    context_count_max = max(0, int(context_count_max or 0))
    rng = random.Random((int(base_seed) << 20) ^ (int(step_index) << 2) ^ 0x531C)
    anchor_form = "block" if rng.random() < anchor_block_prob else "single"
    context_count = rng.randint(0, context_count_max) if context_count_max else 0
    return {
        "enabled": True,
        "plan_sampler": "anchor_context_v2",
        "anchor_form": anchor_form,
        "anchor_block_prob": anchor_block_prob,
        "block_anchor_frames": frames,
        "anchor_depth": (frames - 1) // 4 if anchor_form == "block" else 0,
        "context_count": int(context_count),
        "bucket": anchor_form + ("_ctx" if context_count else ""),
    }


def _train_clean_history_actual_latent_time(frame_start):
    frame_start = max(0, int(frame_start))
    return 1 + frame_start // 4


def _train_clean_history_block_start_from_candidate(frame_id):
    frame_id = int(frame_id)
    if frame_id < 0:
        return None
    return (frame_id // 4) * 4


# Set once when segment_coverage context selection is requested in training but
# no per-frame extrinsics are available, so the silent fallback to oldest_spread
# is surfaced exactly once instead of spamming a line every step.
_SEGMENT_COVERAGE_NO_EXTR_WARNED = False


def _select_train_clean_history_frame_blocks(
    *,
    candidate_frame_ids,
    generating_first_frame_idx,
    history_count,
    selection_mode="oldest_spread",
    clean_history_extr=None,
    target_extrinsics=None,
):
    history_count = max(1, int(history_count or 1))
    G = int(generating_first_frame_idx)
    if G < 1:
        raise ValueError(
            "generating_first_frame_idx must be positive for clean history, "
            f"got {G}."
        )

    latest_block_start = max(0, G - 4)
    required_pool_count = max(0, history_count - 1)
    pool_block_starts = set()
    for group in _candidate_groups_to_lists(candidate_frame_ids):
        for frame_id in group:
            block_start = _train_clean_history_block_start_from_candidate(frame_id)
            if block_start is None:
                continue
            if block_start < 4:
                continue
            if block_start + 4 > G:
                continue
            if block_start == latest_block_start:
                continue
            pool_block_starts.add(int(block_start))

    pool_sorted = sorted(pool_block_starts)
    mode = str(selection_mode or "oldest_spread").strip().lower()
    is_coverage_mode = mode in _FULL_HISTORY_POOL_MODES
    if is_coverage_mode and clean_history_extr is None and required_pool_count > 0:
        global _SEGMENT_COVERAGE_NO_EXTR_WARNED
        if not _SEGMENT_COVERAGE_NO_EXTR_WARNED:
            print(
                f"[mosaic] WARNING: train_context_selection={mode!r} "
                "but no per-frame extrinsics (w2c) were passed; context-block "
                "selection silently falls back to 'oldest_spread'. Check that "
                "fused_query_inputs['w2c'] is populated. (warned once)",
                flush=True,
            )
            _SEGMENT_COVERAGE_NO_EXTR_WARNED = True
    # Recent context blocks, latest-first, used as fallback fillers. Block starts
    # live in frame-id space (multiples of 4); the latest (G-4) is the anchor and
    # is excluded here.
    recent_blocks = list(range(latest_block_start - 4, 3, -4))
    if mode in _PER3_CONTEXT_MODES and clean_history_extr is not None:
        # TRAIN per_3 ONLY: lay out ONE context slot PER GROUP, in group order
        # (cg0, cg1, ... cg{required_pool_count-1}). Each slot holds group g's
        # best COVERAGE pick; if that group found no distinct pool latent, the
        # slot is filled with the next unused RECENT block instead -- so every
        # group is represented and labelable, and recent fills land at the groups
        # that could not be covered (not appended at the end). The order is NOT
        # time-sorted, which is safe: every per-block array below
        # (frame_blocks / encode_frame_windows / actual_latent_times) and the
        # downstream rope/prope are built per ``selected_block_starts`` entry, so
        # each slot's latent, RoPE time and PRoPE cameras stay 1:1 regardless of
        # ordering. Block starts -> latent space (bs//4) for the pose lookup,
        # back to frame space (*4) after.
        pool_latent = sorted(bs // 4 for bs in pool_sorted)
        cov_latents, cov_groups = _select_per3_coverage_indices(
            pool_latent,
            required_pool_count,
            latest_latent_idx=latest_block_start // 4,
            clean_history_extr=clean_history_extr,
            target_extrinsics=target_extrinsics,
            return_groups=True,
        )
        group_to_block = {
            int(g): int(li) * 4
            for li, g in zip(cov_latents, cov_groups)
            if g is not None
        }
        used = set(group_to_block.values())
        selected_pool = []
        context_group_indices = []
        context_is_recent = []
        fallback = []
        for g in range(required_pool_count):
            block_start = group_to_block.get(g)
            is_recent = False
            if block_start is None:
                block_start = next((c for c in recent_blocks if c not in used), None)
                if block_start is None:
                    # History exhausted (very short clip): no coverage pick AND
                    # no unused recent block left for this slot -- drop it.
                    continue
                is_recent = True
                fallback.append(int(block_start))
            used.add(int(block_start))
            selected_pool.append(int(block_start))
            context_group_indices.append(int(g))
            context_is_recent.append(bool(is_recent))
    else:
        # recent / oldest_spread / segment_coverage: pick, shortfall-append
        # recent blocks, then time-sort. No per-group correspondence.
        if is_coverage_mode and clean_history_extr is not None:
            # segment_coverage ranks pool blocks by camera-pose distance to the
            # generating segment; needs per-frame extrinsics. latent*4+1 maps to
            # the block's representative frame, so the latent-space lookup is
            # correct. oldest_spread / recent ignore the poses (else branch).
            pool_latent = sorted(bs // 4 for bs in pool_sorted)
            selected_latent = _select_context_pool_indices(
                pool_latent,
                required_pool_count,
                mode=mode,
                latest_latent_idx=latest_block_start // 4,
                clean_history_extr=clean_history_extr,
                target_extrinsics=target_extrinsics,
            )
            selected_pool = [int(li) * 4 for li in selected_latent]
        else:
            selected_pool = _select_context_pool_indices(
                pool_sorted,
                required_pool_count,
                mode=mode,
                latest_latent_idx=latest_block_start,
            )
        selected_set = set(selected_pool)
        fallback = []
        if len(selected_pool) < required_pool_count:
            for block_start in recent_blocks:
                block_start = int(block_start)
                if block_start in selected_set:
                    continue
                selected_pool.append(block_start)
                selected_set.add(block_start)
                fallback.append(block_start)
                if len(selected_pool) >= required_pool_count:
                    break
        selected_pool = sorted(selected_pool)
        context_group_indices = [None] * len(selected_pool)
        context_is_recent = [False] * len(selected_pool)

    selected_block_starts = selected_pool + [latest_block_start]
    frame_blocks = [
        [int(start + offset) for offset in range(4)] for start in selected_block_starts
    ]
    encode_frame_windows = [
        [max(0, int(start - 1 + offset)) for offset in range(5)]
        for start in selected_block_starts
    ]
    actual_latent_times = [
        _train_clean_history_actual_latent_time(start)
        for start in selected_block_starts
    ]
    return {
        "mode": "B" if history_count > 1 else "A",
        "selected_block_starts": [int(v) for v in selected_block_starts],
        "pool_block_starts": [int(v) for v in pool_sorted],
        "fallback_block_starts": [int(v) for v in fallback],
        "frame_blocks": frame_blocks,
        "encode_frame_windows": encode_frame_windows,
        "actual_latent_times": [int(v) for v in actual_latent_times],
        "context_group_indices": [
            None if g is None else int(g) for g in context_group_indices
        ],
        "context_is_recent": [bool(v) for v in context_is_recent],
    }


def _build_train_clean_history_rope_time_indices(
    *,
    clean_actual_latent_times,
    generating_first_frame_idx,
    noisy_latent_count,
    mosaic_frame_indices=None,
):
    clean_times = [int(v) for v in clean_actual_latent_times]
    if not clean_times:
        raise ValueError("clean_actual_latent_times must not be empty.")
    noisy_latent_count = max(0, int(noisy_latent_count))
    noisy_actual_start = _train_clean_history_actual_latent_time(
        int(generating_first_frame_idx)
    )
    noisy_times = [noisy_actual_start + i for i in range(noisy_latent_count)]

    if mosaic_frame_indices is None:
        mosaic_offsets = []
    elif torch.is_tensor(mosaic_frame_indices):
        mosaic_offsets = mosaic_frame_indices.detach().cpu().reshape(-1).tolist()
    else:
        mosaic_offsets = np.asarray(mosaic_frame_indices).reshape(-1).tolist()
    mosaic_offsets = [int(offset) for offset in mosaic_offsets]
    invalid_offsets = [
        int(offset)
        for offset in mosaic_offsets
        if int(offset) < 0 or int(offset) >= noisy_latent_count
    ]
    if invalid_offsets:
        raise ValueError(
            "mosaic_frame_indices must be within the noisy latent range "
            f"[0, {noisy_latent_count}); got {invalid_offsets}."
        )
    mosaic_times = [noisy_actual_start + int(offset) for offset in mosaic_offsets]

    bias = 1 - min(clean_times)
    mapped = [t + bias for t in clean_times + mosaic_times + noisy_times]
    if min(mapped) < 0:
        raise ValueError(
            "train clean-history RoPE produced a negative temporal index: " f"{mapped}."
        )
    return torch.as_tensor(mapped, dtype=torch.long)
