"""Few-step causal KV rollout for distilled inference.

Only generation state lives here: scheduler overrides, KV windows, dynamic
visual context, and optional generated-memory registration.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from diffsynth.inference.causal_schedule import (
    hiar_sde_corrupt_clean_latents,
    prepare_causal_dmd_eval_scheduler,
)
from diffsynth.pipelines.wan_video import (
    WAN_VIDEO_PROPE_CAMERA_UNIT,
    init_causal_kv_caches,
    model_fn_causal_kv,
)
from .causal_memory import (
    _CausalKVDynamicContextPool,
    _accelerator_zero_stage,
    _apply_causal_validation_camera_time_scale,
    _build_initial_causal_kv_dynamic_context_pool,
    _camera_array_for_frame_count,
    _decode_generated_chunk_from_prefix_latents,
    _generated_memory_chunk_has_future_consumer,
    _generated_memory_publish_interval_from_args,
    _generated_memory_query_kwargs_from_args,
    _is_da3_insufficient_non_sky_error,
    _maybe_gather_zero3_module,
    _read_validation_gt_depth_frames,
    _registered_memory_K_for_query,
    _validation_clean_frame_indices,
    _validation_dynamic_context_accepts_generated,
    _validation_dynamic_context_enabled_for_mode,
    _validation_noisy_frame_indices,
    _validation_register_depth_source,
    _validation_register_rgb_source,
)
from .cleanup import _release_cached_memory
from .video_io import (
    _decode_latents_to_numpy_frames,
    _encode_frames_per_frame,
    _read_video_gt_frames,
)

def _causal_kv_cache_copy(cache_list):
    """Return a shallow structural copy so write_cache cannot mutate inputs."""
    out = []
    for cache in cache_list or []:
        out.append(
            {
                "k": cache.get("k"),
                "v": cache.get("v"),
                "positions": list(cache.get("positions", cache.get("frames", []))),
                "frames": list(cache.get("frames", [])),
                "chunk_ids": list(
                    cache.get("chunk_ids", [-1] * len(cache.get("frames", [])))
                ),
            }
        )
    return out

def _causal_kv_concat_caches(left, right):
    """Concatenate two KV cache lists at frame granularity without mutation."""
    if not left:
        return _causal_kv_cache_copy(right)
    if not right:
        return _causal_kv_cache_copy(left)
    if len(left) != len(right):
        raise RuntimeError(
            f"KV cache block count mismatch: {len(left)} vs {len(right)}."
        )
    out = []
    for lc, rc in zip(left, right):
        lk, lv = lc.get("k"), lc.get("v")
        rk, rv = rc.get("k"), rc.get("v")
        if lk is None or int(lk.shape[1]) == 0:
            k, v = rk, rv
        elif rk is None or int(rk.shape[1]) == 0:
            k, v = lk, lv
        else:
            k = torch.cat([lk, rk], dim=1)
            v = torch.cat([lv, rv], dim=1)
        out.append(
            {
                "k": k,
                "v": v,
                "positions": list(lc.get("positions", lc.get("frames", [])))
                + list(rc.get("positions", rc.get("frames", []))),
                "frames": list(lc.get("frames", [])) + list(rc.get("frames", [])),
                "chunk_ids": list(
                    lc.get("chunk_ids", [-1] * len(lc.get("frames", [])))
                )
                + list(rc.get("chunk_ids", [-1] * len(rc.get("frames", [])))),
            }
        )
    return out

def _causal_kv_frame_count(cache_list):
    if not cache_list:
        return 0
    first = cache_list[0]
    return len(first.get("frames", []))

def _causal_kv_slice_cache_frames(cache_list, frame_indices, *, context):
    """Slice a KV cache list by frame indices and compact the token tensors."""
    frame_indices = [int(v) for v in frame_indices]
    if not cache_list:
        return []
    out = []
    for block_idx, cache in enumerate(cache_list):
        k, v = cache.get("k"), cache.get("v")
        frames = list(cache.get("frames", []))
        positions = list(cache.get("positions", frames))
        chunk_ids = list(cache.get("chunk_ids", [-1] * len(frames)))
        frame_count = len(frames)
        if frame_count != len(positions) or frame_count != len(chunk_ids):
            raise RuntimeError(
                f"{context}: cache metadata length mismatch at block={block_idx}: "
                f"frames={frame_count}, positions={len(positions)}, "
                f"chunk_ids={len(chunk_ids)}."
            )
        if k is None or v is None or int(k.shape[1]) == 0 or frame_count == 0:
            out.append({"k": k, "v": v, "positions": [], "frames": [], "chunk_ids": []})
            continue
        if int(k.shape[1]) % frame_count != 0:
            raise RuntimeError(
                f"{context}: KV token count {int(k.shape[1])} is not divisible "
                f"by frame_count={frame_count} at block={block_idx}."
            )
        tokens_per_frame = int(k.shape[1]) // frame_count
        token_indices = []
        for frame_idx in frame_indices:
            if frame_idx < 0 or frame_idx >= frame_count:
                raise RuntimeError(
                    f"{context}: frame index {frame_idx} out of range for "
                    f"frame_count={frame_count}."
                )
            start = frame_idx * tokens_per_frame
            token_indices.extend(range(start, start + tokens_per_frame))
        if token_indices:
            token_idx = torch.as_tensor(token_indices, device=k.device, dtype=torch.long)
            k_sel = k.index_select(1, token_idx).contiguous()
            v_sel = v.index_select(1, token_idx).contiguous()
        else:
            k_sel = k[:, :0].contiguous()
            v_sel = v[:, :0].contiguous()
        out.append(
            {
                "k": k_sel,
                "v": v_sel,
                "positions": [positions[idx] for idx in frame_indices],
                "frames": [frames[idx] for idx in frame_indices],
                "chunk_ids": [chunk_ids[idx] for idx in frame_indices],
            }
        )
    return out

def _causal_kv_tail_cache_frames(cache_list, frame_count, *, context):
    frame_count = int(frame_count)
    total = _causal_kv_frame_count(cache_list)
    if frame_count <= 0:
        return _causal_kv_slice_cache_frames(cache_list, [], context=context)
    if frame_count > total:
        raise RuntimeError(
            f"{context}: requested tail frame_count={frame_count}, cache has {total}."
        )
    return _causal_kv_slice_cache_frames(
        cache_list,
        range(total - frame_count, total),
        context=context,
    )

def _causal_kv_trim_rolling_window(
    cache_list,
    *,
    frames_per_chunk,
    window_chunks,
    fixed_initial_anchor=False,
):
    """Keep SF-style local cache: boundary frame + recent generated chunks.

    The cached positions are deliberately preserved. Sliding removes old KV
    frames from visibility; it does not renumber RoPE positions or re-forward
    old chunks. The fixed-anchor ablation instead keeps frame 0 (the original
    C0 KV) and evicts only generated-prefix frames.
    """
    frames_per_chunk = int(frames_per_chunk)
    window_chunks = int(window_chunks)
    max_frames = 1 + max(0, window_chunks - 1) * frames_per_chunk
    out = _causal_kv_cache_copy(cache_list)
    while _causal_kv_frame_count(out) > max_frames:
        total = _causal_kv_frame_count(out)
        if bool(fixed_initial_anchor):
            tail_frames = max(0, max_frames - 1)
            keep = [0]
            if tail_frames:
                keep.extend(range(total - tail_frames, total))
            out = _causal_kv_slice_cache_frames(
                out,
                keep,
                context="causal-kv fixed-initial-anchor trim",
            )
            continue
        if total <= frames_per_chunk:
            return _causal_kv_tail_cache_frames(
                out,
                max_frames,
                context="causal-kv trim fallback",
            )
        # Layout before trim is [anchor, chunk_a, chunk_{a+1}, ...].
        # The next anchor is the last latent frame of the oldest full chunk.
        keep = [frames_per_chunk] + list(range(1 + frames_per_chunk, total))
        out = _causal_kv_slice_cache_frames(
            out,
            keep,
            context="causal-kv rolling-window trim",
        )
    return out

def _causal_kv_trim_rolling_latents(
    latents,
    *,
    frames_per_chunk,
    window_chunks,
    fixed_initial_anchor=False,
):
    """Mirror the rolling-cache trim for latent provenance used by HiAR eval."""
    if latents is None:
        return None
    if latents.ndim != 5:
        raise ValueError(
            "rolling latents must be [B,C,T,H,W], got " f"{tuple(latents.shape)}."
        )
    frames_per_chunk = int(frames_per_chunk)
    window_chunks = int(window_chunks)
    max_frames = 1 + max(0, window_chunks - 1) * frames_per_chunk
    out = latents
    while int(out.shape[2]) > max_frames:
        total = int(out.shape[2])
        if bool(fixed_initial_anchor):
            tail_frames = max(0, max_frames - 1)
            keep = [0]
            if tail_frames:
                keep.extend(range(total - tail_frames, total))
            keep_idx = torch.as_tensor(keep, device=out.device, dtype=torch.long)
            out = out.index_select(2, keep_idx).contiguous()
            continue
        if total <= frames_per_chunk:
            out = out[:, :, -max_frames:].contiguous()
            break
        keep = [frames_per_chunk] + list(range(1 + frames_per_chunk, total))
        keep_idx = torch.as_tensor(keep, device=out.device, dtype=torch.long)
        out = out.index_select(2, keep_idx).contiguous()
    return out

def _causal_dmd_validation_transition(
    scheduler,
    model_output,
    timestep,
    sample,
    *,
    next_timestep=None,
    renoise=None,
):
    """Apply the Stage3 transition, with an opt-in validation-only ablation."""
    transition_mode = str(
        os.environ.get("ZKROLL_CAUSAL_DMD_TRANSITION_MODE", "x0_renoise")
        or "x0_renoise"
    ).strip().lower()
    if transition_mode == "ode":
        return scheduler.step(model_output, timestep, sample, to_final=False)
    if transition_mode != "x0_renoise":
        raise ValueError(
            "ZKROLL_CAUSAL_DMD_TRANSITION_MODE must be 'x0_renoise' or 'ode', "
            f"got {transition_mode!r}."
        )
    x0 = scheduler.step(model_output, timestep, sample, to_final=True)
    if next_timestep is None:
        return x0
    if renoise is None:
        raise ValueError("renoise is required when next_timestep is provided.")
    return scheduler.add_noise(x0, renoise, next_timestep)

def _causal_validation_resolve_section_prompts(section_prompts, fallback_prompt, args):
    prompts = list(section_prompts or [])
    if not prompts:
        prompts = [fallback_prompt or ""]
    if bool(getattr(args, "causal_validation_fixed_first_prompt", False)):
        prompts = [prompts[0] for _ in prompts]
    return prompts

def _run_causal_kv_rollout(
    unwrapped_model,
    args,
    data,
    val_dataset,
    handler,
    history_latents,
    H_lat,
    W_lat,
    H_img,
    W_img,
    vae_decode_tiled,
    num_validation_blocks,
    latent_window_size,
    use_mosaic,
    batch_idx,
    accelerator=None,
):
    """Run Stage3 few-step generation with a bounded causal KV cache.

    C0 bootstraps the memory bank, every generated chunk is registered back into
    ``FrustumHandler``, and later chunks query only memory accumulated so far.
    """
    pipe = unwrapped_model.pipe
    device = pipe.device
    dtype = pipe.torch_dtype
    dit = pipe.dit
    validation_info = data.get("info") or {}
    cs = int(getattr(args, "causal_chunk_size", 3))
    if latent_window_size % cs != 0:
        raise RuntimeError(
            f"causal-kv: latent_window_size={latent_window_size} not divisible by chunk={cs}"
        )
    chunks_per_window = latent_window_size // cs
    ccc = int(getattr(args, "causal_context_chunks", 0) or 0) or chunks_per_window
    total_chunks = num_validation_blocks * chunks_per_window
    total_n = total_chunks * cs
    fixed_initial_anchor = bool(
        getattr(args, "causal_validation_fixed_initial_anchor", False)
    )
    if total_chunks > ccc:
        anchor_policy = (
            "fixed original C0 anchor"
            if fixed_initial_anchor
            else "advancing boundary anchor"
        )
        print(
            f"[causal-kv] multi-block: total_chunks={total_chunks} > context_chunks={ccc} "
            f"-> sliding-window eviction + {anchor_policy}, GLOBAL absolute "
            "RoPE/PRoPE positions.",
            flush=True,
        )
    h, w = H_lat // 2, W_lat // 2
    validation_memory_mode = _normalize_causal_validation_memory_mode(
        getattr(args, "causal_validation_memory_mode", "c0_plus_generated")
    )
    validation_empty_memory = validation_memory_mode == "empty"
    validation_gt_memory = validation_memory_mode == "gt"
    validation_c0_only_memory = validation_memory_mode == "c0_only"
    t2v_no_ref = bool(getattr(args, "causal_validation_t2v_no_ref", False))
    if fixed_initial_anchor and t2v_no_ref:
        raise ValueError(
            "--causal_validation_fixed_initial_anchor requires an initial C0 "
            "anchor and cannot be combined with --causal_validation_t2v_no_ref."
        )
    if t2v_no_ref and not validation_empty_memory:
        raise ValueError(
            "--causal_validation_t2v_no_ref is a no-reference T2V diagnostic "
            "and must run with --causal_validation_memory_mode empty."
        )
    cfg_scale = _effective_validation_cfg_scale(
        args,
        dmd_validation=True,
    )
    generated_memory_publish_interval = _generated_memory_publish_interval_from_args(args)
    memory_publish_interval = (
        1 if validation_gt_memory else int(generated_memory_publish_interval)
    )
    pending_generated_memory = []
    published_rgb_count = 0
    memory_disabled_after_c0_failure = False
    memory_registration_stats = {
        "requested_mode": validation_memory_mode,
        "c0_registered": None,
        "c0_registration_failures": 0,
        "generated_registration_failures": 0,
        "generated_registration_failure_chunks": [],
        "generated_published_chunks": [],
        "gt_prefix_published_chunks": [],
        "published_rgb_count": 0,
        "memory_disabled_after_c0_failure": False,
    }
    data["_causal_validation_memory_stats"] = memory_registration_stats
    channels = int(history_latents.shape[1])
    register_rgb_source = _validation_register_rgb_source(args)
    register_depth_source = _validation_register_depth_source(args)
    if register_rgb_source not in {"generated", "gt"}:
        raise ValueError(
            "causal_validation_register_rgb_source must be 'generated' or 'gt', "
            f"got {register_rgb_source!r}."
        )
    if register_depth_source not in {"online", "dataset"}:
        raise ValueError(
            "causal_validation_register_depth_source must be 'online' or 'dataset', "
            f"got {register_depth_source!r}."
        )
    if register_depth_source == "dataset" and not bool(
        getattr(args, "force_using_input_extrinics", False)
    ):
        raise ValueError(
            "causal_validation_register_depth_source='dataset' requires "
            "--force_using_input_extrinics so dataset depth is registered with "
            "the matching dataset camera intrinsics/extrinsics."
        )
    if (
        register_rgb_source != "generated" or register_depth_source != "online"
    ) and accelerator.is_main_process:
        print(
            "[causal-kv] validation generated-memory registration override: "
            f"rgb_source={register_rgb_source} depth_source={register_depth_source}",
            flush=True,
        )

    # ---- text embeddings (cfg batch = [negative, positive]) ----
    pipe.load_models_to_device(["text_encoder"])

    def _enc(p):
        ids, mask = pipe.tokenizer(p, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(device), mask.to(device)
        emb = pipe.text_encoder(ids, mask)
        seq = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq):
            emb[:, v:] = 0
        return emb

    prompt = data.get("prompt") or ""
    neg_prompt = args.validation_negative_prompt

    # ---- sliding-window KV cache rollout, GLOBAL absolute positions ----
    # RoPE pos, PRoPE frame idx and camera_info are ALL global/absolute
    # (0,1,2,...,22,23,24,... no re-anchor). The cache evicts old chunks
    # (incl. the original C0) keeping only [advancing anchor + last ccc-1
    # chunks]; because the window is bounded (~ccc*cs frames) the RELATIVE
    # RoPE/PRoPE between the current chunk and the window stays in the trained
    # range even though absolute positions extrapolate.
    base_clean = data["clean_latent_indices"].clone().reshape(-1)[:1]
    base_noisy = data["noisy_latent_indices"].clone().reshape(1, -1)[:, :latent_window_size]
    initial_c0 = history_latents[:, :, -1:].to(device=device, dtype=dtype)
    if t2v_no_ref and (accelerator is None or accelerator.is_main_process):
        print(
            "[causal-kv][t2v-no-ref] starting rollout from empty KV cache: "
            "no C0 seed, no dynamic context, no mosaic memory.",
            flush=True,
        )
    ccc = int(getattr(args, "causal_context_chunks", 0) or 0) or (latent_window_size // cs)

    # full-trajectory GLOBAL camera: read_camera_params returns ~one window at a
    # time, so a single tiled read only yields ~latent_window_size frames. Read
    # each block's window (drift clean/noisy indices by latent_window_size) and
    # concatenate the noisy poses -> a true global 0..total_n camera.
    running_clean = data["clean_latent_indices"].clone()
    running_noisy = data["noisy_latent_indices"].clone()
    clean_intr = clean_extr = None
    section_prompts = []
    noisy_intr_parts, noisy_extr_parts = [], []
    for s in range(int(num_validation_blocks)):
        running_clean = running_clean + (latent_window_size if s > 0 else 0)
        running_noisy = running_noisy + (latent_window_size if s > 0 else 0)
        camb = dict(
            val_dataset.read_camera_params(
                {
                    "clean_latent_indices_start": data["clean_latent_indices_start"],
                    "clean_latent_indices": running_clean,
                    "noisy_latent_indices": running_noisy,
                    "clean_latents": initial_c0,
                    "info": data["info"],
                    "lookup": data.get("lookup", {}),
                    "is_starting": (s == 0),
                },
                info=data["info"],
                lookup=data.get("lookup", {}),
            )
        )
        if s == 0:
            clean_intr = camb["clean_latent_indices_prope_intrinsic"]
            clean_extr = camb["clean_latent_indices_prope_extrinsic"]
        section_prompts.append(camb.get("prompt") or prompt)
        noisy_intr_parts.append(camb["noisy_latent_indices_prope_intrinsic"])
        noisy_extr_parts.append(camb["noisy_latent_indices_prope_extrinsic"])
    full_noisy_intr = torch.cat(noisy_intr_parts, dim=0)[: total_n * 4]
    full_noisy_extr = torch.cat(noisy_extr_parts, dim=0)[: total_n * 4]
    full_noisy_intr, full_noisy_extr = _apply_causal_validation_camera_time_scale(
        full_noisy_intr,
        full_noisy_extr,
        args,
        accelerator=accelerator,
    )
    original_section_prompts = tuple(section_prompts or [prompt or ""])
    data["_causal_validation_original_section_prompts"] = original_section_prompts
    section_prompts = _causal_validation_resolve_section_prompts(
        section_prompts,
        prompt,
        args,
    )
    if bool(getattr(args, "causal_validation_fixed_first_prompt", False)):
        if accelerator is None or accelerator.is_main_process:
            print(
                "[causal-kv][diagnostic] fixed-first prompt enabled for "
                f"{len(section_prompts)} validation sections.",
                flush=True,
            )
    data["_causal_validation_section_prompts"] = tuple(section_prompts)
    section_contexts = []
    for section_prompt in section_prompts:
        positive_context = _enc(section_prompt).to(device=device, dtype=dtype)
        if cfg_scale == 1.0:
            # Stage3 student rollout is positive-only. Do not execute a
            # numerically different negative + 1 * (positive - negative) path.
            section_contexts.append(positive_context)
        else:
            section_contexts.append(
                torch.cat([_enc(neg_prompt), positive_context], dim=0).to(
                    device=device,
                    dtype=dtype,
                )
            )
    context = section_contexts[0]

    dynamic_context_pool = None
    dynamic_context_intr = None
    dynamic_context_extr = None
    initial_c0_frames = None
    validation_dynamic_context_enabled = (
        _validation_dynamic_context_enabled_for_mode(
            args,
            validation_memory_mode,
            t2v_no_ref=t2v_no_ref,
        )
    )
    if validation_dynamic_context_enabled:
        decoded_c0_frames = _decode_latents_to_numpy_frames(
            pipe, initial_c0, device, tiled=vae_decode_tiled
        )
        if validation_gt_memory:
            video_path = validation_info.get("video_path")
            if not video_path:
                raise RuntimeError(
                    "GT validation dynamic context requires an original GT "
                    "video_path for the C0 boundary."
                )
            gt_c0_indices = _validation_clean_frame_indices(
                validation_info, int(decoded_c0_frames.shape[0])
            )
            initial_c0_frames = _read_video_gt_frames(
                video_path,
                gt_c0_indices,
                H_img,
                W_img,
                strict=True,
            )
        else:
            initial_c0_frames = decoded_c0_frames
        (
            dynamic_context_pool,
            dynamic_context_intr,
            dynamic_context_extr,
        ) = _build_initial_causal_kv_dynamic_context_pool(
            args=args,
            data=data,
            pipe=pipe,
            device=device,
            H_img=H_img,
            W_img=W_img,
            total_chunks=total_chunks,
            frames_per_chunk=cs,
            full_noisy_extr=full_noisy_extr.detach().float().cpu().numpy(),
            initial_c0_frames=initial_c0_frames,
            initial_anchor_latent=initial_c0.detach().clone(),
            generated_context_publish_interval=memory_publish_interval,
            tiled=vae_decode_tiled,
        )
    initial_context_count = (
        int(dynamic_context_pool.initial_context_count)
        if dynamic_context_pool is not None
        else 0
    )
    camera_clean_intr = clean_intr[:0] if t2v_no_ref else clean_intr
    camera_clean_extr = clean_extr[:0] if t2v_no_ref else clean_extr
    if (not t2v_no_ref) and dynamic_context_intr is not None and dynamic_context_extr is not None:
        dyn_intr_t = torch.as_tensor(
            dynamic_context_intr,
            device=clean_intr.device,
            dtype=clean_intr.dtype,
        )
        dyn_extr_t = torch.as_tensor(
            dynamic_context_extr,
            device=clean_extr.device,
            dtype=clean_extr.dtype,
        )
        camera_clean_intr = torch.cat([dyn_intr_t, clean_intr], dim=0)
        camera_clean_extr = torch.cat([dyn_extr_t, clean_extr], dim=0)

    def _anchor_position_for_context_start(context_start_chunk):
        if dynamic_context_pool is None:
            return 1 if int(context_start_chunk) == 0 else int(context_start_chunk) * cs + 1
        if int(context_start_chunk) == 0:
            return int(dynamic_context_pool.anchor_position)
        return dynamic_context_pool.position_for_noisy_index(int(context_start_chunk) * cs)

    def _anchor_camera_for_context_start(context_start_chunk):
        if dynamic_context_pool is None:
            return 0 if int(context_start_chunk) == 0 else int(context_start_chunk) * cs
        if int(context_start_chunk) == 0:
            return int(dynamic_context_pool.anchor_camera_frame)
        return dynamic_context_pool.camera_frame_for_noisy_index(
            int(context_start_chunk) * cs
        )

    def _noisy_position(noisy_index):
        # no-ref T2V has no clean/anchor slot before noisy tokens. Keep the
        # temporal axis 1-based for generated latents, but do not reserve the
        # extra C0 offset used by TI2V/ref-conditioned rollout.
        if t2v_no_ref:
            return int(noisy_index)
        if dynamic_context_pool is None:
            return int(noisy_index) + 1
        return dynamic_context_pool.position_for_noisy_index(noisy_index)

    def _noisy_camera_frame(noisy_index):
        # PRoPE camera_frame indexes compact camera_info slots, not temporal
        # RoPE positions. In no-ref T2V, slot 0 is the first noisy camera.
        if t2v_no_ref:
            return max(0, int(noisy_index) - 1)
        if dynamic_context_pool is None:
            return int(noisy_index)
        return dynamic_context_pool.camera_frame_for_noisy_index(noisy_index)

    validation_rope_stats = {
        "generating_first_frame_idx": int(
            (data.get("info") or {}).get("generating_first_frame_idx", 0) or 0
        ),
        "dynamic_context_enabled": bool(dynamic_context_pool is not None),
        "initial_context_count": int(initial_context_count),
        "initial_anchor_position": (
            None if t2v_no_ref else int(_anchor_position_for_context_start(0))
        ),
        "first_noisy_position": int(_noisy_position(1)),
        "generated_context_publish_interval": int(
            dynamic_context_pool.generated_context_publish_interval
            if dynamic_context_pool is not None
            else 1
        ),
        "force_context_original_anchor": bool(
            dynamic_context_pool is not None
            and getattr(dynamic_context_pool, "force_context_original_anchor", False)
        ),
        "fixed_initial_rolling_anchor": bool(fixed_initial_anchor),
        "generated_context_published_chunks": [],
        "chunks": [],
    }
    data["_causal_validation_rope_stats"] = validation_rope_stats

    generated_context_publish_interval = int(
        validation_rope_stats["generated_context_publish_interval"]
    )
    force_context_original_anchor = bool(
        validation_rope_stats["force_context_original_anchor"]
    )
    if (
        dynamic_context_pool is not None
        and generated_context_publish_interval > 1
        and not force_context_original_anchor
    ):
        print(
            "[causal-kv][context] generated-context publish interval="
            f"{generated_context_publish_interval}; generated context is "
            "query-visible only after each completed group.",
            flush=True,
        )

    camera_info = WAN_VIDEO_PROPE_CAMERA_UNIT.process(
        pipe=None,
        use_prope=True,
        h=h,
        w=w,
        dtype=dtype,
        device=device,
        first_frame_count=0 if t2v_no_ref else initial_context_count + 1,
        mosaic_frame_count=0,
        clean_latent_indices_prope_intrinsic=camera_clean_intr,
        clean_latent_indices_prope_extrinsic=camera_clean_extr,
        noisy_latent_indices_prope_intrinsic=full_noisy_intr,
        noisy_latent_indices_prope_extrinsic=full_noisy_extr,
        trans_scale=getattr(dit, "trans_scale", 50.0),
    )["camera_info"]
    # carrier for the mosaic query below (full-trajectory extrinsics)
    cam = {
        "clean_latent_indices_prope_extrinsic": clean_extr,
        "noisy_latent_indices_prope_extrinsic": full_noisy_extr,
    }

    # All causal stages use the same validation memory modes:
    #   c0_plus_generated: deploy-time inference, C0/generated memory bank.
    #   gt: diagnosis mode, growing online C0 + GT-prefix memory bank.
    #   c0_only: diagnosis mode, C0/reference memory only, no generated register.
    #   empty: diagnosis mode, pass no mosaic latent and skip memory registration.
    mosaic_query_latents = None
    if use_mosaic:
        if validation_empty_memory:
            print(
                "[causal-kv] validation empty memory: passing no "
                "mosaic latent and skipping C0/generated memory registration.",
                flush=True,
            )
        else:
            if handler is None:
                raise RuntimeError(
                    "[causal-kv] FrustumHandler is required for online-memory "
                    "validation."
                )
            if validation_gt_memory:
                print(
                    "[causal-kv] validation GT memory: registering original GT "
                    "C0, then publishing each completed GT chunk immediately; "
                    "chunk i can query only bootstrap + GT chunks [0,i).",
                    flush=True,
                )
            c0_frames = (
                initial_c0_frames
                if initial_c0_frames is not None
                else _decode_latents_to_numpy_frames(
                    pipe, initial_c0, device, tiled=vae_decode_tiled
                )
            )
            c0_count = int(c0_frames.shape[0])
            c0_frame_indices = _validation_clean_frame_indices(data["info"], c0_count)
            c0_register_depths = None
            c0_uses_gt_rgb = bool(
                validation_gt_memory or register_rgb_source == "gt"
            )
            c0_uses_dataset_depth = bool(
                validation_gt_memory or register_depth_source == "dataset"
            )
            if c0_uses_gt_rgb:
                c0_frames = _read_video_gt_frames(
                    data["info"]["video_path"],
                    c0_frame_indices,
                    H_img,
                    W_img,
                    strict=True,
                )
            if c0_uses_dataset_depth:
                c0_register_depths = _read_validation_gt_depth_frames(
                    data["info"],
                    c0_frame_indices,
                )
            try:
                handler.register_source_sequence(
                    device,
                    c0_frames,
                    extrinsics=_camera_array_for_frame_count(
                        clean_extr,
                        c0_count,
                        name="clean_extrinsic",
                        selection="tail",
                    ),
                    intrinsics=_camera_array_for_frame_count(
                        clean_intr,
                        c0_count,
                        name="clean_intrinsic",
                        selection="tail",
                    ),
                    depths=c0_register_depths,
                    start_index=0,
                    force_using_input_extrinics=True,
                )
            except Exception as exc:
                if not _is_da3_insufficient_non_sky_error(exc):
                    raise
                print(
                    "[causal-kv][WARN] Skipping initial C0 memory registration "
                    f"after DA3 depth alignment failure: {exc}",
                    flush=True,
                )
                memory_disabled_after_c0_failure = True
                memory_registration_stats["c0_registered"] = False
                memory_registration_stats["c0_registration_failures"] += 1
                memory_registration_stats["memory_disabled_after_c0_failure"] = True
                pending_generated_memory.clear()
                _release_cached_memory()
            else:
                mosaic_query_latents = _encode_frames_per_frame(
                    pipe, c0_frames, device, tiled=vae_decode_tiled
                )
                memory_registration_stats["c0_registered"] = True
                memory_registration_stats["published_rgb_count"] = int(c0_count)
            if validation_c0_only_memory:
                print(
                    "[causal-kv] validation C0-only memory: registered only "
                    "the initial reference frame; generated chunks will not "
                    "be registered back into mosaic memory.",
                    flush=True,
                )
            elif (not validation_gt_memory) and generated_memory_publish_interval > 1:
                print(
                    f"[causal-kv] generated-memory publish interval="
                    f"{generated_memory_publish_interval}; generated chunks are "
                    "query-visible only after each completed group.",
                    flush=True,
                )

    def _publish_pending_generated_memory(force=False):
        nonlocal mosaic_query_latents, published_rgb_count
        if memory_disabled_after_c0_failure:
            pending_generated_memory.clear()
            return
        if not pending_generated_memory:
            return
        if (not force) and len(pending_generated_memory) < memory_publish_interval:
            return
        published_chunks = []
        for item in pending_generated_memory:
            try:
                handler.register_source_sequence(
                    device,
                    item["frames"],
                    extrinsics=item["extrinsics"],
                    intrinsics=item["intrinsics"],
                    depths=item.get("depths"),
                    force_using_input_extrinics=True,
                )
            except Exception as exc:
                if not _is_da3_insufficient_non_sky_error(exc):
                    raise
                print(
                    "[causal-kv][WARN] Skipping generated memory registration "
                    f"for chunk={int(item['chunk_idx'])} after DA3 depth alignment "
                    f"failure: {exc}",
                    flush=True,
                )
                memory_registration_stats["generated_registration_failures"] += 1
                memory_registration_stats["generated_registration_failure_chunks"].append(
                    int(item["chunk_idx"])
                )
                _release_cached_memory()
                continue
            if mosaic_query_latents is None:
                mosaic_query_latents = item["latents"]
            else:
                mosaic_query_latents = torch.cat(
                    [mosaic_query_latents, item["latents"]], dim=2
                )
            published_chunks.append(int(item["chunk_idx"]))
            source = str(item.get("source") or "generated")
            if source == "gt_prefix":
                memory_registration_stats["gt_prefix_published_chunks"].append(
                    int(item["chunk_idx"])
                )
            else:
                memory_registration_stats["generated_published_chunks"].append(
                    int(item["chunk_idx"])
                )
            if "rgb_end" in item:
                published_rgb_count = max(
                    int(published_rgb_count), int(item["rgb_end"])
                )
            else:
                published_rgb_count += int(item["frames"].shape[0])
            memory_registration_stats["published_rgb_count"] = int(published_rgb_count)
        pending_generated_memory.clear()
        if memory_publish_interval > 1:
            print(
                f"[causal-kv] published generated memory chunks={published_chunks} "
                f"force={bool(force)}",
                flush=True,
            )

    pipe.load_models_to_device(pipe.in_iteration_models)
    nblocks = len(dit.blocks)
    rolling_cache = init_causal_kv_caches(nblocks)
    rolling_latents = None
    if not t2v_no_ref:
        initial_anchor_position = _anchor_position_for_context_start(0)
        initial_anchor_camera_frame = _anchor_camera_for_context_start(0)
        with _maybe_gather_zero3_module(accelerator, dit, label="dit"):
            model_fn_causal_kv(
                dit,
                latents_chunk=initial_c0,
                timestep_frames=torch.zeros(1, device=device, dtype=dtype),
                context=context,
                camera_info=camera_info,
                cur_positions=[int(initial_anchor_position)],
                cur_frames=[int(initial_anchor_camera_frame)],
                caches=rolling_cache,
                write_cache=True,
                cur_cache_chunk_ids=[-1],
            )
    scheduler = pipe.scheduler
    timesteps, sigmas = prepare_causal_dmd_eval_scheduler(
        scheduler,
        getattr(args, "causal_dmd_denoising_step_list", (1000, 750, 500, 250)),
        timestep_wrap=bool(getattr(args, "causal_dmd_timestep_wrap", True)),
        device=device,
        model_dtype=dtype,
    )
    scheduler.timesteps = timesteps.detach().to(device="cpu", dtype=torch.float32)
    scheduler.sigmas = sigmas.detach().to(device="cpu", dtype=torch.float32)
    scheduler.training = False
    print(
        f"[inference][causal_dmd] effective {len(timesteps)}-step contract "
        f"model_dtype={dtype} "
        f"timesteps={tuple(float(v) for v in scheduler.timesteps.tolist())} "
        f"sigmas={tuple(float(v) for v in scheduler.sigmas.tolist())}",
        flush=True,
    )
    prefix_chunks_by_step_raw = str(
        getattr(args, "causal_validation_prefix_chunks_by_step", "") or ""
    ).strip()
    prefix_chunks_by_step = None
    if prefix_chunks_by_step_raw:
        try:
            prefix_chunks_by_step = tuple(
                int(value.strip())
                for value in prefix_chunks_by_step_raw.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "causal_validation_prefix_chunks_by_step must be a comma-separated "
                f"integer list, got {prefix_chunks_by_step_raw!r}."
            ) from exc
        if len(prefix_chunks_by_step) != len(scheduler.timesteps):
            raise ValueError(
                "causal_validation_prefix_chunks_by_step must contain exactly one "
                f"value per denoising step; got {len(prefix_chunks_by_step)} values "
                f"for {len(scheduler.timesteps)} steps."
            )
        if any(value < 1 or value > ccc for value in prefix_chunks_by_step):
            raise ValueError(
                "causal_validation_prefix_chunks_by_step values must lie in "
                f"[1, causal_context_chunks={ccc}], got {prefix_chunks_by_step}."
            )
        print(
            "[causal-kv][diagnostic] timestep-dependent continuous-prefix "
            f"windows={prefix_chunks_by_step}; persistent/cache-fill window "
            f"remains causal_context_chunks={ccc}.",
            flush=True,
        )
    prefix_noise_mode = str(
        getattr(args, "causal_validation_prefix_noise_mode", "none") or "none"
    ).strip().lower()
    if prefix_noise_mode not in {"none", "hiar_sde"}:
        raise ValueError(
            "causal_validation_prefix_noise_mode must be 'none' or 'hiar_sde', "
            f"got {prefix_noise_mode!r}."
        )
    prefix_noise_dynamic_context = bool(
        getattr(args, "causal_validation_prefix_noise_dynamic_context", False)
    )
    prefix_noise_scales_raw = str(
        getattr(args, "causal_validation_prefix_noise_scales_by_step", "") or ""
    ).strip()
    if prefix_noise_scales_raw:
        try:
            prefix_noise_scales_by_step = tuple(
                float(value.strip())
                for value in prefix_noise_scales_raw.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "causal_validation_prefix_noise_scales_by_step must be a "
                f"comma-separated float list, got {prefix_noise_scales_raw!r}."
            ) from exc
        if len(prefix_noise_scales_by_step) != len(scheduler.timesteps):
            raise ValueError(
                "causal_validation_prefix_noise_scales_by_step must contain "
                f"exactly one value per denoising step; got "
                f"{len(prefix_noise_scales_by_step)} values for "
                f"{len(scheduler.timesteps)} steps."
            )
        if any(
            (not math.isfinite(value)) or value < 0.0 or value > 1.0
            for value in prefix_noise_scales_by_step
        ):
            raise ValueError(
                "causal_validation_prefix_noise_scales_by_step values must be "
                f"finite and lie in [0, 1], got {prefix_noise_scales_by_step}."
            )
    else:
        prefix_noise_scales_by_step = tuple(
            1.0 for _ in scheduler.timesteps
        )
    if prefix_noise_mode != "none":
        if t2v_no_ref:
            raise ValueError(
                "HiAR prefix corruption requires a clean moving anchor and is "
                "incompatible with causal_validation_t2v_no_ref."
            )
        if float(cfg_scale) != 1.0:
            raise ValueError("HiAR prefix corruption requires CFG=1 (no CFG).")
        if prefix_chunks_by_step is not None:
            raise ValueError(
                "HiAR prefix corruption cannot be combined with prefix-window "
                "cropping; test one prefix intervention at a time."
            )
        print(
            "[causal-kv][hiar] timestep-conditioned SDE prefix enabled: moving "
            "anchor stays clean; generated rolling prefix is rebuilt at the next "
            "denoising timestep; dynamic_context_noise="
            f"{prefix_noise_dynamic_context}; corruption_scales="
            f"{prefix_noise_scales_by_step}.",
            flush=True,
        )
    elif prefix_noise_dynamic_context:
        raise ValueError(
            "causal_validation_prefix_noise_dynamic_context requires "
            "causal_validation_prefix_noise_mode=hiar_sde."
        )
    elif prefix_noise_scales_raw:
        raise ValueError(
            "causal_validation_prefix_noise_scales_by_step requires "
            "causal_validation_prefix_noise_mode=hiar_sde."
        )
    if prefix_noise_mode == "hiar_sde":
        rolling_latents = initial_c0.detach().clone()
    mosaic_guidance_scale_raw = getattr(
        args, "causal_validation_mosaic_guidance_scale", 1.0
    )
    if mosaic_guidance_scale_raw is None:
        mosaic_guidance_scale_raw = 1.0
    mosaic_guidance_scale = float(mosaic_guidance_scale_raw)
    if not math.isfinite(mosaic_guidance_scale) or mosaic_guidance_scale <= 0.0:
        raise ValueError(
            "causal_validation_mosaic_guidance_scale must be finite and > 0, "
            f"got {mosaic_guidance_scale}."
        )
    mosaic_repeat_by_step_raw = str(
        getattr(args, "causal_validation_mosaic_token_repeat_by_step", "") or ""
    ).strip()
    mosaic_repeat_by_step = None
    if mosaic_repeat_by_step_raw:
        try:
            mosaic_repeat_by_step = tuple(
                int(value.strip())
                for value in mosaic_repeat_by_step_raw.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "causal_validation_mosaic_token_repeat_by_step must be a "
                f"comma-separated integer list, got {mosaic_repeat_by_step_raw!r}."
            ) from exc
        if len(mosaic_repeat_by_step) != len(scheduler.timesteps):
            raise ValueError(
                "causal_validation_mosaic_token_repeat_by_step must contain "
                f"exactly one value per denoising step; got "
                f"{len(mosaic_repeat_by_step)} values for "
                f"{len(scheduler.timesteps)} steps."
            )
        if any(value < 1 or value > 16 for value in mosaic_repeat_by_step):
            raise ValueError(
                "causal_validation_mosaic_token_repeat_by_step values must lie "
                f"in [1, 16], got {mosaic_repeat_by_step}."
            )
        print(
            "[causal-kv][diagnostic] timestep-dependent mosaic token repeats="
            f"{mosaic_repeat_by_step}; rolling prefix and memory pool are unchanged.",
            flush=True,
        )
    mosaic_context_frames = int(
        getattr(args, "causal_validation_mosaic_context_frames", 0) or 0
    )
    if mosaic_context_frames not in {0, 1, 3}:
        raise ValueError(
            "causal_validation_mosaic_context_frames must be one of 0, 1, or 3, "
            f"got {mosaic_context_frames}."
        )
    if mosaic_context_frames and (not use_mosaic or validation_empty_memory):
        raise ValueError(
            "causal_validation_mosaic_context_frames requires an enabled, "
            "non-empty validation mosaic memory mode."
        )
    mosaic_context_topk = int(
        getattr(args, "causal_validation_mosaic_context_topk", 0) or 0
    )
    if mosaic_context_topk < 0:
        raise ValueError(
            "causal_validation_mosaic_context_topk must be >= 0, got "
            f"{mosaic_context_topk}."
        )
    if mosaic_context_topk and not mosaic_context_frames:
        raise ValueError(
            "causal_validation_mosaic_context_topk requires "
            "causal_validation_mosaic_context_frames > 0."
        )
    mosaic_context_guidance_scale = float(
        getattr(args, "causal_validation_mosaic_context_guidance_scale", 1.0)
    )
    if not math.isfinite(mosaic_context_guidance_scale):
        raise ValueError(
            "causal_validation_mosaic_context_guidance_scale must be finite, "
            f"got {mosaic_context_guidance_scale}."
        )
    if not 0.0 <= mosaic_context_guidance_scale <= 2.0:
        raise ValueError(
            "causal_validation_mosaic_context_guidance_scale must lie in "
            f"[0, 2], got {mosaic_context_guidance_scale}."
        )
    if mosaic_context_guidance_scale != 1.0 and not mosaic_context_frames:
        raise ValueError(
            "causal_validation_mosaic_context_guidance_scale != 1 requires "
            "causal_validation_mosaic_context_frames > 0."
        )
    mosaic_context_nonlocal_min_age_chunks = int(
        getattr(
            args,
            "causal_validation_mosaic_context_nonlocal_min_age_chunks",
            0,
        )
        or 0
    )
    if mosaic_context_nonlocal_min_age_chunks < 0:
        raise ValueError(
            "causal_validation_mosaic_context_nonlocal_min_age_chunks must be "
            f">= 0, got {mosaic_context_nonlocal_min_age_chunks}."
        )
    if mosaic_context_nonlocal_min_age_chunks and not mosaic_context_topk:
        raise ValueError(
            "causal_validation_mosaic_context_nonlocal_min_age_chunks requires "
            "causal_validation_mosaic_context_topk > 0 so candidate ages are "
            "well-defined."
        )
    mosaic_context_step_mask_raw = str(
        getattr(args, "causal_validation_mosaic_context_step_mask", "") or ""
    ).strip()
    if mosaic_context_step_mask_raw:
        if not mosaic_context_frames:
            raise ValueError(
                "causal_validation_mosaic_context_step_mask requires "
                "causal_validation_mosaic_context_frames > 0."
            )
        try:
            mosaic_context_step_mask = tuple(
                int(value.strip())
                for value in mosaic_context_step_mask_raw.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise ValueError(
                "causal_validation_mosaic_context_step_mask must be a "
                f"comma-separated 0/1 list, got {mosaic_context_step_mask_raw!r}."
            ) from exc
        if len(mosaic_context_step_mask) != len(scheduler.timesteps):
            raise ValueError(
                "causal_validation_mosaic_context_step_mask must contain "
                f"exactly one value per denoising step; got "
                f"{len(mosaic_context_step_mask)} values for "
                f"{len(scheduler.timesteps)} steps."
            )
        if any(value not in {0, 1} for value in mosaic_context_step_mask):
            raise ValueError(
                "causal_validation_mosaic_context_step_mask values must be "
                f"0 or 1, got {mosaic_context_step_mask}."
            )
        if not any(mosaic_context_step_mask):
            raise ValueError(
                "causal_validation_mosaic_context_step_mask must enable at "
                "least one denoising step."
            )
    else:
        mosaic_context_step_mask = tuple(
            1 for _ in scheduler.timesteps
        ) if mosaic_context_frames else tuple()
    if prefix_noise_mode != "none" and (
        mosaic_guidance_scale != 1.0
        or mosaic_repeat_by_step is not None
        or mosaic_context_frames
        or mosaic_context_topk
        or mosaic_context_guidance_scale != 1.0
        or mosaic_context_nonlocal_min_age_chunks
    ):
        raise ValueError(
            "HiAR prefix corruption must run as a clean single-variable ablation: "
            "paired memory guidance, token repeat, and promoted/top-k mosaic "
            "context diagnostics must all be disabled."
        )
    if mosaic_context_frames:
        mosaic_context_source = (
            f"top-{mosaic_context_topk} warped candidates"
            if mosaic_context_topk
            else "ordinary fused mosaic"
        )
        print(
            "[causal-kv][diagnostic] promoting mosaic to chunk-local "
            f"clean context frames={mosaic_context_frames}; the standard mosaic "
            "channel, rolling prefix, query, and memory pool remain enabled; "
            f"context source={mosaic_context_source}; "
            f"denoising-step mask={mosaic_context_step_mask}; "
            f"paired context guidance scale={mosaic_context_guidance_scale}; "
            "nonlocal minimum age chunks="
            f"{mosaic_context_nonlocal_min_age_chunks}.",
            flush=True,
        )
    mosaic_guidance_stats = {
        "scale": float(mosaic_guidance_scale),
        "chunks": [],
    }
    mosaic_context_guidance_stats = {
        "scale": float(mosaic_context_guidance_scale),
        "chunks": [],
    }
    data["_causal_validation_prefix_chunks_by_step"] = (
        list(prefix_chunks_by_step) if prefix_chunks_by_step is not None else []
    )
    hiar_prefix_noise_stats = {
        "mode": str(prefix_noise_mode),
        "dynamic_context_noised": bool(prefix_noise_dynamic_context),
        "noise_scales_by_step": [
            float(value) for value in prefix_noise_scales_by_step
        ],
        "chunks": [],
    }
    data["_causal_validation_hiar_prefix_noise"] = hiar_prefix_noise_stats
    data["_causal_validation_mosaic_token_repeat_by_step"] = (
        list(mosaic_repeat_by_step) if mosaic_repeat_by_step is not None else []
    )
    data["_causal_validation_mosaic_context_frames"] = int(mosaic_context_frames)
    data["_causal_validation_mosaic_context_topk"] = int(mosaic_context_topk)
    data["_causal_validation_mosaic_context_guidance_stats"] = (
        mosaic_context_guidance_stats
    )
    data["_causal_validation_mosaic_context_nonlocal_min_age_chunks"] = int(
        mosaic_context_nonlocal_min_age_chunks
    )
    data["_causal_validation_mosaic_context_step_mask"] = list(
        mosaic_context_step_mask
    )
    data["_causal_validation_mosaic_guidance_stats"] = mosaic_guidance_stats

    def _hiar_sde_corrupt(
        clean_latents,
        context_timestep,
        *,
        seed,
        keep_first_clean,
        corruption_scale,
    ):
        clean_latents = clean_latents.to(device=device, dtype=dtype)
        context_timestep = float(context_timestep)
        corruption_scale = float(corruption_scale)
        if context_timestep <= 0.0 or corruption_scale <= 0.0:
            return clean_latents.clone()
        noise_gen = torch.Generator(device=device)
        noise_gen.manual_seed(int(seed))
        noise = torch.randn(
            clean_latents.shape,
            generator=noise_gen,
            device=device,
            dtype=dtype,
        )
        return hiar_sde_corrupt_clean_latents(
            scheduler,
            clean_latents,
            context_timestep,
            keep_first_clean=bool(keep_first_clean),
            corruption_scale=corruption_scale,
            noise=noise,
        )

    all_clean, all_mosaic = [], []
    for i in range(total_chunks):
        section_idx = min(
            int(i) // max(1, latent_window_size // cs),
            len(section_contexts) - 1,
        )
        context = section_contexts[section_idx]
        rolling_positions = list(
            rolling_cache[0].get("positions", rolling_cache[0].get("frames", []))
        )
        rolling_camera_frames = list(rolling_cache[0].get("frames", []))
        if rolling_latents is not None and int(rolling_latents.shape[2]) != len(
            rolling_positions
        ):
            raise RuntimeError(
                "[causal-kv] rolling latent/cache provenance diverged before "
                f"chunk={i}: latent_frames={int(rolling_latents.shape[2])}, "
                f"cache_frames={len(rolling_positions)}."
            )
        if not rolling_positions or not rolling_camera_frames:
            if not (t2v_no_ref and i == 0):
                raise RuntimeError(
                    f"[causal-kv] rolling cache is empty before chunk={i}; "
                    "initial anchor seed failed."
                )
            rolling_anchor_position = -1
            rolling_anchor_camera_frame = -1
        else:
            rolling_anchor_position = int(rolling_positions[0])
            rolling_anchor_camera_frame = int(rolling_camera_frames[0])
        cur_positions = [
            _noisy_position(idx)
            for idx in range(1 + i * cs, 1 + i * cs + cs)
        ]
        cur_camera_frames = [
            _noisy_camera_frame(idx)
            for idx in range(1 + i * cs, 1 + i * cs + cs)
        ]
        # M_i: ONLY the current chunk's mosaic (1:1 with N_i), not accumulated.
        # Holes in M_i are masked out of N_i's keys inside model_fn_causal_kv.
        m_i = m_pos = m_fr = None
        mosaic_context_i = None
        mosaic_context_topk_valid_ratio = None
        mosaic_context_candidate_frame_ids = []
        mosaic_context_candidate_age_frames = []
        mosaic_context_active_i = False
        mfc = 0
        if use_mosaic and validation_empty_memory:
            # Match training nomosaic exactly: no mosaic tokens are prepended.
            m_i = None
            mfc = 0
            m_fr = None
            m_pos = None
        elif use_mosaic and mosaic_query_latents is not None:
            rgb_start = i * cs * 4
            rgb_end = rgb_start + cs * 4
            # FrustumHandler.align_w2c_trajectory aligns the first query pose to
            # handler.last_extrinsic. Once generated memory has been published,
            # handler.last_extrinsic is the latest generated frame, not C0. Use
            # that same frame as the clean anchor so chunk-level k=1 rollout and
            # section-level validation query in the same coordinate frame.
            if int(published_rgb_count) <= 0:
                query_anchor_extr = clean_extr[-1:]
                query_window_rgb_start = 0
            else:
                query_anchor_extr = full_noisy_extr[
                    published_rgb_count - 1 : published_rgb_count
                ]
                query_window_rgb_start = int(published_rgb_count)
            q_extr = torch.cat(
                [
                    query_anchor_extr.repeat(4, 1, 1),
                    full_noisy_extr[query_window_rgb_start:rgb_end],
                ],
                dim=0,
            )
            q_extr_np = handler.align_w2c_trajectory(
                q_extr.detach().float().cpu().numpy()[3:]
            )[1:]
            local_start = rgb_start - query_window_rgb_start
            local_end = rgb_end - query_window_rgb_start
            query_extrinsics = q_extr_np[local_start:local_end]
            source_latents = mosaic_query_latents[0].clone()
            memory_K_arg = _registered_memory_K_for_query(
                handler,
                source_latents,
                context="[causal-kv]",
            )
            query_K_arg = full_noisy_intr[rgb_start:rgb_end].detach().float().cpu().numpy()
            generated_query_kwargs = _generated_memory_query_kwargs_from_args(args)
            queried_result = handler.query_hits_mode_new(
                device,
                query_extrinsics,
                source_latents,
                **generated_query_kwargs,
                memory_K=memory_K_arg,
                query_K=query_K_arg,
                return_candidate_frame_ids=bool(mosaic_context_topk),
            )
            if mosaic_context_topk:
                queried, candidate_frame_ids, _ = queried_result
                context_candidate_frame_ids = [
                    list(frame_ids)[:mosaic_context_topk]
                    for frame_ids in candidate_frame_ids
                ]
                center_group_index = len(context_candidate_frame_ids) // 2
                mosaic_context_candidate_frame_ids = [
                    int(frame_id)
                    for frame_id in context_candidate_frame_ids[center_group_index]
                    if int(frame_id) >= 0
                ]
                newest_registered_frame_id = len(handler.extrinsics) - 1
                mosaic_context_candidate_age_frames = [
                    int(newest_registered_frame_id - frame_id)
                    for frame_id in mosaic_context_candidate_frame_ids
                ]
                mosaic_context_active_i = True
                if mosaic_context_nonlocal_min_age_chunks:
                    min_age_frames = (
                        int(mosaic_context_nonlocal_min_age_chunks) * cs * 4
                    )
                    mosaic_context_active_i = bool(
                        mosaic_context_candidate_age_frames
                        and min(mosaic_context_candidate_age_frames)
                        >= min_age_frames
                    )
                if mosaic_context_active_i:
                    context_queried, _ = handler.fuse_candidates(
                        query_extrinsics=query_extrinsics,
                        candidate_frame_ids=context_candidate_frame_ids,
                        latents=source_latents,
                        w2c=np.asarray(handler.extrinsics),
                        depths=np.asarray(handler.depths),
                        memory_K=memory_K_arg,
                        query_K=query_K_arg,
                        fuse_mode=generated_query_kwargs["fuse_mode"],
                        return_revgrid=False,
                        zbuffer_depth_preference=generated_query_kwargs[
                            "zbuffer_depth_preference"
                        ],
                        interpolation_mode=generated_query_kwargs[
                            "interpolation_mode"
                        ],
                        latent_merge_4frames=False,
                        query_reference_frame=generated_query_kwargs[
                            "query_reference_frame"
                        ],
                    )
                    context_valid = (
                        context_queried.abs().amax(dim=0, keepdim=True) > 0
                    )
                    mosaic_context_topk_valid_ratio = float(
                        context_valid.float().mean().item()
                    )
                    context_queried = torch.where(
                        context_valid,
                        context_queried,
                        queried,
                    )
                    mosaic_context_i = context_queried.unsqueeze(0).to(
                        device=device, dtype=dtype
                    )
            else:
                queried = queried_result
                mosaic_context_active_i = bool(mosaic_context_frames)
            if isinstance(queried, (tuple, list)):
                queried = queried[0]
            if int(queried.shape[1]) != cs:
                raise RuntimeError(
                    f"[causal-kv] queried mosaic T={int(queried.shape[1])} "
                    f"does not match chunk size={cs}."
                )
            m_i = queried.unsqueeze(0).to(device=device, dtype=dtype)
            mfc = int(m_i.shape[2])
            m_fr = list(cur_camera_frames)
            m_pos = list(cur_positions)

        selected_context_entries = []
        if dynamic_context_pool is not None:
            rgb_start = i * cs * 4
            rgb_end = rgb_start + cs * 4
            selected_context_entries = dynamic_context_pool.select_for_chunk(
                i,
                full_noisy_extr[rgb_start:rgb_end].detach().float().cpu().numpy(),
                exclude_positions={rolling_anchor_position},
                exclude_camera_frames={rolling_anchor_camera_frame},
            )
        validation_rope_stats["chunks"].append(
            {
                "chunk_idx": int(i),
                "rolling_cache_frame_count": int(
                    _causal_kv_frame_count(rolling_cache)
                ),
                "context_cache_frame_count": int(len(selected_context_entries)),
                "rolling_anchor_position": int(rolling_anchor_position),
                "noisy_positions": [int(v) for v in cur_positions],
                "context_positions": [
                    int(entry["position"]) for entry in selected_context_entries
                ],
                "context_sources": [
                    str(entry.get("source") or "unknown")
                    for entry in selected_context_entries
                ],
                "mosaic_context_frame_count": int(
                    min(mfc, mosaic_context_frames)
                    if mosaic_context_active_i and mosaic_context_frames > 1
                    else int(
                        mosaic_context_active_i
                        and mfc > 0
                        and mosaic_context_frames == 1
                    )
                ),
                "mosaic_context_active": bool(mosaic_context_active_i),
                "mosaic_context_candidate_frame_ids": list(
                    mosaic_context_candidate_frame_ids
                ),
                "mosaic_context_candidate_age_frames": list(
                    mosaic_context_candidate_age_frames
                ),
                "mosaic_context_topk_valid_ratio": (
                    mosaic_context_topk_valid_ratio
                ),
            }
        )

        with _maybe_gather_zero3_module(accelerator, dit, label="dit"):
            context_cache = None
            if selected_context_entries:
                seed_latents = torch.cat(
                    [
                        entry["latent"].to(device=device, dtype=dtype)
                        for entry in selected_context_entries
                    ],
                    dim=2,
                ).contiguous()
                seed_positions = [
                    int(entry["position"]) for entry in selected_context_entries
                ]
                seed_camera_frames = [
                    int(entry["camera_frame"]) for entry in selected_context_entries
                ]
                seed_chunk_ids = [int(i) for _ in selected_context_entries]
                context_cache = init_causal_kv_caches(nblocks)
                model_fn_causal_kv(
                    dit,
                    latents_chunk=seed_latents,
                    timestep_frames=torch.zeros(
                        int(seed_latents.shape[2]), device=device, dtype=dtype
                    ),
                    context=context,
                    camera_info=camera_info,
                    cur_positions=seed_positions,
                    cur_frames=seed_camera_frames,
                    caches=context_cache,
                    write_cache=True,
                    cur_cache_chunk_ids=seed_chunk_ids,
                )
            base_context_cache = context_cache
            mosaic_context_cache = None
            if (
                mosaic_context_frames
                and mosaic_context_active_i
                and m_i is not None
            ):
                mosaic_context_source_i = (
                    mosaic_context_i if mosaic_context_i is not None else m_i
                )
                if mosaic_context_frames == 1:
                    mosaic_context_indices = [int(mfc) // 2]
                else:
                    mosaic_context_indices = list(range(int(mfc)))
                mosaic_context_latents = mosaic_context_source_i[
                    :, :, mosaic_context_indices
                ].contiguous()
                mosaic_context_positions = [
                    int(m_pos[idx]) for idx in mosaic_context_indices
                ]
                mosaic_context_camera_frames = [
                    int(m_fr[idx]) for idx in mosaic_context_indices
                ]
                mosaic_context_cache = init_causal_kv_caches(nblocks)
                model_fn_causal_kv(
                    dit,
                    latents_chunk=mosaic_context_latents,
                    timestep_frames=torch.zeros(
                        len(mosaic_context_indices), device=device, dtype=dtype
                    ),
                    context=context,
                    camera_info=camera_info,
                    cur_positions=mosaic_context_positions,
                    cur_frames=mosaic_context_camera_frames,
                    caches=mosaic_context_cache,
                    write_cache=True,
                    cur_cache_chunk_ids=[int(i)] * len(mosaic_context_indices),
                )
                context_cache = _causal_kv_concat_caches(
                    context_cache,
                    mosaic_context_cache,
                )
            # Assemble read cache = [C_i | rolling anchor/generated-prefix].
            # Generated-prefix KV is global history (chunk_id=-1), while C_i
            # remains chunk-local and is filtered by cache_read_chunk_id.
            read_cache = _causal_kv_concat_caches(context_cache, rolling_cache)
            read_cache_positions = list(
                read_cache[0].get("positions", read_cache[0].get("frames", []))
            )
            read_cache_frames = list(read_cache[0].get("frames", []))
            needs_mosaic_context_free_cache = bool(
                mosaic_context_cache is not None
                and (
                    not all(mosaic_context_step_mask)
                    or mosaic_context_guidance_scale != 1.0
                )
            )
            if needs_mosaic_context_free_cache:
                read_cache_without_mosaic_context = _causal_kv_concat_caches(
                    base_context_cache,
                    rolling_cache,
                )
                read_cache_without_mosaic_context_positions = list(
                    read_cache_without_mosaic_context[0].get(
                        "positions",
                        read_cache_without_mosaic_context[0].get("frames", []),
                    )
                )
                read_cache_without_mosaic_context_frames = list(
                    read_cache_without_mosaic_context[0].get("frames", [])
                )
            else:
                read_cache_without_mosaic_context = read_cache
                read_cache_without_mosaic_context_positions = read_cache_positions
                read_cache_without_mosaic_context_frames = read_cache_frames

            hiar_prediction_cache_step_index = None
            hiar_prediction_cache_step = None

            def _build_hiar_prediction_cache(hiar_step_idx):
                if rolling_latents is None or not rolling_positions:
                    raise RuntimeError(
                        "HiAR prefix corruption requires non-empty rolling latent "
                        f"provenance before chunk={i}."
                    )
                next_context_t = (
                    float(scheduler.timesteps[hiar_step_idx + 1])
                    if hiar_step_idx + 1 < len(scheduler.timesteps)
                    else 0.0
                )
                corruption_scale = float(
                    prefix_noise_scales_by_step[hiar_step_idx]
                )
                effective_context_t = next_context_t * corruption_scale
                noised_rolling = _hiar_sde_corrupt(
                    rolling_latents,
                    next_context_t,
                    seed=(
                        int(args.validation_seed)
                        + int(batch_idx) * 1000000
                        + int(i) * 10000
                        + int(hiar_step_idx) * 101
                        + 7000000
                    ),
                    keep_first_clean=True,
                    corruption_scale=corruption_scale,
                )
                rolling_timestep_frames = torch.full(
                    (int(noised_rolling.shape[2]),),
                    float(effective_context_t),
                    device=device,
                    dtype=dtype,
                )
                rolling_timestep_frames[:1] = 0
                hiar_rolling_cache = init_causal_kv_caches(nblocks)
                model_fn_causal_kv(
                    dit,
                    latents_chunk=noised_rolling,
                    timestep_frames=rolling_timestep_frames,
                    context=context,
                    camera_info=camera_info,
                    cur_positions=rolling_positions,
                    cur_frames=rolling_camera_frames,
                    caches=hiar_rolling_cache,
                    write_cache=True,
                    cur_cache_chunk_ids=[-1] * len(rolling_positions),
                )

                hiar_context_cache = base_context_cache
                if prefix_noise_dynamic_context and selected_context_entries:
                    clean_dynamic_context = torch.cat(
                        [
                            entry["latent"].to(device=device, dtype=dtype)
                            for entry in selected_context_entries
                        ],
                        dim=2,
                    ).contiguous()
                    noised_dynamic_context = _hiar_sde_corrupt(
                        clean_dynamic_context,
                        next_context_t,
                        seed=(
                            int(args.validation_seed)
                            + int(batch_idx) * 1000000
                            + int(i) * 10000
                            + int(hiar_step_idx) * 101
                            + 8000000
                        ),
                        keep_first_clean=False,
                        corruption_scale=prefix_noise_scales_by_step[
                            hiar_step_idx
                        ],
                    )
                    hiar_context_cache = init_causal_kv_caches(nblocks)
                    model_fn_causal_kv(
                        dit,
                        latents_chunk=noised_dynamic_context,
                        timestep_frames=torch.full(
                            (int(noised_dynamic_context.shape[2]),),
                            float(effective_context_t),
                            device=device,
                            dtype=dtype,
                        ),
                        context=context,
                        camera_info=camera_info,
                        cur_positions=[
                            int(entry["position"])
                            for entry in selected_context_entries
                        ],
                        cur_frames=[
                            int(entry["camera_frame"])
                            for entry in selected_context_entries
                        ],
                        caches=hiar_context_cache,
                        write_cache=True,
                        cur_cache_chunk_ids=[int(i)] * len(selected_context_entries),
                    )

                hiar_cache = _causal_kv_concat_caches(
                    hiar_context_cache,
                    hiar_rolling_cache,
                )
                return (
                    hiar_cache,
                    list(
                        hiar_cache[0].get(
                            "positions", hiar_cache[0].get("frames", [])
                        )
                    ),
                    list(hiar_cache[0].get("frames", [])),
                )

            if prefix_noise_mode == "hiar_sde":
                hiar_prefix_noise_stats["chunks"].append(
                    {
                        "chunk_idx": int(i),
                        "rolling_prefix_frames": int(rolling_latents.shape[2]),
                        "dynamic_context_frames": int(len(selected_context_entries)),
                        "context_timesteps": [
                            (
                                float(scheduler.timesteps[step_idx + 1])
                                if step_idx + 1 < len(scheduler.timesteps)
                                else 0.0
                            )
                            for step_idx in range(len(scheduler.timesteps))
                        ],
                        "corruption_scales": [
                            float(value)
                            for value in prefix_noise_scales_by_step
                        ],
                        "effective_context_timesteps": [
                            (
                                float(scheduler.timesteps[step_idx + 1])
                                * float(prefix_noise_scales_by_step[step_idx])
                                if step_idx + 1 < len(scheduler.timesteps)
                                else 0.0
                            )
                            for step_idx in range(len(scheduler.timesteps))
                        ],
                    }
                )

            prediction_cache_views = {}
            prediction_cache_views_without_mosaic_context = {}
            if prefix_chunks_by_step is not None:
                for prefix_chunks in sorted(set(prefix_chunks_by_step)):
                    rolling_view = _causal_kv_trim_rolling_window(
                        rolling_cache,
                        frames_per_chunk=cs,
                        window_chunks=int(prefix_chunks),
                    )
                    cache_view = _causal_kv_concat_caches(
                        context_cache,
                        rolling_view,
                    )
                    prediction_cache_views[int(prefix_chunks)] = (
                        cache_view,
                        list(
                            cache_view[0].get(
                                "positions", cache_view[0].get("frames", [])
                            )
                        ),
                        list(cache_view[0].get("frames", [])),
                    )
                    if needs_mosaic_context_free_cache:
                        cache_view_without_mosaic_context = _causal_kv_concat_caches(
                            base_context_cache,
                            rolling_view,
                        )
                        prediction_cache_views_without_mosaic_context[
                            int(prefix_chunks)
                        ] = (
                            cache_view_without_mosaic_context,
                            list(
                                cache_view_without_mosaic_context[0].get(
                                    "positions",
                                    cache_view_without_mosaic_context[0].get(
                                        "frames", []
                                    ),
                                )
                            ),
                            list(
                                cache_view_without_mosaic_context[0].get(
                                    "frames", []
                                )
                            ),
                        )
                    else:
                        prediction_cache_views_without_mosaic_context[
                            int(prefix_chunks)
                        ] = prediction_cache_views[int(prefix_chunks)]

            gen = torch.Generator().manual_seed(
                int(args.validation_seed) + batch_idx * 1000 + i
            )
            latents = torch.randn(
                1, channels, cs, H_lat, W_lat, generator=gen
            ).to(device=device, dtype=dtype)
            from tqdm import tqdm as _tqdm

            def _kv_pred(latents_in, tval, *, step_index=None):
                tval = float(tval)
                use_mosaic_context_step = bool(
                    mosaic_context_cache is not None
                    and mosaic_context_step_mask[int(step_index)]
                )

                def _prediction_cache(with_mosaic_context):
                    nonlocal hiar_prediction_cache_step_index
                    nonlocal hiar_prediction_cache_step
                    if prefix_noise_mode == "hiar_sde":
                        if step_index is None:
                            raise RuntimeError(
                                "step_index is required for HiAR prefix corruption."
                            )
                        if hiar_prediction_cache_step_index != int(step_index):
                            hiar_prediction_cache_step = _build_hiar_prediction_cache(
                                int(step_index)
                            )
                            hiar_prediction_cache_step_index = int(step_index)
                        return hiar_prediction_cache_step
                    if with_mosaic_context:
                        cache_bundle = (
                            read_cache,
                            read_cache_positions,
                            read_cache_frames,
                        )
                    else:
                        cache_bundle = (
                            read_cache_without_mosaic_context,
                            read_cache_without_mosaic_context_positions,
                            read_cache_without_mosaic_context_frames,
                        )
                    if prefix_chunks_by_step is None:
                        return cache_bundle
                    if step_index is None:
                        raise RuntimeError(
                            "step_index is required for timestep-dependent prefix "
                            "validation."
                        )
                    prefix_chunks = int(prefix_chunks_by_step[int(step_index)])
                    cache_views = (
                        prediction_cache_views
                        if with_mosaic_context
                        else prediction_cache_views_without_mosaic_context
                    )
                    return cache_views[prefix_chunks]

                cache_with_context = _prediction_cache(True)
                cache_without_context = _prediction_cache(False)

                mosaic_repeat = (
                    int(mosaic_repeat_by_step[int(step_index)])
                    if mosaic_repeat_by_step is not None
                    else 1
                )
                mosaic_latent_step = m_i
                mosaic_positions_step = m_pos
                mosaic_frames_step = m_fr
                if mosaic_latent_step is not None and mosaic_repeat > 1:
                    mosaic_latent_step = mosaic_latent_step.repeat(
                        1, 1, mosaic_repeat, 1, 1
                    )
                    mosaic_positions_step = list(m_pos) * mosaic_repeat
                    mosaic_frames_step = list(m_fr) * mosaic_repeat

                def _predict_with_mosaic(
                    mosaic_latent,
                    mosaic_positions,
                    mosaic_frames,
                    cache_bundle,
                ):
                    pred_cache, pred_cache_positions, pred_cache_frames = cache_bundle
                    mosaic_frame_count = (
                        int(mosaic_latent.shape[2])
                        if mosaic_latent is not None
                        else 0
                    )
                    ts = torch.cat(
                        [
                            torch.zeros(
                                mosaic_frame_count,
                                device=device,
                                dtype=dtype,
                            ),
                            torch.full((cs,), tval, device=device, dtype=dtype),
                        ]
                    )
                    return model_fn_causal_kv(
                        dit,
                        latents_chunk=latents_in,
                        timestep_frames=ts,
                        context=context,
                        camera_info=camera_info,
                        cur_positions=cur_positions,
                        cur_frames=cur_camera_frames,
                        caches=pred_cache,
                        mosaic_latent=mosaic_latent,
                        mosaic_positions=(
                            mosaic_positions if mosaic_latent is not None else None
                        ),
                        mosaic_frames=(
                            mosaic_frames if mosaic_latent is not None else None
                        ),
                        cache_positions=pred_cache_positions,
                        cache_frames=pred_cache_frames,
                        cache_read_chunk_id=int(i),
                        write_cache=False,
                    )

                def _predict_with_context_guidance(
                    mosaic_latent,
                    mosaic_positions,
                    mosaic_frames,
                ):
                    if not use_mosaic_context_step:
                        return _predict_with_mosaic(
                            mosaic_latent,
                            mosaic_positions,
                            mosaic_frames,
                            cache_without_context,
                        )
                    pred_with_context = _predict_with_mosaic(
                        mosaic_latent,
                        mosaic_positions,
                        mosaic_frames,
                        cache_with_context,
                    )
                    if mosaic_context_guidance_scale == 1.0:
                        return pred_with_context
                    pred_without_context = _predict_with_mosaic(
                        mosaic_latent,
                        mosaic_positions,
                        mosaic_frames,
                        cache_without_context,
                    )
                    delta = pred_with_context - pred_without_context
                    denom = pred_with_context.float().norm().clamp_min(1e-12)
                    mosaic_context_guidance_stats["chunks"].append(
                        {
                            "chunk_idx": int(i),
                            "step_index": int(step_index),
                            "timestep": float(tval),
                            "mosaic_present": bool(mosaic_latent is not None),
                            "relative_prediction_delta": float(
                                delta.float().norm().div(denom).item()
                            ),
                        }
                    )
                    return (
                        pred_without_context
                        + mosaic_context_guidance_scale * delta
                    )

                pred = _predict_with_context_guidance(
                    mosaic_latent_step,
                    mosaic_positions_step,
                    mosaic_frames_step,
                )
                if mosaic_guidance_scale != 1.0 and m_i is not None:
                    pred_without_mosaic = _predict_with_context_guidance(
                        None, None, None
                    )
                    delta = pred - pred_without_mosaic
                    denom = pred.float().norm().clamp_min(1e-12)
                    mosaic_guidance_stats["chunks"].append(
                        {
                            "chunk_idx": int(i),
                            "step_index": int(step_index),
                            "timestep": float(tval),
                            "mosaic_token_repeat": int(mosaic_repeat),
                            "relative_prediction_delta": float(
                                delta.float().norm().div(denom).item()
                            ),
                        }
                    )
                    pred = pred_without_mosaic + mosaic_guidance_scale * delta
                if cfg_scale == 1.0:
                    return pred
                noise_nega, noise_posi = pred.chunk(2, dim=0)
                return noise_nega + cfg_scale * (noise_posi - noise_nega)

            steps = scheduler.timesteps
            renoise_gen = torch.Generator().manual_seed(
                int(args.validation_seed) + batch_idx * 1000 + i + 50000
            )
            cur = latents
            for j, t in enumerate(
                _tqdm(
                    steps,
                    desc=f"[causal-dmd] chunk {i + 1}/{total_chunks}",
                    leave=False,
                )
            ):
                v = _kv_pred(cur, float(t), step_index=j)
                if j < len(steps) - 1:
                    new_noise = torch.randn(
                        cur.shape,
                        generator=renoise_gen,
                    ).to(device=device, dtype=dtype)
                    cur = _causal_dmd_validation_transition(
                        scheduler,
                        v,
                        t,
                        cur,
                        next_timestep=steps[j + 1],
                        renoise=new_noise,
                    )
                else:
                    cur = _causal_dmd_validation_transition(
                        scheduler,
                        v,
                        t,
                        cur,
                    )
            latents = cur
            # Cache-fill uses the final denoising step's context visibility. This
            # prevents a high-noise-only diagnostic context from leaking into the
            # clean prefix KV persisted for the next chunk.
            cache_fill_uses_mosaic_context = bool(
                mosaic_context_cache is not None
                and mosaic_context_step_mask
                and mosaic_context_step_mask[-1]
            )
            if cache_fill_uses_mosaic_context:
                cache_fill_read_cache = read_cache
                cache_fill_positions = read_cache_positions
                cache_fill_frames = read_cache_frames
            else:
                cache_fill_read_cache = read_cache_without_mosaic_context
                cache_fill_positions = read_cache_without_mosaic_context_positions
                cache_fill_frames = read_cache_without_mosaic_context_frames
            # cache-fill -> append CUR to rolling_cache -> SF-style local trim.
            model_fn_causal_kv(
                dit,
                latents_chunk=latents,
                timestep_frames=torch.zeros(mfc + cs, device=device, dtype=dtype),
                context=context,
                camera_info=camera_info,
                cur_positions=cur_positions,
                cur_frames=cur_camera_frames,
                caches=cache_fill_read_cache,
                mosaic_latent=m_i,
                mosaic_positions=m_pos,
                mosaic_frames=m_fr,
                cache_positions=cache_fill_positions,
                cache_frames=cache_fill_frames,
                cache_read_chunk_id=int(i),
                write_cache=True,
            )
        all_clean.append(latents.detach().cpu())
        all_mosaic.append(
            m_i.detach().cpu()
            if m_i is not None
            else torch.zeros_like(latents).cpu()
        )
        current_chunk_cache = _causal_kv_tail_cache_frames(
            cache_fill_read_cache,
            cs,
            context=f"[causal-kv] extract generated chunk cache chunk={i}",
        )
        rolling_cache = _causal_kv_concat_caches(rolling_cache, current_chunk_cache)
        rolling_cache = _causal_kv_trim_rolling_window(
            rolling_cache,
            frames_per_chunk=cs,
            window_chunks=ccc,
            fixed_initial_anchor=fixed_initial_anchor,
        )
        if rolling_latents is not None:
            rolling_latents = torch.cat(
                [rolling_latents, latents.detach()], dim=2
            ).contiguous()
            rolling_latents = _causal_kv_trim_rolling_latents(
                rolling_latents,
                frames_per_chunk=cs,
                window_chunks=ccc,
                fixed_initial_anchor=fixed_initial_anchor,
            )
            if int(rolling_latents.shape[2]) != _causal_kv_frame_count(
                rolling_cache
            ):
                raise RuntimeError(
                    "[causal-kv] rolling latent/cache provenance diverged after "
                    f"chunk={i}: latent_frames={int(rolling_latents.shape[2])}, "
                    f"cache_frames={_causal_kv_frame_count(rolling_cache)}."
                )

        has_future_query = int(i) + 1 < int(total_chunks)
        dynamic_context_accepts_generated = (
            _validation_dynamic_context_accepts_generated(validation_memory_mode)
            and not force_context_original_anchor
        )
        dynamic_context_needs_decode = bool(
            dynamic_context_pool is not None
            and dynamic_context_accepts_generated
            and _generated_memory_chunk_has_future_consumer(
                i,
                total_chunks,
                generated_context_publish_interval,
            )
        )
        dynamic_context_needs_gt_prefix = bool(
            has_future_query
            and dynamic_context_pool is not None
            and validation_gt_memory
            and not force_context_original_anchor
        )
        generated_memory_has_consumer = bool(
            use_mosaic
            and not validation_empty_memory
            and not validation_gt_memory
            and not validation_c0_only_memory
            and not memory_disabled_after_c0_failure
            and _generated_memory_chunk_has_future_consumer(
                i,
                total_chunks,
                generated_memory_publish_interval,
            )
        )
        gt_memory_has_consumer = bool(
            use_mosaic
            and validation_gt_memory
            and has_future_query
            and not memory_disabled_after_c0_failure
        )
        needs_generated_decode = bool(
            dynamic_context_needs_decode or generated_memory_has_consumer
        )
        generated_frames = None
        if needs_generated_decode:
            current_chunk_cpu = latents.detach().to(device="cpu", dtype=latents.dtype)
            if i == 0 and t2v_no_ref:
                decode_prefix_latents = current_chunk_cpu
            elif i == 0:
                decode_prefix_latents = torch.cat(
                    [initial_c0.detach().to(device="cpu", dtype=latents.dtype), current_chunk_cpu],
                    dim=2,
                )
            else:
                prev_anchor = all_clean[-2][:, :, -1:, :, :].to(device="cpu", dtype=latents.dtype)
                decode_prefix_latents = torch.cat([prev_anchor, current_chunk_cpu], dim=2)
            generated_frames = _decode_generated_chunk_from_prefix_latents(
                pipe,
                decode_prefix_latents,
                noisy_start=0,
                noisy_count=cs,
                device=device,
                tiled=vae_decode_tiled,
                context="[causal-kv] generated chunk decode",
            )
            decode_prefix_latents = None
            current_chunk_cpu = None

        gt_prefix_frames = None
        gt_prefix_frame_indices = None
        if dynamic_context_needs_gt_prefix or gt_memory_has_consumer:
            video_path = validation_info.get("video_path")
            if not video_path:
                raise RuntimeError(
                    "GT validation prefix registration requires "
                    "data['info']['video_path']."
                )
            rgb_start = int(i) * cs * 4
            gt_prefix_frame_indices = _validation_noisy_frame_indices(
                validation_info,
                rgb_start,
                cs * 4,
            )
            gt_prefix_frames = _read_video_gt_frames(
                video_path,
                gt_prefix_frame_indices,
                H_img,
                W_img,
                strict=True,
            )

        context_register_frames = generated_frames
        context_register_source = "generated"
        if dynamic_context_needs_gt_prefix:
            context_register_frames = gt_prefix_frames
            context_register_source = "gt_prefix"

        if (
            dynamic_context_pool is not None
            and context_register_frames is not None
            and (
                dynamic_context_accepts_generated
                or dynamic_context_needs_gt_prefix
            )
        ):
            rgb_start = i * cs * 4
            context_publish_result = dynamic_context_pool.add_generated_chunk(
                pipe=pipe,
                generated_frames=context_register_frames,
                chunk_idx=i,
                frames_per_chunk=cs,
                rgb_start=rgb_start,
                full_noisy_extr=full_noisy_extr.detach().float().cpu().numpy(),
                full_noisy_intr=full_noisy_intr.detach().float().cpu().numpy(),
                device=device,
                tiled=vae_decode_tiled,
                source=context_register_source,
            )
            published_context_chunks = list(
                (context_publish_result or {}).get(
                    "published_context_chunks", []
                )
            )
            if published_context_chunks:
                validation_rope_stats[
                    "generated_context_published_chunks"
                ].extend(int(v) for v in published_context_chunks)
                if generated_context_publish_interval > 1:
                    print(
                        "[causal-kv][context] published generated context chunks="
                        f"{published_context_chunks}",
                        flush=True,
                    )

        if (
            use_mosaic
            and not validation_empty_memory
            and not validation_c0_only_memory
            and (generated_memory_has_consumer or gt_memory_has_consumer)
        ):
            rgb_start = i * cs * 4
            if validation_gt_memory:
                register_frames = gt_prefix_frames
                register_frame_indices = gt_prefix_frame_indices
                register_source = "gt_prefix"
                if register_frames is None or register_frame_indices is None:
                    raise RuntimeError(
                        "GT-prefix memory publication lost its original RGB/frame "
                        f"indices for chunk={i}."
                    )
                register_depths = _read_validation_gt_depth_frames(
                    data["info"],
                    register_frame_indices,
                )
            else:
                register_frames = generated_frames
                register_source = "generated"
                if register_frames is None:
                    raise RuntimeError(
                        "generated-memory publication requires decoded generated "
                        f"frames for chunk={i}."
                    )
                register_depths = None
                register_frame_indices = _validation_noisy_frame_indices(
                    data["info"],
                    rgb_start,
                    int(register_frames.shape[0]),
                )
                if register_rgb_source == "gt":
                    register_frames = _read_video_gt_frames(
                        data["info"]["video_path"],
                        register_frame_indices,
                        H_img,
                        W_img,
                        strict=True,
                    )
                if register_depth_source == "dataset":
                    register_depths = _read_validation_gt_depth_frames(
                        data["info"],
                        register_frame_indices,
                    )
            rgb_end = rgb_start + int(register_frames.shape[0])
            if rgb_end > int(full_noisy_extr.shape[0]) or rgb_end > int(full_noisy_intr.shape[0]):
                raise RuntimeError(
                    "[causal-kv] online-memory camera slice "
                    f"[{rgb_start}:{rgb_end}] exceeds camera lengths "
                    f"extr={int(full_noisy_extr.shape[0])} intr={int(full_noisy_intr.shape[0])}."
                )
            new_query_latents = _encode_frames_per_frame(
                pipe,
                register_frames,
                device,
                tiled=vae_decode_tiled,
            )
            pending_generated_memory.append(
                {
                    "chunk_idx": int(i),
                    "source": str(register_source),
                    "frames": register_frames,
                    "extrinsics": full_noisy_extr[rgb_start:rgb_end].detach().float().cpu().numpy(),
                    "intrinsics": full_noisy_intr[rgb_start:rgb_end].detach().float().cpu().numpy(),
                    "depths": register_depths,
                    "latents": new_query_latents,
                    "rgb_end": int(rgb_end),
                }
            )
            _publish_pending_generated_memory()

    if dynamic_context_pool is not None:
        validation_rope_stats["generated_context_pending_chunks_at_end"] = list(
            dynamic_context_pool.pending_generated_chunk_indices
        )
        validation_rope_stats["generated_context_visible_entry_count"] = int(
            len(dynamic_context_pool.generated_entries)
        )

    # A residual group at the end has no consumer, so skip its VAE/DA3 work.
    pending_generated_memory.clear()
    if t2v_no_ref:
        history_latents = torch.cat(all_clean, dim=2)
        mosaic_latent_history = torch.cat(all_mosaic, dim=2)
    else:
        history_latents = torch.cat([initial_c0.detach().cpu()] + all_clean, dim=2)
        mosaic_latent_history = torch.cat(
            [initial_c0.detach().cpu()] + all_mosaic, dim=2
        )
    scheduler.set_timesteps(1000, training=False)
    return history_latents, mosaic_latent_history

def _normalize_causal_validation_memory_mode(value):
    mode = str(value or "c0_plus_generated").strip().lower().replace("-", "_")
    if mode in {"generated", "stage3_dmd_c0_plus_generated"}:
        return "c0_plus_generated"
    if mode in {"gt", "gt_memory", "ground_truth", "stage3_dmd_gt"}:
        return "gt"
    if mode in {
        "c0",
        "c0_only",
        "c0_memory",
        "ref",
        "ref_only",
        "reference",
        "reference_only",
        "stage3_dmd_c0_only",
    }:
        return "c0_only"
    if mode in {"empty", "none", "no_memory", "without_memory", "empty_memory"}:
        return "empty"
    if mode != "c0_plus_generated":
        raise ValueError(
            "causal_validation_memory_mode must be 'c0_plus_generated', 'gt', "
            f"'c0_only', or 'empty', got {value!r}."
        )
    return mode

def _effective_validation_cfg_scale(args, *, dmd_validation):
    requested_scale = float(getattr(args, "validation_cfg_scale", 1.0))
    if not math.isfinite(requested_scale) or requested_scale <= 0.0:
        raise ValueError(
            "validation_cfg_scale must be finite and > 0, "
            f"got {requested_scale}."
        )
    if dmd_validation and not bool(
        getattr(args, "causal_dmd_validation_use_cfg", False)
    ):
        return 1.0
    return requested_scale

def _validation_memory_mode_uses_runtime_pool(value):
    """Whether validation must build/query a live FrustumHandler pool."""

    return _normalize_causal_validation_memory_mode(value) != "empty"

def _validation_memory_mode_uses_online_depth(value):
    """Whether the live pool needs a registration depth estimator.

    GT mode registers dataset RGB/depth pairs, while C0/generated modes retain
    the configured online registration estimator contract.
    """

    mode = _normalize_causal_validation_memory_mode(value)
    return mode not in {"empty", "gt"}

def _normalize_causal_memory_bootstrap_mode(value):
    mode = str(value or "c0_only").strip().lower().replace("-", "_")
    if mode in {"c0", "c0_only", "ref", "reference", "anchor", "anchor_only"}:
        return "c0_only"
    if mode in {"history", "history_plus_c0", "legacy", "old"}:
        return "history"
    if mode in {"mixed", "mix", "random", "rand", "sample"}:
        return "mixed"
    raise ValueError(
        "causal_memory_bootstrap_mode must be 'c0_only', 'history', or 'mixed', "
        f"got {value!r}."
    )
