from functools import partial
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


# Camera tiling layouts (consumed by ``_prepare_apply_fns``).
#
# * ``"full"`` -- legacy behaviour: tile every sub-frame matrix across the
#   WHOLE head_dim via ``_apply_tiled_projmat``. Overlaps x/y-RoPE.
# * ``"sf13"`` -- contiguous 2-camera layout for the ``prope_disable_t_rope``
#   path: subsample the 4 VAE sub-frame cameras to the 2nd and 4th (0-based
#   ``{1, 3}``) and write each as a contiguous ``dims_per_cam``-wide block at
#   the FRONT of head_dim (sub-frame 1 -> ``[0:16)``, sub-frame 3 ->
#   ``[16:32)``), each block = ``dims_per_cam // 4`` four-vectors all getting
#   that matrix. The tail is left identity. Under ``prope_disable_t_rope`` the
#   front of head_dim is rope-free, so the camera lives in a clean subspace
#   disjoint from x/y-RoPE (no conjugation entanglement).
PROPE_CAMERA_LAYOUTS = {
    "full": None,
    "sf13": {"subframes": (1, 3), "dims_per_cam": 16},
}


def prope_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    viewmats,
    view_change_positions=None,
    rope3_each=None,
    camera_layout="full",
    scale=None,
    **kwargs,
) -> torch.Tensor:
    """Apply PRoPE transforms around torch SDPA.

    `viewmats` is a tuple of `(P, P_T, P_inv)` with shape
    `(batch, cameras, 4, 4, 4)`. The old `prope_diagonal_mask` and
    `attn_mask` hooks are intentionally not part of this interface.
    `camera_layout` selects the head_dim tiling (see ``PROPE_CAMERA_LAYOUTS``).
    """
    batch, num_heads, seqlen, head_dim = q.shape
    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        view_change_positions=view_change_positions,
        rope3_each=rope3_each,
        camera_layout=camera_layout,
    )
    out = F.scaled_dot_product_attention(
        query=apply_fn_q(q),
        key=apply_fn_kv(k),
        value=apply_fn_kv(v),
        scale=scale,
        **kwargs,
    )
    out = apply_fn_o(out)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


def _prepare_apply_fns(
    head_dim: int,
    viewmats,
    view_change_positions=None,
    rope3_each=None,
    camera_layout="full",
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    P, P_T, P_inv = viewmats
    if camera_layout not in PROPE_CAMERA_LAYOUTS:
        raise ValueError(
            f"unknown prope camera_layout {camera_layout!r}; expected one of "
            f"{sorted(PROPE_CAMERA_LAYOUTS)}."
        )
    spec = PROPE_CAMERA_LAYOUTS[camera_layout]
    if spec is None:
        transforms_q = [(partial(_apply_tiled_projmat, matrix=P_T), head_dim)]
        transforms_kv = [(partial(_apply_tiled_projmat, matrix=P_inv), head_dim)]
        transforms_o = [(partial(_apply_tiled_projmat, matrix=P), head_dim)]
        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
        apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
        apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    else:
        sub, dpc = spec["subframes"], spec["dims_per_cam"]
        apply_fn_q = partial(
            _apply_contiguous_projmat, matrix=P_T, subframes=sub, dims_per_cam=dpc
        )
        apply_fn_kv = partial(
            _apply_contiguous_projmat, matrix=P_inv, subframes=sub, dims_per_cam=dpc
        )
        apply_fn_o = partial(
            _apply_contiguous_projmat, matrix=P, subframes=sub, dims_per_cam=dpc
        )

    if view_change_positions is not None:
        each = rope3_each if rope3_each is not None else ROPE3_EACH
        coeffs3 = make_rope3_coeffs(view_change_positions, each, device=P.device)
        proj_q, proj_kv, proj_o = apply_fn_q, apply_fn_kv, apply_fn_o
        apply_fn_q = lambda x: _apply_rope3_slice(
            proj_q(x), coeffs3, each, inverse=False
        )
        apply_fn_kv = lambda x: _apply_rope3_slice(
            proj_kv(x), coeffs3, each, inverse=False
        )
        apply_fn_o = lambda x: proj_o(
            _apply_rope3_slice(x, coeffs3, each, inverse=True)
        )

    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    feat_pc = rearrange(
        feats,
        "b h (s t) (p q j r) -> b h s t r p q j",
        s=matrix.shape[1],
        p=4,
        q=4,
        j=4,
    )
    out_pc = torch.einsum("bscij,bhstrpcj->bhstrpci", matrix, feat_pc)
    return rearrange(out_pc, "b h s t r p q j -> b h (s t) (p q j r)")


def _apply_contiguous_projmat(
    feats: torch.Tensor,
    matrix: torch.Tensor,
    subframes: Tuple[int, ...],
    dims_per_cam: int = 16,
) -> torch.Tensor:
    """Contiguous per-sub-frame PRoPE tiling (see ``PROPE_CAMERA_LAYOUTS``).

    ``matrix`` is ``(batch, frames, sub_frames, 4, 4)``. Each selected
    sub-frame's matrix is written into its own contiguous block at the front
    of head_dim: sub-frame ``subframes[k]`` -> dims
    ``[k*dims_per_cam : (k+1)*dims_per_cam)``, holding ``dims_per_cam // 4``
    four-vectors that all get that matrix. Tail dims pass through unchanged.
    """
    if dims_per_cam % 4 != 0:
        raise ValueError(f"dims_per_cam must be a multiple of 4, got {dims_per_cam}.")
    n = len(subframes)
    cam_dims = n * dims_per_cam
    head_dim = feats.shape[-1]
    if cam_dims > head_dim:
        raise ValueError(
            f"contiguous camera needs {cam_dims} dims but head_dim={head_dim}."
        )
    num_sub = matrix.shape[2]
    if max(subframes) >= num_sub:
        raise ValueError(
            f"sub-frame index {max(subframes)} out of range; matrix has "
            f"{num_sub} sub-frames."
        )
    fourvec = dims_per_cam // 4
    cam, rest = feats[..., :cam_dims], feats[..., cam_dims:]
    cam = rearrange(
        cam,
        "b h (s t) (n f j) -> b h s t n f j",
        s=matrix.shape[1],
        n=n,
        f=fourvec,
        j=4,
    )
    idx = torch.as_tensor(subframes, device=matrix.device, dtype=torch.long)
    m_sel = matrix.index_select(2, idx)  # (b, s, n, 4, 4)
    out = torch.einsum("bsnij,bhstnfj->bhstnfi", m_sel, cam)
    out = rearrange(out, "b h s t n f j -> b h (s t) (n f j)")
    return torch.cat([out, rest], dim=-1)


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat([func(x_block) for func, x_block in zip(funcs, x_blocks)], dim=-1)
    assert out.shape == feats.shape
    return out


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    assert transforms.shape[-2:] == (4, 4)
    r_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", r_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def lift_k(intrinsics: torch.Tensor) -> torch.Tensor:
    assert intrinsics.shape[-2:] == (3, 3)
    out = torch.zeros(
        intrinsics.shape[:-2] + (4, 4),
        device=intrinsics.device,
        dtype=intrinsics.dtype,
    )
    out[..., :3, :3] = intrinsics
    out[..., 3, 3] = 1.0
    return out


ROPE3_EACH = 8


def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim, device):
    if positions.dim() == 1:
        positions = positions[None]
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (-torch.arange(num_freqs, device=device, dtype=positions.dtype) / num_freqs)
    )
    angles = positions[:, None, :, None] * freqs[None, None, None, :]
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(feats, coeffs, inverse=False):
    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        if feats.shape[2] % cos.shape[2] != 0:
            raise ValueError(
                "view_change_positions length must divide feature sequence length, "
                f"got coeffs={cos.shape[2]} feats={feats.shape[2]}."
            )
        n = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n, 1)
        sin = sin.repeat(1, 1, n, 1)
    half = feats.shape[-1] // 2
    x_in, y_in = feats[..., :half], feats[..., half:]
    if not inverse:
        return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
    return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)


def make_rope3_coeffs(view_change_positions, rope_each, device, freq_base=100.0):
    p = view_change_positions.to(device)
    pos = torch.stack(
        [torch.log(p[..., 0].clamp_min(1e-6)), p[..., 1], p[..., 2]], dim=-1
    )
    return [
        _rope_precompute_coeffs(pos[..., i], freq_base, 1.0, rope_each, device)
        for i in range(3)
    ]


def _apply_rope3_slice(feats, coeffs3, rope_each, inverse=False):
    if coeffs3 is None:
        return feats
    n = 3 * rope_each
    if feats.shape[-1] < n:
        raise ValueError(
            f"head_dim={feats.shape[-1]} is too small for 3VAL RoPE slice {n}."
        )
    head, tail = feats[..., : feats.shape[-1] - n], feats[..., feats.shape[-1] - n :]
    blocks = list(tail.split(rope_each, dim=-1))
    rolled = [
        _rope_apply_coeffs(blocks[i], coeffs3[i], inverse=inverse) for i in range(3)
    ]
    return torch.cat([head] + rolled, dim=-1)


def invert_k(intrinsics: torch.Tensor) -> torch.Tensor:
    assert intrinsics.shape[-2:] == (3, 3)
    out = torch.zeros_like(intrinsics)
    out[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    out[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    out[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    out[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out
