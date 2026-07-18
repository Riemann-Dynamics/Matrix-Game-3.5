import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from .prope_attention import prope_dot_product_attention
from ..core.gradient import gradient_checkpoint_forward
from .wantodance import WanToDanceRotaryEmbedding, WanToDanceMusicEncoderLayer

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode=False,
    attn_mask=None,
):
    if attn_mask is not None or compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_with_step(
    dim: int,
    end: int = 1024,
    theta: float = 10000.0,
    step: int = 2,
):
    """
    Fractional-step RoPE precompute (higher spatial resolution than integer grid).

    Produces a table over positions 0, 1/step, 2/step, …, (end*step-1)/step
    so the returned length is ``end * step``.
    ``freqs_cis[step * i]`` equals ``precompute_freqs_cis(dim, end, theta)[i]``.
    """
    idx = torch.arange(0, dim, 2)[: (dim // 2)].double()
    freqs = 1.0 / (theta ** (idx / dim))
    t = torch.arange(end * step).double() / float(step)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    ql_h_freqs_cis = precompute_freqs_cis_with_step(dim // 3, end, theta, step=16)
    ql_w_freqs_cis = precompute_freqs_cis_with_step(dim // 3, end, theta, step=16)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis, ql_h_freqs_cis, ql_w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:
            return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attn_mask=None):
        x = flash_attention(
            q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask
        )
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        use_prope: bool = False,
        prope_disable_native_rope: bool = False,
        prope_disable_t_rope: bool = False,
        prope_camera_layout: str = "full",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_prope = use_prope
        self.prope_disable_native_rope = prope_disable_native_rope
        self.prope_disable_t_rope = prope_disable_t_rope
        # Camera tiling for PRoPE attention (see PROPE_CAMERA_LAYOUTS):
        # "full" = legacy tiling over all head_dim; "sf13" = contiguous
        # 2-camera (sub-frames {1,3}) layout in the rope-free t-band.
        self.prope_camera_layout = prope_camera_layout
        # Complex-pair count of the temporal RoPE band; mirrors the
        # head_dim split in `precompute_freqs_cis_3d` (t gets
        # head_dim - 2*(head_dim//3) real dims, x/y get head_dim//3 each).
        self.rope_t_pairs = (self.head_dim - 2 * (self.head_dim // 3)) // 2

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, attn_mask=None, camera_info=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        use_prope_attention = self.use_prope and camera_info is not None
        if not (use_prope_attention and self.prope_disable_native_rope):
            rope_freqs = freqs
            if use_prope_attention and self.prope_disable_t_rope:
                # PRoPE blocks keep x/y RoPE (intra-frame spatial structure)
                # but drop the temporal band: cross-frame addressing is
                # delegated to the camera matrices, and the t-band channels
                # become a RoPE-phase-free subspace for them. 1+0j is the
                # identity rotation.
                rope_freqs = freqs.clone()
                rope_freqs[..., : self.rope_t_pairs] = 1
            q = rope_apply(q, rope_freqs, self.num_heads)
            k = rope_apply(k, rope_freqs, self.num_heads)
        if use_prope_attention:
            # print("Using PROPE")
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
            view_change_positions = camera_info[2] if len(camera_info) > 2 else None
            x = prope_dot_product_attention(
                q,
                k,
                v,
                viewmats=camera_info[1],
                view_change_positions=view_change_positions,
                camera_layout=getattr(self, "prope_camera_layout", "full"),
                attn_mask=attn_mask,
            )
            x = rearrange(x, "b n s d -> b s (n d)")
        else:
            x = self.attn(q, k, v, attn_mask=attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        use_prope: bool = False,
        prope_disable_native_rope: bool = False,
        prope_disable_t_rope: bool = False,
        prope_camera_layout: str = "full",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_prope = use_prope
        self.prope_disable_native_rope = prope_disable_native_rope
        self.prope_disable_t_rope = prope_disable_t_rope
        self.prope_camera_layout = prope_camera_layout

        self.self_attn = SelfAttention(
            dim,
            num_heads,
            eps,
            use_prope=use_prope,
            prope_disable_native_rope=prope_disable_native_rope,
            prope_disable_t_rope=prope_disable_t_rope,
            prope_camera_layout=prope_camera_layout,
        )
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        attn_mask=None,
        camera_info=None,
        cross_attn_keep_mask=None,
    ):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        camera_info = camera_info if self.use_prope else None
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(
                input_x, freqs, attn_mask=attn_mask, camera_info=camera_info
            ),
        )
        ca = self.cross_attn(self.norm3(x), context)
        if cross_attn_keep_mask is not None:
            ca = ca * cross_attn_keep_mask.to(device=ca.device, dtype=ca.dtype).view(
                1, -1, 1
            )
        x = x + ca
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
                + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
            ).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


def wantodance_torch_dfs(model: nn.Module, parent_name="root"):
    module_names, modules = [], []
    current_name = parent_name if parent_name else "root"
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f"{parent_name}.{name}"
        else:
            child_name = name
        child_modules, child_names = wantodance_torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class WanToDanceInjector(nn.Module):
    def __init__(
        self,
        all_modules,
        all_modules_names,
        dim=2048,
        num_heads=32,
        inject_layer=[0, 27],
    ):
        super().__init__()
        self.injected_block_id = {}
        injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f"root.transformer_blocks.{inject_id}" == mod_name:
                        self.injected_block_id[inject_id] = injector_id
                        injector_id += 1

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(
                    dim,
                    elementwise_affine=False,
                    eps=1e-6,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(
                    dim,
                    elementwise_affine=False,
                    eps=1e-6,
                )
                for _ in range(injector_id)
            ]
        )


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
        use_prope: bool = False,
        prope_disable_native_rope: bool = False,
        prope_disable_t_rope: bool = False,
        prope_camera_layout: str = "full",
        clean_latent_noise_enabled: bool = False,
        clean_latent_noise_prob: float = 0.2,
        clean_latent_noise_magnitude: float = 0.03,
        mosaic_latent_noise_enabled: bool = False,
        mosaic_latent_noise_prob: float = 0.2,
        mosaic_latent_noise_magnitude: float = 0.03,
        context_latent_noise_enabled: bool = False,
        context_latent_noise_prob: float = 0.2,
        context_latent_noise_magnitude: float = 0.03,
        subject_ref_memory_enabled: bool = False,
        subject_ref_memory_max_refs: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.use_prope = use_prope
        self.prope_disable_native_rope = prope_disable_native_rope
        self.prope_disable_t_rope = prope_disable_t_rope
        self.prope_camera_layout = prope_camera_layout
        # Training-only clean-latent noise augmentation. When enabled and the
        # model is in training mode, the pipeline perturbs ONLY the most
        # recent clean latent frame (the last frame of `first_frame_latents`;
        # mosaic latents and older clean-history/context latents are left
        # untouched) with probability `clean_latent_noise_prob`, adding
        # Gaussian noise N(0, magnitude^2). Read by
        # `model_fn_wan_video` via getattr, so checkpoints without these
        # attributes default to disabled.
        self.clean_latent_noise_enabled = bool(clean_latent_noise_enabled)
        self.clean_latent_noise_prob = float(clean_latent_noise_prob)
        self.clean_latent_noise_magnitude = float(clean_latent_noise_magnitude)
        # Training-only mosaic-latent noise augmentation (parallel to the
        # clean-latent noise above). Perturbs only the valid (non-hole) mosaic
        # latents while the model is in training mode; read by
        # ``model_fn_wan_video`` via getattr, so checkpoints without these
        # attributes default to disabled.
        self.mosaic_latent_noise_enabled = bool(mosaic_latent_noise_enabled)
        self.mosaic_latent_noise_prob = float(mosaic_latent_noise_prob)
        self.mosaic_latent_noise_magnitude = float(mosaic_latent_noise_magnitude)
        # Training-only context-latent noise augmentation (parallel to the
        # clean-latent noise above; targets the context latents, i.e. every
        # clean latent before the anchor). Read by ``model_fn_wan_video`` via
        # getattr, so checkpoints without these attrs default to disabled.
        self.context_latent_noise_enabled = bool(context_latent_noise_enabled)
        self.context_latent_noise_prob = float(context_latent_noise_prob)
        self.context_latent_noise_magnitude = float(context_latent_noise_magnitude)
        self.subject_ref_memory_enabled = False

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    has_image_input,
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    use_prope=use_prope,
                    prope_disable_native_rope=prope_disable_native_rope,
                    prope_disable_t_rope=prope_disable_t_rope,
                    prope_camera_layout=prope_camera_layout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        if wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            end = int(22350 / 8 + 0.5)  # 149f * 30fps * 5s = 22350
            self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(
                in_dim_control_adapter,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
            )
        else:
            self.control_adapter = None

        self.prepare_wantodance(
            in_dim,
            dim,
            num_heads,
            has_image_pos_emb,
            out_dim,
            patch_size,
            eps,
            wantodance_enable_music_inject,
            wantodance_music_inject_layers,
            wantodance_enable_refimage,
            wantodance_enable_refface,
            wantodance_enable_global,
            wantodance_enable_dynamicfps,
            wantodance_enable_unimodel,
        )
        if subject_ref_memory_enabled:
            self.enable_subject_ref_memory(subject_ref_memory_max_refs)

    def enable_subject_ref_memory(self, max_refs: int = 2):
        max_refs = max(1, int(max_refs))
        if self.subject_ref_memory_enabled:
            if int(self.subject_ref_index_embedding.shape[0]) != max_refs:
                raise ValueError(
                    "subject_ref_memory is already enabled with "
                    f"{int(self.subject_ref_index_embedding.shape[0])} refs, "
                    f"cannot re-enable with {max_refs} refs."
                )
            return
        self.subject_ref_memory_enabled = True
        self.subject_ref_memory_max_refs = max_refs
        self.subject_ref_memory_local_pos_size = 64
        ref_param = next(self.parameters(), None)
        device = ref_param.device if ref_param is not None else None
        dtype = ref_param.dtype if ref_param is not None else None
        self.subject_ref_index_embedding = nn.Parameter(
            torch.zeros(max_refs, self.dim, device=device, dtype=dtype)
        )
        self.subject_ref_type_embedding = nn.Parameter(
            torch.zeros(1, self.dim, device=device, dtype=dtype)
        )
        self.subject_ref_local_h_embedding = nn.Parameter(
            torch.zeros(
                self.subject_ref_memory_local_pos_size,
                self.dim,
                device=device,
                dtype=dtype,
            )
        )
        self.subject_ref_local_w_embedding = nn.Parameter(
            torch.zeros(
                self.subject_ref_memory_local_pos_size,
                self.dim,
                device=device,
                dtype=dtype,
            )
        )

    def prepare_wantodance(
        self,
        in_dim,
        dim,
        num_heads,
        has_image_pos_emb,
        out_dim,
        patch_size,
        eps,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers=[0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        if wantodance_enable_music_inject:
            all_modules, all_modules_names = wantodance_torch_dfs(
                self.blocks, parent_name="root.transformer_blocks"
            )
            self.music_injector = WanToDanceInjector(
                all_modules,
                all_modules_names,
                dim=dim,
                num_heads=num_heads,
                inject_layer=wantodance_music_inject_layers,
            )
        if wantodance_enable_refimage:
            self.img_emb_refimage = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        if wantodance_enable_refface:
            self.img_emb_refface = MLP(
                1280, dim, has_pos_emb=has_image_pos_emb
            )  # clip_feature_dim = 1280
        if (
            wantodance_enable_global
            or wantodance_enable_dynamicfps
            or wantodance_enable_unimodel
        ):
            music_feature_dim = 35
            ff_size = 1024
            dropout = 0.1
            latent_dim = 256
            nhead = 4
            activation = F.gelu
            rotary = WanToDanceRotaryEmbedding(dim=latent_dim)
            self.music_projection = nn.Linear(music_feature_dim, latent_dim)
            self.music_encoder = nn.Sequential()
            for _ in range(2):
                self.music_encoder.append(
                    WanToDanceMusicEncoderLayer(
                        d_model=latent_dim,
                        nhead=nhead,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=rotary,
                        device="cuda",
                    )
                )
        if wantodance_enable_unimodel:
            self.patch_embedding_global = nn.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size
            )
        if wantodance_enable_unimodel:
            self.head_global = Head(dim, out_dim, patch_size, eps)
        self.wantodance_enable_music_inject = wantodance_enable_music_inject
        self.wantodance_enable_refimage = wantodance_enable_refimage
        self.wantodance_enable_refface = wantodance_enable_refface
        self.wantodance_enable_global = wantodance_enable_global
        self.wantodance_enable_dynamicfps = wantodance_enable_dynamicfps
        self.wantodance_enable_unimodel = wantodance_enable_unimodel

    def wantodance_after_transformer_block(self, block_idx, hidden_states):
        if self.wantodance_enable_music_inject:
            if block_idx in self.music_injector.injected_block_id.keys():
                audio_attn_id = self.music_injector.injected_block_id[block_idx]
                audio_emb = self.merged_audio_emb  # b f n c
                num_frames = audio_emb.shape[1]
                input_hidden_states = hidden_states.clone()  # b (f h w) c
                input_hidden_states = rearrange(
                    input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames
                )
                attn_hidden_states = self.music_injector.injector_pre_norm_feat[
                    audio_attn_id
                ](input_hidden_states)
                audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
                attn_audio_emb = audio_emb
                residual_out = self.music_injector.injector[audio_attn_id](
                    attn_hidden_states, attn_audio_emb
                )
                residual_out = rearrange(
                    residual_out, "(b t) n c -> b (t n) c", t=num_frames
                )
                hidden_states = hidden_states + residual_out
        return hidden_states

    def patchify(
        self,
        x: torch.Tensor,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        enable_wantodance_global=False,
    ):
        if enable_wantodance_global:
            x = self.patch_embedding_global(x)
        else:
            x = self.patch_embedding(x)
        if (
            self.control_adapter is not None
            and control_camera_latents_input is not None
        ):
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x,
                    context,
                    t_mod,
                    freqs,
                )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
