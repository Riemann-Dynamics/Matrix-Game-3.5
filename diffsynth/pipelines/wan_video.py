import contextlib
import os, torch, types
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Tuple, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from transformers import Wav2Vec2Processor

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.prope_attention import invert_k, invert_se3, lift_k
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel
from ..models.wav2vec import WanS2VAudioEncoder
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel

def vis_mask(mask, path="mask.png", s=8, line=2):
    m = np.asarray(mask).astype(bool)            # 形状 t,h,w
    t, h, w = m.shape
    img = np.full((h*s, t*w*s + (t-1)*line, 3), 255, np.uint8)
    for i in range(t):
        block = np.kron(m[i], np.ones((s, s), np.uint8)) * 255   # h*s, w*s
        x = i * (w*s + line)
        img[:, x:x+w*s] = block[..., None]       # 灰度铺到3通道
        if i < t-1:
            img[:, x+w*s:x+w*s+line] = (255, 0, 0)               # 红线分区
    Image.fromarray(img).save(path)

def _resolve_mosaic_frame_indices(
    mosaic_frame_indices,
    *,
    noisy_frame_count: int,
    mosaic_frame_count: int,
    device,
):
    if mosaic_frame_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if mosaic_frame_indices is None:
        if mosaic_frame_count != noisy_frame_count:
            raise ValueError(
                "mosaic_frame_indices is required when mosaic_frame_count "
                f"({mosaic_frame_count}) differs from noisy_frame_count ({noisy_frame_count})."
            )
        return torch.arange(noisy_frame_count, dtype=torch.long, device=device)
    if not torch.is_tensor(mosaic_frame_indices):
        mosaic_frame_indices = torch.as_tensor(mosaic_frame_indices)
    indices = mosaic_frame_indices.to(device=device, dtype=torch.long).reshape(-1)
    if int(indices.numel()) != int(mosaic_frame_count):
        raise ValueError(
            "mosaic_frame_indices length must match mosaic_frame_count, got "
            f"{int(indices.numel())} vs {int(mosaic_frame_count)}."
        )
    if indices.numel() and (
        int(indices.min().item()) < 0
        or int(indices.max().item()) >= int(noisy_frame_count)
    ):
        raise ValueError(
            "mosaic_frame_indices must be within the noisy latent range "
            f"[0, {int(noisy_frame_count)})."
        )
    return indices


def _resolve_latent_rope_time_indices(
    latent_rope_time_indices,
    *,
    first_frame_count: int,
    mosaic_frame_count: int,
    noisy_frame_count: int,
    mosaic_frame_indices,
    device,
):
    total_count = (
        int(first_frame_count) + int(mosaic_frame_count) + int(noisy_frame_count)
    )
    if latent_rope_time_indices is not None:
        if not torch.is_tensor(latent_rope_time_indices):
            latent_rope_time_indices = torch.as_tensor(latent_rope_time_indices)
        indices = latent_rope_time_indices.to(device=device, dtype=torch.long).reshape(
            -1
        )
        if int(indices.numel()) != int(total_count):
            raise ValueError(
                "latent_rope_time_indices length must match "
                "first_frame_count + mosaic_frame_count + noisy_frame_count, got "
                f"{int(indices.numel())} vs {int(total_count)}."
            )
        if indices.numel() and int(indices.min().item()) < 0:
            raise ValueError("latent_rope_time_indices must be non-negative.")
        return indices

    first_times = torch.arange(int(first_frame_count), dtype=torch.long, device=device)
    noisy_times = int(first_frame_count) + torch.arange(
        int(noisy_frame_count), dtype=torch.long, device=device
    )
    if int(mosaic_frame_count) > 0:
        mosaic_times = int(first_frame_count) + _resolve_mosaic_frame_indices(
            mosaic_frame_indices,
            noisy_frame_count=int(noisy_frame_count),
            mosaic_frame_count=int(mosaic_frame_count),
            device=device,
        )
    else:
        mosaic_times = torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat([first_times, mosaic_times, noisy_times], dim=0)


def _build_mosaic_cross_attn_keep_mask(
    *,
    prefix_memory_token_count: int = 0,
    reference_token_count: int,
    first_frame_count: int,
    mosaic_frame_count: int,
    noisy_frame_count: int,
    tokens_per_frame: int,
    device,
):
    prefix_memory_token_count = int(prefix_memory_token_count)
    total = prefix_memory_token_count + int(reference_token_count) + (
        int(first_frame_count) + int(mosaic_frame_count) + int(noisy_frame_count)
    ) * int(tokens_per_frame)
    mask = torch.ones(total, dtype=torch.bool, device=device)
    if prefix_memory_token_count > 0:
        mask[:prefix_memory_token_count] = False
    if mosaic_frame_count > 0:
        start = (
            prefix_memory_token_count
            + int(reference_token_count)
            + int(first_frame_count) * int(tokens_per_frame)
        )
        end = start + int(mosaic_frame_count) * int(tokens_per_frame)
        mask[start:end] = False
    return mask


def _build_subject_ref_memory_tokens(
    dit: WanModel,
    subject_ref_latents: Optional[torch.Tensor],
    *,
    batch_size: int,
    video_h: int,
    video_w: int,
    subject_ref_slot_ratio: float,
    subject_ref_time_gap: int,
    device,
    dtype,
):
    if subject_ref_latents is None:
        return None
    if not getattr(dit, "subject_ref_memory_enabled", False):
        return None
    required_attrs = (
        "subject_ref_index_embedding",
        "subject_ref_type_embedding",
        "subject_ref_local_h_embedding",
        "subject_ref_local_w_embedding",
    )
    missing_attrs = [name for name in required_attrs if not hasattr(dit, name)]
    if missing_attrs:
        raise ValueError(
            "subject_ref_latents were provided, but the DiT has no "
            f"{missing_attrs}. Enable subject ref memory before loading."
        )
    refs = subject_ref_latents
    if not torch.is_tensor(refs):
        refs = torch.as_tensor(refs)
    if refs.ndim == 5:
        # Materialized training data stores (R, C, 1, H, W). Inference may
        # pass (B, C, R, H, W); batch size > 1 is intentionally unsupported
        # for the current variable-ref-count path.
        if int(refs.shape[2]) == 1 and int(refs.shape[0]) != 1:
            refs = refs.permute(2, 1, 0, 3, 4).contiguous()
        elif int(refs.shape[0]) in (1, int(batch_size)):
            refs = refs.contiguous()
        else:
            raise ValueError(
                "subject_ref_latents expects (R,C,1,H,W) or (1,C,R,H,W), got "
                f"{tuple(refs.shape)}."
            )
    elif refs.ndim == 4:
        refs = refs.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    else:
        raise ValueError(
            "subject_ref_latents expects 4 or 5 dims, got "
            f"{tuple(refs.shape)}."
        )
    if int(refs.shape[0]) not in (1, int(batch_size)):
        raise ValueError(
            "subject_ref_latents batch size must be 1 or match model batch, got "
            f"{int(refs.shape[0])} vs {int(batch_size)}."
        )
    ref_count = int(refs.shape[2])
    if ref_count <= 0:
        return None
    max_refs = int(dit.subject_ref_index_embedding.shape[0])
    if ref_count > max_refs:
        refs = refs[:, :, :max_refs]
        ref_count = max_refs
    refs = refs.to(device=device, dtype=dtype)
    ref_x = dit.patchify(refs)
    if int(ref_x.shape[0]) == 1 and int(batch_size) > 1:
        ref_x = ref_x.expand(int(batch_size), -1, -1, -1, -1)
    _, _, _, ref_h, ref_w = ref_x.shape

    ratio = min(1.0, max(0.01, float(subject_ref_slot_ratio)))
    slot_size = int(round(min(int(video_h), int(video_w)) * ratio))
    if slot_size <= 0:
        return None
    slot_h = max(1, int(round(ref_h * slot_size / float(video_h))))
    slot_w = max(1, int(round(ref_w * slot_size / float(video_w))))
    slot_h = min(slot_h, ref_h)
    slot_w = min(slot_w, ref_w)
    h_start = int(ref_h - slot_h)
    w_start = int(ref_w - slot_w)

    ref_x = ref_x[:, :, :, h_start:ref_h, w_start:ref_w]
    ref_x = rearrange(ref_x, "b c r h w -> b r h w c").contiguous()

    ref_index_pos = dit.subject_ref_index_embedding[:ref_count].to(
        device=device, dtype=ref_x.dtype
    )
    ref_type_pos = dit.subject_ref_type_embedding.to(device=device, dtype=ref_x.dtype)
    local_h_pos = _subject_ref_local_pos(
        dit.subject_ref_local_h_embedding, slot_h, device=device, dtype=ref_x.dtype
    )
    local_w_pos = _subject_ref_local_pos(
        dit.subject_ref_local_w_embedding, slot_w, device=device, dtype=ref_x.dtype
    )
    local_pos = (
        local_h_pos.view(1, 1, slot_h, 1, -1)
        + local_w_pos.view(1, 1, 1, slot_w, -1)
    )
    ref_x = (
        ref_x
        + ref_type_pos.view(1, 1, 1, 1, -1)
        + ref_index_pos.view(1, ref_count, 1, 1, -1)
        + local_pos
    )
    ref_x = rearrange(ref_x, "b r h w c -> b (r h w) c").contiguous()

    ref_freqs = _build_subject_ref_time_freqs(
        dit,
        ref_count=ref_count,
        slot_h=slot_h,
        slot_w=slot_w,
        subject_ref_time_gap=subject_ref_time_gap,
        device=device,
    )
    return {
        "x": ref_x,
        "freqs": ref_freqs,
        "token_count": int(ref_x.shape[1]),
        "ref_count": ref_count,
        "slot_grid": (int(slot_h), int(slot_w)),
        "slot_start": (int(h_start), int(w_start)),
    }


def _subject_ref_local_pos(table, length: int, *, device, dtype):
    table = table.to(device=device, dtype=dtype)
    length = int(length)
    if length <= int(table.shape[0]):
        return table[:length]
    pos = torch.nn.functional.interpolate(
        table.float().transpose(0, 1).unsqueeze(0),
        size=length,
        mode="linear",
        align_corners=True,
    )
    return pos.squeeze(0).transpose(0, 1).to(dtype=dtype)


def _build_subject_ref_time_freqs(
    dit: WanModel,
    *,
    ref_count: int,
    slot_h: int,
    slot_w: int,
    subject_ref_time_gap: int,
    device,
):
    freq_dev = dit.freqs[0].device
    time_gap = max(1, int(subject_ref_time_gap))
    ref_time_indices = (
        torch.arange(1, ref_count + 1, device=freq_dev, dtype=torch.long) * time_gap
    ).clamp(max=int(dit.freqs[0].shape[0]) - 1)
    time_freqs = dit.freqs[0][ref_time_indices].conj()
    h_freqs = dit.freqs[1][:1].expand(slot_h, -1)
    w_freqs = dit.freqs[2][:1].expand(slot_w, -1)
    ref_freqs = torch.cat(
        [
            time_freqs
            .view(ref_count, 1, 1, -1)
            .expand(ref_count, slot_h, slot_w, -1),
            h_freqs.view(1, slot_h, 1, -1).expand(ref_count, slot_h, slot_w, -1),
            w_freqs.view(1, 1, slot_w, -1).expand(ref_count, slot_h, slot_w, -1),
        ],
        dim=-1,
    )
    return ref_freqs.reshape(ref_count * slot_h * slot_w, 1, -1).to(device)


# Single source of truth for the PRoPE camera input keys. Every layer that
# special-cases these tensors (pipeline unit input_params, the training
# module's fp32 exemption from the blanket bf16 cast, the pipeline-input
# pass-through) must import THIS tuple -- a key added in one hand-written
# copy but missed in another silently re-introduces bf16 input quantization
# of absolute extrinsics (catastrophic cancellation of relative motion).
WAN_VIDEO_PROPE_CAMERA_KEYS = (
    "clean_latent_indices_prope_intrinsic",
    "clean_latent_indices_prope_extrinsic",
    "noisy_latent_indices_prope_intrinsic",
    "noisy_latent_indices_prope_extrinsic",
)
WAN_VIDEO_PROPE_CLEAN_CAMERA_KEYS = tuple(
    key for key in WAN_VIDEO_PROPE_CAMERA_KEYS if key.startswith("clean_")
)


def _negative_no_context_inputs_shared(inputs_shared):
    """Build negative CFG shared inputs with only the latest clean context."""
    negative_shared = dict(inputs_shared)
    first_frame_latents = negative_shared.get("first_frame_latents")
    if torch.is_tensor(first_frame_latents) and int(first_frame_latents.shape[2]) > 1:
        negative_shared["first_frame_latents"] = first_frame_latents[:, :, -1:]

    for name in WAN_VIDEO_PROPE_CLEAN_CAMERA_KEYS:
        value = negative_shared.get(name)
        if torch.is_tensor(value) and int(value.shape[0]) > 4:
            negative_shared[name] = value[-4:]

    # The full-context RoPE table has the wrong length after trimming the
    # clean prefix. Let model_fn_wan_video rebuild the single-clean timeline.
    negative_shared["latent_rope_time_indices"] = None
    return negative_shared


def _drop_holes_reindex_prope_camera_info(
    camera_info,
    *,
    full_frame_count: int,
    tokens_per_frame: int,
    keep_idx_latent: torch.Tensor,
):
    if camera_info is None:
        return None
    if len(camera_info) < 2 or camera_info[1] is None:
        return camera_info

    w2c_info = camera_info[0]
    viewmats_st = camera_info[1]
    view_change_positions = camera_info[2] if len(camera_info) > 2 else None
    p_st, pt_st, pinv_st = viewmats_st
    b_cam, s_cam = p_st.shape[0], p_st.shape[1]
    if int(s_cam) != int(full_frame_count):
        raise ValueError(
            "PROPE viewmats temporal length "
            f"{int(s_cam)} does not match expected "
            f"{int(full_frame_count)} (first+mosaic+noisy)."
        )

    def _broadcast_then_select(mat):
        rest = mat.shape[2:]
        return (
            mat.unsqueeze(2)
            .expand(b_cam, s_cam, int(tokens_per_frame), *rest)
            .reshape(b_cam, s_cam * int(tokens_per_frame), *rest)
            .index_select(1, keep_idx_latent)
            .contiguous()
        )

    viewmats_pt = (
        _broadcast_then_select(p_st),
        _broadcast_then_select(pt_st),
        _broadcast_then_select(pinv_st),
    )
    if view_change_positions is None:
        return (w2c_info, viewmats_pt)

    expected_view_change_tokens = int(full_frame_count) * int(tokens_per_frame)
    if int(view_change_positions.shape[1]) != expected_view_change_tokens:
        raise ValueError(
            "PROPE view_change_positions length "
            f"{int(view_change_positions.shape[1])} does not match expected "
            f"{expected_view_change_tokens} (first+mosaic+noisy tokens)."
        )
    view_change_positions = view_change_positions.index_select(
        1, keep_idx_latent
    ).contiguous()
    return (w2c_info, viewmats_pt, view_change_positions)


def _reindex_token_prope_camera_info(camera_info, keep_idx: torch.Tensor):
    if camera_info is None:
        return None
    if len(camera_info) < 2 or camera_info[1] is None:
        return camera_info
    w2c_info = camera_info[0]
    p, p_t, p_inv = camera_info[1]
    viewmats = (
        p.index_select(1, keep_idx).contiguous(),
        p_t.index_select(1, keep_idx).contiguous(),
        p_inv.index_select(1, keep_idx).contiguous(),
    )
    if len(camera_info) <= 2 or camera_info[2] is None:
        return (w2c_info, viewmats)
    view_change_positions = camera_info[2].index_select(1, keep_idx).contiguous()
    return (w2c_info, viewmats, view_change_positions)


def _prepend_subject_ref_prope_camera_info(
    camera_info,
    *,
    prefix_token_count: int,
    tokens_per_frame: int,
    frame_count: Optional[int] = None,
    mode: str = "identity",
    clean_anchor_token_index: Optional[int] = None,
):
    if camera_info is None or int(prefix_token_count) <= 0:
        return camera_info
    if len(camera_info) < 2 or camera_info[1] is None:
        return camera_info

    w2c_info = camera_info[0]
    viewmats = camera_info[1]
    view_change_positions = camera_info[2] if len(camera_info) > 2 else None
    p, p_t, p_inv = viewmats
    b_cam = int(p.shape[0])
    token_count = int(prefix_token_count)
    mode = str(mode or "identity").strip().lower()
    if mode not in {"identity", "clean_anchor"}:
        raise ValueError(
            "subject_ref_prope_mode must be 'identity' or 'clean_anchor', "
            f"got {mode!r}."
        )

    def _as_token_viewmats(mat):
        rest = mat.shape[2:]
        if frame_count is not None and int(mat.shape[1]) == int(frame_count):
            return (
                mat.unsqueeze(2)
                .expand(b_cam, int(mat.shape[1]), int(tokens_per_frame), *rest)
                .reshape(b_cam, int(mat.shape[1]) * int(tokens_per_frame), *rest)
                .contiguous()
            )
        if frame_count is not None and int(mat.shape[1]) == (
            int(frame_count) * int(tokens_per_frame)
        ):
            return mat
        return mat

    def _prepend_refs(mat):
        mat_pt = _as_token_viewmats(mat)
        if mode == "clean_anchor":
            if int(mat_pt.shape[1]) <= 0:
                raise ValueError(
                    "Cannot use subject_ref_prope_mode='clean_anchor' with "
                    "empty PROPE viewmats."
                )
            anchor_idx = int(clean_anchor_token_index or 0)
            if anchor_idx < 0 or anchor_idx >= int(mat_pt.shape[1]):
                raise ValueError(
                    "subject_ref_prope_mode='clean_anchor' anchor token index "
                    f"{anchor_idx} is outside PROPE token length "
                    f"{int(mat_pt.shape[1])}."
                )
            prefix = mat_pt[:, anchor_idx : anchor_idx + 1].expand(
                b_cam, token_count, *mat_pt.shape[2:]
            )
        else:
            eye = torch.eye(
                mat_pt.shape[-1],
                device=mat_pt.device,
                dtype=mat_pt.dtype,
            )
            view_shape = (
                (1, 1)
                + tuple(1 for _ in mat_pt.shape[2:-2])
                + (mat_pt.shape[-2], mat_pt.shape[-1])
            )
            prefix = eye.view(view_shape).expand(
                b_cam, token_count, *mat_pt.shape[2:]
            )
        return torch.cat([prefix, mat_pt], dim=1).contiguous()

    viewmats_pt = (
        _prepend_refs(p),
        _prepend_refs(p_t),
        _prepend_refs(p_inv),
    )
    if view_change_positions is None:
        return (w2c_info, viewmats_pt)

    if mode == "clean_anchor":
        if int(view_change_positions.shape[1]) <= 0:
            raise ValueError(
                "Cannot use subject_ref_prope_mode='clean_anchor' with empty "
                "PROPE view_change_positions."
            )
        anchor_idx = int(clean_anchor_token_index or 0)
        if anchor_idx < 0 or anchor_idx >= int(view_change_positions.shape[1]):
            raise ValueError(
                "subject_ref_prope_mode='clean_anchor' anchor token index "
                f"{anchor_idx} is outside PROPE view-change token length "
                f"{int(view_change_positions.shape[1])}."
            )
        prefix_view_change = view_change_positions[:, anchor_idx : anchor_idx + 1].expand(
            b_cam, token_count, 3
        )
    else:
        prefix_view_change = torch.zeros(
            b_cam,
            token_count,
            3,
            device=view_change_positions.device,
            dtype=view_change_positions.dtype,
        )
        prefix_view_change[..., 0] = 1.0
    view_change_positions = torch.cat(
        [prefix_view_change, view_change_positions], dim=1
    ).contiguous()
    return (w2c_info, viewmats_pt, view_change_positions)


class WanVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.audio_encoder: WanS2VAudioEncoder = None
        self.in_iteration_models = (
            "dit",
            "motion_controller",
            "vace",
            "animate_adapter",
            "vap",
        )
        self.in_iteration_models_2 = (
            "dit2",
            "motion_controller",
            "vace2",
            "animate_adapter",
            "vap",
        )
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_MosaicLatent(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_AnimateVideoSplit(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateFacePixelValues(),
            WanVideoUnit_AnimateInpaint(),
            WanVideoUnit_VAP(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_LongCatVideo(),
            WanVideoUnit_WanToDance_ProcessInputs(),
            WanVideoUnit_WanToDance_RefImageEmbedder(),
            WanVideoUnit_WanToDance_ImageKeyframesEmbedder(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video
        self.compilable_models = ["dit", "dit2"]

    def enable_usp(self):
        from ..utils.xfuser import (
            get_sequence_parallel_world_size,
            usp_attn_forward,
            usp_dit_forward,
            usp_vace_forward,
        )

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        if self.vace is not None:
            for block in self.vace.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
            self.vace.forward = types.MethodType(usp_vace_forward, self.vace)
        if self.vace2 is not None:
            for block in self.vace2.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
            self.vace2.forward = types.MethodType(usp_vace_forward, self.vace2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"
        ),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_t5_umt5-xxl-enc-bf16.safetensors",
                ),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
                ),
                "Wan2.1_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.1_VAE.safetensors",
                ),
                "Wan2.2_VAE.pth": (
                    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
                    "Wan2.2_VAE.safetensors",
                ),
            }
            for model_config in model_configs:
                if (
                    model_config.origin_file_pattern is None
                    or model_config.model_id is None
                ):
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern][0]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[
                        model_config.origin_file_pattern
                    ][0]
                    model_config.origin_file_pattern = redirect_dict[
                        model_config.origin_file_pattern
                    ][1]

        if use_usp:
            from ..utils.xfuser import initialize_usp

            initialize_usp(device)
            import torch.distributed as dist
            from ..core.device.npu_compatible_device import get_device_name

            if dist.is_available() and dist.is_initialized():
                device = get_device_name()
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_pool.fetch_model("wan_video_motion_controller")
        vace = model_pool.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.vap = model_pool.fetch_model("wan_video_vap")
        pipe.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_config.path, seq_len=512, clean="whitespace"
            )
        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary()
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(
                audio_processor_config.path
            )

        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @contextlib.contextmanager
    def _temporary_no_prope(self, *dits):
        states = []
        try:
            for dit in dits:
                if dit is None:
                    continue
                states.append((dit, "use_prope", getattr(dit, "use_prope", None)))
                dit.use_prope = False
                for block in getattr(dit, "blocks", []):
                    states.append(
                        (block, "use_prope", getattr(block, "use_prope", None))
                    )
                    block.use_prope = False
                    self_attn = getattr(block, "self_attn", None)
                    if self_attn is not None:
                        states.append(
                            (
                                self_attn,
                                "use_prope",
                                getattr(self_attn, "use_prope", None),
                            )
                        )
                        self_attn.use_prope = False
            yield
        finally:
            for obj, attr, value in reversed(states):
                if value is None:
                    try:
                        delattr(obj, attr)
                    except AttributeError:
                        pass
                else:
                    setattr(obj, attr, value)

    @contextlib.contextmanager
    def _temporary_prope_flag(self, attr, value, *dits):
        """Temporarily force a PRoPE flag on dit/blocks/self_attn; ``None``
        leaves the stamped module attributes untouched."""
        if value is None:
            yield
            return
        states = []
        try:
            value = bool(value)
            for dit in dits:
                if dit is None:
                    continue
                states.append((dit, attr, getattr(dit, attr, None)))
                setattr(dit, attr, value)
                for block in getattr(dit, "blocks", []):
                    states.append((block, attr, getattr(block, attr, None)))
                    setattr(block, attr, value)
                    self_attn = getattr(block, "self_attn", None)
                    if self_attn is not None:
                        states.append(
                            (self_attn, attr, getattr(self_attn, attr, None))
                        )
                        setattr(self_attn, attr, value)
            yield
        finally:
            for obj, attr_name, old_value in reversed(states):
                if old_value is None:
                    try:
                        delattr(obj, attr_name)
                    except AttributeError:
                        pass
                else:
                    setattr(obj, attr_name, old_value)

    def _temporary_prope_native_rope(self, prope_disable_native_rope, *dits):
        return self._temporary_prope_flag(
            "prope_disable_native_rope", prope_disable_native_rope, *dits
        )

    def _temporary_prope_t_rope(self, prope_disable_t_rope, *dits):
        return self._temporary_prope_flag(
            "prope_disable_t_rope", prope_disable_t_rope, *dits
        )

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = "",
        negative_prompt: str = "",
        # Image-to-video
        input_image: Image.Image = None,
        # First-last-frame-to-video
        end_image: Image.Image = None,
        # Video-to-video
        input_video: list[Image.Image] = None,
        denoising_strength: float = 1.0,
        # Speech-to-video
        input_audio: np.array = None,
        audio_embeds: torch.Tensor = None,
        audio_sample_rate: int = 16000,
        s2v_pose_video: list[Image.Image] = None,
        s2v_pose_latents: torch.Tensor = None,
        motion_video: list[Image.Image] = None,
        # ControlNet
        control_video: list[Image.Image] = None,
        reference_image: Image.Image = None,
        # Camera control
        camera_control_direction: Literal[
            "Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"
        ] = None,
        camera_control_speed: float = 1 / 54,
        camera_control_origin: tuple = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        # VACE
        vace_video: list[Image.Image] = None,
        vace_video_mask: Image.Image = None,
        vace_reference_image: Image.Image = None,
        vace_scale: float = 1.0,
        # Animate
        animate_pose_video: list[Image.Image] = None,
        animate_face_video: list[Image.Image] = None,
        animate_inpaint_video: list[Image.Image] = None,
        animate_mask_video: list[Image.Image] = None,
        # VAP
        vap_video: list[Image.Image] = None,
        vap_prompt: str = " ",
        negative_vap_prompt: str = " ",
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Shape
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        # Classifier-free guidance
        cfg_scale: float = 5.0,
        cfg_merge: bool = False,
        negative_no_prope: bool = False,
        negative_no_context: bool = False,
        # Boundary
        switch_DiT_boundary: float = 0.875,
        # Scheduler
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        # Speed control
        motion_bucket_id: int = None,
        # LongCat-Video
        longcat_video: list[Image.Image] = None,
        # VAE tiling
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        # Sliding window
        sliding_window_size: int = None,
        sliding_window_stride: int = None,
        # Teacache
        tea_cache_l1_thresh: float = None,
        tea_cache_model_id: str = "",
        # WanToDance
        wantodance_music_path: str = None,
        wantodance_reference_image: Image.Image = None,
        wantodance_fps: float = 30,
        wantodance_keyframes: list[Image.Image] = None,
        wantodance_keyframes_mask: list[int] = None,
        # Pre-encoded first-frame latent. When provided, the
        # `WanVideoUnit_ImageEmbedderFused` step is bypassed (because
        # `input_image` is None) and this latent is used directly as the
        # condition prefix in `WanVideoUnit_LatentSequence`. Skips the
        # decode->encode VAE round-trip used when only `input_image` is given.
        first_frame_latents: Optional[torch.Tensor] = None,
        mosaic_latent: Optional[torch.Tensor] = None,
        mosaic_timestep_zero: bool = True,
        mosaic_revgrid: Optional[np.ndarray] = None,
        mosaic_use_revgrid_rope: bool = False,
        mosaic_view_change: Optional[torch.Tensor] = None,
        mosaic_view_change_prope: bool = False,
        mosaic_mask_holes: bool = True,
        mosaic_drop_holes: bool = False,
        mosaic_frame_indices: Optional[torch.Tensor] = None,
        latent_rope_time_indices: Optional[torch.Tensor] = None,
        subject_ref_latents: Optional[torch.Tensor] = None,
        subject_ref_slot_ratio: float = 0.5,
        subject_ref_time_gap: int = 1,
        subject_ref_prope_mode: str = "identity",
        prope_disable_native_rope: Optional[bool] = None,
        prope_disable_t_rope: Optional[bool] = None,
        clean_latent_indices_prope_intrinsic: Optional[torch.Tensor] = None,
        clean_latent_indices_prope_extrinsic: Optional[torch.Tensor] = None,
        noisy_latent_indices_prope_intrinsic: Optional[torch.Tensor] = None,
        noisy_latent_indices_prope_extrinsic: Optional[torch.Tensor] = None,
        framewise_decoding: bool = False,
        return_latent: bool = False,
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Literal["quantized", "floatpoint"] = "quantized",
    ):
        if mosaic_latent is not None and (
            sliding_window_size is not None or sliding_window_stride is not None
        ):
            raise ValueError(
                "mosaic_latent currently does not support sliding-window inference."
            )
        if negative_no_prope and cfg_merge:
            raise ValueError(
                "negative_no_prope requires cfg_merge=False because the negative "
                "CFG branch must run as a separate model call."
            )
        if negative_no_context and cfg_merge:
            raise ValueError(
                "negative_no_context requires cfg_merge=False because the negative "
                "CFG branch must use a different clean context length."
            )
        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "negative_no_prope": negative_no_prope,
            "negative_no_context": negative_no_context,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio,
            "audio_sample_rate": audio_sample_rate,
            "s2v_pose_video": s2v_pose_video,
            "audio_embeds": audio_embeds,
            "s2v_pose_latents": s2v_pose_latents,
            "motion_video": motion_video,
            "animate_pose_video": animate_pose_video,
            "animate_face_video": animate_face_video,
            "animate_inpaint_video": animate_inpaint_video,
            "animate_mask_video": animate_mask_video,
            "vap_video": vap_video,
            "wantodance_music_path": wantodance_music_path,
            "wantodance_reference_image": wantodance_reference_image,
            "wantodance_fps": wantodance_fps,
            "wantodance_keyframes": wantodance_keyframes,
            "wantodance_keyframes_mask": wantodance_keyframes_mask,
            "first_frame_latents": first_frame_latents,
            "mosaic_latent": mosaic_latent,
            "mosaic_timestep_zero": mosaic_timestep_zero,
            "mosaic_revgrid": mosaic_revgrid,
            "mosaic_use_revgrid_rope": mosaic_use_revgrid_rope,
            "mosaic_view_change": mosaic_view_change,
            "mosaic_view_change_prope": mosaic_view_change_prope,
            "mosaic_mask_holes": mosaic_mask_holes,
            "mosaic_drop_holes": mosaic_drop_holes,
            "mosaic_frame_indices": mosaic_frame_indices,
            "latent_rope_time_indices": latent_rope_time_indices,
            "subject_ref_latents": subject_ref_latents,
            "subject_ref_slot_ratio": subject_ref_slot_ratio,
            "subject_ref_time_gap": subject_ref_time_gap,
            "subject_ref_prope_mode": subject_ref_prope_mode,
            "prope_disable_native_rope": prope_disable_native_rope,
            "prope_disable_t_rope": prope_disable_t_rope,
            "clean_latent_indices_prope_intrinsic": clean_latent_indices_prope_intrinsic,
            "clean_latent_indices_prope_extrinsic": clean_latent_indices_prope_extrinsic,
            "noisy_latent_indices_prope_intrinsic": noisy_latent_indices_prope_intrinsic,
            "noisy_latent_indices_prope_extrinsic": noisy_latent_indices_prope_extrinsic,
            "framewise_decoding": framewise_decoding,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        model_fn_wan_video._rope_logged = False
        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps)
        ):
            # Switch DiT if necessary
            if (
                timestep.item() < switch_DiT_boundary * 1000
                and self.dit2 is not None
                and not models["dit"] is self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2

            # Timestep
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )

            # Inference
            with (
                self._temporary_prope_native_rope(
                    prope_disable_native_rope,
                    self.dit,
                    self.dit2,
                ),
                self._temporary_prope_t_rope(
                    prope_disable_t_rope,
                    self.dit,
                    self.dit2,
                ),
            ):
                noise_pred_posi = self.model_fn(
                    **models, **inputs_shared, **inputs_posi, timestep=timestep
                )
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    inputs_shared_nega = (
                        _negative_no_context_inputs_shared(inputs_shared)
                        if negative_no_context
                        else inputs_shared
                    )
                    if negative_no_prope:
                        with (
                            self._temporary_no_prope(self.dit, self.dit2),
                            self._temporary_prope_native_rope(
                                prope_disable_native_rope,
                                self.dit,
                                self.dit2,
                            ),
                            self._temporary_prope_t_rope(
                                prope_disable_t_rope,
                                self.dit,
                                self.dit2,
                            ),
                        ):
                            noise_pred_nega = self.model_fn(
                                **models,
                                **inputs_shared_nega,
                                **inputs_nega,
                                timestep=timestep,
                            )
                    else:
                        with (
                            self._temporary_prope_native_rope(
                                prope_disable_native_rope,
                                self.dit,
                                self.dit2,
                            ),
                            self._temporary_prope_t_rope(
                                prope_disable_t_rope,
                                self.dit,
                                self.dit2,
                            ),
                        ):
                            noise_pred_nega = self.model_fn(
                                **models,
                                **inputs_shared_nega,
                                **inputs_nega,
                                timestep=timestep,
                            )
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )

        # VACE (TODO: remove it)
        if vace_reference_image is not None or (
            animate_pose_video is not None and animate_face_video is not None
        ):
            if vace_reference_image is not None and isinstance(
                vace_reference_image, list
            ):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )
        if return_latent:
            self.load_models_to_device([])
            out_latents = inputs_shared["latents"]
            first_frame_latents_out = inputs_shared.get("first_frame_latents")
            if first_frame_latents_out is not None:
                first_frame_latents_out = first_frame_latents_out.to(
                    device=out_latents.device, dtype=out_latents.dtype
                )
                out_latents = torch.cat([first_frame_latents_out, out_latents], dim=2)
            return out_latents

        # Decode
        self.load_models_to_device(["vae"])
        if framewise_decoding:
            video = self.vae.decode_framewise(
                inputs_shared["latents"], device=self.device
            )
        else:
            video = self.vae.decode(
                inputs_shared["latents"],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        self.load_models_to_device([])
        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "seed",
                "rand_device",
                "vace_reference_image",
            ),
            output_params=("noise",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        num_frames,
        seed,
        rand_device,
        vace_reference_image,
    ):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = (
                len(vace_reference_image)
                if isinstance(vace_reference_image, list)
                else 1
            )
            length += f
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
                "framewise_decoding",
            ),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
        framewise_decoding,
    ):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        if framewise_decoding:
            input_latents = pipe.vae.encode_framewise(input_video, device=pipe.device)
        else:
            input_latents = pipe.vae.encode(
                input_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(
                vace_reference_image, device=pipe.device
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(
                input_latents, noise, timestep=pipe.scheduler.timesteps[0]
            )
            return {"latents": latents}


class WanVideoUnit_MosaicLatent(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("mosaic_latent", "latents", "mosaic_frame_indices"),
            output_params=("mosaic_latent", "mosaic_frame_indices"),
        )

    def process(
        self, pipe: WanVideoPipeline, mosaic_latent, latents, mosaic_frame_indices=None
    ):
        if mosaic_latent is None:
            return {"mosaic_latent": None, "mosaic_frame_indices": mosaic_frame_indices}

        mosaic_latent = mosaic_latent.to(device=latents.device, dtype=latents.dtype)
        if (
            mosaic_latent.shape[:2] != latents.shape[:2]
            or mosaic_latent.shape[-2:] != latents.shape[-2:]
        ):
            raise ValueError(
                "mosaic_latent must share batch/channel/spatial shape with latents, got "
                f"{tuple(mosaic_latent.shape)} vs {tuple(latents.shape)}."
            )
        if mosaic_latent.shape[2] > latents.shape[2]:
            raise ValueError(
                "mosaic_latent temporal length must be <= latents temporal length, got "
                f"{mosaic_latent.shape[2]} vs {latents.shape[2]}."
            )
        indices = _resolve_mosaic_frame_indices(
            mosaic_frame_indices,
            noisy_frame_count=int(latents.shape[2]),
            mosaic_frame_count=int(mosaic_latent.shape[2]),
            device=latents.device,
        )
        return {"mosaic_latent": mosaic_latent, "mosaic_frame_indices": indices}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",),
        )

    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",),
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if (
            input_image is None
            or pipe.image_encoder is None
            or not pipe.dit.require_clip_embedding
        ):
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(
            pipe.device
        )
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(
                pipe.device
            )
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat(
                    [clip_context, pipe.image_encoder.encode_image([end_image])], dim=1
                )
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("y",),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(
            pipe.device
        )
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(
                pipe.device
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=(
                "latents",
                "fuse_vae_embedding_in_latents",
                "first_frame_latents",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(
            0, 1
        )
        z = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return {"fuse_vae_embedding_in_latents": True, "first_frame_latents": z}


class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "control_video",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "clip_feature",
                "y",
                "latents",
            ),
            output_params=("clip_feature", "y"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        control_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        clip_feature,
        y,
        latents,
    ):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(
            control_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim - control_latents.shape[1] - latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros(
                (1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device
            )
            y = torch.zeros(
                (1, y_dim, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            output_params=("reference_latents", "clip_feature"),
            onload_model_names=("vae", "image_encoder"),
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}


class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "camera_control_direction",
                "camera_control_speed",
                "camera_control_origin",
                "latents",
                "input_image",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("control_camera_latents_input", "y"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        num_frames,
        camera_control_direction,
        camera_control_speed,
        camera_control_origin,
        latents,
        input_image,
        tiled,
        tile_size,
        tile_stride,
    ):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = (
            pipe.dit.control_adapter.process_camera_coordinates(
                camera_control_direction,
                num_frames,
                height,
                width,
                camera_control_speed,
                camera_control_origin,
            )
        )

        control_camera_video = (
            camera_control_plucker_embedding[:num_frames]
            .permute([3, 0, 1, 2])
            .unsqueeze(0)
        )
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(
                    control_camera_video[:, :, 0:1], repeats=4, dim=2
                ),
                control_camera_video[:, :, 1:],
            ],
            dim=2,
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, 4, c, h, w)
            .transpose(2, 3)
        )
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, c * 4, h, w)
            .transpose(1, 2)
        )
        control_camera_latents_input = control_camera_latents.to(
            device=pipe.device, dtype=pipe.torch_dtype
        )

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(
                pipe.device
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )
            y = pipe.vae.encode(
                [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}


class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("motion_bucket_id",), output_params=("motion_bucket_id",)
        )

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        return {"motion_bucket_id": motion_bucket_id}


class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video",
                "vace_video_mask",
                "vace_reference_image",
                "vace_scale",
                "height",
                "width",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("vace_context", "vace_scale"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video,
        vace_video_mask,
        vace_reference_image,
        vace_scale,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if (
            vace_video is not None
            or vace_video_mask is not None
            or vace_reference_image is not None
        ):
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros(
                    (1, 3, num_frames, height, width),
                    dtype=pipe.torch_dtype,
                    device=pipe.device,
                )
            else:
                vace_video = pipe.preprocess_video(vace_video)

            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(
                    vace_video_mask, min_value=0, max_value=1
                )

            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(
                inactive,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(
                reactive,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            vace_mask_latents = rearrange(
                vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8
            )
            vace_mask_latents = torch.nn.functional.interpolate(
                vace_mask_latents,
                size=(
                    (vace_mask_latents.shape[2] + 3) // 4,
                    vace_mask_latents.shape[3],
                    vace_mask_latents.shape[4],
                ),
                mode="nearest-exact",
            )

            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image, list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j : j + 1])
                vace_reference_image = new_vace_ref_images

                vace_reference_latents = pipe.vae.encode(
                    vace_reference_image,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat(
                    (vace_reference_latents, torch.zeros_like(vace_reference_latents)),
                    dim=1,
                )
                vace_reference_latents = [
                    u.unsqueeze(0) for u in vace_reference_latents
                ]

                vace_video_latents = torch.concat(
                    (*vace_reference_latents, vace_video_latents), dim=2
                )
                vace_mask_latents = torch.concat(
                    (torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents),
                    dim=2,
                )

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_VAP(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder", "vae", "image_encoder"),
            input_params=(
                "vap_video",
                "vap_prompt",
                "negative_vap_prompt",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("vap_clip_feature", "vap_hidden_state", "context_vap"),
        )

    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("vap_video") is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            # 1. encode vap prompt
            pipe.load_models_to_device(["text_encoder"])
            vap_prompt, negative_vap_prompt = inputs_posi.get(
                "vap_prompt", ""
            ), inputs_nega.get("negative_vap_prompt", "")
            vap_prompt_emb = self.encode_prompt(pipe, vap_prompt)
            negative_vap_prompt_emb = self.encode_prompt(pipe, negative_vap_prompt)
            inputs_posi.update({"context_vap": vap_prompt_emb})
            inputs_nega.update({"context_vap": negative_vap_prompt_emb})
            # 2. prepare vap image clip embedding
            pipe.load_models_to_device(["vae", "image_encoder"])
            vap_video, end_image = inputs_shared.get("vap_video"), inputs_shared.get(
                "end_image"
            )

            num_frames, height, width = (
                inputs_shared.get("num_frames"),
                inputs_shared.get("height"),
                inputs_shared.get("width"),
            )

            image_vap = pipe.preprocess_image(vap_video[0].resize((width, height))).to(
                pipe.device
            )

            vap_clip_context = pipe.image_encoder.encode_image([image_vap])
            if end_image is not None:
                vap_end_image = pipe.preprocess_image(
                    vap_video[-1].resize((width, height))
                ).to(pipe.device)
                if pipe.dit.has_image_pos_emb:
                    vap_clip_context = torch.concat(
                        [
                            vap_clip_context,
                            pipe.image_encoder.encode_image([vap_end_image]),
                        ],
                        dim=1,
                    )
            vap_clip_context = vap_clip_context.to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            inputs_shared.update({"vap_clip_feature": vap_clip_context})

            # 3. prepare vap latents
            msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
            msk[:, 1:] = 0
            if end_image is not None:
                msk[:, -1:] = 1
                last_image_vap = pipe.preprocess_image(
                    vap_video[-1].resize((width, height))
                ).to(pipe.device)
                vae_input = torch.concat(
                    [
                        image_vap.transpose(0, 1),
                        torch.zeros(3, num_frames - 2, height, width).to(
                            image_vap.device
                        ),
                        last_image_vap.transpose(0, 1),
                    ],
                    dim=1,
                )
            else:
                vae_input = torch.concat(
                    [
                        image_vap.transpose(0, 1),
                        torch.zeros(3, num_frames - 1, height, width).to(
                            image_vap.device
                        ),
                    ],
                    dim=1,
                )

            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)[0]

            tiled, tile_size, tile_stride = (
                inputs_shared.get("tiled"),
                inputs_shared.get("tile_size"),
                inputs_shared.get("tile_stride"),
            )

            y = pipe.vae.encode(
                [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_video = pipe.preprocess_video(vap_video)
            vap_latent = pipe.vae.encode(
                vap_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_latent = torch.concat([vap_latent, y], dim=1).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            inputs_shared.update({"vap_hidden_state": vap_latent})

            return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(), output_params=("use_unified_sequence_parallel",)
        )

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "num_inference_steps": "num_inference_steps",
                "tea_cache_l1_thresh": "tea_cache_l1_thresh",
                "tea_cache_model_id": "tea_cache_model_id",
            },
            input_params_nega={
                "num_inference_steps": "num_inference_steps",
                "tea_cache_l1_thresh": "tea_cache_l1_thresh",
                "tea_cache_model_id": "tea_cache_model_id",
            },
            output_params=("tea_cache",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        num_inference_steps,
        tea_cache_l1_thresh,
        tea_cache_model_id,
    ):
        if tea_cache_l1_thresh is None:
            return {}
        return {
            "tea_cache": TeaCache(
                num_inference_steps,
                rel_l1_thresh=tea_cache_l1_thresh,
                model_id=tea_cache_model_id,
            )
        }


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat(
                    (tensor_shared, tensor_shared), dim=0
                )
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=(
                "audio_encoder",
                "vae",
            ),
            input_params=(
                "input_audio",
                "audio_embeds",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "audio_sample_rate",
                "s2v_pose_video",
                "s2v_pose_latents",
                "motion_video",
            ),
            output_params=(
                "audio_embeds",
                "motion_latents",
                "drop_motion_frames",
                "s2v_pose_latents",
            ),
        )

    def process_audio(
        self,
        pipe: WanVideoPipeline,
        input_audio,
        audio_sample_rate,
        num_frames,
        fps=16,
        audio_embeds=None,
        return_all=False,
    ):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(
            input_audio,
            audio_sample_rate,
            pipe.audio_processor,
            fps=fps,
            batch_frames=num_frames - 1,
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        motion_video=None,
    ):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None:
            assert (
                motion_video.shape[2] == motion_frames
            ), f"motion video must have {motion_frames} frames, but got {motion_video.shape[2]}"
            motion_latents = motion_video
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros(
                [1, 3, motion_frames, height, width],
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(
            motion_latents,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(
        self,
        pipe: WanVideoPipeline,
        s2v_pose_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        s2v_pose_latents=None,
        num_repeats=1,
        return_all=False,
    ):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[
            :, :, : infer_frames * num_repeats
        ]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat(
            [
                input_video,
                -torch.ones(
                    1,
                    3,
                    padding_frames,
                    height,
                    width,
                    device=input_video.device,
                    dtype=input_video.dtype,
                ),
            ],
            dim=2,
        )
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(
                cond,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:, :, 1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (
            (
                inputs_shared.get("input_audio") is None
                and inputs_shared.get("audio_embeds") is None
            )
            or pipe.audio_encoder is None
            or pipe.audio_processor is None
        ):
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = (
            inputs_shared.get("num_frames"),
            inputs_shared.get("height"),
            inputs_shared.get("width"),
            inputs_shared.get("tiled"),
            inputs_shared.get("tile_size"),
            inputs_shared.get("tile_stride"),
        )
        input_audio, audio_embeds, audio_sample_rate = (
            inputs_shared.pop("input_audio", None),
            inputs_shared.pop("audio_embeds", None),
            inputs_shared.get("audio_sample_rate", 16000),
        )
        s2v_pose_video, s2v_pose_latents, motion_video = (
            inputs_shared.pop("s2v_pose_video", None),
            inputs_shared.pop("s2v_pose_latents", None),
            inputs_shared.pop("motion_video", None),
        )

        audio_input_positive = self.process_audio(
            pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds
        )
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(
            self.process_motion_latents(
                pipe, height, width, tiled, tile_size, tile_stride, motion_video
            )
        )
        inputs_shared.update(
            self.process_pose_cond(
                pipe,
                s2v_pose_video,
                num_frames,
                height,
                width,
                tiled,
                tile_size,
                tile_stride,
                s2v_pose_latents=s2v_pose_latents,
            )
        )
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(
        pipe: WanVideoPipeline,
        input_audio=None,
        audio_sample_rate=16000,
        s2v_pose_video=None,
        num_frames=81,
        height=448,
        width=832,
        fps=16,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        assert (
            pipe.audio_encoder is not None and pipe.audio_processor is not None
        ), "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = (
            shapes["height"],
            shapes["width"],
            shapes["num_frames"],
        )
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(
            pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True
        )
        pose_latents = unit.process_pose_cond(
            pipe,
            s2v_pose_video,
            num_frames,
            height,
            width,
            num_repeats=len(audio_embeds),
            return_all=True,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("latents", "motion_latents", "drop_motion_frames")
        )

    def process(
        self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames
    ):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:, :, 1:]], dim=2)
        return {"latents": latents}


class WanVideoUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "animate_pose_video",
                "animate_face_video",
                "animate_inpaint_video",
                "animate_mask_video",
            ),
            output_params=(
                "animate_pose_video",
                "animate_face_video",
                "animate_inpaint_video",
                "animate_mask_video",
            ),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        animate_pose_video,
        animate_face_video,
        animate_inpaint_video,
        animate_mask_video,
    ):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[: len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[: len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[: len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[: len(input_video) - 4]
        return {
            "animate_pose_video": animate_pose_video,
            "animate_face_video": animate_face_video,
            "animate_inpaint_video": animate_inpaint_video,
            "animate_mask_video": animate_mask_video,
        }


class WanVideoUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            output_params=("pose_latents",),
            onload_model_names=("vae",),
        )

    def process(
        self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride
    ):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(
            animate_pose_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("animate_face_video",),
            output_params=("face_pixel_values"),
        )

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(
            inputs_shared["animate_face_video"]
        )
        inputs_nega["face_pixel_values"] = (
            torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        )
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "animate_inpaint_video",
                "animate_mask_video",
                "input_image",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("y",),
            onload_model_names=("vae",),
        )

    def get_i2v_mask(
        self,
        lat_t,
        lat_h,
        lat_w,
        mask_len=1,
        mask_pixel_values=None,
        device=get_device_type(),
    ):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(
        self,
        pipe: WanVideoPipeline,
        animate_inpaint_video,
        animate_mask_video,
        input_image,
        tiled,
        tile_size,
        tile_stride,
    ):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(
            bg_pixel_values,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape

        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(
            ref_pixel_values,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(
            dtype=torch.bfloat16, device=pipe.device
        )

        mask_pixel_values = 1 - pipe.preprocess_video(
            animate_mask_video, max_value=1, min_value=0
        )
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(
            mask_pixel_values, size=(lat_h, lat_w), mode="nearest"
        )
        mask_pixel_values = rearrange(
            mask_pixel_values, "(b t) c h w -> b t c h w", b=1
        )[:, :, 0]
        msk_reft = self.get_i2v_mask(
            lat_t,
            lat_h,
            lat_w,
            0,
            mask_pixel_values=mask_pixel_values,
            device=pipe.device,
        )

        y_reft = torch.concat([msk_reft, y_reft]).to(
            dtype=torch.bfloat16, device=pipe.device
        )
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class WanVideoUnit_LongCatVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("longcat_video",),
            output_params=("longcat_latents",),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipeline, longcat_video):
        if longcat_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        longcat_video = pipe.preprocess_video(longcat_video)
        longcat_latents = pipe.vae.encode(longcat_video, device=pipe.device).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        return {"longcat_latents": longcat_latents}


class WanVideoUnit_WanToDance_ProcessInputs(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
        )

    def get_music_base_feature(self, music_path, fps=30):
        import librosa

        hop_length = 512
        sr = fps * hop_length
        data, sr = librosa.load(music_path, sr=sr)
        sr = 22050
        envelope = librosa.onset.onset_strength(y=data, sr=sr)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20).T
        chroma = librosa.feature.chroma_cens(
            y=data, sr=sr, hop_length=hop_length, n_chroma=12
        ).T
        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=sr, hop_length=hop_length
        )
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0
        start_bpm = librosa.beat.tempo(y=librosa.load(music_path)[0])[0]
        _, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=sr,
            hop_length=hop_length,
            start_bpm=start_bpm,
            tightness=100,
        )
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0
        audio_feature = np.concatenate(
            [
                envelope[:, None],
                mfcc,
                chroma,
                peak_onehot[:, None],
                beat_onehot[:, None],
            ],
            axis=-1,
        )
        return torch.from_numpy(audio_feature)

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (
            hasattr(pipe.dit, "wantodance_enable_global")
            and pipe.dit.wantodance_enable_global
        ):
            inputs_nega["skip_9th_layer"] = True
        if inputs_shared.get("wantodance_music_path", None) is not None:
            inputs_shared["music_feature"] = self.get_music_base_feature(
                inputs_shared["wantodance_music_path"]
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_WanToDance_RefImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "wantodance_reference_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("wantodance_refimage_feature",),
            onload_model_names=("image_encoder", "vae"),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        wantodance_reference_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if wantodance_reference_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(wantodance_reference_image, list):
            wantodance_reference_image = wantodance_reference_image[0]
        image = pipe.preprocess_image(
            wantodance_reference_image.resize((width, height))
        ).to(
            pipe.device
        )  # B,C,H,W;B=1
        refimage_feature = pipe.image_encoder.encode_image([image])
        refimage_feature = refimage_feature.to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        return {"wantodance_refimage_feature": refimage_feature}


class WanVideoUnit_WanToDance_ImageKeyframesEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "wantodance_keyframes",
                "wantodance_keyframes_mask",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=("clip_feature", "y"),
            onload_model_names=("image_encoder", "vae"),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        wantodance_keyframes,
        wantodance_keyframes_mask,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if wantodance_keyframes is None:
            return {}
        wantodance_keyframes_mask = torch.tensor(wantodance_keyframes_mask)
        pipe.load_models_to_device(self.onload_model_names)
        images = []
        for input_image in wantodance_keyframes:
            input_image = pipe.preprocess_image(input_image.resize((width, height))).to(
                pipe.device
            )
            images.append(input_image)

        clip_context = pipe.image_encoder.encode_image(
            images[:1]
        )  # 取第一帧作为clip输入
        msk = torch.zeros(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, wantodance_keyframes_mask == 1, :, :] = torch.ones(
            1, height // 8, width // 8, device=pipe.device
        )  # set keyframes mask to 1

        images = [image.transpose(0, 1) for image in images]  # 3, num_frames, h, w
        images = torch.concat(images, dim=1)
        vae_input = images

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )  # expand first frame mask, N to N + 3
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            "Wan2.1-T2V-14B": [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            "Wan2.1-I2V-14B-480P": [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ],
            "Wan2.1-I2V-14B-720P": [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x

        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(
                (torch.arange(border_width) + shift) / border_width, dims=(0,)
            )
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask

    def run(
        self,
        model_fn,
        sliding_window_size,
        sliding_window_stride,
        computation_device,
        computation_dtype,
        model_kwargs,
        tensor_names,
        batch_size=None,
    ):
        tensor_names = [
            tensor_name
            for tensor_name in tensor_names
            if model_kwargs.get(tensor_name) is not None
        ]
        tensor_dict = {
            tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names
        }
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = (
            tensor_dict[tensor_names[0]].device,
            tensor_dict[tensor_names[0]].dtype,
        )
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if (
                t - sliding_window_stride >= 0
                and t - sliding_window_stride + sliding_window_size >= T
            ):
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update(
                {
                    tensor_name: tensor_dict[tensor_name][:, :, t:t_:, :].to(
                        device=computation_device, dtype=computation_dtype
                    )
                    for tensor_name in tensor_names
                }
            )
            model_output = model_fn(**model_kwargs).to(
                device=data_device, dtype=data_dtype
            )
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,),
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t:t_, :, :] += model_output * mask
            weight[:, :, t:t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def wantodance_get_single_freqs(freqs, frame_num, fps):
    total_frame = int(30.0 / (fps + 1e-6) * frame_num + 0.5)
    interval_frame = 30.0 / (fps + 1e-6)
    freqs_0 = freqs[:total_frame]
    freqs_new = torch.zeros(
        (frame_num, freqs_0.shape[1]), device=freqs_0.device, dtype=freqs_0.dtype
    )
    freqs_new[0] = freqs_0[0]
    freqs_new[-1] = freqs_0[total_frame - 1]
    for i in range(1, frame_num - 1):
        pos = i * interval_frame
        low_idx = int(pos)
        high_idx = min(low_idx + 1, total_frame - 1)
        weight_high = pos - low_idx
        weight_low = 1.0 - weight_high
        freqs_new[i] = freqs_0[low_idx] * weight_low + freqs_0[high_idx] * weight_high
    return freqs_new


class WanVideoUnit_LatentSequence(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "latents",
                "first_frame_latents",
                "mosaic_latent",
                "mosaic_frame_indices",
            ),
            output_params=(
                "latents",
                "first_frame_count",
                "mosaic_frame_count",
                "condition_frame_count",
                "mosaic_frame_indices",
            ),
        )

    def process(
        self,
        pipe,
        latents,
        first_frame_latents=None,
        mosaic_latent=None,
        mosaic_frame_indices=None,
    ):
        first_frame_count = 0
        mosaic_frame_count = 0
        latent_sequence = []

        if first_frame_latents is not None:
            first_frame_latents = first_frame_latents.to(
                device=latents.device, dtype=latents.dtype
            )
            if (
                first_frame_latents.shape[:2] != latents.shape[:2]
                or first_frame_latents.shape[-2:] != latents.shape[-2:]
            ):
                raise ValueError(
                    "first_frame_latents must share batch/channel/spatial shape with latents, got "
                    f"{tuple(first_frame_latents.shape)} vs {tuple(latents.shape)}."
                )
            first_frame_count = first_frame_latents.shape[2]
            latent_sequence.append(first_frame_latents)

        if mosaic_latent is not None:
            mosaic_latent = mosaic_latent.to(device=latents.device, dtype=latents.dtype)
            if (
                mosaic_latent.shape[:2] != latents.shape[:2]
                or mosaic_latent.shape[-2:] != latents.shape[-2:]
            ):
                raise ValueError(
                    "mosaic_latent and latents must share batch/channel/spatial shape, got "
                    f"{tuple(mosaic_latent.shape)} vs {tuple(latents.shape)}."
                )
            if mosaic_latent.shape[2] > latents.shape[2]:
                raise ValueError(
                    "mosaic_latent temporal length must be <= latents temporal length, got "
                    f"{mosaic_latent.shape[2]} vs {latents.shape[2]}."
                )
            mosaic_frame_count = mosaic_latent.shape[2]
            mosaic_frame_indices = _resolve_mosaic_frame_indices(
                mosaic_frame_indices,
                noisy_frame_count=int(latents.shape[2]),
                mosaic_frame_count=int(mosaic_frame_count),
                device=latents.device,
            )
            latent_sequence.append(mosaic_latent)
        else:
            mosaic_frame_indices = _resolve_mosaic_frame_indices(
                mosaic_frame_indices,
                noisy_frame_count=int(latents.shape[2]),
                mosaic_frame_count=0,
                device=latents.device,
            )

        latent_sequence.append(latents)
        if len(latent_sequence) > 1:
            latents = torch.cat(latent_sequence, dim=2)
        return {
            "latents": latents,
            "first_frame_count": first_frame_count,
            "mosaic_frame_count": mosaic_frame_count,
            "condition_frame_count": first_frame_count + mosaic_frame_count,
            "mosaic_frame_indices": mosaic_frame_indices,
        }


WAN_VIDEO_LATENT_SEQUENCE_UNIT = WanVideoUnit_LatentSequence()


class WanVideoUnit_PropeCamera(PipelineUnit):
    VAE_HW_SCALING = 16
    VAE_T_SCALING = 4

    def __init__(self):
        super().__init__(
            input_params=WAN_VIDEO_PROPE_CAMERA_KEYS + ("mosaic_frame_indices",),
            output_params=("camera_info",),
        )

    def process(
        self,
        pipe,
        use_prope,
        h,
        w,
        dtype,
        device,
        first_frame_count,
        mosaic_frame_count,
        clean_latent_indices_prope_intrinsic=None,
        clean_latent_indices_prope_extrinsic=None,
        noisy_latent_indices_prope_intrinsic=None,
        noisy_latent_indices_prope_extrinsic=None,
        mosaic_frame_indices=None,
        mosaic_view_change=None,
        use_mosaic_view_change_prope=False,
        trans_scale=50.0,
    ):
        if not use_prope:
            return {"camera_info": None}
        if (
            noisy_latent_indices_prope_intrinsic is None
            or noisy_latent_indices_prope_extrinsic is None
        ):
            return {"camera_info": None}

        # Camera-matrix composition runs in fp32 regardless of the model
        # dtype: K normalization, SE(3) inversion and the P/P_T/P_inv
        # einsums suffer catastrophic cancellation in bf16 (relative
        # translations between nearby frames lose 13-32% accuracy). Only
        # the final products are cast back to ``dtype`` for the attention.
        noisy_intrinsic = self._camera_tensor(
            noisy_latent_indices_prope_intrinsic, device, torch.float32
        )
        noisy_extrinsic = self._camera_tensor(
            noisy_latent_indices_prope_extrinsic, device, torch.float32
        )
        intrinsic_parts = []
        extrinsic_parts = []

        if first_frame_count > 0:
            if (
                clean_latent_indices_prope_intrinsic is None
                or clean_latent_indices_prope_extrinsic is None
            ):
                return {"camera_info": None}
            clean_intrinsic = self._camera_tensor(
                clean_latent_indices_prope_intrinsic, device, torch.float32
            )
            clean_extrinsic = self._camera_tensor(
                clean_latent_indices_prope_extrinsic, device, torch.float32
            )
            intrinsic_parts.append(
                clean_intrinsic[:, : first_frame_count * self.VAE_T_SCALING]
            )
            extrinsic_parts.append(
                clean_extrinsic[:, : first_frame_count * self.VAE_T_SCALING]
            )

        if mosaic_frame_count > 0:
            noisy_latent_count = noisy_intrinsic.shape[1] // self.VAE_T_SCALING
            indices = _resolve_mosaic_frame_indices(
                mosaic_frame_indices,
                noisy_frame_count=int(noisy_latent_count),
                mosaic_frame_count=int(mosaic_frame_count),
                device=noisy_intrinsic.device,
            )
            frame_offsets = (
                indices[:, None] * self.VAE_T_SCALING
                + torch.arange(
                    self.VAE_T_SCALING, dtype=torch.long, device=noisy_intrinsic.device
                )[None, :]
            )
            flat_offsets = frame_offsets.reshape(-1)
            intrinsic_parts.append(noisy_intrinsic.index_select(1, flat_offsets))
            extrinsic_parts.append(noisy_extrinsic.index_select(1, flat_offsets))

        intrinsic_parts.append(noisy_intrinsic)
        extrinsic_parts.append(noisy_extrinsic)
        prope_intrinsic = torch.cat(intrinsic_parts, dim=1).reshape(
            noisy_intrinsic.shape[0], -1, self.VAE_T_SCALING, 3, 3
        )
        prope_extrinsic = torch.cat(extrinsic_parts, dim=1).reshape(
            noisy_extrinsic.shape[0], -1, self.VAE_T_SCALING, 4, 4
        )

        w2c = prope_extrinsic.clone()
        # Recenter the world on the FIRST NOISY camera. PRoPE consumes
        # only pairwise products P_i P_j^{-1}, which are exactly invariant
        # to a common world shift (W_i T (W_j T)^{-1} = W_i W_j^{-1}), so
        # this changes nothing semantically -- but it makes the bf16
        # quantization of the final matrices act on window-local
        # translations instead of large absolute world coordinates, which
        # otherwise wipe out the small inter-frame motion by cancellation.
        # The reference must be the noisy window (always present, always
        # window-local), NOT slot 0: with pool history enabled slot 0 is
        # the OLDEST context camera, which can sit far from the generating
        # window and would leave the noisy translations large again.
        # NOTE: the shift-invariance argument above holds for the legacy
        # linear trans_scale (a global world rescale). For the nonlinear
        # "log"/"tanh" modes per-frame compression does NOT commute with
        # world shifts, so the recenter becomes part of the encoding
        # definition; train and inference stay consistent because both go
        # through this unit.
        noisy_first = int(first_frame_count) + int(mosaic_frame_count)
        c_ref = invert_se3(w2c[:, noisy_first, 0])[..., :3, 3]
        w2c[..., :3, 3] = w2c[..., :3, 3] + torch.einsum(
            "bstij,bj->bsti", w2c[..., :3, :3], c_ref
        )
        w2c[..., :3, 3] = self._apply_trans_scale(w2c[..., :3, 3], trans_scale)
        ks_norm = torch.zeros_like(prope_intrinsic)
        image_width = w * self.VAE_HW_SCALING * 2
        image_height = h * self.VAE_HW_SCALING * 2
        ks_norm[..., 0, 0] = prope_intrinsic[..., 0, 0] / image_width
        ks_norm[..., 1, 1] = prope_intrinsic[..., 1, 1] / image_height
        ks_norm[..., 0, 2] = prope_intrinsic[..., 0, 2] / image_width - 0.5
        ks_norm[..., 1, 2] = prope_intrinsic[..., 1, 2] / image_height - 0.5
        ks_norm[..., 2, 2] = 1.0
        p = torch.einsum("...ij,...jk->...ik", lift_k(ks_norm), w2c)
        p_t = p.transpose(-1, -2)
        p_inv = torch.einsum(
            "...ij,...jk->...ik", invert_se3(w2c), lift_k(invert_k(ks_norm))
        )
        # fp32 composition done -- hand the attention the model dtype.
        w2c = w2c.to(dtype)
        p = p.to(dtype)
        p_t = p_t.to(dtype)
        p_inv = p_inv.to(dtype)
        view_change_positions = self._build_view_change_positions(
            use_mosaic_view_change_prope=use_mosaic_view_change_prope,
            mosaic_view_change=mosaic_view_change,
            batch_size=noisy_intrinsic.shape[0],
            first_frame_count=first_frame_count,
            mosaic_frame_count=mosaic_frame_count,
            noisy_frame_count=noisy_intrinsic.shape[1] // self.VAE_T_SCALING,
            h=h,
            w=w,
            device=device,
            dtype=dtype,
        )
        if view_change_positions is not None:
            return {"camera_info": (w2c, (p, p_t, p_inv), view_change_positions)}
        return {"camera_info": (w2c, (p, p_t, p_inv))}

    @staticmethod
    def _apply_trans_scale(t, trans_scale):
        """Compress the (recentered) w2c translation vectors ``t`` (..., 3).

        Numeric ``trans_scale`` keeps the legacy behaviour ``t / trans_scale``
        (a global world rescale; must match the dataset's extrinsic units).
        ``"log"``/``"logd4"``/``"tanh"`` apply a direction-preserving radial
        compression of the magnitude (``||t|| -> log1p(||t||)`` resp.
        ``log1p(||t||) / 4`` resp. ``tanh(||t||)``): small window-local motions
        pass through near-identity instead of being crushed by a fixed divisor,
        while far history baselines are log-compressed (unbounded but
        slow-growing) resp. saturated to 1. ``"logd4"`` is ``"log"`` with the
        whole compressed magnitude divided by 4 (equivalently log base
        ``e**4``): same dynamic range as ``"log"`` but the far baseline is
        re-anchored near ~1 instead of ~4, so large pose gaps no longer inflate
        the attention logits.
        """
        if isinstance(trans_scale, str):
            mode = trans_scale.strip().lower()
            norm = t.norm(dim=-1, keepdim=True)
            if mode == "log":
                factor = torch.log1p(norm) / norm.clamp_min(1e-8)
            elif mode == "logd4":
                factor = torch.log1p(norm) / norm.clamp_min(1e-8) / 4.0
            elif mode == "tanh":
                factor = torch.tanh(norm) / norm.clamp_min(1e-8)
            else:
                try:
                    return t / float(mode)
                except ValueError:
                    raise ValueError(
                        "trans_scale must be a number, 'log', 'logd4' or "
                        f"'tanh', got {trans_scale!r}."
                    ) from None
            return t * factor
        return t / float(trans_scale)

    @staticmethod
    def _canonical_view_change(batch_size, frame_count, h, w, device, dtype):
        out = torch.zeros(
            batch_size,
            int(frame_count),
            int(h),
            int(w),
            3,
            device=device,
            dtype=dtype,
        )
        out[..., 0] = 1.0
        return out

    def _build_view_change_positions(
        self,
        *,
        use_mosaic_view_change_prope,
        mosaic_view_change,
        batch_size,
        first_frame_count,
        mosaic_frame_count,
        noisy_frame_count,
        h,
        w,
        device,
        dtype,
    ):
        if not use_mosaic_view_change_prope:
            return None

        parts = []
        if first_frame_count > 0:
            parts.append(
                self._canonical_view_change(
                    batch_size, first_frame_count, h, w, device, dtype
                )
            )

        if mosaic_frame_count > 0:
            if mosaic_view_change is None:
                raise ValueError(
                    "mosaic_view_change is required when mosaic_view_change_prope "
                    "is enabled and mosaic tokens are present."
                )
            vc = torch.as_tensor(mosaic_view_change, device=device, dtype=dtype)
            if vc.ndim == 4:
                vc = vc.unsqueeze(0)
            if vc.ndim != 5 or vc.shape[-1] != 3:
                raise ValueError(
                    "mosaic_view_change must have shape (T,H,W,3) or "
                    f"(B,T,H,W,3); got {tuple(vc.shape)}."
                )
            if int(vc.shape[1]) != int(mosaic_frame_count):
                raise ValueError(
                    "mosaic_view_change temporal length must match mosaic_frame_count, "
                    f"got {int(vc.shape[1])} vs {int(mosaic_frame_count)}."
                )
            if int(vc.shape[2]) == int(h) * 2 and int(vc.shape[3]) == int(w) * 2:
                vc = vc[:, :, ::2, ::2]
            if int(vc.shape[2]) != int(h) or int(vc.shape[3]) != int(w):
                raise ValueError(
                    "mosaic_view_change spatial shape must match patch grid "
                    f"({int(h)},{int(w)}) or doubled latent grid "
                    f"({int(h) * 2},{int(w) * 2}); got "
                    f"({int(vc.shape[2])},{int(vc.shape[3])})."
                )
            if int(vc.shape[0]) == 1 and int(batch_size) > 1:
                vc = vc.expand(int(batch_size), -1, -1, -1, -1)
            elif int(vc.shape[0]) != int(batch_size):
                raise ValueError(
                    "mosaic_view_change batch size must be 1 or match model batch, "
                    f"got {int(vc.shape[0])} vs {int(batch_size)}."
                )
            valid = torch.isfinite(vc).all(dim=-1, keepdim=True) & (vc[..., 0:1] > 0)
            neutral = torch.zeros_like(vc)
            neutral[..., 0] = 1.0
            parts.append(torch.where(valid, vc, neutral))

        parts.append(
            self._canonical_view_change(
                batch_size, noisy_frame_count, h, w, device, dtype
            )
        )
        return torch.cat(parts, dim=1).reshape(int(batch_size), -1, 3)

    @staticmethod
    def _camera_tensor(value, device, dtype):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 3:
            value = value.unsqueeze(0)
        return value.to(device=device, dtype=dtype)


WAN_VIDEO_PROPE_CAMERA_UNIT = WanVideoUnit_PropeCamera()


def _kv_freqs(dit, temporal_positions, h, w, device):
    """RoPE freqs for a list of frames at the given temporal positions (one int
    per frame), spatial grid h x w. Mirrors the standard (non-revgrid) freqs
    build in ``model_fn_wan_video``: temporal(p) (x) freqs[1][:h] (x) freqs[2][:w]."""
    parts = []
    for p in temporal_positions:
        parts.append(
            torch.cat(
                [
                    dit.freqs[0][int(p)].view(1, 1, 1, -1).expand(1, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1),
                ],
                dim=-1,
            )
        )
    return (
        torch.cat(parts, dim=0)
        .reshape(len(temporal_positions) * h * w, 1, -1)
        .to(device)
    )


def model_fn_causal_kv(
    dit,
    *,
    latents_chunk,
    timestep_frames,
    context,
    camera_info,
    cur_positions,
    cur_frames,
    caches,
    mosaic_latent=None,
    mosaic_positions=None,
    mosaic_frames=None,
    cache_positions=None,
    cache_frames=None,
    cache_read_chunk_id=None,
    cur_cache_chunk_ids=None,
    write_cache=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    """One AR chunk forward with the per-block KV cache.

    ``latents_chunk`` is the CUR latents (C0 seed, or the current N_i chunk),
    ``mosaic_latent`` (optional) is the frozen ``M_{0..i}`` prepended. CUR attends
    to ``[cache(clean prefix), M, CUR]``. Returns the CUR prediction (B,C,F,H,W).
    ``*_positions`` are the CURRENT-window temporal RoPE positions; ``*_frames``
    index ``camera_info`` for PRoPE (positions == frames when camera is built
    per-window like the window-recompute path)."""
    if mosaic_latent is not None:
        x_in = torch.cat([mosaic_latent, latents_chunk], dim=2)
        n_mosaic_frames = int(mosaic_latent.shape[2])
    else:
        x_in = latents_chunk
        n_mosaic_frames = 0
    cur_positions = list(cur_positions)
    cur_frames = list(cur_frames)
    if len(cur_positions) != len(cur_frames):
        raise ValueError(
            "cur_positions length must match cur_frames, "
            f"got {len(cur_positions)} vs {len(cur_frames)}."
        )
    if mosaic_latent is not None:
        if mosaic_positions is None:
            mosaic_positions = list(mosaic_frames) if mosaic_frames is not None else []
        else:
            mosaic_positions = list(mosaic_positions)
        if mosaic_frames is None:
            mosaic_frames = list(mosaic_positions)
        else:
            mosaic_frames = list(mosaic_frames)
        if len(mosaic_positions) != n_mosaic_frames or len(mosaic_frames) != n_mosaic_frames:
            raise ValueError(
                "mosaic_positions/mosaic_frames length must match mosaic_latent T, "
                f"got positions={len(mosaic_positions)} frames={len(mosaic_frames)} "
                f"T={n_mosaic_frames}."
            )
    else:
        mosaic_positions = []
        mosaic_frames = []
    if cache_positions is None and caches:
        cache_positions = caches[0].get("positions")
    if cache_frames is None and caches:
        cache_frames = caches[0].get("frames")
    if cache_positions is None:
        cache_positions = cache_frames
    if cache_frames is None:
        cache_frames = cache_positions
    cache_positions = [] if cache_positions is None else list(cache_positions)
    cache_frames = [] if cache_frames is None else list(cache_frames)
    if len(cache_positions) != len(cache_frames):
        raise ValueError(
            "cache_positions length must match cache_frames, "
            f"got {len(cache_positions)} vs {len(cache_frames)}."
        )
    x = dit.patchify(x_in)
    f, h, w = x.shape[2:]
    tpf = h * w
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    mosaic_tokens = n_mosaic_frames * tpf

    timestep = timestep_frames.to(device=x.device).repeat_interleave(tpf)
    mosaic_hole_token_mask = None
    mosaic_hole_keep = None  # (mosaic_tokens,) True=keep -> attention mask for N
    # Match the window-recompute path: hole mosaic tokens are zeroed, their RoPE
    # freqs are zeroed below, their timestep is set to 1000, AND they are masked
    # out of the noisy chunk's keys (causal + hole mask stacked).
    if mosaic_latent is not None and mosaic_tokens > 0:
        mz = (mosaic_latent == 0).all(dim=(0, 1))  # (Tm, H_lat, W_lat)
        Tm, Hf, Wf = mz.shape
        hole_patch = (
            mz.reshape(Tm, Hf // 2, 2, Wf // 2, 2).all(dim=(2, 4)).flatten()
        )  # (mosaic_tokens,)
        full_mask = torch.zeros(
            timestep.shape[0], dtype=torch.bool, device=timestep.device
        )
        full_mask[:mosaic_tokens] = hole_patch.to(timestep.device)
        mosaic_hole_token_mask = full_mask
        mosaic_hole_keep = (~hole_patch).to(x.device)
        x[:, mosaic_hole_token_mask.to(x.device)] = 0
        timestep = timestep.clone()
        timestep[mosaic_hole_token_mask] = 1000
    t = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
    )
    t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    context = dit.text_embedding(context)
    if x.shape[0] != context.shape[0]:
        x = x.repeat(context.shape[0], 1, 1)

    freqs_positions = list(mosaic_positions) + list(cur_positions)
    freqs_cur = _kv_freqs(dit, freqs_positions, h, w, x.device)
    if mosaic_hole_token_mask is not None:
        freqs_cur = freqs_cur.clone()
        freqs_cur[mosaic_hole_token_mask.to(freqs_cur.device)] = 0
    cache_freqs = (
        _kv_freqs(dit, cache_positions, h, w, x.device) if cache_positions else None
    )
    mframes = list(mosaic_frames)
    cframes = list(cache_frames)

    for bi, block in enumerate(dit.blocks):
        cache_i = caches[bi]
        causal_kv_config = {
            "num_heads": block.self_attn.num_heads,
            "mosaic_tokens": mosaic_tokens,
            "cur_positions": list(cur_positions),
            "cur_frames": list(cur_frames),
            "mosaic_frames": mframes,
            "cache": cache_i,
            "cache_freqs": cache_freqs,
            "cache_frames": cframes,
            "cache_read_chunk_id": cache_read_chunk_id,
            "cur_cache_chunk_ids": cur_cache_chunk_ids,
            "write_cache": write_cache,
            "hole_keep": mosaic_hole_keep,
        }
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
            context,
            t_mod,
            freqs_cur,
            None,
            camera_info,
            None,
            causal_kv_config,
        )

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x[:, :, n_mosaic_frames:]


def init_causal_kv_caches(num_blocks):
    """Per-block empty KV caches (filled lazily by ``causal_self_attention_kv``)."""
    return [
        {"k": None, "v": None, "positions": [], "frames": [], "chunk_ids": []}
        for _ in range(num_blocks)
    ]


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state=None,
    vap_clip_feature=None,
    context_vap=None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    wantodance_refimage_feature=None,
    wantodance_fps: float = 30.0,
    music_feature=None,
    skip_9th_layer: bool = False,
    mosaic_latent: Optional[torch.Tensor] = None,
    mosaic_timestep_zero: bool = True,
    mosaic_revgrid: Optional[np.ndarray] = None,
    mosaic_use_revgrid_rope: bool = False,
    mosaic_view_change: Optional[torch.Tensor] = None,
    mosaic_view_change_prope: bool = False,
    mosaic_mask_holes: bool = True,
    mosaic_drop_holes: bool = False,
    mosaic_frame_indices=None,
    latent_rope_time_indices=None,
    mosaic_debug: bool = False,
    mosaic_sequence_debug: Optional[dict] = None,
    first_frame_latents: Optional[torch.Tensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    subject_ref_latents: Optional[torch.Tensor] = None,
    subject_ref_slot_ratio: float = 0.5,
    subject_ref_time_gap: int = 1,
    subject_ref_prope_mode: str = "identity",
    clean_latent_indices_prope_intrinsic: Optional[torch.Tensor] = None,
    clean_latent_indices_prope_extrinsic: Optional[torch.Tensor] = None,
    noisy_latent_indices_prope_intrinsic: Optional[torch.Tensor] = None,
    noisy_latent_indices_prope_extrinsic: Optional[torch.Tensor] = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        if subject_ref_latents is not None:
            raise ValueError("subject_ref_latents are not supported with sliding_window.")
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )
    if use_unified_sequence_parallel and subject_ref_latents is not None:
        raise ValueError(
            "subject_ref_latents are not yet supported with unified sequence parallel."
        )
    # LongCat-Video
    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )

    # Training-only clean-latent noise augmentation (see WanModel.__init__).
    # Perturbs ONLY the most recent clean latent frame (last frame of
    # `first_frame_latents`); mosaic latents and older clean-history/context
    # latents are left untouched. Disabled by default and never active
    # outside training mode.
    if (
        first_frame_latents is not None
        and dit.training
        and getattr(dit, "clean_latent_noise_enabled", False)
    ):
        noise_prob = float(getattr(dit, "clean_latent_noise_prob", 0.2))
        noise_magnitude = float(getattr(dit, "clean_latent_noise_magnitude", 0.03))
        if noise_magnitude > 0.0 and torch.rand(()).item() < noise_prob:
            first_frame_latents = first_frame_latents.clone()
            last_frame = first_frame_latents[:, :, -1:]
            normal_noise = torch.randn_like(last_frame)
            first_frame_latents[:, :, -1:] = last_frame + normal_noise * noise_magnitude

    # Training-only context-latent noise augmentation, parallel to the clean-
    # latent noise above but targeting the CONTEXT latents -- every clean
    # latent BEFORE the anchor (``first_frame_latents[:, :, :-1]``). Only fires
    # when context blocks are present this step (more than the single anchor
    # latent, i.e. train_context_count_max > 0 and the sampled context > 0).
    # The anchor (last frame) is left to the clean-latent noise above; the two
    # slices are disjoint so they compose. Disabled by default; train-mode only.
    if (
        first_frame_latents is not None
        and first_frame_latents.shape[2] > 1
        and dit.training
        and getattr(dit, "context_latent_noise_enabled", False)
    ):
        ctx_noise_prob = float(getattr(dit, "context_latent_noise_prob", 0.2))
        ctx_noise_magnitude = float(
            getattr(dit, "context_latent_noise_magnitude", 0.03)
        )
        if ctx_noise_magnitude > 0.0 and torch.rand(()).item() < ctx_noise_prob:
            first_frame_latents = first_frame_latents.clone()
            context_frames = first_frame_latents[:, :, :-1]
            context_noise = torch.randn_like(context_frames)
            first_frame_latents[:, :, :-1] = (
                context_frames + context_noise * ctx_noise_magnitude
            )

    # Training-only mosaic exposure-bias augmentation, parallel to the clean-
    # latent noise above. Perturbs ONLY the valid (non-hole) mosaic latents so
    # the model is trained on a slightly-corrupted mosaic conditioning signal,
    # narrowing the gap to inference where the mosaic is warped from
    # model-generated (not GT) frames. Holes stay EXACTLY zero so the
    # patch-level hole detection / drop below is unaffected. Applied before the
    # sequence is assembled and before hole detection so both consume the
    # perturbed (yet hole-preserving) mosaic latent.
    if (
        mosaic_latent is not None
        and dit.training
        and getattr(dit, "mosaic_latent_noise_enabled", False)
    ):
        m_noise_prob = float(getattr(dit, "mosaic_latent_noise_prob", 0.2))
        m_noise_magnitude = float(getattr(dit, "mosaic_latent_noise_magnitude", 0.03))
        if m_noise_magnitude > 0.0 and torch.rand(()).item() < m_noise_prob:
            mosaic_latent = mosaic_latent.clone()
            # A latent position is a hole iff it is zero across all batch and
            # channel dims -- the same reduction the hole-patch detection uses
            # below -- so masking the noise by it keeps holes at exactly zero.
            nonhole = ~(mosaic_latent == 0).all(dim=(0, 1), keepdim=True)
            mosaic_noise = torch.randn_like(mosaic_latent) * m_noise_magnitude
            mosaic_latent = mosaic_latent + mosaic_noise * nonhole.to(
                mosaic_latent.dtype
            )

    mosaic_hole_mask = None
    latent_sequence = WAN_VIDEO_LATENT_SEQUENCE_UNIT.process(
        pipe=None,
        latents=latents,
        first_frame_latents=first_frame_latents,
        mosaic_latent=mosaic_latent,
        mosaic_frame_indices=mosaic_frame_indices,
    )
    latents = latent_sequence["latents"]
    first_frame_count = latent_sequence["first_frame_count"]
    mosaic_frame_count = latent_sequence["mosaic_frame_count"]
    condition_frame_count = latent_sequence["condition_frame_count"]
    mosaic_frame_indices = latent_sequence["mosaic_frame_indices"]
    noisy_frame_count = int(latents.shape[2] - condition_frame_count)

    if mosaic_latent is not None and mosaic_mask_holes:
        # Patch-level hole judgement aligned with the 1x2x2 patch_embedding
        # conv: a mosaic 2x2 patch is a hole iff ALL 4 latents inside are
        # zero. The previous `[:, ::2, ::2]` shortcut only inspected the
        # top-left latent of each patch, which mis-classified patches that
        # had a non-zero TR/BL/BR latent as holes.
        all_zero = (mosaic_latent == 0).all(dim=(0, 1))  # (T_lat, H_lat, W_lat)
        T_lat, H_lat_full, W_lat_full = all_zero.shape
        hole_patch = (
            all_zero.reshape(T_lat, H_lat_full // 2, 2, W_lat_full // 2, 2)
            .all(dim=(2, 4))  # (T_lat, H_lat//2, W_lat//2)
            .flatten()
        )
        prefix = (
            torch.zeros(
                first_frame_count * hole_patch.shape[0] // mosaic_frame_count,
                dtype=torch.bool,
                device=hole_patch.device,
            )
            if first_frame_count > 0
            else None
        )
        suffix = torch.zeros(
            (latents.shape[2] - condition_frame_count)
            * hole_patch.shape[0]
            // mosaic_frame_count,
            dtype=torch.bool,
            device=hole_patch.device,
        )
        masks = []
        if prefix is not None:
            masks.append(prefix)
        masks.extend([hole_patch, suffix])
        mosaic_hole_mask = torch.cat(masks, dim=0)

    # Timestep
    #
    # We need per-frame (separated) timesteps whenever there is a condition slot
    # in the latent sequence: the condition frames must stay at t=0 while only
    # the noisy tail receives the current diffusion step.
    #
    # The legacy condition was `fuse_vae_embedding_in_latents OR (mosaic_timestep_zero AND mosaic_frame_count > 0)`.
    # That breaks for the only_prope inference path where the caller supplies
    # `first_frame_latents` directly (so `WanVideoUnit_ImageEmbedderFused`
    # short-circuits and never sets `fuse_vae_embedding_in_latents=True`) and
    # there is no mosaic latent. Falling through to the uniform-timestep branch
    # would assign the noisy step to the clean first frame too, corrupting the
    # conditioning signal and producing an obvious jump between the user-passed
    # first frame and the model-generated frames.
    #
    # Adding `first_frame_count > 0` makes the branch fire whenever any clean
    # condition is present, mirroring training (which sets the flag explicitly).
    if dit.seperated_timestep and (
        fuse_vae_embedding_in_latents
        or first_frame_count > 0
        or (mosaic_timestep_zero and mosaic_frame_count > 0)
    ):
        patch_count_per_frame = latents.shape[3] * latents.shape[4] // 4
        timestep_scalar = timestep.reshape(-1)[0]
        if condition_frame_count > 0 and mosaic_timestep_zero:
            noisy_steps = (
                torch.ones(
                    (noisy_frame_count,), dtype=latents.dtype, device=latents.device
                )
                * timestep_scalar
            )
            frame_steps = torch.cat(
                [
                    torch.zeros(
                        (condition_frame_count,),
                        dtype=latents.dtype,
                        device=latents.device,
                    ),
                    noisy_steps,
                ],
                dim=0,
            )
            timestep = frame_steps.repeat_interleave(patch_count_per_frame)
        else:
            timestep = torch.concat(
                [
                    torch.zeros(
                        (1, patch_count_per_frame),
                        dtype=latents.dtype,
                        device=latents.device,
                    ),
                    torch.ones(
                        (latents.shape[2] - 1, patch_count_per_frame),
                        dtype=latents.dtype,
                        device=latents.device,
                    )
                    * timestep_scalar,
                ]
            ).flatten()
        if mosaic_hole_mask is not None:
            timestep = timestep.clone()
            timestep[mosaic_hole_mask] = 1000
            # vis_mask(mosaic_hole_mask.reshape(-1,22,40).cpu().data.numpy())
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        )
        if (
            use_unified_sequence_parallel
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [
                torch.nn.functional.pad(
                    chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0
                )
                for chunk in t_chunks
            ]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Camera control
    if (
        hasattr(dit, "wantodance_enable_global")
        and dit.wantodance_enable_global
        and int(wantodance_fps + 0.5) != 30
    ):
        x = dit.patchify(x, control_camera_latents_input, enable_wantodance_global=True)
    else:
        x = dit.patchify(x, control_camera_latents_input)

    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(
            x, pose_latents, face_pixel_values
        )

    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    subject_ref_memory = None
    subject_ref_token_count = 0
    subject_ref_prefix_token_count = 0
    if subject_ref_latents is not None:
        if reference_latents is not None:
            raise ValueError(
                "subject_ref_latents cannot be used together with reference_latents."
            )
        subject_ref_memory = _build_subject_ref_memory_tokens(
            dit,
            subject_ref_latents,
            batch_size=int(x.shape[0]),
            video_h=int(height) if height is not None else int(latents.shape[3] * 16),
            video_w=int(width) if width is not None else int(latents.shape[4] * 16),
            subject_ref_slot_ratio=subject_ref_slot_ratio,
            subject_ref_time_gap=subject_ref_time_gap,
            device=x.device,
            dtype=x.dtype,
        )
        if subject_ref_memory is not None:
            subject_ref_token_count = int(subject_ref_memory["token_count"])
            subject_ref_prefix_token_count = subject_ref_token_count

    prope_camera = WAN_VIDEO_PROPE_CAMERA_UNIT.process(
        pipe=None,
        use_prope=getattr(dit, "use_prope", False),
        h=h,
        w=w,
        dtype=x.dtype,
        device=x.device,
        first_frame_count=first_frame_count,
        mosaic_frame_count=mosaic_frame_count,
        clean_latent_indices_prope_intrinsic=clean_latent_indices_prope_intrinsic,
        clean_latent_indices_prope_extrinsic=clean_latent_indices_prope_extrinsic,
        noisy_latent_indices_prope_intrinsic=noisy_latent_indices_prope_intrinsic,
        noisy_latent_indices_prope_extrinsic=noisy_latent_indices_prope_extrinsic,
        mosaic_frame_indices=mosaic_frame_indices,
        mosaic_view_change=mosaic_view_change,
        use_mosaic_view_change_prope=mosaic_view_change_prope,
        trans_scale=getattr(dit, "trans_scale", 50.0),
    )
    camera_info = prope_camera["camera_info"]
    full_frame_count = first_frame_count + mosaic_frame_count + noisy_frame_count
    if subject_ref_prefix_token_count > 0:
        ref_zero_timestep = torch.zeros(
            subject_ref_prefix_token_count,
            dtype=latents.dtype,
            device=x.device,
        )
        ref_t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, ref_zero_timestep).unsqueeze(0)
        )
        ref_t_mod = dit.time_projection(ref_t).unflatten(2, (6, dit.dim))
        if t_mod.ndim != 4:
            raise ValueError(
                "subject_ref prefix memory requires per-token t_mod; "
                f"got shape {tuple(t_mod.shape)}."
            )
        t = torch.cat([ref_t.to(dtype=t.dtype), t], dim=1)
        t_mod = torch.cat([ref_t_mod.to(dtype=t_mod.dtype), t_mod], dim=1)
        x = torch.cat([subject_ref_memory["x"], x], dim=1)
        camera_info = _prepend_subject_ref_prope_camera_info(
            camera_info,
            prefix_token_count=subject_ref_prefix_token_count,
            tokens_per_frame=h * w,
            frame_count=full_frame_count,
            mode=subject_ref_prope_mode,
            clean_anchor_token_index=max(0, int(first_frame_count) - 1) * int(h * w),
        )
    mosaic_debug_enabled = (
        bool(mosaic_debug) or os.environ.get("MOSAIC_VERBOSE", "0") == "1"
    )
    if mosaic_debug_enabled and mosaic_frame_count > 0:
        camera_frames = None if camera_info is None else int(camera_info[0].shape[1])
        print(
            f"[mosaic_interval:pipeline.camera] first={first_frame_count} "
            f"mosaic={mosaic_frame_count} noisy={noisy_frame_count} "
            f"indices={mosaic_frame_indices.detach().cpu().tolist()} "
            f"camera_frames={camera_frames}"
        )

    freq_dev = dit.freqs[0].device
    rope_time_indices = _resolve_latent_rope_time_indices(
        latent_rope_time_indices,
        first_frame_count=first_frame_count,
        mosaic_frame_count=mosaic_frame_count,
        noisy_frame_count=noisy_frame_count,
        mosaic_frame_indices=mosaic_frame_indices,
        device=freq_dev,
    )
    if rope_time_indices.numel() and int(rope_time_indices.max().item()) >= int(
        dit.freqs[0].shape[0]
    ):
        raise ValueError(
            "latent_rope_time_indices contains a temporal index outside "
            f"dit.freqs[0], max={int(rope_time_indices.max().item())}, "
            f"available={int(dit.freqs[0].shape[0])}."
        )
    first_rope_times = rope_time_indices[:first_frame_count]
    mosaic_rope_times = rope_time_indices[
        first_frame_count : first_frame_count + mosaic_frame_count
    ]
    noisy_rope_times = rope_time_indices[first_frame_count + mosaic_frame_count :]

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    if (
        mosaic_use_revgrid_rope
        and mosaic_frame_count > 0
        and mosaic_revgrid is not None
        and len(dit.freqs) >= 5
    ):
        freq_chunks = []
        if reference_latents is not None:
            ref_freqs = torch.cat(
                [
                    dit.freqs[0][:1].view(1, 1, 1, -1).expand(1, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1),
                ],
                dim=-1,
            )
            freq_chunks.append(ref_freqs)
        if first_frame_count > 0:
            first_freqs = torch.cat(
                [
                    dit.freqs[0][first_rope_times]
                    .view(first_frame_count, 1, 1, -1)
                    .expand(first_frame_count, h, w, -1),
                    dit.freqs[1][:h]
                    .view(1, h, 1, -1)
                    .expand(first_frame_count, h, w, -1),
                    dit.freqs[2][:w]
                    .view(1, 1, w, -1)
                    .expand(first_frame_count, h, w, -1),
                ],
                dim=-1,
            )
            freq_chunks.append(first_freqs)

        noisy_freqs = torch.cat(
            [
                dit.freqs[0][noisy_rope_times]
                .view(noisy_frame_count, 1, 1, -1)
                .expand(noisy_frame_count, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(noisy_frame_count, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(noisy_frame_count, h, w, -1),
            ],
            dim=-1,
        )

        rg_t = torch.from_numpy(np.asarray(mosaic_revgrid, dtype=np.float32)).to(
            freq_dev
        )
        if int(rg_t.shape[0]) != int(mosaic_frame_count):
            raise ValueError(
                "mosaic_revgrid temporal length must match mosaic_frame_count, got "
                f"{int(rg_t.shape[0])} vs {int(mosaic_frame_count)}."
            )
        mosaic_rope_frames = []
        for idx in range(mosaic_frame_count):
            rope_t = (
                dit.freqs[0][mosaic_rope_times[idx]]
                .view(1, 1, 1, -1)
                .expand(1, h, w, -1)
            )
            rg = rg_t[idx, ::2, ::2]  # (h, w, 2)  x, y in latent coords
            ql_h = (rg[..., 1] * 8).long().clamp(0, dit.freqs[3].shape[0] - 1)
            ql_w = (rg[..., 0] * 8).long().clamp(0, dit.freqs[4].shape[0] - 1)
            rope_h = dit.freqs[3][ql_h].reshape(1, h, w, -1)
            rope_w = dit.freqs[4][ql_w].reshape(1, h, w, -1)
            invalid = (rg[..., 0] < 0) | (rg[..., 1] < 0)
            if invalid.any():
                inv = invalid.view(1, h, w, 1)
                rope_h = torch.where(
                    inv, dit.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1), rope_h
                )
                rope_w = torch.where(
                    inv, dit.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1), rope_w
                )
            mosaic_rope_frames.append(torch.cat([rope_t, rope_h, rope_w], dim=-1))

        mosaic_freqs = torch.cat(mosaic_rope_frames, dim=0)
        freq_chunks.extend([mosaic_freqs, noisy_freqs])

        if (
            not getattr(model_fn_wan_video, "_rope_logged", False)
            and os.environ.get("MOSAIC_VERBOSE", "0") == "1"
        ):
            model_fn_wan_video._rope_logged = True
            std_freqs = torch.cat(
                [
                    dit.freqs[0][mosaic_rope_times]
                    .view(mosaic_frame_count, 1, 1, -1)
                    .expand(mosaic_frame_count, h, w, -1),
                    dit.freqs[1][:h]
                    .view(1, h, 1, -1)
                    .expand(mosaic_frame_count, h, w, -1),
                    dit.freqs[2][:w]
                    .view(1, 1, w, -1)
                    .expand(mosaic_frame_count, h, w, -1),
                ],
                dim=-1,
            )
            diff = (mosaic_freqs - std_freqs).abs()
            print(
                f"[rope debug] mosaic_freqs vs standard_grid: "
                f"max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}, "
                f"revgrid shape={mosaic_revgrid.shape}, "
                f"invalid_pct={(rg_t < 0).any(dim=-1).float().mean().item()*100:.1f}%"
            )

        freqs = torch.cat(freq_chunks, dim=0).reshape(f * h * w, 1, -1).to(x.device)
    else:
        freq_chunks = []
        if reference_latents is not None:
            freq_chunks.append(
                torch.cat(
                    [
                        dit.freqs[0][:1].view(1, 1, 1, -1).expand(1, h, w, -1),
                        dit.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),
                        dit.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1),
                    ],
                    dim=-1,
                )
            )
        if first_frame_count > 0:
            freq_chunks.append(
                torch.cat(
                    [
                        dit.freqs[0][first_rope_times]
                        .view(first_frame_count, 1, 1, -1)
                        .expand(first_frame_count, h, w, -1),
                        dit.freqs[1][:h]
                        .view(1, h, 1, -1)
                        .expand(first_frame_count, h, w, -1),
                        dit.freqs[2][:w]
                        .view(1, 1, w, -1)
                        .expand(first_frame_count, h, w, -1),
                    ],
                    dim=-1,
                )
            )
        if mosaic_frame_count > 0:
            freq_chunks.append(
                torch.cat(
                    [
                        dit.freqs[0][mosaic_rope_times]
                        .view(mosaic_frame_count, 1, 1, -1)
                        .expand(mosaic_frame_count, h, w, -1),
                        dit.freqs[1][:h]
                        .view(1, h, 1, -1)
                        .expand(mosaic_frame_count, h, w, -1),
                        dit.freqs[2][:w]
                        .view(1, 1, w, -1)
                        .expand(mosaic_frame_count, h, w, -1),
                    ],
                    dim=-1,
                )
            )
        freq_chunks.append(
            torch.cat(
                [
                    dit.freqs[0][noisy_rope_times]
                    .view(noisy_frame_count, 1, 1, -1)
                    .expand(noisy_frame_count, h, w, -1),
                    dit.freqs[1][:h]
                    .view(1, h, 1, -1)
                    .expand(noisy_frame_count, h, w, -1),
                    dit.freqs[2][:w]
                    .view(1, 1, w, -1)
                    .expand(noisy_frame_count, h, w, -1),
                ],
                dim=-1,
            )
        )
        freqs = torch.cat(freq_chunks, dim=0).reshape(f * h * w, 1, -1).to(x.device)
    if subject_ref_prefix_token_count > 0:
        freqs = torch.cat(
            [subject_ref_memory["freqs"].to(device=x.device), freqs], dim=0
        )
    if mosaic_debug_enabled and mosaic_frame_count > 0:
        revgrid_invalid = None
        if mosaic_revgrid is not None:
            _rg_dbg = np.asarray(mosaic_revgrid)
            revgrid_invalid = float((_rg_dbg < 0).any(axis=-1).mean() * 100.0)
        print(
            f"[mosaic_interval:pipeline.rope] first={first_frame_count} "
            f"mosaic={mosaic_frame_count} noisy={noisy_frame_count} "
            f"freqs={tuple(freqs.shape)} revgrid_invalid_pct={revgrid_invalid}"
        )

    # VAP
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, "b c f h w -> b (f h w) c").contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(
            timestep.dtype
        )
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f, h, w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    # WanToDance
    if hasattr(dit, "wantodance_enable_global") and dit.wantodance_enable_global:
        if wantodance_refimage_feature is not None:
            refimage_feature_embedding = dit.img_emb_refimage(
                wantodance_refimage_feature
            )
            context = torch.cat([refimage_feature_embedding, context], dim=1)
        if (dit.wantodance_enable_dynamicfps or dit.wantodance_enable_unimodel) and int(
            wantodance_fps + 0.5
        ) != 30:
            freqs_0 = wantodance_get_single_freqs(dit.freqs[0], f, wantodance_fps)
            freqs = (
                torch.cat(
                    [
                        freqs_0.view(f, 1, 1, -1).expand(f, h, w, -1),
                        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                    ],
                    dim=-1,
                )
                .reshape(f * h * w, 1, -1)
                .to(x.device)
            )
        if (
            dit.wantodance_enable_global
            or dit.wantodance_enable_dynamicfps
            or dit.wantodance_enable_unimodel
        ):
            if use_unified_sequence_parallel:
                length = (
                    int(
                        float(music_feature.shape[0])
                        / get_sequence_parallel_world_size()
                    )
                    * get_sequence_parallel_world_size()
                )
                music_feature = music_feature[:length]
                music_feature = torch.chunk(
                    music_feature, get_sequence_parallel_world_size(), dim=0
                )[get_sequence_parallel_rank()]
            if not dit.training:
                dit.music_encoder.to(x.device, dtype=x.dtype)  # only evaluation
            music_feature = music_feature.to(x.device, dtype=x.dtype)
            music_feature = dit.music_projection(music_feature)
            music_feature = dit.music_encoder(music_feature)
            if music_feature.dim() == 2:
                music_feature = music_feature.unsqueeze(0)
            if use_unified_sequence_parallel:
                if dist.is_initialized() and dist.get_world_size() > 1:
                    music_feature = get_sp_group().all_gather(music_feature, dim=1)
            music_feature = music_feature.unsqueeze(1)  # [1, 1, 149, 4800]
            N = 149
            M = 4800
            music_feature = torch.nn.functional.interpolate(
                music_feature, size=(N, M), mode="bilinear"
            )
            music_feature = music_feature.squeeze(1)  # shape: [1, 149, 4800]
        if music_feature is not None:
            if music_feature.dim() == 2:
                music_feature = music_feature.unsqueeze(0)
            music_feature = music_feature.to(x.device, dtype=x.dtype)
            interp_mode = "bilinear"
            if interp_mode == "bilinear":
                frame_num = (
                    latents.shape[2] if len(latents.shape) == 5 else latents.shape[1]
                )  # 21
                context_shape_end = context.shape[2]  ## 14B 5120
                music_feature = music_feature.unsqueeze(1)  # shape: [1, 1, 149, 4800]
                if use_unified_sequence_parallel:
                    N = (
                        int(float(frame_num * 8) / get_sequence_parallel_world_size())
                        * get_sequence_parallel_world_size()
                    )
                else:
                    N = frame_num * 8
                music_feature = torch.nn.functional.interpolate(
                    music_feature, size=(N, context_shape_end), mode="bilinear"
                )
                music_feature = music_feature.squeeze(
                    1
                )  # shape: [1, N, context_shape_end]
                if use_unified_sequence_parallel:
                    dit.merged_audio_emb = torch.chunk(
                        music_feature, get_sequence_parallel_world_size(), dim=1
                    )[get_sequence_parallel_rank()]
                else:
                    dit.merged_audio_emb = music_feature
            else:
                dit.merged_audio_emb = music_feature

    legacy_reference_token_count = h * w if reference_latents is not None else 0
    reference_token_count = int(legacy_reference_token_count)
    cross_attn_keep_mask = None
    if mosaic_frame_count > 0 or subject_ref_prefix_token_count > 0:
        cross_attn_keep_mask = _build_mosaic_cross_attn_keep_mask(
            prefix_memory_token_count=subject_ref_prefix_token_count,
            reference_token_count=reference_token_count,
            first_frame_count=first_frame_count,
            mosaic_frame_count=mosaic_frame_count,
            noisy_frame_count=noisy_frame_count,
            tokens_per_frame=h * w,
            device=x.device,
        )
        if mosaic_debug_enabled:
            ref_prefix_start = 0
            ref_prefix_end = subject_ref_prefix_token_count
            first_start = ref_prefix_end + reference_token_count
            mosaic_start = first_start + first_frame_count * h * w
            mosaic_end = mosaic_start + mosaic_frame_count * h * w
            noisy_end = mosaic_end + noisy_frame_count * h * w
            print(
                f"[mosaic_interval:pipeline.cross_attn_mask] "
                f"subject_ref_tokens=[{ref_prefix_start},{ref_prefix_end}) "
                f"first_tokens=[{first_start},{mosaic_start}) "
                f"mosaic_tokens=[{mosaic_start},{mosaic_end}) "
                f"noisy_tokens=[{mosaic_end},{noisy_end}) "
                f"keep={int(cross_attn_keep_mask.sum().item())} "
                f"zero={int((~cross_attn_keep_mask).sum().item())}"
            )

    # Mosaic hole masking: by default mask-in-place (legacy behaviour). When
    # ``mosaic_drop_holes=True`` we actually drop the hole tokens from the
    # transformer sequence: index_select x, freqs, t_mod (if seq-extended),
    # cross_attn_keep_mask, and re-broadcast PROPE viewmats from (s, t) to
    # per-token (s_kept,) so the downstream attention sees a strictly
    # shorter sequence with no zero-padding.
    mosaic_attn_mask = None
    mosaic_kept_frame_ids_per_token: Optional[torch.Tensor] = None
    mosaic_token_slice: Optional[Tuple[int, int]] = None
    drop_holes_keep_idx_full: Optional[torch.Tensor] = None
    drop_holes_pre_seq_len: Optional[int] = None
    if isinstance(mosaic_sequence_debug, dict):
        tokens_per_frame_int = int(h * w)
        reference_tokens = int(reference_token_count)
        first_tokens = int(first_frame_count) * tokens_per_frame_int
        mosaic_tokens_before = int(mosaic_frame_count) * tokens_per_frame_int
        noisy_tokens = int(noisy_frame_count) * tokens_per_frame_int
        seq_len_before_drop = int(x.shape[1])
        mosaic_hole_tokens = (
            int(mosaic_hole_mask.sum().item()) if mosaic_hole_mask is not None else 0
        )
        mosaic_content_tokens = max(0, mosaic_tokens_before - mosaic_hole_tokens)
        cross_attn_masked_tokens = (
            int((~cross_attn_keep_mask).sum().item())
            if cross_attn_keep_mask is not None
            else 0
        )
        if int(mosaic_frame_count) <= 0:
            status = "no_mosaic"
        elif mosaic_hole_mask is None:
            status = "mosaic_no_hole_mask"
        elif bool(mosaic_drop_holes):
            status = "drop_holes"
        else:
            status = "mask_holes"
        mosaic_sequence_debug.update(
            {
                "status": status,
                "drop_holes_enabled": bool(mosaic_drop_holes),
                "seq_len_before_drop": seq_len_before_drop,
                "seq_len_after_drop": seq_len_before_drop,
                "dropped_tokens": 0,
                "drop_ratio": 0.0,
                "keep_ratio": 1.0,
                "reference_tokens": reference_tokens,
                "first_tokens": first_tokens,
                "mosaic_tokens_before": mosaic_tokens_before,
                "mosaic_tokens_after": mosaic_content_tokens,
                "mosaic_hole_tokens": mosaic_hole_tokens,
                "noisy_tokens": noisy_tokens,
                "tokens_per_frame": tokens_per_frame_int,
                "mosaic_frame_count": int(mosaic_frame_count),
                "noisy_frame_count": int(noisy_frame_count),
                "cross_attn_masked_tokens": cross_attn_masked_tokens,
                "subject_ref_tokens": int(subject_ref_token_count),
                "legacy_reference_tokens": int(legacy_reference_token_count),
            }
        )
    if mosaic_hole_mask is not None:
        _hm = mosaic_hole_mask.to(x.device)
        hm_parts = []
        if subject_ref_prefix_token_count > 0:
            hm_parts.append(
                torch.zeros(
                    subject_ref_prefix_token_count,
                    dtype=torch.bool,
                    device=x.device,
                )
            )
        if reference_token_count > 0:
            hm_parts.append(
                torch.zeros(
                    reference_token_count,
                    dtype=torch.bool,
                    device=x.device,
                )
            )
        hm_parts.append(_hm)
        _hm_full = torch.cat(hm_parts, dim=0)

        if bool(mosaic_drop_holes):
            keep_full = ~_hm_full
            keep_idx_full = torch.nonzero(keep_full, as_tuple=False).squeeze(-1)
            keep_idx_latent = torch.nonzero(~_hm, as_tuple=False).squeeze(-1)
            drop_holes_keep_idx_full = keep_idx_full
            drop_holes_pre_seq_len = int(x.shape[1])

            x = x.index_select(1, keep_idx_full)
            freqs = freqs.index_select(0, keep_idx_full)
            if cross_attn_keep_mask is not None:
                cross_attn_keep_mask = cross_attn_keep_mask.index_select(
                    0, keep_idx_full
                )
            time_keep_idx = (
                keep_idx_full
                if subject_ref_prefix_token_count > 0
                else keep_idx_latent
            )
            if t_mod.ndim == 4:
                t_mod = t_mod.index_select(1, time_keep_idx)
            if t.ndim == 3:
                t = t.index_select(1, time_keep_idx)

            tokens_per_frame_int = h * w
            full_frame_count = (
                first_frame_count + mosaic_frame_count + noisy_frame_count
            )
            frame_ids_latent = torch.arange(
                full_frame_count, device=x.device, dtype=torch.long
            ).repeat_interleave(tokens_per_frame_int)
            mosaic_kept_frame_ids_per_token = frame_ids_latent.index_select(
                0, keep_idx_latent
            )

            first_count = int(first_frame_count) * tokens_per_frame_int
            full_mosaic_count = int(mosaic_frame_count) * tokens_per_frame_int
            mosaic_keep_latent = (~_hm)[first_count : first_count + full_mosaic_count]
            mosaic_kept_count = int(mosaic_keep_latent.sum().item())
            mosaic_start_new = (
                int(subject_ref_prefix_token_count)
                + int(reference_token_count)
                + first_count
            )
            mosaic_end_new = mosaic_start_new + mosaic_kept_count
            mosaic_token_slice = (mosaic_start_new, mosaic_end_new)
            if isinstance(mosaic_sequence_debug, dict):
                seq_len_after_drop = int(x.shape[1])
                dropped_tokens = max(
                    0,
                    int(drop_holes_pre_seq_len or seq_len_after_drop)
                    - seq_len_after_drop,
                )
                seq_len_before_drop = int(
                    mosaic_sequence_debug.get(
                        "seq_len_before_drop",
                        drop_holes_pre_seq_len or seq_len_after_drop,
                    )
                )
                mosaic_sequence_debug.update(
                    {
                        "seq_len_after_drop": seq_len_after_drop,
                        "dropped_tokens": dropped_tokens,
                        "drop_ratio": (
                            dropped_tokens / seq_len_before_drop
                            if seq_len_before_drop > 0
                            else 0.0
                        ),
                        "keep_ratio": (
                            seq_len_after_drop / seq_len_before_drop
                            if seq_len_before_drop > 0
                            else 1.0
                        ),
                        "mosaic_tokens_after": mosaic_kept_count,
                        "mosaic_token_slice_start": mosaic_start_new,
                        "mosaic_token_slice_end": mosaic_end_new,
                    }
                )

            # Re-broadcast PROPE viewmats from (b, s, ...) to per-token
            # (b, s_kept, ...) using the same kept indices: each frame's
            # (p, p_t, p_inv) gets repeated ``tokens_per_frame`` times and
            # then sliced down to the kept positions. This keeps the
            # noisy-aligned PROPE concept intact (mosaic[i] still reuses
            # noisy[i]'s camera at the same temporal position) while
            # accommodating the dropped tokens. View-change ProPE carries
            # per-token positions in a third tuple element and must be sliced
            # with the same latent keep indices.
            if subject_ref_prefix_token_count > 0:
                camera_info = _reindex_token_prope_camera_info(
                    camera_info,
                    keep_idx_full,
                )
            else:
                camera_info = _drop_holes_reindex_prope_camera_info(
                    camera_info,
                    full_frame_count=full_frame_count,
                    tokens_per_frame=tokens_per_frame_int,
                    keep_idx_latent=keep_idx_latent,
                )
        else:
            x[:, _hm_full] = 0
            freqs[_hm_full] = 0
            mosaic_attn_mask = (~_hm_full).view(1, 1, 1, -1)

    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [
                torch.nn.functional.pad(
                    chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0
                )
                for chunk in chunks
            ]
            x = chunks[get_sequence_parallel_rank()]
            if cross_attn_keep_mask is not None:
                mask_chunks = torch.chunk(
                    cross_attn_keep_mask, get_sequence_parallel_world_size(), dim=0
                )
                mask_chunks = [
                    torch.nn.functional.pad(
                        chunk, (0, mask_chunks[0].shape[0] - chunk.shape[0]), value=True
                    )
                    for chunk in mask_chunks
                ]
                cross_attn_keep_mask = mask_chunks[get_sequence_parallel_rank()]
                if mosaic_debug_enabled:
                    print(
                        f"[mosaic_interval:pipeline.cross_attn_mask_sp] "
                        f"rank={get_sequence_parallel_rank()} "
                        f"chunk_len={int(cross_attn_keep_mask.shape[0])}"
                    )

    if vace_context is not None:
        vace_hints = vace(
            x,
            vace_context,
            context,
            t_mod,
            freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
    if tea_cache_update:
        x = tea_cache.update(x)
    else:

        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)

            return custom_forward

        # Block
        for block_id, block in enumerate(dit.blocks):
            if skip_9th_layer:
                # This is only used in WanToDance
                if block_id == 9:
                    continue
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x,
                            context,
                            t_mod,
                            freqs,
                            x_vap,
                            context_vap,
                            t_mod_vap,
                            freqs_vap,
                            block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x,
                        context,
                        t_mod,
                        freqs,
                        x_vap,
                        context_vap,
                        t_mod_vap,
                        freqs_vap,
                        block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(
                        block,
                        x,
                        context,
                        t_mod,
                        freqs,
                        x_vap,
                        context_vap,
                        t_mod_vap,
                        freqs_vap,
                        block_id,
                    )
            else:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x,
                    context,
                    t_mod,
                    freqs,
                    mosaic_attn_mask,
                    camera_info,
                    cross_attn_keep_mask,
                )

            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                x = x + current_vace_hint * vace_scale

            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)

            # WanToDance
            if (
                hasattr(dit, "wantodance_enable_music_inject")
                and dit.wantodance_enable_music_inject
            ):
                x = dit.wantodance_after_transformer_block(block_id, x)
        if tea_cache is not None:
            tea_cache.store(x)

    if (
        hasattr(dit, "wantodance_enable_unimodel")
        and dit.wantodance_enable_unimodel
        and int(wantodance_fps + 0.5) != 30
    ):
        x = dit.head_global(x, t)
    else:
        x = dit.head(x, t)

    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    if drop_holes_keep_idx_full is not None and drop_holes_pre_seq_len is not None:
        # Scatter the per-kept-token head outputs back into a full-length
        # sequence with zeros at hole positions, so the downstream
        # ``dit.unpatchify`` can reshape against the unmodified (f, h, w)
        # grid without seeing a shorter-than-expected x.
        full_out = x.new_zeros(x.shape[0], drop_holes_pre_seq_len, x.shape[2])
        full_out.index_copy_(1, drop_holes_keep_idx_full, x)
        x = full_out
    if subject_ref_prefix_token_count > 0:
        x = x[:, subject_ref_prefix_token_count:]
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1] :]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    if condition_frame_count > 0:
        x = x[:, :, condition_frame_count:, :, :]
    return x

def model_fn_longcat_video(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    longcat_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    if longcat_latents is not None:
        latents[:, :, : longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0
    context = context.unsqueeze(0)
    encoder_attention_mask = torch.any(context != 0, dim=-1)[:, 0].to(torch.int64)
    output = dit(
        latents,
        timestep,
        context,
        encoder_attention_mask,
        num_cond_latents=num_cond_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    output = -output
    output = output.to(latents.dtype)
    return output


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = (
        torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    )
    x, (f, h, w) = dit.patchify(
        dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents)
    )
    seq_len_x = seq_len_x_global = x.shape[
        1
    ]  # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = (
        torch.cat(
            [torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1
        )
        .to(torch.long)
        .to(x.device)
    )
    # freqs
    pre_compute_freqs = rope_precompute(
        x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads),
        grid_sizes,
        dit.freqs,
        start=None,
    )
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(
        x,
        pre_compute_freqs,
        mask,
        motion_latents,
        drop_motion_frames=drop_motion_frames,
        add_last_motion=2,
    )

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat(
        [timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)]
    )
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = (
        dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)
    )

    if (
        use_unified_sequence_parallel
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        world_size, sp_rank = (
            get_sequence_parallel_world_size(),
            get_sequence_parallel_rank(),
        )
        assert (
            x.shape[1] % world_size == 0
        ), f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(
            torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy()
        )
        seq_len_x_list = [
            min(max(0, seq_len_x - seg_idxs[i]), x.shape[1])
            for i in range(len(seg_idxs) - 1)
        ]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        x = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
            context,
            t_mod,
            seq_len_x,
            pre_compute_freqs[0],
        )
        x = gradient_checkpoint_forward(
            lambda x: dit.after_transformer_block(
                block_id, x, audio_emb_global, merged_audio_emb, seq_len_x
            ),
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
        )

    if (
        use_unified_sequence_parallel
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x
