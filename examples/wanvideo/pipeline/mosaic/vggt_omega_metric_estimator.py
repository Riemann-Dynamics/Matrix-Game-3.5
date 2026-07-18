# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT-Omega *metric* estimator: a DA3-compatible drop-in for FrustumHandler.

This wraps a VGGT-Omega model plus the DepthAnything-3 (DA3) metric branch
behind the exact runtime contract that
``frustum.frustum_handler.FrustumHandler`` expects from a
``depth_anything_3.api.DepthAnything3`` instance, so it can be selected as an
alternative registration estimator via
``--registration_estimator vggt_omega_metric`` (see
``mosaic/config.py`` and ``mosaic/validation.py``).

Contract mirrored (see ``register_source_sequence`` / ``reestimate_depth`` in
``frustum/frustum_handler.py``):

* ``.to(device)`` / ``.cuda()`` / ``.cpu()`` / ``.eval()`` return ``self``.
* ``.inference(frames, use_ray_pose=..., process_res=..., **_)`` returns a
  ``Prediction``-shaped object with numpy fields, exactly like DA3:
    - ``.depth``      ``(N, H, W)``  float32, **metric meters**
    - ``.extrinsics`` ``(N, 3, 4)``  float32, **world-to-camera** (OpenCV)
    - ``.intrinsics`` ``(N, 3, 3)``  float32, **pixel-space** at the VGGT
      processing resolution
  The handler reads only ``extrinsics[:, :3, :]`` (so 3x4 matches DA3's own
  top-3-rows usage) and re-normalises intrinsics to the frame resolution via
  ``ensure_pixel_intrinsics``; depth is spatially resampled to the latent grid.
* ``.da3_compatible = True`` so the handler routes this through its DA3 branch
  (the handler checks this flag in addition to ``isinstance(DepthAnything3)``),
  without ``frustum/`` having to import this module.

NOTE: unlike standalone DA3, ``vggt_omega_metric`` does **not** remove the DA3
dependency. VGGT-Omega's depth and camera translation are in a single arbitrary
relative scale; the global metric scale ``s`` is recovered by aligning VGGT's
depth to DA3's metric branch and then multiplied into both depth and the camera
translation (rotation and intrinsics untouched). So selecting this estimator
loads TWO models: VGGT-Omega (~1B) and the DA3 metric ViT-L branch (the giant
anyview ViT-G branch of the nested checkpoint is dropped to save VRAM).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


# Genie cluster default; overridable via --vggt_omega_ckpt or $VGGT_OMEGA_CKPT.
_DEFAULT_VGGT_OMEGA_CKPT = (
    "runjia.qian/huggingface_models/VGGT-Omega/vggt_omega_1b_512.pt"
)

# Below this magnitude the recovered metric scale ``s`` is treated as degenerate
# (a collapsed / sign-flipped least-squares solution) rather than a valid metric
# alignment, even if the runner reported "success".
_MIN_VALID_SCALE = 1e-4

# ``_ensure_paths_on_syspath`` only needs to run once; cache so the per-call
# ``inference()`` invocation does not re-stat the filesystem every registration.
_PATHS_ENSURED = False


def _is_main_process() -> bool:
    """Best-effort rank-0 check used to gate noisy prints under DDP.

    Each rank validates an independent sample, so an unguarded ``print`` is
    emitted ``world_size`` times. Defaults to ``True`` (print) when rank cannot
    be determined, so single-process runs are unaffected.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    for env_name in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        val = os.environ.get(env_name)
        if val is not None:
            try:
                return int(val) == 0
            except ValueError:
                return True
    return True


def _ensure_paths_on_syspath():
    """Make ``vggt_omega`` and the in-repo ``depth_anything_3`` importable.

    ``vggt_omega`` is run from source (not pip-installed), and
    ``vggt_omega.metric.scale_alignment`` imports ``depth_anything_3`` at module
    load time, so both source roots must be on ``sys.path`` *before* we import
    the metric package. The DA3 package lives under
    ``third_party/depth-anything-3/src`` (see its pyproject
    ``packages = ["src/depth_anything_3"]``); the parent dir is kept too for
    parity with the legacy ``_init_da3_depth_estimator``.
    """
    global _PATHS_ENSURED
    if _PATHS_ENSURED:
        return
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    rels = (
        "vggt-omega",
        os.path.join("third_party", "depth-anything-3", "src"),
        os.path.join("third_party", "depth-anything-3"),
    )
    for rel in rels:
        p = os.path.join(repo_root, rel)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    _PATHS_ENSURED = True


@dataclass
class VGGTOmegaPrediction:
    """A DA3 ``Prediction``-shaped result, the subset FrustumHandler reads."""

    depth: np.ndarray  # (N, H, W) float32, metric meters
    extrinsics: np.ndarray  # (N, 3, 4) float32, world-to-camera (OpenCV)
    intrinsics: np.ndarray  # (N, 3, 3) float32, pixel-space at VGGT res
    scale: float = 1.0  # the global metric scale s (diagnostic only)


class VGGTOmegaMetricEstimator:
    """DA3-compatible wrapper around VGGT-Omega + DA3 metric anchoring."""

    # Read by the handler for logging (`getattr(est, "model_name", "unknown")`).
    model_name = "vggt-omega-metric"
    # Flag the handler checks to route us through its DA3 inference branch.
    da3_compatible = True

    def __init__(
        self,
        vggt_model,
        da3_model,
        *,
        image_resolution: int = 512,
        patch_size: int = 16,
        da3_process_res: int = 504,
        device=None,
    ):
        self.vggt_model = vggt_model
        self.da3_model = da3_model
        self.image_resolution = int(image_resolution)
        self.patch_size = int(patch_size)
        self.da3_process_res = int(da3_process_res)
        # Fail fast on a bad resolution at construction rather than letting
        # ``_preprocess_pil_images`` raise mid-validation, after both ~1B models
        # are already loaded onto the GPU.
        if self.image_resolution <= 0 or self.patch_size <= 0:
            raise ValueError(
                f"image_resolution ({self.image_resolution}) and patch_size "
                f"({self.patch_size}) must be positive."
            )
        if self.image_resolution % self.patch_size != 0:
            raise ValueError(
                f"image_resolution ({self.image_resolution}) must be divisible "
                f"by patch_size ({self.patch_size})."
            )
        # Last successfully-aligned metric scale, carried forward as the fallback
        # when DA3<->VGGT alignment fails on a degenerate (e.g. all-sky) segment,
        # so one bad section cannot abort the whole validation / deadlock DDP.
        self._last_good_scale = 1.0
        self.device = (
            torch.device(device) if device is not None else None
        )

    # ------------------------------------------------------------------ #
    # Lifecycle surface the handler touches (mirror nn.Module semantics). #
    # ------------------------------------------------------------------ #
    def to(self, device):
        device = (
            device
            if isinstance(device, torch.device)
            else torch.device(device)
        )
        if self.device is not None and self.device == device:
            # Already on the target device; skip the redundant parameter-tree
            # walk over two ~1B models (the handler calls ``.to(device)`` before
            # every registration).
            return self
        if device.type == "cuda":
            # Mirror the legacy DA3 path: pin the rank-local device so any
            # subsequently-allocated tensors (and a later bare ``.cuda()``) land
            # on the same GPU as the weights.
            torch.cuda.set_device(device)
        self.vggt_model.to(device)
        self.da3_model.to(device)
        # The DA3 API places its inputs on ``model.device`` (see
        # run_da3_metric), and ``.to()`` alone does not update that attribute.
        self.da3_model.device = device
        self.device = device
        return self

    def cuda(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                device = torch.device("cuda")
        return self.to(device)

    def cpu(self):
        return self.to(torch.device("cpu"))

    def eval(self):
        self.vggt_model.eval()
        self.da3_model.eval()
        return self

    def train(self, mode: bool = True):
        # Estimator is inference-only; keep submodules in eval regardless.
        return self

    # ------------------------------------------------------------------ #
    # DA3-shaped inference.                                               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def inference(self, frames, *, use_ray_pose=True, process_res=None, **_ignored):
        """Run VGGT-Omega + DA3 metric anchoring on ``frames``.

        ``use_ray_pose`` / ``process_res`` are accepted for signature parity
        with DA3 but ignored: VGGT-Omega uses its own ``image_resolution`` and
        the DA3 metric branch uses ``da3_process_res`` (both fixed at
        construction). Returns a :class:`VGGTOmegaPrediction`.
        """
        _ensure_paths_on_syspath()
        from vggt_omega.utils.load_fn import _preprocess_pil_images
        from vggt_omega.metric import vggt_omega_metric_inference

        pil_frames = self._to_pil_uint8_list(frames)

        # VGGT-Omega input tensor (N, 3, H, W) on CPU, then to the model device.
        images = _preprocess_pil_images(
            pil_frames, "balanced", self.image_resolution, self.patch_size
        )
        device = self._infer_device()
        images = images.to(device)

        # A degenerate section (mostly sky / too few alignment pixels) makes the
        # DA3<->VGGT scale alignment raise. Hand it the last-good scale as a
        # fallback so it degrades gracefully instead of aborting the validation
        # step and deadlocking other ranks at the next collective op.
        out = vggt_omega_metric_inference(
            images=images,
            original_rgb=pil_frames,
            vggt_model=self.vggt_model,
            da3_model=self.da3_model,
            da3_process_res=self.da3_process_res,
            fallback_scale_on_failure=self._last_good_scale,
        )

        # depth: (1, N, H, W, 1) -> (N, H, W); ext: (1, N, 3, 4) -> (N, 3, 4);
        # intr: (1, N, 3, 3) -> (N, 3, 3).
        depth = self._np(out["depth"][0, ..., 0])
        extrinsics = self._np(out["extrinsic_w2c"][0])
        intrinsics = self._np(out["intrinsic"][0])
        scale_val = out["scale"]
        scale = (
            float(scale_val.item())
            if torch.is_tensor(scale_val)
            else float(scale_val)
        )
        diagnostics = out.get("diagnostics") or {}
        # Treat a non-finite or near-zero/negative scale as a failed alignment
        # even when the runner reported success: such a least-squares solution
        # collapses or sign-flips the depth, and silently caching it would poison
        # ``_last_good_scale`` for every later (genuinely-failing) segment.
        alignment_failed = bool(diagnostics.get("alignment_failed"))
        scale_degenerate = not (math.isfinite(scale) and scale > _MIN_VALID_SCALE)
        if alignment_failed or scale_degenerate:
            reason = (
                diagnostics.get("alignment_error", "unknown error")
                if alignment_failed
                else f"degenerate scale s={scale:.6g}"
            )
            # Re-anchor depth/translation to the last-good metric scale when the
            # bad scale is recoverable (the runner multiplied depth by ``scale``,
            # so rescale by ``last_good / scale``). A scale at/under the floor is
            # unrecoverable (depth already collapsed toward zero), so leave the
            # frame as-is rather than divide by ~0.
            if math.isfinite(scale) and abs(scale) > _MIN_VALID_SCALE:
                correction = self._last_good_scale / scale
                depth = depth * correction
                extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * correction
            scale = self._last_good_scale
            # NB: do NOT update ``_last_good_scale`` here — that is the poisoning
            # this guard exists to prevent.
            if _is_main_process():
                print(
                    "[vggt_omega_metric] WARNING: metric alignment unreliable "
                    f"for this segment ({reason}); reused last-good scale "
                    f"s={scale:.4f}. This segment's geometry may be non-metric."
                )
        else:
            self._last_good_scale = scale
        return VGGTOmegaPrediction(
            depth=depth,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            scale=scale,
        )

    # ------------------------------------------------------------------ #
    # Helpers.                                                            #
    # ------------------------------------------------------------------ #
    def _infer_device(self):
        if self.device is not None:
            return self.device
        try:
            return next(self.vggt_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _np(t):
        if torch.is_tensor(t):
            return t.detach().to("cpu", dtype=torch.float32).numpy()
        return np.asarray(t, dtype=np.float32)

    @staticmethod
    def _to_pil_uint8_list(frames):
        """Normalise a frame sequence to a list of uint8 RGB ``PIL.Image``.

        Accepts the handler's two call styles: a list of PIL images
        (register_source_sequence) or a list of numpy ``(H, W, 3)`` arrays
        (reestimate_depth). Float frames in [0, 1] are scaled to [0, 255].
        """
        pil = []
        for f in frames:
            if isinstance(f, Image.Image):
                pil.append(f if f.mode == "RGB" else f.convert("RGB"))
                continue
            arr = np.asarray(f)
            if arr.size == 0:
                raise ValueError(
                    "_to_pil_uint8_list received an empty (zero-size) frame."
                )
            if arr.dtype != np.uint8:
                # Map NaN/inf to finite values first so a single bad pixel cannot
                # flip the [0, 1] vs [0, 255] heuristic (NaN <= 1.0 is False) and
                # blacken the whole frame; clip before the uint8 cast so any
                # out-of-range float saturates instead of wrapping modulo 256.
                arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
                if float(arr.max()) <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            if img.mode != "RGB":
                img = img.convert("RGB")
            pil.append(img)
        return pil


def _resolve_ckpt_path(ckpt_path):
    """Resolve a possibly repo-root-relative checkpoint path.

    ``torch.load`` / ``os.path.isfile`` resolve relative paths against the
    process CWD, but the shipped default (``runjia.qian/...``) is relative to the
    repo root. So if the path is not absolute and does not exist under the CWD,
    retry it under the repo root before giving up — otherwise a launch from any
    other working directory spuriously fails even though the file exists.
    """
    if os.path.isabs(ckpt_path) or os.path.exists(ckpt_path):
        return ckpt_path
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    candidate = os.path.join(repo_root, ckpt_path)
    return candidate if os.path.exists(candidate) else ckpt_path


def _strip_ddp_prefix(state_dict):
    """Strip a uniform ``module.`` prefix from a DDP-saved state_dict.

    Keeps ``load_state_dict(strict=True)`` (genuine architecture mismatches still
    fail loudly) while accepting the wrapped-key checkpoints the loader's
    ``weights_only=False`` comment explicitly anticipates. Only ``module.`` (the
    unambiguous DDP wrapper) is stripped, and only when *every* key carries it.
    """
    if isinstance(state_dict, dict) and state_dict and all(
        isinstance(k, str) and k.startswith("module.") for k in state_dict
    ):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_vggt_omega_model(ckpt_path, device):
    """Instantiate VGGT-Omega and load a raw state_dict checkpoint."""
    _ensure_paths_on_syspath()
    from vggt_omega.models import VGGTOmega

    ckpt_path = _resolve_ckpt_path(ckpt_path)
    if not (os.path.isfile(ckpt_path) or os.path.isdir(ckpt_path)):
        # torch.load does NOT resolve HF hub ids, so a non-filesystem path only
        # leads to a confusing low-level error after the ~1B model is already
        # allocated. Fail fast with an actionable message instead.
        raise FileNotFoundError(
            f"[vggt_omega_metric] VGGT-Omega checkpoint not found: {ckpt_path!r}. "
            "Pass a valid local path via --vggt_omega_ckpt or $VGGT_OMEGA_CKPT "
            "(a filesystem path, not an HF hub id)."
        )
    if _is_main_process():
        print(f"Loading VGGT-Omega checkpoint: {ckpt_path}")
    model = VGGTOmega().eval()
    # weights_only=False: these checkpoints may be training-style dicts wrapping
    # the weights as {"state_dict": ...} alongside non-tensor metadata, which the
    # torch>=2.6 weights_only=True default would reject before we can unwrap it.
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(
        state_dict["state_dict"], dict
    ):
        state_dict = state_dict["state_dict"]
    state_dict = _strip_ddp_prefix(state_dict)
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    return model


def init_vggt_omega_metric_estimator(
    device=None,
    *,
    vggt_ckpt=None,
    image_resolution=512,
    da3_process_res=504,
    patch_size=16,
    da3_init_fn=None,
):
    """Build a :class:`VGGTOmegaMetricEstimator` on ``device``.

    Args:
        device: target torch device (or ``"cuda"`` / ``None``).
        vggt_ckpt: VGGT-Omega checkpoint path. ``$VGGT_OMEGA_CKPT`` takes
            precedence if set (so it can override an argparse/config default,
            which is always truthy), then ``vggt_ckpt``, then
            :data:`_DEFAULT_VGGT_OMEGA_CKPT`. A repo-root-relative path is
            resolved against the repo root, not just the CWD.
        image_resolution: VGGT-Omega preprocessing resolution (must match the
            checkpoint; the default 512 ckpt is ``vggt_omega_1b_512.pt``).
        da3_process_res: DA3 metric-branch processing resolution.
        patch_size: VGGT-Omega patch size; ``image_resolution`` must be a
            multiple of it (validated up front).
        da3_init_fn: callable ``(device=...) -> DepthAnything3`` used to load the
            DA3 metric branch. Injected by the caller (``video_io`` passes
            ``_init_da3_depth_estimator``) so the DA3 weights/resolution match
            the standalone-DA3 path exactly and we avoid a circular import.
            The giant ``anyview`` (``.model.da3``) branch is dropped afterwards.
    """
    _ensure_paths_on_syspath()

    # Validate the resolution BEFORE loading two ~1B models, so a bad config
    # fails fast instead of crashing minutes later in VGGTOmegaMetricEstimator
    # construction after the checkpoints are already on the GPU.
    image_resolution = int(image_resolution)
    patch_size = int(patch_size)
    if image_resolution <= 0 or patch_size <= 0:
        raise ValueError(
            f"image_resolution ({image_resolution}) and patch_size "
            f"({patch_size}) must be positive."
        )
    if image_resolution % patch_size != 0:
        raise ValueError(
            f"--vggt_omega_image_resolution ({image_resolution}) must be "
            f"divisible by the VGGT-Omega patch size ({patch_size})."
        )

    if da3_init_fn is None:
        raise ValueError(
            "init_vggt_omega_metric_estimator requires da3_init_fn (e.g. "
            "mosaic.video_io._init_da3_depth_estimator) to load the DA3 metric "
            "branch."
        )

    # $VGGT_OMEGA_CKPT takes precedence so it can actually override the config
    # default (the argparse/YAML default is non-empty and would otherwise always
    # short-circuit the ``or`` chain, making the documented env override dead).
    ckpt = (
        os.environ.get("VGGT_OMEGA_CKPT")
        or vggt_ckpt
        or _DEFAULT_VGGT_OMEGA_CKPT
    )

    # Load BOTH models on CPU first, then drop DA3's giant ViT-G anyview branch
    # before anything reaches the GPU — so peak VRAM never holds
    # VGGT + ViT-G + ViT-L at once. A single ``estimator.to(device)`` below moves
    # only the kept branches onto the GPU.
    vggt_model = _load_vggt_omega_model(ckpt, device=None)
    da3_model = da3_init_fn(device=None)
    da3_model.eval()
    # vggt_omega_metric needs DA3's *metric* branch (model.model.da3_metric),
    # which only exists in a NESTED DA3 checkpoint. The DA3 weights come from
    # $DA3_MODEL_PATH / $DA3_MODEL_ID (shared with the standalone da3 path), so a
    # mono checkpoint is a valid-but-wrong selection here; fail fast at init
    # rather than at the first inference() deep in the validation loop.
    inner = getattr(da3_model, "model", None)
    if inner is None or not hasattr(inner, "da3_metric"):
        raise RuntimeError(
            "registration_estimator='vggt_omega_metric' requires a NESTED DA3 "
            "checkpoint exposing model.model.da3_metric (e.g. "
            "DA3NESTED-GIANT-LARGE-1.1), but the DA3 weights loaded via "
            "$DA3_MODEL_PATH / $DA3_MODEL_ID have no metric branch. Point those "
            "env vars at a nested checkpoint."
        )
    # Drop the giant anyview branch (ViT-G); we only need the metric ViT-L.
    # (Still on CPU here, so no GPU allocation to free.)
    if hasattr(inner, "da3"):
        del inner.da3

    estimator = VGGTOmegaMetricEstimator(
        vggt_model,
        da3_model,
        image_resolution=image_resolution,
        patch_size=patch_size,
        da3_process_res=da3_process_res,
        device=None,
    )
    if device is not None:
        estimator.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return estimator
