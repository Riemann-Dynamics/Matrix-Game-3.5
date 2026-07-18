"""Helpers for the ``--val_image_path`` validation option.

Kept dependency-light (``os`` / ``glob`` / ``numpy`` / ``PIL`` only -- no
``torch``, no ``.common``) so the file-discovery and crop/resize logic stays
unit-testable without the full training stack. The VAE encode + clean-latent
substitution that consumes these helpers lives in ``validation.py``.
"""

import os

import numpy as np
from PIL import Image

# Still-image formats the option accepts (matched case-insensitively).
VAL_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _resolve_val_image_pool(val_image_path):
    """Resolve ``--val_image_path`` into an ordered list of image files.

    Returns ``None`` when unset. A single file yields ``[file]`` (every rank
    shares the one image). A directory yields the sorted png/jpg/jpeg files
    inside it -- the order is deterministic so the per-rank stripe is
    reproducible across runs. Raises on an unsupported file extension, a
    directory with no images, or a path that does not exist.
    """
    if not val_image_path:
        return None
    path = str(val_image_path)
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in VAL_IMAGE_EXTENSIONS:
            raise ValueError(
                "--val_image_path file must be one of "
                f"{VAL_IMAGE_EXTENSIONS}, got {path!r}."
            )
        return [path]
    if os.path.isdir(path):
        # os.listdir (not glob) so directory names containing glob
        # metacharacters ('[', ']', '*', '?') are treated literally.
        files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if os.path.splitext(name)[1].lower() in VAL_IMAGE_EXTENSIONS
            and os.path.isfile(os.path.join(path, name))
        )
        if not files:
            raise ValueError(
                "--val_image_path directory contains no png/jpg/jpeg files: "
                f"{path!r}."
            )
        return files
    raise FileNotFoundError(f"--val_image_path does not exist: {path!r}.")


def _select_val_image_for_sample(pool, *, batch_idx, num_processes, proc_rank):
    """Pick one image for a ``(global rank, batch)`` validation sample.

    Mirrors the validation rank stripe used elsewhere in the loop
    (``sample_index = batch_idx * num_processes + proc_rank``): images are
    handed out in folder order across the global rank x batch grid, so distinct
    ranks read distinct images with no overlap until the pool is exhausted,
    then wrap around from the start. ``proc_rank`` must be the GLOBAL rank
    (``accelerator.process_index``) so images do not collide across nodes. A
    single-file pool returns that file for every rank.
    """
    if not pool:
        raise ValueError("val image pool is empty.")
    num_processes = max(1, int(num_processes))
    sample_index = int(batch_idx) * num_processes + int(proc_rank)
    return pool[sample_index % len(pool)]


def _load_and_preprocess_val_image(image_path, *, height, width):
    """Load an image and cover-resize + center-crop to ``width`` x ``height``.

    No-stretch (aspect-preserving) scaling that fills the whole target frame
    while cropping as little as possible -- identical semantics to the
    dataset's ``ImageCropAndResize`` (``scale = max(width / w, height / h)``
    followed by a centre crop). Returns an ``(height, width, 3)`` uint8 RGB
    array, ready to hand to ``_encode_frames_per_frame`` exactly like a
    dataset frame.
    """
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"target size must be positive, got {(height, width)}.")
    with Image.open(image_path) as opened:
        img = opened.convert("RGB")
        src_w, src_h = img.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"image {image_path!r} has invalid size {(src_w, src_h)}.")
        scale = max(width / src_w, height / src_h)
        # Clamp up to the target so float rounding can never leave a side
        # shorter than the crop window.
        resized_w = max(width, round(src_w * scale))
        resized_h = max(height, round(src_h * scale))
        img = img.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        left = (resized_w - width) // 2
        top = (resized_h - height) // 2
        img = img.crop((left, top, left + width, top + height))
    frame = np.asarray(img, dtype=np.uint8)
    if frame.shape != (height, width, 3):
        raise RuntimeError(
            f"val image preprocess produced shape {frame.shape}, "
            f"expected {(height, width, 3)} for {image_path!r}."
        )
    return frame
