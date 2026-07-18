import json
import os
from collections import OrderedDict

import numpy as np
import torch

from .unified_dataset import (
    DA3MosaicVideoDataset,
    _derive_seed,
    _dynamic_mask_postprocess_module,
)


IMAGENET_MEAN_RGB = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json in {path}:{line_no}: {exc}") from exc
    return rows


def _sample_reference_indices(
    candidate_count,
    pairwise_similarity,
    *,
    num_refs,
    rng,
    dissimilar_top_k,
    max_similarity,
):
    if candidate_count <= 0 or num_refs <= 0:
        return []
    k = min(int(num_refs), int(candidate_count))
    if k <= 1:
        return [int(rng.integers(0, candidate_count))]

    sim = np.asarray(pairwise_similarity, dtype=np.float32)
    if sim.shape != (candidate_count, candidate_count):
        # Fall back to random unique refs if the similarity file is stale.
        return [int(v) for v in rng.choice(candidate_count, size=k, replace=False)]
    sim = np.nan_to_num(sim, nan=1.0, posinf=1.0, neginf=-1.0)
    sim = np.clip(sim, -1.0, 1.0)

    selected = [int(rng.integers(0, candidate_count))]
    remaining = set(range(candidate_count))
    remaining.discard(selected[0])
    top_k = max(1, int(dissimilar_top_k))
    max_sim_threshold = float(max_similarity)
    while remaining and len(selected) < k:
        scored = [
            (max(float(sim[int(idx), int(sel)]) for sel in selected), int(idx))
            for idx in sorted(remaining)
        ]
        scored.sort(key=lambda item: item[0])
        valid = [idx for max_sim, idx in scored if max_sim <= max_sim_threshold]
        choice_pool = valid[: min(top_k, len(valid))] if valid else [
            idx for _, idx in scored[: min(top_k, len(scored))]
        ]
        choice = int(choice_pool[int(rng.integers(0, len(choice_pool)))])
        selected.append(choice)
        remaining.discard(choice)
    return selected


def _safe_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


class SubjectRefMemoryDA3MosaicVideoDataset(DA3MosaicVideoDataset):
    """DA3 video dataset with VAE reference-memory data preparation.

    This first-step dataset only prepares reference RGB canvases and
    protagonist patch masks. Transformer-side reference memory is added in a
    later step. Base DA3 behavior is unchanged when this dataset type is not
    selected.
    """

    def __init__(
        self,
        *args,
        subject_ref_dir_name="protagonist_refs",
        subject_num_refs_max=2,
        subject_dissimilar_top_k=8,
        subject_max_similarity=0.94,
        subject_shuffle_refs=True,
        subject_ref_dropout_prob=0.0,
        subject_ref_canvas_slot_ratio=0.5,
        subject_ref_background="imagenet_mean",
        subject_context_patch_mask_prob=0.5,
        subject_context_patch_mask_ratio_min=0.25,
        subject_context_patch_mask_ratio_max=0.75,
        subject_context_patch_mask_fill="background_patch",
        subject_context_patch_mask_temporal_dilate_radius=0,
        subject_object_loss_temporal_dilate_radius=0,
        subject_ref_cache_max=32,
        **kwargs,
    ):
        self.subject_ref_dir_name = str(subject_ref_dir_name or "protagonist_refs")
        self.subject_num_refs_max = max(0, int(subject_num_refs_max))
        self.subject_dissimilar_top_k = max(1, int(subject_dissimilar_top_k))
        self.subject_max_similarity = float(subject_max_similarity)
        self.subject_shuffle_refs = bool(subject_shuffle_refs)
        self.subject_ref_dropout_prob = min(
            1.0, max(0.0, float(subject_ref_dropout_prob))
        )
        self.subject_ref_canvas_slot_ratio = min(
            1.0,
            max(0.05, float(subject_ref_canvas_slot_ratio)),
        )
        self.subject_ref_background = str(
            subject_ref_background or "imagenet_mean"
        ).strip().lower()
        self.subject_context_patch_mask_prob = min(
            1.0, max(0.0, float(subject_context_patch_mask_prob))
        )
        self.subject_context_patch_mask_ratio_min = min(
            1.0, max(0.0, float(subject_context_patch_mask_ratio_min))
        )
        self.subject_context_patch_mask_ratio_max = min(
            1.0, max(0.0, float(subject_context_patch_mask_ratio_max))
        )
        if (
            self.subject_context_patch_mask_ratio_max
            < self.subject_context_patch_mask_ratio_min
        ):
            self.subject_context_patch_mask_ratio_min, self.subject_context_patch_mask_ratio_max = (
                self.subject_context_patch_mask_ratio_max,
                self.subject_context_patch_mask_ratio_min,
            )
        self.subject_context_patch_mask_fill = str(
            subject_context_patch_mask_fill or "background_patch"
        ).strip().lower()
        self.subject_context_patch_mask_temporal_dilate_radius = max(
            0, int(subject_context_patch_mask_temporal_dilate_radius)
        )
        self.subject_object_loss_temporal_dilate_radius = max(
            0, int(subject_object_loss_temporal_dilate_radius)
        )
        self.subject_ref_cache_max = max(1, int(subject_ref_cache_max))
        self._subject_ref_cache = OrderedDict()
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        rng = self._get_sample_rng()
        sample_ref_dropout = True
        if bool(getattr(self, "is_val_dataset", False)):
            info = data.get("info") or {}
            ref_seed_key = (
                info.get("clean_name")
                or info.get("video_path")
                or info.get("latent_path")
                or int(index)
            )
            rng = np.random.default_rng(
                _derive_seed(self.seed, "validation_subject_ref", ref_seed_key)
            )
            sample_ref_dropout = False
        ref_pack = self._sample_subject_ref_canvases(
            data, rng, sample_dropout=sample_ref_dropout
        )
        data.update(ref_pack)
        object_loss_pack = self._build_subject_object_loss_mask(data)
        data.update(object_loss_pack)
        return data

    def _dataset_cache_inputs_dict(self):
        base = super()._dataset_cache_inputs_dict()
        base.update(
            {
                "dataset_kind": "da3_video_subject_ref",
                "subject_ref_dir_name": self.subject_ref_dir_name,
                "subject_num_refs_max": int(self.subject_num_refs_max),
                "subject_dissimilar_top_k": int(self.subject_dissimilar_top_k),
                "subject_max_similarity": float(self.subject_max_similarity),
                "subject_shuffle_refs": bool(self.subject_shuffle_refs),
                "subject_ref_dropout_prob": float(self.subject_ref_dropout_prob),
                "subject_ref_canvas_slot_ratio": float(
                    self.subject_ref_canvas_slot_ratio
                ),
                "subject_ref_background": self.subject_ref_background,
                "subject_object_loss_temporal_dilate_radius": int(
                    self.subject_object_loss_temporal_dilate_radius
                ),
            }
        )
        return base

    def _scene_dir_from_info(self, info):
        scene_dir = info.get("dirname") or info.get("scene_dir")
        if scene_dir:
            return str(scene_dir)
        video_path = info.get("video_path")
        if video_path:
            return os.path.dirname(os.path.dirname(str(video_path)))
        return ""

    def _subject_ref_dir(self, info):
        scene_dir = self._scene_dir_from_info(info)
        if not scene_dir:
            return ""
        return os.path.join(
            scene_dir,
            self.dynamic_object_mask_subdir,
            self.subject_ref_dir_name,
        )

    def _dynamic_mask_path_from_info(self, info):
        mask_path = info.get("dynamic_mask_path")
        if mask_path:
            return str(mask_path)
        scene_dir = self._scene_dir_from_info(info)
        if scene_dir:
            return self._default_dynamic_mask_path_for_scene_dir(scene_dir)
        return ""

    @staticmethod
    def _source_hw_from_record(row):
        mask_rle = row.get("mask_rle")
        if isinstance(mask_rle, dict):
            size = mask_rle.get("size")
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                try:
                    return int(size[0]), int(size[1])
                except (TypeError, ValueError):
                    pass
        return None

    def _load_subject_refs(self, ref_dir):
        if not ref_dir:
            return None
        cached = self._subject_ref_cache.get(ref_dir)
        if cached is not None:
            self._subject_ref_cache.move_to_end(ref_dir)
            return cached

        candidates_path = os.path.join(ref_dir, "candidates.jsonl")
        if not os.path.exists(candidates_path):
            return None
        candidates = _read_jsonl(candidates_path)
        candidates = [row for row in candidates if row.get("frame_idx") is not None]
        if not candidates:
            return None

        pairwise = None
        pairwise_path = os.path.join(ref_dir, "pairwise_similarity.npy")
        if os.path.exists(pairwise_path):
            try:
                pairwise = np.load(pairwise_path).astype(np.float32, copy=False)
            except Exception:
                pairwise = None

        selected_track_ids = []
        track_path = os.path.join(ref_dir, "track.json")
        if os.path.exists(track_path):
            with open(track_path, "r", encoding="utf-8") as f:
                track = json.load(f) or {}
            for value in track.get("selected_track_ids") or []:
                try:
                    selected_track_ids.append(int(value))
                except (TypeError, ValueError):
                    continue
            if not selected_track_ids and track.get("track_id") is not None:
                try:
                    selected_track_ids.append(int(track["track_id"]))
                except (TypeError, ValueError):
                    pass

        payload = {
            "candidates": candidates,
            "pairwise": pairwise,
            "selected_track_ids": selected_track_ids,
            "ref_dir": ref_dir,
        }
        self._subject_ref_cache[ref_dir] = payload
        self._subject_ref_cache.move_to_end(ref_dir)
        while len(self._subject_ref_cache) > self.subject_ref_cache_max:
            self._subject_ref_cache.popitem(last=False)
        return payload

    @staticmethod
    def _normal_to_uint8_canvas_value(background):
        if str(background).strip().lower() == "zero":
            return np.asarray([127.5, 127.5, 127.5], dtype=np.float32)
        if str(background).strip().lower() == "black":
            return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        return IMAGENET_MEAN_RGB

    @staticmethod
    def _pil_read_rgb(path):
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB")).copy()

    @staticmethod
    def _pil_read_mask(path):
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("L")).copy()

    def _read_ref_rgb(self, path):
        import time
        import cv2

        errors = []
        for attempt in range(3):
            try:
                return self._pil_read_rgb(path)
            except Exception as exc:
                errors.append(f"attempt={attempt + 1} pil={type(exc).__name__}: {exc}")
            image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if image_bgr is not None:
                return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            errors.append(f"attempt={attempt + 1} cv2=None")
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
        size = os.path.getsize(path) if os.path.exists(path) else -1
        raise RuntimeError(
            "failed to read subject ref cutout after cv2/PIL retries: "
            f"{path} size={size} errors={errors}"
        )

    def _read_ref_mask(self, path):
        import time
        import cv2

        errors = []
        for attempt in range(3):
            try:
                return self._pil_read_mask(path)
            except Exception as exc:
                errors.append(f"attempt={attempt + 1} pil={type(exc).__name__}: {exc}")
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask
            errors.append(f"attempt={attempt + 1} cv2=None")
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
        size = os.path.getsize(path) if os.path.exists(path) else -1
        raise RuntimeError(
            "failed to read subject ref mask after cv2/PIL retries: "
            f"{path} size={size} errors={errors}"
        )

    def _load_saved_ref_cutout_rgb(self, ref_dir, row):
        image_path = row.get("image_path")
        if not image_path:
            raise ValueError(f"subject ref candidate missing image_path: {row}")
        path = os.path.join(str(ref_dir), str(image_path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"subject ref cutout does not exist: {path}")
        image_rgb = self._read_ref_rgb(path)

        mask_path = row.get("mask_path")
        if not mask_path:
            raise ValueError(f"subject ref candidate missing mask_path: {row}")
        full_mask_path = os.path.join(str(ref_dir), str(mask_path))
        if not os.path.exists(full_mask_path):
            raise FileNotFoundError(f"subject ref mask does not exist: {full_mask_path}")
        mask = self._read_ref_mask(full_mask_path)
        if mask.shape[:2] != image_rgb.shape[:2]:
            import cv2

            mask = cv2.resize(
                mask,
                (int(image_rgb.shape[1]), int(image_rgb.shape[0])),
                interpolation=cv2.INTER_NEAREST,
            )
        return image_rgb, mask > 0

    def _build_subject_ref_canvas_from_cutout(self, cutout_rgb, mask_np, row):
        import cv2

        height = int(self.height)
        width = int(self.width)
        bg = self._normal_to_uint8_canvas_value(self.subject_ref_background)
        crop = np.asarray(cutout_rgb, dtype=np.float32)
        if crop.ndim != 3 or crop.shape[-1] != 3:
            return None, {"status": "bad_cutout_shape"}
        if mask_np is not None:
            crop_mask = np.asarray(mask_np, dtype=bool)
            if crop_mask.shape[:2] != crop.shape[:2]:
                crop_mask = cv2.resize(
                    crop_mask.astype(np.uint8),
                    (int(crop.shape[1]), int(crop.shape[0])),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
        else:
            crop_mask = np.ones(crop.shape[:2], dtype=bool)

        slot_size = max(
            1,
            int(round(min(height, width) * self.subject_ref_canvas_slot_ratio)),
        )
        slot_size = min(slot_size, height, width)
        resized = cv2.resize(
            np.clip(crop, 0.0, 255.0).astype(np.uint8),
            (slot_size, slot_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        resized_mask = cv2.resize(
            crop_mask.astype(np.uint8),
            (slot_size, slot_size),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

        canvas = np.empty((height, width, 3), dtype=np.float32)
        canvas[:, :] = bg.reshape(1, 1, 3)
        y0 = height - slot_size
        x0 = width - slot_size
        canvas[y0:height, x0:width] = resized
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float()
        tensor = tensor.mul(2.0 / 255.0).sub(1.0).clamp(-1.0, 1.0)
        debug = {
            "status": "ok",
            "source": "saved_cutout",
            "image_path": str(row.get("image_path") or ""),
            "mask_path": str(row.get("mask_path") or ""),
            "slot_xyxy": [int(x0), int(y0), int(width), int(height)],
            "slot_size": int(slot_size),
            "mask_pixel_count": int(crop_mask.sum()),
            "slot_mask_pixel_count": int(resized_mask.sum()),
            "foreground_enhance": "precomputed",
            "background": self.subject_ref_background,
        }
        return tensor, debug

    def _sample_subject_ref_canvases(self, data, rng, *, sample_dropout=True):
        info = data.get("info") or {}
        empty = {
            "subject_ref_images": torch.zeros(
                0, 3, int(self.height), int(self.width), dtype=torch.float32
            ),
            "subject_ref_info": {
                "enabled": False,
                "status": "disabled" if self.subject_num_refs_max <= 0 else "missing_refs",
                "actual_ref_count": 0,
                "selected_ref_indices": [],
                "selected_frame_indices": [],
                "num_available_refs": 0,
            },
        }
        if self.subject_num_refs_max <= 0:
            return empty
        if sample_dropout and float(self.subject_ref_dropout_prob) > 0.0:
            dropped = bool(rng.random() < float(self.subject_ref_dropout_prob))
            if dropped:
                empty["subject_ref_info"].update(
                    {
                        "status": "dropped",
                        "dropout_prob": float(self.subject_ref_dropout_prob),
                    }
                )
                return empty
        ref_dir = self._subject_ref_dir(info)
        refs = self._load_subject_refs(ref_dir)
        if refs is None:
            empty["subject_ref_info"].update({"ref_dir": ref_dir})
            return empty
        candidates = refs["candidates"]
        if not candidates:
            empty["subject_ref_info"].update({"ref_dir": ref_dir})
            return empty

        pairwise = refs.get("pairwise")
        if pairwise is None:
            selected = [
                int(v)
                for v in rng.choice(
                    len(candidates),
                    size=min(self.subject_num_refs_max, len(candidates)),
                    replace=False,
                )
            ]
        else:
            selected = _sample_reference_indices(
                len(candidates),
                pairwise,
                num_refs=min(self.subject_num_refs_max, len(candidates)),
                rng=rng,
                dissimilar_top_k=self.subject_dissimilar_top_k,
                max_similarity=self.subject_max_similarity,
            )
        if self.subject_shuffle_refs and len(selected) > 1:
            selected = [int(v) for v in rng.permutation(selected).tolist()]
        selected_rows = [candidates[int(idx)] for idx in selected]
        frame_ids = [int(row["frame_idx"]) for row in selected_rows]
        if not frame_ids:
            empty["subject_ref_info"].update({"ref_dir": ref_dir})
            return empty

        canvases = []
        ref_records = []
        for row in selected_rows:
            frame_idx = int(row["frame_idx"])
            cutout_rgb, cutout_mask = self._load_saved_ref_cutout_rgb(
                ref_dir,
                row,
            )
            canvas, canvas_info = self._build_subject_ref_canvas_from_cutout(
                cutout_rgb,
                cutout_mask,
                row,
            )
            ref_record = {
                "ref_id": _safe_int(row.get("ref_id"), len(ref_records)),
                "frame_idx": int(frame_idx),
                "track_id": _safe_int(row.get("track_id"), -1),
                "image_path": str(row.get("image_path") or ""),
                "mask_path": str(row.get("mask_path") or ""),
                "candidate_crop_xyxy": [
                    int(v) for v in (row.get("crop_xyxy") or [])
                ],
                "candidate_bbox_xyxy": [
                    float(v) for v in (row.get("bbox_xyxy") or [])
                ],
                "mask_status": "saved_cutout_mask",
                **(canvas_info or {}),
            }
            ref_records.append(ref_record)
            if canvas is not None:
                canvases.append(canvas)
        if not canvases:
            empty["subject_ref_info"].update({"ref_dir": ref_dir})
            return empty
        ref_images = torch.stack(canvases, dim=0).to(dtype=torch.float32)
        return {
            "subject_ref_images": ref_images,
            "subject_ref_info": {
                "enabled": True,
                "status": "ok",
                "ref_dir": ref_dir,
                "actual_ref_count": int(ref_images.shape[0]),
                "num_available_refs": int(len(candidates)),
                "selected_ref_indices": [int(v) for v in selected],
                "selected_frame_indices": [int(v) for v in frame_ids],
                "selected_track_ids": [int(v) for v in refs["selected_track_ids"]],
                "canvas_slot_ratio": float(self.subject_ref_canvas_slot_ratio),
                "background": self.subject_ref_background,
                "ref_records": ref_records,
            },
        }

    def _build_target_image_masks(self, data):
        info = data.get("info") or {}
        mask_path = info.get("dynamic_mask_path")
        if not mask_path:
            scene_dir = self._scene_dir_from_info(info)
            if scene_dir:
                mask_path = self._default_dynamic_mask_path_for_scene_dir(scene_dir)
        if not mask_path:
            return None, "no_mask_path"
        records_by_frame = self._load_dynamic_mask_records(mask_path)
        if not records_by_frame:
            return None, "no_mask_records"

        fused = data.get("fused_query_inputs") or {}
        frame_ids = fused.get("query_frame_indices")
        if frame_ids is None:
            return None, "no_query_frame_indices"
        frame_ids = [int(v) for v in np.asarray(frame_ids).reshape(-1).tolist()]
        if not frame_ids:
            return None, "empty_query_frame_indices"

        height = int(self.height)
        width = int(self.width)
        postprocess = _dynamic_mask_postprocess_module()
        frame_masks = []
        for frame_idx in frame_ids:
            mask_np = np.zeros((height, width), dtype=bool)
            for row in records_by_frame.get(int(frame_idx), ()):
                source_hw = self._source_hw_from_record(row) or (height, width)
                image_mask = postprocess.mask_from_record(
                    row,
                    height=int(source_hw[0]),
                    width=int(source_hw[1]),
                    use_bbox_fallback=False,
                )
                if image_mask is None:
                    continue
                image_mask = self._resize_bool_mask_for_training(
                    image_mask, height=height, width=width
                )
                mask_np |= image_mask
            frame_masks.append(mask_np)
        if not frame_masks:
            return None, "no_target_frames"
        masks = np.stack(frame_masks, axis=0)
        radius = int(self.subject_object_loss_temporal_dilate_radius)
        if radius > 0:
            masks = postprocess.apply_temporal_dilation_to_frame_masks(
                masks, radius=radius
            )
        return masks, ""

    @staticmethod
    def _latent_to_dit_patch_mask(latent_mask):
        latent_mask = np.asarray(latent_mask, dtype=bool)
        h_lat, w_lat = latent_mask.shape[-2:]
        h_patch = int(h_lat) // 2
        w_patch = int(w_lat) // 2
        cropped = latent_mask[: h_patch * 2, : w_patch * 2]
        if cropped.size == 0:
            return np.zeros((h_patch, w_patch), dtype=bool)
        return cropped.reshape(h_patch, 2, w_patch, 2).any(axis=(1, 3))

    def _build_subject_object_loss_mask(self, data):
        base_debug = {
            "enabled": False,
            "status": "disabled",
            "temporal_dilate_radius": int(
                self.subject_object_loss_temporal_dilate_radius
            ),
        }
        image_masks, reason = self._build_target_image_masks(data)
        if image_masks is None or not image_masks.any():
            return {
                "subject_object_loss_mask_info": {
                    **base_debug,
                    "status": reason or "empty_mask",
                }
            }
        latent_window_size = int(self.latent_window_size)
        expected_frames = int(latent_window_size) * self.VAE_T_SCALING
        if int(image_masks.shape[0]) < expected_frames:
            return {
                "subject_object_loss_mask_info": {
                    **base_debug,
                    "status": "too_few_target_frames",
                    "target_frame_count": int(image_masks.shape[0]),
                    "expected_frame_count": int(expected_frames),
                }
            }
        image_masks = image_masks[:expected_frames]
        h_lat = int(self.height) // self.VAE_HW_SCALING
        w_lat = int(self.width) // self.VAE_HW_SCALING
        postprocess = _dynamic_mask_postprocess_module()
        latent_masks = []
        for latent_idx in range(latent_window_size):
            start = int(latent_idx) * self.VAE_T_SCALING
            stop = start + self.VAE_T_SCALING
            image_mask = np.asarray(image_masks[start:stop], dtype=bool).any(axis=0)
            latent_mask = postprocess.image_mask_to_latent_mask(
                image_mask, h_lat, w_lat
            )
            latent_masks.append(latent_mask.astype(bool, copy=False))
        latent_masks = np.stack(latent_masks, axis=0)
        if not latent_masks.any():
            return {
                "subject_object_loss_mask_info": {
                    **base_debug,
                    "status": "empty_latent_mask",
                }
            }
        patch_masks = np.stack(
            [self._latent_to_dit_patch_mask(mask) for mask in latent_masks],
            axis=0,
        )
        fused = data.get("fused_query_inputs") or {}
        frame_ids = [
            int(v)
            for v in np.asarray(fused.get("query_frame_indices", [])).reshape(-1)[
                :expected_frames
            ]
        ]
        return {
            "subject_object_loss_mask": torch.from_numpy(latent_masks).bool(),
            "subject_object_loss_patch_mask": torch.from_numpy(patch_masks).bool(),
            "subject_object_loss_mask_info": {
                **base_debug,
                "enabled": True,
                "status": "ok",
                "target_frame_ids": frame_ids,
                "target_frame_count": int(expected_frames),
                "latent_count": int(latent_masks.shape[0]),
                "latent_mask_count": int(latent_masks.sum()),
                "patch_mask_count": int(patch_masks.sum()),
                "latent_mask_ratio": float(latent_masks.mean()),
                "patch_mask_ratio": float(patch_masks.mean()),
            },
        }
