"""Dynamic-object masks for mosaic memory filtering.

The helpers here are deliberately optional: importing this module does not
import ultralytics. The YOLO dependency is loaded only when validation enables
dynamic-object filtering.
"""

from __future__ import annotations

import os
import queue
import traceback
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from examples.wanvideo.pipeline.dynamic_mask_postprocess import (
    dilate_bool_mask,
    image_mask_to_latent_mask,
    postprocess_dynamic_mask_records,
    records_to_frame_masks,
)


DEFAULT_DYNAMIC_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
)


def parse_class_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_DYNAMIC_CLASSES
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = [str(part).strip() for part in value]
    return tuple(part for part in parts if part)


def _model_names_to_dict(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {int(i): str(v) for i, v in enumerate(names)}
    return {}


def _class_ids_for_names(names, class_names: Iterable[str]) -> list[int]:
    wanted = {str(name).strip().lower() for name in class_names if str(name).strip()}
    model_names = _model_names_to_dict(names)
    return sorted(
        int(idx) for idx, name in model_names.items() if str(name).lower() in wanted
    )


def _resize_bool_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    import cv2

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    return (
        cv2.resize(mask_u8, (int(out_w), int(out_h)), interpolation=cv2.INTER_NEAREST)
        > 0
    )


def overlay_masks_on_frames(frames_np: np.ndarray, masks_np: np.ndarray) -> list[np.ndarray]:
    frames = np.asarray(frames_np, dtype=np.uint8)
    masks = np.asarray(masks_np, dtype=bool)
    out = []
    for frame, mask in zip(frames, masks):
        mask_img = _resize_bool_nearest(mask, frame.shape[0], frame.shape[1])
        overlay = frame.astype(np.float32, copy=True)
        red = np.zeros_like(overlay)
        red[..., 0] = 255.0
        overlay[mask_img] = 0.45 * overlay[mask_img] + 0.55 * red[mask_img]
        out.append(np.clip(overlay, 0, 255).astype(np.uint8))
    return out


@dataclass
class DynamicObjectMaskResult:
    source_valid_masks: np.ndarray
    dynamic_latent_masks: np.ndarray
    masked_ratio: float
    per_frame_masked_ratio: list[float]
    class_ids: list[int]
    class_names: list[str]


def _result_to_payload(result: DynamicObjectMaskResult) -> dict:
    return {
        "source_valid_masks": result.source_valid_masks,
        "dynamic_latent_masks": result.dynamic_latent_masks,
        "masked_ratio": float(result.masked_ratio),
        "per_frame_masked_ratio": list(result.per_frame_masked_ratio),
        "class_ids": list(result.class_ids),
        "class_names": list(result.class_names),
    }


def _result_from_payload(payload: dict) -> DynamicObjectMaskResult:
    return DynamicObjectMaskResult(
        source_valid_masks=np.asarray(payload["source_valid_masks"], dtype=bool),
        dynamic_latent_masks=np.asarray(payload["dynamic_latent_masks"], dtype=bool),
        masked_ratio=float(payload.get("masked_ratio", 0.0)),
        per_frame_masked_ratio=[
            float(v) for v in payload.get("per_frame_masked_ratio", [])
        ],
        class_ids=[int(v) for v in payload.get("class_ids", [])],
        class_names=[str(v) for v in payload.get("class_names", [])],
    )


class YoloDynamicObjectMasker:
    def __init__(
        self,
        model_path: str,
        *,
        class_names: Sequence[str] | str | None = None,
        conf: float = 0.25,
        imgsz: int = 960,
        device: str | int | None = None,
        debug_device: bool = False,
        dilate_latent: int = 0,
        temporal_dilate_radius: int = 0,
        track_gap_fill_max_gap: int = 0,
        tracker: str = "bytetrack.yaml",
    ):
        if not model_path:
            raise ValueError("model_path is required when dynamic-object filtering is enabled.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Dynamic-object filtering requires ultralytics. Install it or disable "
                "--validation_dynamic_object_filter."
            ) from exc

        self.class_names = parse_class_names(class_names)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.debug_device = bool(debug_device)
        self.dilate_latent = int(dilate_latent)
        self.temporal_dilate_radius = int(temporal_dilate_radius)
        self.track_gap_fill_max_gap = int(track_gap_fill_max_gap)
        self.tracker = str(tracker)
        self._debug_device_log("init-before-yolo")
        self.model = YOLO(str(model_path))
        self._debug_device_log("init-after-yolo")
        self.class_ids = _class_ids_for_names(getattr(self.model, "names", None), self.class_names)
        if not self.class_ids:
            names = _model_names_to_dict(getattr(self.model, "names", None))
            raise ValueError(
                "None of the requested dynamic classes exist in the YOLO model. "
                f"requested={list(self.class_names)} model_names={names}"
            )

    def _model_device(self) -> str:
        try:
            model = getattr(self.model, "model", None)
            if model is None:
                return "none"
            return str(next(model.parameters()).device)
        except StopIteration:
            return "no_parameters"
        except Exception as exc:
            return f"unknown:{exc.__class__.__name__}:{exc}"

    def _cuda_state(self) -> str:
        try:
            import torch

            if not torch.cuda.is_available():
                return "cuda_unavailable"
            current = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(current) / (1024**2)
            reserved = torch.cuda.memory_reserved(current) / (1024**2)
            return (
                f"current_cuda={current} "
                f"allocated_mb={allocated:.1f} reserved_mb={reserved:.1f}"
            )
        except Exception as exc:
            return f"cuda_unknown:{exc.__class__.__name__}:{exc}"

    def _debug_device_log(self, event: str, *, frame_idx: int | None = None) -> None:
        if not self.debug_device:
            return
        extra = "" if frame_idx is None else f" frame_idx={int(frame_idx)}"
        print(
            f"[yolo-device] event={event}{extra} requested_device={self.device} "
            f"{self._cuda_state()} model_device={self._model_device()}",
            flush=True,
        )

    def masks_for_frames(
        self,
        frames_np: np.ndarray,
        *,
        H_lat: int,
        W_lat: int,
        batch_idx: int | None = None,
        section_idx: int | None = None,
        clean_name: str | None = None,
    ) -> DynamicObjectMaskResult:
        frames = np.asarray(frames_np, dtype=np.uint8)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Expected frames with shape (T,H,W,3), got {tuple(frames.shape)}."
            )
        H, W = int(frames.shape[1]), int(frames.shape[2])
        from PIL import Image

        # Ultralytics treats numpy arrays as OpenCV/BGR inputs in common paths.
        # The validation decoder gives RGB frames, so pass PIL images to keep
        # channel semantics unambiguous.
        yolo_inputs = [Image.fromarray(frame, mode="RGB") for frame in frames]
        try:
            import torch

            if isinstance(self.device, str) and self.device.startswith("cuda:"):
                torch.cuda.set_device(torch.device(self.device))
        except Exception:
            pass
        frame_records = []
        names = _model_names_to_dict(getattr(self.model, "names", None))
        # Track sequentially so Ultralytics keeps a real per-frame tracker
        # state.  Passing a list of images can be treated as an image batch by
        # some versions, which is not equivalent to video tracking.
        if hasattr(self.model, "predictor"):
            self.model.predictor = None
        for frame_idx, image in enumerate(yolo_inputs):
            self._debug_device_log("track-before", frame_idx=frame_idx)
            results = self.model.track(
                image,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                classes=self.class_ids,
                verbose=False,
            )
            self._debug_device_log("track-after", frame_idx=frame_idx)
            if not results:
                continue
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or getattr(boxes, "xyxy", None) is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            track_arr = None
            if getattr(boxes, "id", None) is not None and boxes.id is not None:
                track_arr = boxes.id.detach().cpu().numpy().astype(np.int64)
            conf_arr = (
                boxes.conf.detach().cpu().numpy()
                if getattr(boxes, "conf", None) is not None
                else np.ones((xyxy.shape[0],), dtype=np.float32)
            )
            cls_arr = (
                boxes.cls.detach().cpu().numpy().astype(np.int64)
                if getattr(boxes, "cls", None) is not None
                else np.full((xyxy.shape[0],), -1, dtype=np.int64)
            )
            keep = np.isin(cls_arr, np.asarray(self.class_ids, dtype=np.int64))

            masks = getattr(result, "masks", None)
            mask_arr = None
            if masks is not None and getattr(masks, "data", None) is not None:
                mask_arr = masks.data.detach().cpu().numpy()

            for det_idx, (box, conf, cls_id, use_det) in enumerate(
                zip(xyxy, conf_arr, cls_arr, keep)
            ):
                if not bool(use_det):
                    continue
                x0, y0, x1, y1 = [float(v) for v in box]
                record = {
                    "frame_idx": int(frame_idx),
                    "det_idx": int(det_idx),
                    "track_id": (
                        int(track_arr[det_idx])
                        if track_arr is not None and det_idx < len(track_arr)
                        else None
                    ),
                    "class_id": int(cls_id),
                    "class_name": names.get(int(cls_id), str(cls_id)),
                    "conf": float(conf),
                    "x1": x0,
                    "y1": y0,
                    "x2": x1,
                    "y2": y1,
                    "box_area": float(max(0.0, x1 - x0) * max(0.0, y1 - y0)),
                    "postprocess_source": "yolo",
                }
                if mask_arr is not None and det_idx < len(mask_arr):
                    one = mask_arr[det_idx]
                    if one.shape != (H, W):
                        one = _resize_bool_nearest(one > 0.5, H, W)
                    one = np.asarray(one > 0.5, dtype=bool)
                    record["mask"] = one
                    record["mask_area"] = int(one.sum())
                    record["has_mask"] = True
                else:
                    record["mask_area"] = 0
                    record["has_mask"] = False
                frame_records.append(record)

        if int(self.track_gap_fill_max_gap) > 0:
            frame_records = postprocess_dynamic_mask_records(
                frame_records,
                frame_count=int(frames.shape[0]),
                height=H,
                width=W,
                track_gap_fill_max_gap=int(self.track_gap_fill_max_gap),
                encode_gap_fill_rle=False,
            )

        image_masks = records_to_frame_masks(
            frame_records,
            frame_count=int(frames.shape[0]),
            height=H,
            width=W,
            temporal_dilate_radius=int(self.temporal_dilate_radius),
            use_bbox_fallback=True,
        )

        dynamic_latent_masks = []
        for image_mask in image_masks:
            latent_mask = image_mask_to_latent_mask(image_mask, int(H_lat), int(W_lat))
            latent_mask = dilate_bool_mask(latent_mask, self.dilate_latent)
            dynamic_latent_masks.append(latent_mask)

        if dynamic_latent_masks:
            dynamic = np.stack(dynamic_latent_masks, axis=0).astype(bool)
        else:
            dynamic = np.zeros((0, int(H_lat), int(W_lat)), dtype=bool)
        valid = ~dynamic
        return DynamicObjectMaskResult(
            source_valid_masks=valid,
            dynamic_latent_masks=dynamic,
            masked_ratio=float(dynamic.mean()) if dynamic.size else 0.0,
            per_frame_masked_ratio=[float(x.mean()) for x in dynamic],
            class_ids=list(self.class_ids),
            class_names=[names.get(int(i), str(i)) for i in self.class_ids],
        )


def _physical_gpu_for_local_rank() -> str | None:
    local_rank = 0
    for env_name in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        try:
            local_rank = int(os.environ.get(env_name, "0"))
            break
        except ValueError:
            continue
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return str(local_rank)
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if not devices:
        return str(local_rank)
    return devices[local_rank % len(devices)]


def _normalize_worker_device(device: str | int | None, physical_gpu: str | None):
    raw = "" if device is None else str(device).strip()
    raw_lower = raw.lower()
    if raw_lower == "cpu":
        return "cpu", None
    # Treat empty/cuda/gpu/cuda:N/N all as GPU requests. The subprocess has
    # CUDA_VISIBLE_DEVICES narrowed to one physical GPU, so cuda:0 is the only
    # stable device name inside the worker.
    if physical_gpu:
        return "cuda:0", str(physical_gpu)
    if raw_lower in ("", "cuda", "gpu") or raw_lower.startswith("cuda:"):
        return ("cuda" if raw_lower in ("", "gpu") else raw), None
    return raw, None


def _yolo_worker_loop(config: dict, request_q, response_q) -> None:
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested_device = config.get("device")
    yolo_device, physical_gpu = _normalize_worker_device(
        requested_device,
        config.get("physical_gpu"),
    )
    if physical_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

    try:
        masker = YoloDynamicObjectMasker(
            str(config.get("model_path") or ""),
            class_names=config.get("class_names"),
            conf=float(config.get("conf", 0.25)),
            imgsz=int(config.get("imgsz", 960)),
            device=yolo_device,
            debug_device=bool(config.get("debug_device", False)),
            dilate_latent=int(config.get("dilate_latent", 0)),
            temporal_dilate_radius=int(config.get("temporal_dilate_radius", 0)),
            track_gap_fill_max_gap=int(config.get("track_gap_fill_max_gap", 0)),
            tracker=str(config.get("tracker", "bytetrack.yaml")),
        )
        response_q.put(
            {
                "type": "ready",
                "class_ids": list(masker.class_ids),
                "class_names": list(masker.class_names),
                "device": yolo_device,
                "requested_device": requested_device,
                "physical_gpu": physical_gpu,
                "original_visible": original_visible,
                "worker_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
    except Exception:
        response_q.put({"type": "error", "traceback": traceback.format_exc()})
        return

    while True:
        req = request_q.get()
        if req is None:
            response_q.put({"type": "closed"})
            return
        try:
            if req.get("type") != "masks_for_frames":
                raise ValueError(f"Unknown YOLO worker request: {req!r}")
            result = masker.masks_for_frames(
                np.asarray(req["frames"], dtype=np.uint8),
                H_lat=int(req["H_lat"]),
                W_lat=int(req["W_lat"]),
                batch_idx=(
                    None
                    if req.get("batch_idx") is None
                    else int(req.get("batch_idx"))
                ),
                section_idx=(
                    None
                    if req.get("section_idx") is None
                    else int(req.get("section_idx"))
                ),
                clean_name=(
                    None
                    if req.get("clean_name") is None
                    else str(req.get("clean_name"))
                ),
            )
            response_q.put({"type": "result", "payload": _result_to_payload(result)})
        except Exception:
            response_q.put({"type": "error", "traceback": traceback.format_exc()})


class YoloDynamicObjectMaskerSubprocess:
    """YOLO masker client backed by a subprocess with rank-local GPU masking.

    The validation process may already own heavy CUDA state from Wan/DA3.  This
    client starts a clean spawned worker, optionally narrows its
    CUDA_VISIBLE_DEVICES to the current rank's physical GPU, and makes
    Ultralytics see only ``cuda:0`` inside that worker.
    """

    def __init__(
        self,
        model_path: str,
        *,
        class_names: Sequence[str] | str | None = None,
        conf: float = 0.25,
        imgsz: int = 960,
        device: str | int | None = None,
        debug_device: bool = False,
        dilate_latent: int = 0,
        temporal_dilate_radius: int = 0,
        track_gap_fill_max_gap: int = 0,
        tracker: str = "bytetrack.yaml",
        timeout_sec: float = 300.0,
    ):
        import multiprocessing as mp

        self.class_names = parse_class_names(class_names)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.requested_device = device
        self.debug_device = bool(debug_device)
        self.dilate_latent = int(dilate_latent)
        self.temporal_dilate_radius = int(temporal_dilate_radius)
        self.track_gap_fill_max_gap = int(track_gap_fill_max_gap)
        self.tracker = str(tracker)
        self.timeout_sec = float(timeout_sec)
        device_str = "" if device is None else str(device)
        self.physical_gpu = (
            None if device_str.strip().lower() == "cpu" else _physical_gpu_for_local_rank()
        )

        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue(maxsize=1)
        self._response_q = ctx.Queue(maxsize=1)
        self._proc = ctx.Process(
            target=_yolo_worker_loop,
            args=(
                {
                    "model_path": str(model_path),
                    "class_names": class_names,
                    "conf": self.conf,
                    "imgsz": self.imgsz,
                    "device": device_str,
                    "debug_device": self.debug_device,
                    "dilate_latent": self.dilate_latent,
                    "temporal_dilate_radius": self.temporal_dilate_radius,
                    "track_gap_fill_max_gap": self.track_gap_fill_max_gap,
                    "tracker": self.tracker,
                    "physical_gpu": self.physical_gpu,
                },
                self._request_q,
                self._response_q,
            ),
            daemon=True,
        )
        self._proc.start()
        ready = self._recv("YOLO worker startup")
        if ready.get("type") != "ready":
            raise RuntimeError(
                "YOLO worker failed to start:\n" + str(ready.get("traceback", ready))
            )
        self.class_ids = [int(v) for v in ready.get("class_ids", [])]
        self.worker_class_names = [str(v) for v in ready.get("class_names", [])]
        self.worker_device = ready.get("device")
        self.worker_visible = ready.get("worker_visible")
        print(
            "[yolo-worker] ready "
            f"pid={self._proc.pid} requested_device={self.requested_device!r} "
            f"worker_device={self.worker_device!r} physical_gpu={self.physical_gpu!r} "
            f"worker_visible={self.worker_visible!r} class_ids={self.class_ids}",
            flush=True,
        )

    @property
    def device(self):
        return self.worker_device

    def _recv(self, context: str) -> dict:
        try:
            msg = self._response_q.get(timeout=self.timeout_sec)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out waiting for {context}.") from exc
        if isinstance(msg, dict) and msg.get("type") == "error":
            raise RuntimeError(
                f"YOLO worker error during {context}:\n{msg.get('traceback', '')}"
            )
        return msg

    def masks_for_frames(
        self,
        frames_np: np.ndarray,
        *,
        H_lat: int,
        W_lat: int,
        batch_idx: int | None = None,
        section_idx: int | None = None,
        clean_name: str | None = None,
    ) -> DynamicObjectMaskResult:
        self._request_q.put(
            {
                "type": "masks_for_frames",
                "frames": np.asarray(frames_np, dtype=np.uint8),
                "H_lat": int(H_lat),
                "W_lat": int(W_lat),
                "batch_idx": None if batch_idx is None else int(batch_idx),
                "section_idx": None if section_idx is None else int(section_idx),
                "clean_name": None if clean_name is None else str(clean_name),
            }
        )
        msg = self._recv("YOLO masks_for_frames")
        if msg.get("type") != "result":
            raise RuntimeError(f"Unexpected YOLO worker response: {msg!r}")
        return _result_from_payload(msg["payload"])

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        if proc.is_alive():
            try:
                self._request_q.put(None, timeout=1.0)
                proc.join(timeout=10.0)
            except Exception:
                pass
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
