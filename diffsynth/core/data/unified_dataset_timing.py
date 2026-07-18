import contextlib
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from . import unified_dataset as _base
from .unified_dataset import *  # noqa: F401,F403


class _NoopDatasetTimingReporter:
    enabled = False

    @contextlib.contextmanager
    def record(self, record_type, **metadata):
        yield self

    @contextlib.contextmanager
    def scope(self, name, **metadata):
        yield self

    def set_metadata(self, **kwargs):
        return None


NOOP_DATASET_TIMING_REPORTER = _NoopDatasetTimingReporter()


class DatasetTimingReporter:
    """JSON reporter for timing work done inside Dataset workers."""

    def __init__(
        self,
        output_dir,
        *,
        enabled=True,
        rank=0,
        sample_every=1,
        filename_prefix="dataset_timing",
    ):
        self.output_dir = output_dir
        self.enabled = bool(enabled and output_dir)
        self.rank = int(rank)
        self.sample_every = max(1, int(sample_every))
        self.filename_prefix = str(filename_prefix)
        self._active_record = None
        self._active_record_start = None
        self._records_seen = 0
        self._json_path = None
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    def _worker_id(self):
        try:
            info = torch.utils.data.get_worker_info()
        except (AttributeError, RuntimeError):
            info = None
        return "main" if info is None else f"worker{int(info.id):03d}"

    @property
    def json_path(self):
        if self._json_path is None:
            worker = self._worker_id()
            pid = os.getpid()
            name = (
                f"{self.filename_prefix}_rank{self.rank:03d}_" f"{worker}_pid{pid}.json"
            )
            self._json_path = os.path.join(self.output_dir, name)
        return self._json_path

    @contextlib.contextmanager
    def record(self, record_type, **metadata):
        if not self.enabled:
            yield self
            return
        self._records_seen += 1
        if self._records_seen % self.sample_every != 0:
            yield NOOP_DATASET_TIMING_REPORTER
            return
        if self._active_record is not None:
            with self.scope(record_type, **metadata):
                yield self
            return

        start = time.perf_counter()
        self._active_record_start = start
        self._active_record = {
            "type": str(record_type),
            "rank": self.rank,
            "worker": self._worker_id(),
            "pid": os.getpid(),
            "metadata": dict(metadata),
            "events": [],
        }
        try:
            yield self
        finally:
            end = time.perf_counter()
            record = self._active_record
            self._active_record = None
            self._active_record_start = None
            record["duration_s"] = max(0.0, end - start)
            record["event_totals_s"] = self._event_totals(record["events"])
            record["event_counts"] = self._event_counts(record["events"])
            self._write_record(record)

    @contextlib.contextmanager
    def scope(self, name, **metadata):
        if not self.enabled or self._active_record is None:
            yield self
            return
        start = time.perf_counter()
        try:
            yield self
        finally:
            end = time.perf_counter()
            self._active_record["events"].append(
                {
                    "name": str(name),
                    "start_ms": round((start - self._active_record_start) * 1000.0, 3),
                    "duration_s": max(0.0, end - start),
                    "metadata": dict(metadata),
                }
            )

    def set_metadata(self, **kwargs):
        if self.enabled and self._active_record is not None:
            self._active_record["metadata"].update(kwargs)

    def _write_record(self, record):
        records = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    records = payload
            except (OSError, ValueError):
                records = []
        records.append(record)
        tmp_path = self.json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.json_path)

    def _event_totals(self, events):
        totals = defaultdict(float)
        for event in events:
            totals[event["name"]] += float(event["duration_s"])
        return dict(sorted(totals.items()))

    def _event_counts(self, events):
        counts = defaultdict(int)
        for event in events:
            counts[event["name"]] += 1
        return dict(sorted(counts.items()))


class DatasetTimingFrustumHandlerProxy:
    _INTERNAL_TIMING_METHODS = {
        "_coerce_memory_K": "dataset.frustum.coerce_memory_K",
        "_select_group_ids_by_projection_iou": "dataset.frustum.select_groups_projection_iou",
        "_build_group_representative_geometry": "dataset.frustum.build_representative_geometry",
        "_rank_candidates_by_angle_then_translation": "dataset.frustum.rank_candidates",
        "_score_group_candidate_by_projection_iou": "dataset.frustum.score_projection_iou",
        "_project_memory_to_query": "dataset.frustum.project_memory_to_query",
        "_downsample_depth_to_latent": "dataset.frustum.downsample_depth",
        "_compute_canvas_iou_score": "dataset.frustum.compute_canvas_iou",
        "_clipped_proj_bbox": "dataset.frustum.clipped_proj_bbox",
        "_pick_candidates_with_nms": "dataset.frustum.pick_candidates_nms",
    }

    def __init__(self, handler, reporter):
        self._handler = handler
        self._reporter = reporter or NOOP_DATASET_TIMING_REPORTER

    def __getattr__(self, name):
        return getattr(self._handler, name)

    @contextlib.contextmanager
    def _timed_internal_methods(self):
        originals = {}
        try:
            for method_name, event_name in self._INTERNAL_TIMING_METHODS.items():
                if not hasattr(self._handler, method_name):
                    continue
                original = getattr(self._handler, method_name)
                if not callable(original):
                    continue
                originals[method_name] = original

                def _make_wrapper(_original, _event_name, _method_name):
                    def _wrapper(*args, **kwargs):
                        with self._reporter.scope(
                            _event_name,
                            method=_method_name,
                        ):
                            return _original(*args, **kwargs)

                    return _wrapper

                setattr(
                    self._handler,
                    method_name,
                    _make_wrapper(original, event_name, method_name),
                )
            yield
        finally:
            for method_name, original in originals.items():
                setattr(self._handler, method_name, original)

    def select_candidates(self, **kwargs):
        query = kwargs.get("query_extrinsics")
        query_count = len(query) if query is not None else None
        with self._reporter.scope(
            "dataset.frustum_select_candidates",
            query_count=query_count,
            total_latents=kwargs.get("total_latents"),
            selection_mode=kwargs.get("selection_mode", "projection_iou"),
            candidates_per_query_group=kwargs.get("candidates_per_query_group"),
        ):
            with self._timed_internal_methods():
                return self._handler.select_candidates(**kwargs)


class DatasetTimingMixin:
    def _init_dataset_timing(
        self,
        *,
        dataset_timing_enabled=False,
        dataset_timing_dir="",
        dataset_timing_rank=0,
        dataset_timing_sample_every=1,
    ):
        self.dataset_timing_enabled = bool(dataset_timing_enabled)
        self.dataset_timing_dir = str(dataset_timing_dir or "")
        self.dataset_timing_rank = int(dataset_timing_rank)
        self.dataset_timing_sample_every = max(1, int(dataset_timing_sample_every))
        self._dataset_timing_reporter = None

    def _timing_reporter(self):
        if not self.dataset_timing_enabled:
            return NOOP_DATASET_TIMING_REPORTER
        if self._dataset_timing_reporter is None:
            self._dataset_timing_reporter = DatasetTimingReporter(
                self.dataset_timing_dir,
                enabled=True,
                rank=self.dataset_timing_rank,
                sample_every=self.dataset_timing_sample_every,
            )
        return self._dataset_timing_reporter

    def _summarize_timing_data(self, data):
        if not isinstance(data, dict):
            return {}
        fused = data.get("fused_query_inputs") or {}
        memory_frames_by_id = data.get("memory_frames_by_id") or {}
        cached_paths = data.get("cached_memory_paths_by_id") or {}
        candidate_groups = fused.get("candidate_frame_ids") or []
        candidate_count = sum(len(group) for group in candidate_groups)
        return {
            "clean_name": (data.get("info") or {}).get("clean_name"),
            "cache_hits": len(cached_paths),
            "cache_misses": len(memory_frames_by_id),
            "candidate_groups": len(candidate_groups),
            "candidate_count": int(candidate_count),
            "memory_total_frames": int(fused.get("memory_total_frames", 0) or 0),
        }


class _TimingVideoDatasetMixin(DatasetTimingMixin):
    def __init__(self, *args, **kwargs):
        timing_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key.startswith("dataset_timing_")
        }
        self._init_dataset_timing(**timing_kwargs)
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        reporter = self._timing_reporter()
        with reporter.record(
            "dataset.__getitem__",
            dataset_cls=self.__class__.__name__,
            index=int(index),
            epoch_id=int(getattr(self, "_epoch_id", 0)),
            rank=int(getattr(self, "rank", 0)),
        ) as active:
            data = super().__getitem__(index)
            active.set_metadata(**self._summarize_timing_data(data))
            return data

    def _select_path(self, index):
        with self._timing_reporter().scope("dataset.select_path", index=int(index)):
            return super()._select_path(index)

    def _sample_blocks(self):
        with self._timing_reporter().scope("dataset.sample_blocks"):
            return super()._sample_blocks()

    def _load_extrinsics(self, path):
        with self._timing_reporter().scope("dataset.load_extrinsics"):
            return super()._load_extrinsics(path)

    def _load_intrinsics(self, path, num_frames):
        with self._timing_reporter().scope(
            "dataset.load_intrinsics", num_frames=int(num_frames)
        ):
            return super()._load_intrinsics(path, num_frames)

    def normalize_and_scale_intrinsics(self, intrinsics, *args, **kwargs):
        with self._timing_reporter().scope("dataset.scale_intrinsics"):
            return super().normalize_and_scale_intrinsics(intrinsics, *args, **kwargs)

    def _load_depths_for_memory(
        self,
        depth_path,
        frame_count,
        frame_ids=None,
        *,
        depth_format=None,
        depth_metadata=None,
        require_depth_format=None,
    ):
        with self._timing_reporter().scope(
            "dataset.load_depths_for_memory",
            frame_count=int(frame_count),
            selected_frame_count=(
                -1 if frame_ids is None else len({int(fid) for fid in frame_ids})
            ),
        ):
            return super()._load_depths_for_memory(
                depth_path,
                frame_count,
                frame_ids=frame_ids,
                depth_format=depth_format,
                depth_metadata=depth_metadata,
                require_depth_format=require_depth_format,
            )

    def _resolve_prompt_for_scene(self, scene_hash, frame_min=None, frame_max=None):
        with self._timing_reporter().scope("dataset.resolve_prompt"):
            return super()._resolve_prompt_for_scene(
                scene_hash, frame_min=frame_min, frame_max=frame_max
            )

    def _get_frustum_handler(self):
        with self._timing_reporter().scope("dataset.get_frustum_handler"):
            if self.handler is None:
                from frustum.frustum_handler_timing import TimingFrustumHandler

                self.handler = TimingFrustumHandler(
                    np.eye(3),
                    latent_stride=16,
                    timing_reporter=self._timing_reporter(),
                )
            elif hasattr(self.handler, "set_timing_reporter"):
                self.handler.set_timing_reporter(self._timing_reporter())
            return self.handler

    def _read_video_frames(self, video_path, frame_indices, *, return_metadata=False):
        reporter = self._timing_reporter()
        with reporter.scope(
            "dataset.read_video_frames",
            frame_count=len(frame_indices),
        ):
            from decord import VideoReader, cpu

            with reporter.scope("dataset.read_video_frames.init_reader"):
                reader = VideoReader(str(video_path), ctx=cpu())
            with reporter.scope(
                "dataset.read_video_frames.get_batch",
                frame_count=len(frame_indices),
            ):
                frames = reader.get_batch(list(frame_indices)).asnumpy()
            metadata = {
                "source_height": int(frames.shape[1]),
                "source_width": int(frames.shape[2]),
            }
            out = _base._video_frames_uint8_to_normalized_tensor(
                frames,
                height=self.height,
                width=self.width,
                timing_reporter=reporter,
                timing_prefix="dataset.read_video_frames",
            )
            if return_metadata:
                return out, metadata
            return out

    def read_camera_params(self, data, info, lookup=None):
        with self._timing_reporter().record(
            "dataset.read_camera_params",
            dataset_cls=self.__class__.__name__,
            clean_name=(info or {}).get("clean_name"),
        ):
            return super().read_camera_params(data, info, lookup=lookup)


class TimingDA3MosaicVideoDataset(
    _TimingVideoDatasetMixin, _base.DA3MosaicVideoDataset
):
    pass


DA3MosaicVideoDataset = TimingDA3MosaicVideoDataset
