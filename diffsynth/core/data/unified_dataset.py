from .operators import *
import contextlib
import glob
import hashlib
import importlib.util
import os
import random
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch, json, pandas

# Bumping this invalidates every on-disk dataset cache. Bump whenever the
# payload schema or any logic that affects the scan output changes in a way
# that would make older caches return wrong data.
#
# v2: legacy mosaic latent scan discovered ``depth/video.zip`` co-located
#     with ``pose/`` and ``intrinsics/`` and records it in ``info["depth_path"]``.
#     Older caches stored ``depth_path=None`` for the same scenes, which would
#     silently disable the frustum query path -> mosaic conditioning -> wrong
#     training signal. Bumping the version forces a re-scan.
# v3: video-flow datasets add a brand-new ``info`` payload shape
#     (``video_path`` / ``frame_count`` instead of ``latent_path`` /
#     ``latent``); the cache fingerprint now also folds in
#     ``height``/``width``/``vae_clean_context_blocks_max``/
#     ``memory_latent_cache_version``/``dataset_kind`` so a video-flow
#     scan never collides with a latent-flow scan in the same cache dir.
# v4: Train/val split + ``self.path`` shuffle now use seed-derived
#     ``random.Random`` instances instead of mutating the global
#     ``random`` state. The split decision is unchanged for the same
#     ``--seed`` (key derivation is keyed on ``"split"`` only) but the
#     post-scan path order is now ``(seed, "path_shuffle",
#     dataset_pass_id)``-derived so old caches whose payload was written
#     under the legacy ``random.seed(seed)`` codepath would mismatch the
#     manifest order on reload. Bump forces a re-scan + re-shuffle.
# v5: Validation datasets now scan only their held-out split instead of
#     scanning the full corpus and relying on runtime ``val_raw_paths``
#     selection. Manifests also carry explicit split/sample keys for
#     future save/load driven eval runs.
DATASET_CACHE_VERSION = 5
VAL_DATASET_PATH_SHUFFLE_SEED = 42


_DYNAMIC_MASK_POSTPROCESS_MODULE = None


def _dynamic_mask_postprocess_module():
    """Load the shared dynamic-mask postprocess utilities from this repo."""
    global _DYNAMIC_MASK_POSTPROCESS_MODULE
    if _DYNAMIC_MASK_POSTPROCESS_MODULE is not None:
        return _DYNAMIC_MASK_POSTPROCESS_MODULE
    try:
        from examples.wanvideo.pipeline import dynamic_mask_postprocess

        _DYNAMIC_MASK_POSTPROCESS_MODULE = dynamic_mask_postprocess
        return _DYNAMIC_MASK_POSTPROCESS_MODULE
    except ModuleNotFoundError:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        module_path = os.path.join(
            repo_root,
            "examples",
            "wanvideo",
            "pipeline",
            "dynamic_mask_postprocess.py",
        )
        if not os.path.exists(module_path):
            raise
        module_name = "_diffsynth_dynamic_mask_postprocess"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load dynamic mask utilities: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _DYNAMIC_MASK_POSTPROCESS_MODULE = module
        return _DYNAMIC_MASK_POSTPROCESS_MODULE


def _derive_seed(*parts) -> int:
    """Stable, well-distributed 32-bit seed derived from a tuple of parts.

    Uses BLAKE2b-32 over a ``|``-joined UTF-8 string of the parts, so any
    callable site that needs a seed for a specific *role* (per-rank, per-
    epoch, per-sample, ...) can compose a unique int without worrying
    about hash collisions or pollutting the global ``random`` state.

    Examples:
        >>> _derive_seed(0, "split")        # canonical train/val split
        >>> _derive_seed(0, "sample", 1, 4, 17)  # pass=0, epoch=1, rank=4, idx=17
    """
    blob = b"|".join(str(p).encode() for p in parts)
    h = hashlib.blake2b(blob, digest_size=4)
    return int.from_bytes(h.digest(), "little")


def _sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def _optional_timing_scope(timing_reporter, name, **metadata):
    if timing_reporter is None or not hasattr(timing_reporter, "scope"):
        yield
        return
    with timing_reporter.scope(name, **metadata):
        yield


def _pingpong_reflect_index(fid, length):
    """Map an arbitrary frame index into ``[0, length-1]`` by ping-pong.

    Instead of clamping out-of-range indices to the last frame, fold them back
    and forth (boustrophedon / triangle wave): ``0, 1, ..., L-1, L-2, ..., 1,
    0, 1, ...``. Used by ``read_camera_params`` when
    ``--validation_camera_pingpong`` is set so a camera trajectory shorter than
    the rollout keeps moving instead of freezing on the final pose. Negative
    indices clamp to the first frame (the pre-existing behaviour for the few
    bootstrap frames that fall before frame 0).
    """
    length = int(length)
    if length <= 1:
        return 0
    fid = max(0, int(fid))
    period = 2 * (length - 1)
    m = fid % period
    return m if m < length else period - m


def _video_frames_uint8_to_normalized_tensor(
    frames_np,
    *,
    height,
    width,
    normalize=True,
    timing_reporter=None,
    timing_prefix="dataset.read_video_frames",
):
    """Convert decord ``uint8`` NHWC frames to ``(C,T,H,W)``.

    Cropping happens before float conversion so unused pixels never pay the
    uint8->float32 conversion cost. When ``normalize`` is false, the returned
    tensor stays uint8 unless resize is required; resized tensors remain float
    in pixel range [0, 255] and are normalized on GPU before VAE encode.
    """
    import torch.nn.functional as F

    frames_np = np.asarray(frames_np)
    if frames_np.ndim != 4 or frames_np.shape[-1] != 3:
        raise ValueError(
            "Expected video frames as uint8 NHWC array with 3 channels, "
            f"got shape {frames_np.shape}."
        )

    height = int(height)
    width = int(width)
    with _optional_timing_scope(
        timing_reporter,
        f"{timing_prefix}.crop_uint8",
        source_height=int(frames_np.shape[1]),
        source_width=int(frames_np.shape[2]),
        target_height=height,
        target_width=width,
    ):
        src_h, src_w = int(frames_np.shape[1]), int(frames_np.shape[2])
        crop_h = min(src_h, height)
        crop_w = min(src_w, width)
        top = max((src_h - crop_h) // 2, 0)
        left = max((src_w - crop_w) // 2, 0)
        frames_np = frames_np[:, top : top + crop_h, left : left + crop_w, :]

    with _optional_timing_scope(
        timing_reporter,
        f"{timing_prefix}.to_tensor",
        frame_count=int(frames_np.shape[0]),
        normalize=bool(normalize),
    ):
        frames = torch.as_tensor(frames_np).permute(0, 3, 1, 2)
        if normalize:
            frames = frames.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            frames.mul_(2.0 / 255.0).sub_(1.0)
        else:
            frames = frames.contiguous(memory_format=torch.contiguous_format)

    with _optional_timing_scope(
        timing_reporter,
        f"{timing_prefix}.resize",
        target_height=height,
        target_width=width,
    ):
        if frames.shape[-2:] != (height, width):
            if not normalize:
                frames = frames.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            frames = F.interpolate(
                frames,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

    with _optional_timing_scope(timing_reporter, f"{timing_prefix}.layout"):
        frames = frames.permute(1, 0, 2, 3)
    return frames


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = (
            {} if special_operator_map is None else special_operator_map
        )
        self.max_data_items = max_data_items
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
    ):
        return RouteByType(
            operator_map=[
                (
                    str,
                    ToAbsolutePath(base_path)
                    >> LoadImage()
                    >> ImageCropAndResize(
                        height,
                        width,
                        max_pixels,
                        height_division_factor,
                        width_division_factor,
                    ),
                ),
                (
                    list,
                    SequencialProcess(
                        ToAbsolutePath(base_path)
                        >> LoadImage()
                        >> ImageCropAndResize(
                            height,
                            width,
                            max_pixels,
                            height_division_factor,
                            width_division_factor,
                        )
                    ),
                ),
            ]
        )

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_rate=24,
        fix_frame_rate=False,
    ):
        return RouteByType(
            operator_map=[
                (
                    str,
                    ToAbsolutePath(base_path)
                    >> RouteByExtensionName(
                        operator_map=[
                            (
                                ("jpg", "jpeg", "png", "webp"),
                                LoadImage()
                                >> ImageCropAndResize(
                                    height,
                                    width,
                                    max_pixels,
                                    height_division_factor,
                                    width_division_factor,
                                )
                                >> ToList(),
                            ),
                            (
                                ("gif",),
                                LoadGIF(
                                    num_frames,
                                    time_division_factor,
                                    time_division_remainder,
                                    frame_processor=ImageCropAndResize(
                                        height,
                                        width,
                                        max_pixels,
                                        height_division_factor,
                                        width_division_factor,
                                    ),
                                ),
                            ),
                            (
                                ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
                                LoadVideo(
                                    num_frames,
                                    time_division_factor,
                                    time_division_remainder,
                                    frame_processor=ImageCropAndResize(
                                        height,
                                        width,
                                        max_pixels,
                                        height_division_factor,
                                        width_division_factor,
                                    ),
                                    frame_rate=frame_rate,
                                    fix_frame_rate=fix_frame_rate,
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, "r") as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat


def _split_path_list(path_or_paths):
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, str):
        return path_or_paths.split(",") if "," in path_or_paths else [path_or_paths]
    return list(path_or_paths)


# The DiT's 3D RoPE temporal frequency table is precomputed with a fixed
# 1024-entry timeline (``precompute_freqs_cis_3d(head_dim)`` in
# ``diffsynth/models/wan_video_dit.py``). Flows that index RoPE with the
# REAL latent timeline (train clean-history modes A/B, validation with
# ``--use_original_latent_indices``) can reach indices up to about
# ``frame_count // 4 + latent_window_size``; past the table length the
# embedding lookup raises a CUDA indexing error mid-run, which is near
# impossible to trace back to one over-long scene. Such scenes are
# filtered at scan time with the dedicated ``rope_time_overflow`` label.
ROPE_FREQS_TABLE_END = 1024


def _rope_time_overflow(
    frame_count, latent_window_size, table_end=ROPE_FREQS_TABLE_END
):
    """Return the (conservative) max real-timeline RoPE index for a scene
    when it would overflow the precomputed freqs table, else ``None``.

    Conservative bound: the largest generating start (``frame_count`` minus
    one noisy window) contributes ``frame_count // 4`` and the noisy window
    itself adds up to ``latent_window_size`` latent steps.
    """
    max_rope_time = int(frame_count) // 4 + int(latent_window_size)
    if max_rope_time >= int(table_end):
        return max_rope_time
    return None


def _normalize_vipe_prompt_type(value):
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("vipe_prompt_type must be 1 or 2.") from exc
    if value not in (1, 2):
        raise ValueError(f"vipe_prompt_type must be 1 or 2, got {value!r}.")
    return value


class _MosaicDatasetBase(UnifiedDataset):
    VAE_HW_SCALING = 16
    VAE_T_SCALING = 4

    def __init__(
        self,
        base_path=None,
        camera_params_path=None,
        depth_path=None,
        steps_per_epoch=1,
        latent_window_size=None,
        is_val_dataset=False,
        val_assign_yaml=None,
        val_ratio=0.05,
        val_raw_paths=None,
        filter_yaml=None,
        include_yaml=None,
        random_start_latent_prob=0.0,
        seed=None,
        rank=0,
        max_data_items=None,
        max_scan_items=None,
        frustum_handler_cls=None,
        dataset_cache_dir=None,
        dataset_pass_id=0,
        max_frames_per_scene=None,
        force_override_extrinsic=None,
        mosaic_intrinsics_mode="per_frame",
        mosaic_query_reference_frame=2,
        mosaic_view_change_prope=False,
        vipe_prompt_type=1,
        dataset_index_path=None,
        min_frame_count=None,
        dataset_compact_mode=False,
        dataset_balance_mode="global",
        dataset_balance_root_probs="",
        val_dataset_path_shuffle_seed=VAL_DATASET_PATH_SHUFFLE_SEED,
        **kwargs,
    ):
        if base_path is None:
            raise ValueError("base_path is required for _MosaicDatasetBase.")
        if camera_params_path is None:
            camera_params_path = kwargs.get("camera_params_path")
        if camera_params_path is None:
            raise ValueError("camera_params_path is required for _MosaicDatasetBase.")
        if depth_path is None:
            depth_path = kwargs.get("depth_path")
        if depth_path is None:
            raise ValueError("depth_path is required for _MosaicDatasetBase.")
        if latent_window_size is None:
            latent_window_size = kwargs.get("latent_window_size")
        if latent_window_size is None:
            raise ValueError("latent_window_size is required for _MosaicDatasetBase.")
        if force_override_extrinsic is None:
            force_override_extrinsic = kwargs.get("force_override_extrinsic")

        self.base_path = _split_path_list(base_path)
        self.camera_params_path = _split_path_list(camera_params_path)
        self.depth_path = _split_path_list(depth_path)
        self.force_override_extrinsic = (
            str(force_override_extrinsic) if force_override_extrinsic else None
        )
        self._force_override_extrinsic_npz_paths = (
            self._discover_force_override_extrinsic_npz_paths(
                self.force_override_extrinsic
            )
        )
        self.metadata_path = None
        self.repeat = 1
        self.data_file_keys = tuple()
        self.main_data_operator = lambda x: x
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {}
        self.max_data_items = max_data_items
        self.load_from_cache = False
        self.data = []
        self.cached_data = []

        self.steps_per_epoch = int(steps_per_epoch)
        self.latent_window_size = int(latent_window_size)
        self.is_val_dataset = bool(is_val_dataset)
        self.mosaic_intrinsics_mode = self._normalize_mosaic_intrinsics_mode(
            mosaic_intrinsics_mode
        )
        self.mosaic_query_reference_frame = (
            self._normalize_mosaic_query_reference_frame(mosaic_query_reference_frame)
        )
        self.mosaic_view_change_prope = bool(mosaic_view_change_prope)
        self.vipe_prompt_type = _normalize_vipe_prompt_type(vipe_prompt_type)
        self.random_start_latent_prob = float(random_start_latent_prob)
        self.seed = seed
        self.rank = int(rank)
        # Bumped (and rolled into derived seeds) every time training
        # resumes from a checkpoint, so the post-resume data sequence is
        # disjoint from -- but still reproducible vs -- the previous run.
        self.dataset_pass_id = int(dataset_pass_id)
        # Test / debug knob: cap each scene to its first N frames. ``None``
        # (or any non-positive value) disables the cap. Applies to:
        #   * scan: ``too_short`` is evaluated against the capped count and
        #     ``info["frame_count"]`` is stored already capped.
        #   * runtime: ``__getitem__`` derives ``G`` / latent windows
        #     from the capped ``info[...]`` values, so generating /
        #     memory frames are guaranteed to fall inside [0, N).
        # Use to overfit on a small temporal slice, or to run end-to-end
        # smoke tests on long clips without paying for the full footage.
        if max_frames_per_scene is None or int(max_frames_per_scene) <= 0:
            self.max_frames_per_scene = None
        else:
            self.max_frames_per_scene = int(max_frames_per_scene)
        self._frustum_handler_cls = frustum_handler_cls
        self.handler = None

        # Cache / manifest plumbing. The cache lives under ``dataset_cache_dir``
        # (e.g. ``./.cache/dataset_info``) and is a JSON snapshot of the full
        # scan output. ``save_manifest`` writes the exact same payload to an
        # arbitrary path, so the log directory's copy and the on-disk cache
        # stay byte-identical and are always in sync.
        self.dataset_cache_dir = dataset_cache_dir
        self._filter_yaml_path = filter_yaml
        self._include_yaml_path = include_yaml
        self._val_assign_yaml_path = val_assign_yaml
        self._val_ratio = float(val_ratio) if val_ratio is not None else None
        self._max_scan_items = max_scan_items
        self.dataset_index_path = (
            str(dataset_index_path) if dataset_index_path else None
        )
        self.dataset_compact_mode = bool(dataset_compact_mode)
        self.dataset_balance_mode = self._normalize_dataset_balance_mode(
            dataset_balance_mode
        )
        self.dataset_balance_root_probs = str(dataset_balance_root_probs or "")
        if val_dataset_path_shuffle_seed is None:
            val_dataset_path_shuffle_seed = VAL_DATASET_PATH_SHUFFLE_SEED
        self.val_dataset_path_shuffle_seed = int(val_dataset_path_shuffle_seed)
        if min_frame_count is None or int(min_frame_count) <= 0:
            self.min_frame_count = None
        else:
            self.min_frame_count = int(min_frame_count)
        self._last_dataset_cache_payload = None
        self._last_dataset_cache_path = None
        self._last_dataset_cache_hit = False

        rawpath = self._collect_latent_paths(filter_yaml)
        rawpath = self._apply_include_yaml(rawpath, include_yaml)
        self.val_raw_paths = []
        if val_raw_paths is not None and self.is_val_dataset:
            self.val_raw_paths = list(
                dict.fromkeys(str(path) for path in val_raw_paths)
            )
            self.rawpath = self._filter_raw_paths_by_keys(rawpath, self.val_raw_paths)
            val_assigns = []
        else:
            self.rawpath, val_assigns = self._split_train_val_paths(
                rawpath,
                val_assign_yaml=val_assign_yaml,
                val_ratio=val_ratio,
            )

        self.reason = {}
        scan_rawpath = (
            self.rawpath[:max_scan_items]
            if max_scan_items is not None
            else self.rawpath
        )
        self.dataset_info = self._build_or_load_dataset_info(
            scan_rawpath,
            val_assigns,
        )
        self.path = sorted(self.dataset_info.keys())
        if self.is_val_dataset:
            self.val_raw_paths = list(
                dict.fromkeys(
                    path for path in self.val_raw_paths if path in self.dataset_info
                )
            )

        # Path order is shuffled with a private ``random.Random`` instance
        # so the global ``random`` state stays pristine. Train datasets use a
        # role-derived seed that advances across resumes; validation datasets
        # intentionally use a separate seed so ``self.path`` order is stable
        # unless the validation shuffle seed is explicitly changed.
        if self.is_val_dataset:
            shuffle_rng = random.Random(self.val_dataset_path_shuffle_seed)
        else:
            shuffle_rng = random.Random(
                _derive_seed(self.seed, "path_shuffle", self.dataset_pass_id)
            )
        shuffle_rng.shuffle(self.path)

        # Runtime validation sampling goes through ``_select_path`` which
        # prefers ``self.val_raw_paths`` when it is non-empty. Keep that
        # actual sampling list aligned with the fixed validation ``self.path``
        # order; otherwise the fixed ``self.path`` shuffle is bypassed.
        if self.is_val_dataset:
            self.val_raw_paths = list(self.path)

        # Sampling-time state (rank-aware index rotation, per-sample RNG
        # derivation). Defaults reproduce the legacy "all ranks see the
        # same scene at the same step" behaviour; ``set_epoch`` is what
        # makes it rank-aware. The trainer calls ``set_epoch`` at the
        # top of each epoch.
        self._epoch_id = 0
        self._world_size = 1
        if self.rank is not None:
            self._rank = int(self.rank)
        else:
            self._rank = 0
        self._epoch_offset = 0
        self._validation_sample_offset = 0
        self._build_sampling_groups()

    @staticmethod
    def _load_yaml_tokens(yaml_path):
        if yaml_path is None:
            return []
        import yaml

        with open(yaml_path, "r") as f:
            payload = yaml.safe_load(f) or []
        if isinstance(payload, dict):
            for key in ("include", "include_hashes", "hashes", "items"):
                if key in payload:
                    payload = payload[key] or []
                    break
            else:
                payload = list(payload.values())
        if isinstance(payload, (str, int, float)):
            payload = [payload]
        return [str(item).strip() for item in payload if str(item).strip()]

    def _apply_include_yaml(self, rawpath, include_yaml):
        include_tokens = self._load_yaml_tokens(include_yaml)
        if not include_tokens:
            return rawpath
        return [
            path
            for path in rawpath
            if any(include_token in path for include_token in include_tokens)
        ]

    @staticmethod
    def _raw_path_to_dataset_key(path):
        clean_name = os.path.basename(path).split(".")[0]
        return f"{os.path.dirname(path)}/{clean_name}"

    def _filter_raw_paths_by_keys(self, rawpath, keys):
        key_set = set(keys or [])
        if not key_set:
            return []
        return [
            path for path in rawpath if self._raw_path_to_dataset_key(path) in key_set
        ]

    def _split_train_val_paths(self, rawpath, val_assign_yaml=None, val_ratio=0.05):
        if val_assign_yaml is not None:
            import yaml

            with open(val_assign_yaml, "r") as f:
                val_assigns = yaml.safe_load(f) or []
            matched, unmatched = [], []
            for path in rawpath:
                if any(assign in path for assign in val_assigns):
                    matched.append(path)
                    self.val_raw_paths.append(self._raw_path_to_dataset_key(path))
                else:
                    unmatched.append(path)
            return (matched if self.is_val_dataset else unmatched), val_assigns

        val_assigns = []
        shuffled = list(rawpath)
        # Stable per-seed train/val split. Deliberately keyed on
        # ``"split"`` only -- NOT on ``dataset_pass_id`` -- so a resumed
        # run keeps the same val set as the original (otherwise val
        # metrics would be computed on a moving target across resumes).
        split_rng = random.Random(_derive_seed(self.seed, "split"))
        split_rng.shuffle(shuffled)
        split = int(len(shuffled) * val_ratio)
        matched = shuffled[:split]
        unmatched = shuffled[split:]
        self.val_raw_paths.extend(
            self._raw_path_to_dataset_key(path) for path in matched
        )
        return (matched if self.is_val_dataset else unmatched), val_assigns

    # ------------------------------------------------------------------
    # Dataset-info cache & manifest
    # ------------------------------------------------------------------

    def _dataset_cache_inputs_dict(self):
        """Serializable snapshot of constructor inputs for this scan.

        Subclasses can override to add backend-specific knobs (e.g. prompt
        roots or sqlite index paths). The dict ends up inside both the cache
        file and the log-dir manifest so the
        operator can grep / diff what produced a given training run.
        """
        inputs = {
            "base_path": list(self.base_path),
            "camera_params_path": list(self.camera_params_path),
            "depth_path": list(self.depth_path),
            "filter_yaml": self._filter_yaml_path,
            "include_yaml": self._include_yaml_path,
            "val_assign_yaml": self._val_assign_yaml_path,
            "val_ratio": self._val_ratio,
            "seed": self.seed,
            "latent_window_size": int(self.latent_window_size),
            "max_scan_items": self._max_scan_items,
            "dataset_index_path": self.dataset_index_path,
            "min_frame_count": self.min_frame_count,
            "dataset_compact_mode": bool(self.dataset_compact_mode),
            "is_val_dataset": bool(self.is_val_dataset),
            # Affects post-scan ``self.path`` order. Folded in so the
            # cached manifest survives ``dataset_pass_id`` bumps without
            # silently reusing a stale shuffle from a previous resume.
            "dataset_pass_id": int(self.dataset_pass_id),
            # Caps each scene's recorded length, so different cap values
            # need different manifest files (otherwise switching the cap
            # would silently reuse a stale ``info["frame_count"]`` /
            # ``info["latent"]``).
            "max_frames_per_scene": self.max_frames_per_scene,
            "mosaic_intrinsics_mode": self.mosaic_intrinsics_mode,
            "mosaic_query_reference_frame": int(self.mosaic_query_reference_frame),
            "mosaic_view_change_prope": bool(self.mosaic_view_change_prope),
            "vipe_prompt_type": int(self.vipe_prompt_type),
        }
        if self.is_val_dataset:
            inputs["val_dataset_path_shuffle_seed"] = int(
                self.val_dataset_path_shuffle_seed
            )
        if self.force_override_extrinsic:
            inputs["force_override_extrinsic"] = self.force_override_extrinsic
        return inputs

    @staticmethod
    def _discover_force_override_extrinsic_npz_paths(path):
        if not path or not os.path.isdir(path):
            return None
        npz_paths = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not npz_paths:
            raise ValueError(
                f"force_override_extrinsic directory has no .npz files: {path}"
            )
        return npz_paths

    @staticmethod
    def _canonical_view_change_like_latents(latents):
        if latents.ndim == 5:
            t, h, w = latents.shape[2], latents.shape[-2], latents.shape[-1]
        elif latents.ndim == 4:
            t, h, w = latents.shape[1], latents.shape[-2], latents.shape[-1]
        else:
            raise ValueError(
                f"Expected latent with 4 or 5 dims, got shape {tuple(latents.shape)}."
            )
        vc = torch.zeros(
            int(t), int(h), int(w), 3, dtype=latents.dtype, device=latents.device
        )
        vc[..., 0] = 1.0
        return vc

    @staticmethod
    def _normalize_mosaic_intrinsics_mode(mode):
        mode = str(mode or "per_frame").strip().lower()
        aliases = {
            "per-frame": "per_frame",
            "perframe": "per_frame",
            "frame": "per_frame",
            "mean": "episode_mean",
            "episode-mean": "episode_mean",
            "episode_kmean": "episode_mean",
            "kmean": "episode_mean",
            "first": "first_frame",
            "first-frame": "first_frame",
            "firstframe": "first_frame",
            "frame0": "first_frame",
            "first_frame_only": "first_frame",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"per_frame", "episode_mean", "first_frame"}:
            raise ValueError(
                "mosaic_intrinsics_mode must be 'per_frame', 'episode_mean', "
                "or 'first_frame', "
                f"got {mode!r}."
            )
        return mode

    @staticmethod
    def _normalize_mosaic_query_reference_frame(frame):
        frame = int(frame)
        if frame < 1 or frame > 4:
            raise ValueError(
                "mosaic_query_reference_frame must be in [1, 4] (1-based), "
                f"got {frame}."
            )
        return frame

    def _intrinsics_temporal_mean_enabled(self):
        return self.mosaic_intrinsics_mode == "episode_mean"

    @staticmethod
    def _normalize_dataset_balance_mode(mode):
        mode = str(mode or "global").strip().lower()
        aliases = {
            "": "global",
            "none": "global",
            "off": "global",
            "false": "global",
            "0": "global",
            "camera-root": "camera_root",
            "camera": "camera_root",
            "root": "camera_root",
            "manual": "camera_root_manual",
            "camera-root-manual": "camera_root_manual",
            "camera_manual": "camera_root_manual",
            "root_manual": "camera_root_manual",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"global", "camera_root", "camera_root_manual"}:
            raise ValueError(
                "dataset_balance_mode must be 'global', 'camera_root', or "
                "'camera_root_manual', "
                f"got {mode!r}."
            )
        return mode

    def _dataset_cache_fingerprint(self, rawpath, val_assigns):
        """Stable hex fingerprint of everything that affects the scan.

        Includes the *exact* list of latent files we're about to scan
        (so adding / removing a sample invalidates the cache) plus the
        camera / depth / prompt roots, the val assignment, the latent
        window threshold (used to drop too-short scenes) and the class
        name. Two datasets with the same fingerprint are guaranteed to
        produce byte-identical ``dataset_info`` from a re-scan, so the
        cache can be safely reused.
        """
        payload = {
            "version": DATASET_CACHE_VERSION,
            "class": type(self).__name__,
            "rawpath_sha": _sha1_text("\n".join(sorted(rawpath))),
            "rawpath_count": len(rawpath),
            "val_assigns": sorted(list(val_assigns or [])),
            "inputs": self._dataset_cache_inputs_dict(),
        }
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:24]

    def _build_or_load_dataset_info(self, rawpath, val_assigns):
        """Cache-wrapped scan.

        On a cache hit the scan is skipped entirely and ``self.reason`` /
        ``self.val_raw_paths`` are repopulated from the cached payload so
        the rest of ``__init__`` is oblivious to whether we scanned or
        loaded.
        """
        fingerprint = self._dataset_cache_fingerprint(rawpath, val_assigns)
        cache_dir = self.dataset_cache_dir
        cache_path = (
            os.path.join(cache_dir, f"{fingerprint}.json") if cache_dir else None
        )

        # Snapshot what val_raw_paths held *before* the scan: the cache
        # records only the delta the scan itself appended (see below).
        pre_val_keys_len = len(self.val_raw_paths)

        if cache_path and os.path.exists(cache_path):
            payload = self._dataset_cache_try_load(cache_path, fingerprint)
            if payload is not None:
                # _build_dataset_info would have populated self.reason
                # and appended to self.val_raw_paths; do the same here.
                self.reason.update(payload.get("reason", {}))
                self.val_raw_paths.extend(payload.get("val_keys_appended", []))
                self._last_dataset_cache_payload = payload
                self._last_dataset_cache_path = cache_path
                self._last_dataset_cache_hit = True
                if self.rank == 0:
                    print(
                        f"[{type(self).__name__}] cache HIT  fp={fingerprint}  "
                        f"kept={len(payload.get('dataset_info', {}))}/"
                        f"{payload.get('rawpath_count', len(rawpath))}  "
                        f"path={cache_path}"
                    )
                return payload["dataset_info"]

        dataset_info = self._build_dataset_info(rawpath, val_assigns)
        val_keys_appended = list(self.val_raw_paths[pre_val_keys_len:])
        payload = self._dataset_cache_make_payload(
            dataset_info=dataset_info,
            fingerprint=fingerprint,
            val_keys_appended=val_keys_appended,
            rawpath=rawpath,
            val_assigns=val_assigns,
        )
        self._last_dataset_cache_payload = payload
        self._last_dataset_cache_path = cache_path
        self._last_dataset_cache_hit = False

        if cache_path and self.rank == 0:
            try:
                self._dataset_cache_write(cache_path, payload)
                print(
                    f"[{type(self).__name__}] cache MISS fp={fingerprint}  "
                    f"kept={len(dataset_info)}/{len(rawpath)}  "
                    f"saved={cache_path}"
                )
            except OSError as exc:
                # Cache failure must never break training; just warn.
                print(
                    f"[{type(self).__name__}] WARNING: cache write to "
                    f"{cache_path} failed: {exc}"
                )
        return dataset_info

    def _dataset_cache_make_payload(
        self,
        *,
        dataset_info,
        fingerprint,
        val_keys_appended,
        rawpath,
        val_assigns,
    ):
        sample_keys = sorted(dataset_info.keys())
        val_keys = sorted(
            dict.fromkeys(
                getattr(self, "_split_validation_keys", None) or self.val_raw_paths
            )
        )
        train_keys = sample_keys if not self.is_val_dataset else []
        return {
            "version": DATASET_CACHE_VERSION,
            "fingerprint": fingerprint,
            "class": type(self).__name__,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "is_val_dataset": bool(self.is_val_dataset),
            "split_role": "val" if self.is_val_dataset else "train",
            "sample_keys": sample_keys,
            "train_keys": train_keys,
            "validation_keys": val_keys,
            "split_seed": getattr(self, "_split_seed", None),
            "split_source_count": getattr(self, "_split_source_count", None),
            "split": {
                "role": "val" if self.is_val_dataset else "train",
                "sample_keys": sample_keys,
                "train_keys": train_keys,
                "validation_keys": val_keys,
                "split_seed": getattr(self, "_split_seed", None),
                "split_source_count": getattr(self, "_split_source_count", None),
            },
            "inputs": self._dataset_cache_inputs_dict(),
            "val_assigns": sorted(list(val_assigns or [])),
            "rawpath_count": len(rawpath),
            "kept_count": len(dataset_info),
            "dataset_info": dataset_info,
            "val_keys_appended": list(val_keys_appended),
            "reason": dict(self.reason),
            "kept_paths": sorted(
                info.get("latent_path", "") for info in dataset_info.values()
            ),
            "kept_clean_names": sorted(
                info.get("clean_name", "") for info in dataset_info.values()
            ),
        }

    def _dataset_cache_try_load(self, cache_path, expected_fingerprint):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != DATASET_CACHE_VERSION:
            return None
        if payload.get("fingerprint") != expected_fingerprint:
            return None
        if not isinstance(payload.get("dataset_info"), dict):
            return None
        return payload

    @staticmethod
    def _dataset_cache_write(cache_path, payload):
        parent = os.path.dirname(cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Atomic write so concurrent ranks / repeated runs never observe
        # a half-written JSON.
        tmp_path = cache_path + f".tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, cache_path)

    def save_manifest(self, path):
        """Write the dataset manifest (== cache payload) to ``path``.

        Same content as the on-disk cache so an operator can grep through
        the log directory to see *exactly* which scenes were kept /
        excluded for any given run, without having to rummage in the
        ``.cache`` directory. Returns ``path`` on success, ``None`` if
        the dataset never produced a cache payload (e.g. when
        ``dataset_cache_dir`` was unset *and* nothing was scanned).
        """
        payload = self._last_dataset_cache_payload
        if payload is None:
            return None
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = path + f".tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return path

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def get_val_raw_paths(self):
        return self.val_raw_paths

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        if self.is_val_dataset:
            return len(self.path)
        return self.steps_per_epoch

    # ------------------------------------------------------------------
    # ``max_frames_per_scene`` cap helpers. Subclasses call these from
    # their ``_build_dataset_info`` so the manifest stores
    # ALREADY-CAPPED frame / latent counts -- the runtime
    # ``__getitem__`` then doesn't need any extra cap logic.
    # ------------------------------------------------------------------

    def _cap_frame_count(self, actual: int) -> int:
        """Return ``min(actual, max_frames_per_scene)`` (or ``actual``
        when no cap is configured). Used by video-flow datasets whose
        ``info["frame_count"]`` is the canonical scene length.
        """
        if self.max_frames_per_scene is None:
            return int(actual)
        return min(int(actual), int(self.max_frames_per_scene))

    # ------------------------------------------------------------------
    # Per-sample reproducible RNG. ``__getitem__`` stashes one of these
    # on ``self`` so any helper down the call chain can pull a draw
    # without having to thread an rng arg through every layer.
    # ------------------------------------------------------------------

    def _make_sample_rng(self, index):
        """Build & stash a per-sample numpy ``Generator``.

        The seed is derived from ``(seed, "sample", dataset_pass_id,
        epoch_id, rank, index)`` so the *exact* same draw sequence is
        produced for the same tuple, and DIFFERENT for any of:
          * different ``--seed``
          * a resume (different ``dataset_pass_id``)
          * different epoch
          * different rank
          * different ``index`` within an epoch.

        The stash also avoids worker-init races: PyTorch DataLoader
        spawns workers with their own python/numpy random states, but
        we deliberately want our randomness to be *uncoupled* from
        those (so two runs with the same args / seed / pass_id but
        different DataLoader worker counts still produce the same
        sample stream).
        """
        s = _derive_seed(
            self.seed,
            "sample",
            self.dataset_pass_id,
            self._epoch_id,
            self._rank,
            int(index),
        )
        rng = np.random.default_rng(s)
        self._sample_rng = rng
        return rng

    def _get_sample_rng(self):
        """Return the rng stashed by ``_make_sample_rng`` for this
        ``__getitem__`` call. Legacy code paths that bypass
        ``__getitem__`` (e.g. ``data_process_task``) get a fresh
        non-deterministic rng so behaviour there is unchanged.
        """
        rng = getattr(self, "_sample_rng", None)
        if rng is None:
            return np.random.default_rng()
        return rng

    def _infer_camera_root_for_info(self, info):
        if not isinstance(info, dict):
            return None
        camera_root = info.get("camera_root")
        if camera_root:
            return str(camera_root)

        dirname = (
            info.get("dirname")
            or info.get("scene_dir")
            or os.path.dirname(str(info.get("extrinsic_path") or ""))
        )
        if not dirname:
            return None
        dirname_abs = os.path.abspath(str(dirname))
        for configured_root in getattr(self, "camera_params_path", []):
            root_abs = os.path.abspath(str(configured_root))
            try:
                if os.path.commonpath([root_abs, dirname_abs]) == root_abs:
                    return str(configured_root)
            except ValueError:
                continue
        return None

    def _camera_root_order(self, info, camera_root):
        if isinstance(info, dict) and info.get("camera_root_order") is not None:
            try:
                return int(info.get("camera_root_order"))
            except (TypeError, ValueError):
                pass
        if camera_root:
            root_abs = os.path.abspath(str(camera_root))
            for idx, configured_root in enumerate(
                getattr(self, "camera_params_path", [])
            ):
                if os.path.abspath(str(configured_root)) == root_abs:
                    return int(idx)
        return None

    @staticmethod
    def _parse_dataset_balance_root_probs(spec):
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(
                "dataset_balance_root_probs must be a non-empty string like "
                "'dl3dv:70,objaverse:30' when dataset_balance_mode="
                "'camera_root_manual'."
            )
        entries = []
        total = 0
        seen_patterns = set()
        for raw in spec.split(","):
            token = raw.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(
                    f"dataset_balance_root_probs entry {token!r} must be "
                    "'<root-pattern>:<percent>'."
                )
            pattern, percent_str = token.rsplit(":", 1)
            pattern = pattern.strip()
            pattern_key = pattern.lower()
            if not pattern_key:
                raise ValueError(
                    f"dataset_balance_root_probs entry {token!r} has an empty "
                    "root pattern."
                )
            if pattern_key in seen_patterns:
                raise ValueError(
                    f"dataset_balance_root_probs pattern {pattern!r} is duplicated."
                )
            seen_patterns.add(pattern_key)
            try:
                percent = int(percent_str.strip())
            except ValueError as exc:
                raise ValueError(
                    f"dataset_balance_root_probs percent in {token!r} must be an int."
                ) from exc
            if percent < 0:
                raise ValueError(
                    f"dataset_balance_root_probs percent in {token!r} must be >= 0."
                )
            entries.append((pattern, percent))
            total += percent
        if not entries:
            raise ValueError("dataset_balance_root_probs produced no entries.")
        if total != 100:
            raise ValueError(
                f"dataset_balance_root_probs percentages must sum to 100, got {total}."
            )
        if all(percent == 0 for _pattern, percent in entries):
            raise ValueError(
                "dataset_balance_root_probs must contain at least one positive "
                "percentage."
            )
        return entries

    def _manual_balance_weights(self, group_names):
        entries = self._parse_dataset_balance_root_probs(
            self.dataset_balance_root_probs
        )
        lower_roots = [str(root).lower() for root in group_names]
        root_to_percent = {}
        for pattern, percent in entries:
            pattern_key = pattern.lower()
            matches = [
                idx
                for idx, root_text in enumerate(lower_roots)
                if pattern_key in root_text
            ]
            if not matches:
                raise ValueError(
                    f"dataset_balance_root_probs pattern {pattern!r} matched no "
                    f"camera roots from {list(group_names)!r}."
                )
            if len(matches) > 1:
                matched_roots = [group_names[idx] for idx in matches]
                raise ValueError(
                    f"dataset_balance_root_probs pattern {pattern!r} matched "
                    f"multiple camera roots: {matched_roots!r}."
                )
            root = group_names[matches[0]]
            if root in root_to_percent:
                raise ValueError(
                    f"camera root {root!r} was matched by multiple "
                    "dataset_balance_root_probs entries."
                )
            root_to_percent[root] = int(percent)

        missing_roots = [root for root in group_names if root not in root_to_percent]
        if missing_roots:
            raise ValueError(
                "dataset_balance_root_probs must cover every camera root; "
                f"unmatched roots: {missing_roots!r}."
            )

        weights = [int(root_to_percent[root]) for root in group_names]
        return weights, root_to_percent

    def _build_sampling_groups(self):
        self._balanced_sampling_groups = None
        self._balanced_sampling_group_names = []
        self._balanced_sampling_group_weights = None
        if self.dataset_balance_mode == "global" or self.is_val_dataset:
            return

        buckets = {}
        orders = {}
        missing_count = 0
        for path in self.path:
            info = self.dataset_info.get(path, {})
            camera_root = self._infer_camera_root_for_info(info)
            if not camera_root:
                missing_count += 1
                continue
            buckets.setdefault(camera_root, []).append(path)
            order = self._camera_root_order(info, camera_root)
            if order is not None:
                orders[camera_root] = min(order, orders.get(camera_root, order))

        if missing_count and getattr(self, "rank", 0) == 0:
            print(
                f"[{type(self).__name__}] dataset_balance_mode="
                f"{self.dataset_balance_mode} "
                f"ignored {missing_count} samples without camera_root metadata."
            )
        if len(buckets) <= 1:
            if self.dataset_balance_mode == "camera_root_manual" and buckets:
                self._manual_balance_weights(list(buckets.keys()))
            if (
                self.dataset_balance_mode in {"camera_root", "camera_root_manual"}
                and getattr(self, "rank", 0) == 0
            ):
                print(
                    f"[{type(self).__name__}] dataset_balance_mode="
                    f"{self.dataset_balance_mode} "
                    f"found {len(buckets)} camera roots; falling back to global order."
                )
            return

        items = sorted(
            buckets.items(),
            key=lambda item: (orders.get(item[0], 10**12), str(item[0])),
        )
        self._balanced_sampling_group_names = [root for root, _paths in items]
        self._balanced_sampling_groups = [paths for _root, paths in items]
        if self.dataset_balance_mode == "camera_root_manual":
            weights, root_to_percent = self._manual_balance_weights(
                self._balanced_sampling_group_names
            )
            self._balanced_sampling_group_weights = weights
        else:
            root_to_percent = None
        if getattr(self, "rank", 0) == 0:
            counts = {
                str(root): len(paths)
                for root, paths in zip(
                    self._balanced_sampling_group_names,
                    self._balanced_sampling_groups,
                )
            }
            details = f" counts={counts}"
            if root_to_percent is not None:
                details += f" manual_percentages={root_to_percent}"
            print(
                f"[{type(self).__name__}] dataset_balance_mode="
                f"{self.dataset_balance_mode} enabled roots={len(items)}"
                f"{details}"
            )

    def _select_balanced_path(self, index):
        groups = getattr(self, "_balanced_sampling_groups", None)
        if not groups:
            return None
        virtual_index = self._world_size * int(index) + self._rank + self._epoch_offset
        weights = getattr(self, "_balanced_sampling_group_weights", None)
        if weights:
            draw_rng = random.Random(
                _derive_seed(
                    self.seed,
                    "root_balance_sample",
                    self.dataset_pass_id,
                    getattr(self, "_epoch_id", 0),
                    virtual_index,
                )
            )
            group_idx = draw_rng.choices(
                range(len(groups)),
                weights=weights,
                k=1,
            )[0]
            group_cycle_index = virtual_index
        else:
            group_idx = virtual_index % len(groups)
            group_cycle_index = virtual_index // len(groups)
        group = groups[group_idx]
        item_idx = group_cycle_index % len(group)
        return group[item_idx]

    def set_epoch(self, epoch_id, rank=None, world_size=None):
        """Refresh per-epoch sampling state.

        Called by the trainer at the top of each epoch so that:

        * ``_select_path`` rotates the path window per epoch (so the
          stream doesn't permanently start at ``self.path[0]``); the
          rank/world_size fields are still kept fresh for the balanced
          and validation branches plus the per-sample RNG.
        * ``__getitem__``'s per-sample RNG can fold the new
          ``epoch_id`` into its derivation, making the runtime
          randomness (G, mode A/B, ``random_start_latent_prob``)
          reproducible for a given ``(seed, dataset_pass_id, epoch,
          rank, step)``.

        All three knobs are optional -- callers that only know the
        epoch number can omit rank/world_size and the dataset will
        keep whatever was last set (or the ``__init__`` defaults
        rank=0, world_size=1).
        """
        self._epoch_id = int(epoch_id)
        if rank is not None:
            self._rank = int(rank)
            self.rank = int(rank)
        if world_size is not None:
            self._world_size = max(1, int(world_size))
        # Per-epoch rotation offset so the global-mode path window (and
        # the balanced branch's cycle position) moves across epochs.
        self._epoch_offset = _derive_seed(
            self.seed, "epoch_offset", self._epoch_id, self.dataset_pass_id
        ) % max(1, len(self.path))

    def set_validation_sample_offset(self, offset):
        self._validation_sample_offset = int(offset)

    def _select_path(self, index):
        if len(self.path) == 0:
            raise RuntimeError(
                "_MosaicDatasetBase is empty. Check base_path, camera_params_path, "
                f"depth_path, and filter reasons: {self.reason}"
            )
        if hasattr(self, "current_video_idx"):
            return self.path[self.current_video_idx]

        if (
            self.is_val_dataset
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and len(self.val_raw_paths) > 0
        ):
            if torch.distributed.get_rank() < len(self.val_raw_paths):
                idx_rank = (
                    self._validation_sample_offset
                    + torch.distributed.get_world_size() * index
                    + torch.distributed.get_rank()
                )
                return self.val_raw_paths[idx_rank % len(self.val_raw_paths)]
            return self.path[time.time_ns() % len(self.path)]

        if self.is_val_dataset and len(self.val_raw_paths) > 0:
            idx = self._validation_sample_offset + int(index)
            return self.val_raw_paths[idx % len(self.val_raw_paths)]

        # Distributed val with no explicit ``val_raw_paths`` (e.g. when
        # ``--val_assign_yaml`` is not used and ``--val_ratio`` falls back to
        # the shuffled split): mirror the rank-aware formula used above so
        # each rank lands on a distinct sample. Without this, all ranks pull
        # ``self.path[0]`` and validation across ranks degenerates to the
        # same scene.
        if (
            self.is_val_dataset
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            idx_rank = (
                self._validation_sample_offset
                + torch.distributed.get_world_size() * index
                + torch.distributed.get_rank()
            )
            return self.path[idx_rank % len(self.path)]

        balanced_path = self._select_balanced_path(index)
        if balanced_path is not None:
            return balanced_path

        # Train (global balance mode): rank-AGNOSTIC mapping with only the
        # per-epoch rotation. Rank decorrelation is already provided by the
        # trainer's per-rank DataLoader shuffle seed plus Accelerate's batch
        # sharding; the old ``world_size * index + rank`` stripe on top of
        # that double-sharded the stream and locked every rank into a fixed
        # ``gcd(world_size, len(path))`` congruence class of the path list
        # for a whole epoch (uneven coverage when world_size does not
        # divide len(path)). ``self.path`` is seed-shuffled at build time,
        # so a contiguous window here is an unbiased sample. Cross-rank
        # same-index collisions can map to the same scene, but the
        # per-sample rng folds the rank, so the actual training windows
        # (G, plan draws) still differ -- harmless for SGD.
        global_idx = (int(index) + self._epoch_offset) % len(self.path)
        return self.path[global_idx]

    def normalize_and_scale_intrinsics(
        self, intrinsics, H_img, W_img, temporal_mean=True
    ):
        if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
            raise ValueError(
                f"intrinsics shape must be (N, 3, 3), but got {intrinsics.shape}"
            )

        processed_intrinsics = intrinsics.astype(np.float32, copy=True)
        if getattr(self, "mosaic_intrinsics_mode", "per_frame") == "first_frame":
            processed_intrinsics = np.repeat(
                processed_intrinsics[:1],
                repeats=processed_intrinsics.shape[0],
                axis=0,
            )
        cx = processed_intrinsics[:, 0, 2]
        cy = processed_intrinsics[:, 1, 2]
        target_cx = float(W_img) * 0.5
        target_cy = float(H_img) * 0.5

        valid_mask = (np.abs(cx) > 1e-6) & (np.abs(cy) > 1e-6)
        if np.any(valid_mask):
            fx_norm = processed_intrinsics[valid_mask, 0, 0] / cx[valid_mask]
            fy_norm = processed_intrinsics[valid_mask, 1, 1] / cy[valid_mask]
            skew_norm = processed_intrinsics[valid_mask, 0, 1] / cx[valid_mask]
            processed_intrinsics[valid_mask, 0, 0] = fx_norm * target_cx
            processed_intrinsics[valid_mask, 1, 1] = fy_norm * target_cy
            processed_intrinsics[valid_mask, 0, 1] = skew_norm * target_cx
            processed_intrinsics[valid_mask, 0, 2] = target_cx
            processed_intrinsics[valid_mask, 1, 2] = target_cy

        if temporal_mean:
            mean_intrinsics = processed_intrinsics.mean(axis=0, keepdims=True)
            processed_intrinsics = np.repeat(
                mean_intrinsics, repeats=processed_intrinsics.shape[0], axis=0
            )
        return processed_intrinsics

    def _get_extrinsic_path(self, info):
        fixed_path = info.get("force_override_extrinsic_path")
        if fixed_path:
            return fixed_path
        override_path = getattr(self, "force_override_extrinsic", None)
        if not override_path or not getattr(self, "is_val_dataset", False):
            return info["extrinsic_path"]
        override_npz_paths = getattr(self, "_force_override_extrinsic_npz_paths", None)
        if override_npz_paths:
            sample_rng = getattr(self, "_sample_rng", None)
            selected_idx = (
                int(sample_rng.integers(0, len(override_npz_paths)))
                if sample_rng is not None
                else 0
            )
            selected_path = override_npz_paths[selected_idx]
            info["force_override_extrinsic_path"] = selected_path
            return selected_path
        if os.path.isdir(override_path):
            raise ValueError(
                f"force_override_extrinsic directory has no .npz files: {override_path}"
            )
        if override_path:
            return override_path
        return info["extrinsic_path"]

    def _get_frustum_handler(self):
        if self.handler is not None:
            return self.handler
        handler_cls = self._frustum_handler_cls
        if handler_cls is None:
            import os, sys

            os.getcwd()
            sys.path.append(os.path.join(os.getcwd()))
            from frustum.frustum_handler import FrustumHandler

            handler_cls = FrustumHandler
        # latent_stride=16 matches the WAN VAE this dataset feeds. Override
        # if you wire in a VAE with a different spatial downsample ratio.
        self.handler = handler_cls(np.eye(3), latent_stride=16)
        return self.handler


class _MosaicVideoSharedDataset(_MosaicDatasetBase):
    def __init__(
        self,
        *args,
        prompt="",
        prompt_path=None,
        vipe_prompt_type=1,
        vipe_prompt_segment_format=False,
        align_generating_to_prompt_segments=False,
        require_depth=False,
        allow_no_prompt=False,
        loose_prompt_match=False,
        mosaic_fuse_mode="fill_stop",
        recent_first_photo_refine=False,
        candidates_per_query_group_train=20,
        mosaic_selection_mode="projection_iou",
        mosaic_candidate_nms_mode=None,
        mosaic_candidate_nms_projection_iou_threshold=0.7,
        mosaic_candidate_nms_min_temporal_gap=0,
        mosaic_candidate_nms_pose_distance_threshold=0.0,
        mosaic_candidate_nms_pool_multiplier=2.0,
        mosaic_coverage_grid_downsample=4,
        mosaic_coverage_pool_stride=2,
        **kwargs,
    ):
        self.default_prompt = prompt
        # Lazy-built scene_hash -> (extrinsic_path, intrinsic_path) lookup.
        # DA3 strict mode populates this directly from sqlite index records.
        self._vipe_camera_index = None
        # Companion index: scene_hash -> path to the indexed depth npz.
        self._vipe_depth_zip_index = None
        # Optional prompt source. ``prompt_path`` mirrors ``camera_params_path``:
        # one or more roots holding ``<group>/<scene>/video.json`` (the DL3DV
        # caption layout). When empty, every sample falls back to
        # ``default_prompt`` for backwards compatibility.
        self._vipe_prompt_roots = _split_path_list(prompt_path) if prompt_path else []
        self._vipe_prompt_index = None
        self._vipe_prompt_path_index = None
        self._vipe_prompt_segment_cache = {}
        self._vipe_prompt_payload_cache = {}
        self.vipe_prompt_type = _normalize_vipe_prompt_type(vipe_prompt_type)
        # Retained only as an informational attribute: prompt JSON forms
        # (list segment entries vs legacy per-frame dicts) are auto-detected
        # in _load_prompt_segments, and this flag NO LONGER implicitly
        # enables align_generating_to_prompt_segments -- the two switches
        # are orthogonal now.
        self.vipe_prompt_segment_format = bool(vipe_prompt_segment_format)
        self.align_generating_to_prompt_segments = bool(
            align_generating_to_prompt_segments
        )
        # Strict DA3 training sets this to True so scenes missing depth are
        # dropped at scan time instead of reaching runtime.
        self.require_depth = bool(require_depth)
        # When True, scenes whose caption JSON is missing / unreadable /
        # has an empty ``detailed`` block are STILL kept at scan time and
        # silently fall back to ``default_prompt`` at __getitem__ time
        # (via ``_resolve_prompt_for_scene`` which already returns the
        # default for missing index entries). Useful for partially
        # captioned datasets where you'd rather train on the visual
        # signal with an empty / generic prompt than drop the scene
        # entirely. Default ``False`` preserves the legacy behaviour
        # where missing-caption scenes get reason="no_prompt" and are excluded.
        self.allow_no_prompt = bool(allow_no_prompt)
        # When True, ``_lookup_prompt_path`` falls back from a strict
        # ``scene_hash == key`` lookup to a longest-prefix match: any
        # indexed key that is a prefix of ``scene_hash`` ending on a
        # token boundary (``_``, ``-``, ``.``) is accepted, with the
        # longest such key winning. The intended use case is scenes
        # whose hash carries an extra trailing suffix that the caption
        # JSON does not, e.g. scene
        # ``8d4iysZ0pOU_0003750_0005550_f301-376`` matched against
        # prompt ``8d4iysZ0pOU_0003750_0005550.json``. Token-boundary
        # anchoring is what keeps unrelated scenes that happen to share
        # a leading character run from colliding ("foo" never matches
        # "foobar"). Default ``False`` preserves strict equality.
        self.loose_prompt_match = bool(loose_prompt_match)
        self.mosaic_fuse_mode = str(mosaic_fuse_mode or "fill_stop")
        # ``random`` is the trainer-side toggle that picks
        # ``fill_stop_zbuffer`` / ``recent_first_smooth`` per materialize
        # call. The dataset cannot reorder the choice once samples are
        # shipped, so it must ship the *superset* of inputs that either
        # mode could need. The candidate budget itself is now governed
        # by ``self.candidates_per_query_group_train`` regardless of
        # mode (see ``__getitem__``); this flag only gates the optional
        # RGB-memory-frame loading needed for ``recent_first_smooth``
        # photo refinement, so ``random`` is grouped with
        # ``recent_first_smooth`` to ensure the frames are available
        # whenever the trainer might resolve to it.
        self._mosaic_fuse_mode_is_recent_first_like = self.mosaic_fuse_mode in (
            "recent_first_smooth",
            "random",
        )
        self.recent_first_photo_refine = bool(recent_first_photo_refine)
        # Sole training-time candidate budget per query group passed to
        # `select_candidates`. Wired from the trainer's
        # `--candidates_per_query_group_train` and applied
        # unconditionally regardless of the resolved fuse mode (the
        # `_mosaic_fuse_mode_is_recent_first_like` flag now only gates
        # RGB-memory-frame loading, not the candidate count). Replaces
        # the previous per-flow hardcoded fallbacks and the old per-dataset
        # `recent_first_candidates_per_query_group`.
        self.candidates_per_query_group_train = max(
            1, int(candidates_per_query_group_train)
        )
        self.mosaic_selection_mode = str(mosaic_selection_mode or "projection_iou")
        self.mosaic_candidate_nms_mode = (
            None
            if mosaic_candidate_nms_mode in (None, "", "none", "None")
            else str(mosaic_candidate_nms_mode)
        )
        self.mosaic_candidate_nms_projection_iou_threshold = float(
            mosaic_candidate_nms_projection_iou_threshold
        )
        self.mosaic_candidate_nms_min_temporal_gap = int(
            mosaic_candidate_nms_min_temporal_gap
        )
        self.mosaic_candidate_nms_pose_distance_threshold = float(
            mosaic_candidate_nms_pose_distance_threshold
        )
        self.mosaic_candidate_nms_pool_multiplier = float(
            mosaic_candidate_nms_pool_multiplier
        )
        self.mosaic_coverage_grid_downsample = max(
            1, int(mosaic_coverage_grid_downsample)
        )
        self.mosaic_coverage_pool_stride = max(1, int(mosaic_coverage_pool_stride))
        super().__init__(*args, vipe_prompt_type=vipe_prompt_type, **kwargs)

    def _dataset_cache_inputs_dict(self):
        # Inherit base inputs and add VIPE-specific knobs so the cache /
        # manifest can distinguish runs that differ only in their prompt
        # / default-prompt / depth-requirement configuration.
        base = super()._dataset_cache_inputs_dict()
        base.update(
            {
                "prompt_path": list(self._vipe_prompt_roots),
                "default_prompt": self.default_prompt,
                "vipe_prompt_type": int(self.vipe_prompt_type),
                "vipe_prompt_segment_format": bool(self.vipe_prompt_segment_format),
                "align_generating_to_prompt_segments": bool(
                    self.align_generating_to_prompt_segments
                ),
                "require_depth": bool(self.require_depth),
                # Folded into the fingerprint because flipping this flag
                # changes which scenes survive the scan, which would
                # otherwise be invisible to the manifest cache.
                "allow_no_prompt": bool(self.allow_no_prompt),
                # Same reasoning as ``allow_no_prompt``: enabling loose
                # matching can rescue scenes that strict lookup would
                # have dropped (and bind different scenes to different
                # JSONs), so the cached manifest must distinguish runs
                # that flipped this flag.
                "loose_prompt_match": bool(self.loose_prompt_match),
                "mosaic_fuse_mode": self.mosaic_fuse_mode,
                "recent_first_photo_refine": bool(self.recent_first_photo_refine),
                "candidates_per_query_group_train": int(
                    self.candidates_per_query_group_train
                ),
                "mosaic_selection_mode": self.mosaic_selection_mode,
                "mosaic_candidate_nms_mode": self.mosaic_candidate_nms_mode,
                "mosaic_candidate_nms_projection_iou_threshold": float(
                    self.mosaic_candidate_nms_projection_iou_threshold
                ),
                "mosaic_candidate_nms_min_temporal_gap": int(
                    self.mosaic_candidate_nms_min_temporal_gap
                ),
                "mosaic_candidate_nms_pose_distance_threshold": float(
                    self.mosaic_candidate_nms_pose_distance_threshold
                ),
                "mosaic_candidate_nms_pool_multiplier": float(
                    self.mosaic_candidate_nms_pool_multiplier
                ),
                "mosaic_coverage_grid_downsample": int(
                    self.mosaic_coverage_grid_downsample
                ),
                "mosaic_coverage_pool_stride": int(
                    self.mosaic_coverage_pool_stride
                ),
            }
        )
        return base

    def _compose_vipe_prompt(self, start, dynamic):
        dynamic = str(dynamic or "").strip()
        if self.vipe_prompt_type == 1:
            return dynamic
        start = str(start or "").strip()
        return "".join(part for part in (start, dynamic) if part).strip()

    def _build_vipe_prompt_index(self):
        """BFS index of ``scene_hash -> *.json`` under each prompt root.

        Two layouts are supported:

        * **DL3DV / per-scene-dir** (legacy):
          ``<root>/<group>/<scene_hash>/video.json``. The scene's
          directory name is the key.
        * **citywalk / flat-file** (added 2026-05): each scene is a single
          JSON file directly under some grouping directory, e.g.
          ``<root>/citywalk/<scene_hash>.json``. The basename minus the
          ``.json`` extension is the key. Useful when an external pipeline
          drops one prompt JSON per scene without wrapping it in a folder.

        BFS rule: while descending with ``os.scandir``, if the current
        directory contains any ``.json`` file children we treat it as a
        scene-container leaf -- index those JSONs (per the two layouts
        above) and stop descending. Otherwise keep descending into
        subdirectories.
        """
        index = {}
        path_index = {}
        for prompt_root in self._vipe_prompt_roots:
            if not prompt_root or not os.path.isdir(prompt_root):
                continue
            stack = [prompt_root]
            while stack:
                current = stack.pop()
                try:
                    with os.scandir(current) as it:
                        entries = list(it)
                except OSError:
                    continue
                json_files = [
                    e
                    for e in entries
                    if e.name.endswith(".json") and e.is_file(follow_symlinks=False)
                ]
                if json_files:
                    dir_scene_name = os.path.basename(os.path.normpath(current))
                    for je in json_files:
                        if je.name == "video.json":
                            # Layout A: directory name == scene_hash.
                            path = os.path.join(current, je.name)
                            index.setdefault(dir_scene_name, path)
                            path_index.setdefault(dir_scene_name, []).append(path)
                        else:
                            # Layout B: file basename (sans .json) == scene_hash.
                            scene_name = je.name[: -len(".json")]
                            path = os.path.join(current, je.name)
                            index.setdefault(scene_name, path)
                            path_index.setdefault(scene_name, []).append(path)
                    # Either layout: this directory is a scene-container,
                    # don't descend further.
                    continue
                stack.extend(e.path for e in entries if e.is_dir(follow_symlinks=False))
        self._vipe_prompt_path_index = path_index
        return index

    # Token-boundary chars accepted as the cut point between an indexed
    # prompt key and the trailing suffix-only-on-the-scene-side. ``/``
    # is included for completeness even though scene hashes practically
    # never contain it. ``_PROMPT_PREFIX_BOUNDARY_CHARS`` is a class
    # attr so subclasses can extend it without touching the lookup.
    _PROMPT_PREFIX_BOUNDARY_CHARS = ("_", "-", ".", "/")

    def _lookup_prompt_path(self, scene_hash):
        """Resolve ``scene_hash`` to a prompt JSON path via the index.

        Strict mode (``loose_prompt_match=False``, the default) is a
        plain ``dict.get(scene_hash)`` against ``_vipe_prompt_index``.

        Loose mode (``loose_prompt_match=True``) additionally accepts
        any indexed key that is a *prefix* of ``scene_hash`` ending on
        a token boundary character (``_``, ``-``, ``.``, ``/``). The
        longest such key wins. Strict equality is always preferred.

        Implementation note: rather than scanning all indexed keys
        (which is O(N) per query and would dominate the scan on large
        datasets), we walk ``scene_hash`` from longest to shortest
        prefix and ask the dict directly. That's at most
        ``len(scene_hash)`` O(1) hash lookups (~50 chars in practice),
        so the total scan cost stays linear in the number of scenes.
        """
        if not self._vipe_prompt_index:
            return None
        path = self._vipe_prompt_index.get(scene_hash)
        if path is not None:
            return path
        if not self.loose_prompt_match:
            return None
        boundary = self._PROMPT_PREFIX_BOUNDARY_CHARS
        # Walk shrinking prefixes; require the cut character (the one
        # *immediately after* the candidate prefix) to be a token
        # boundary so e.g. "foo" never matches "foobar". We start at
        # ``len-1`` because ``len(scene_hash)`` would just repeat the
        # strict lookup above.
        for L in range(len(scene_hash) - 1, 0, -1):
            if scene_hash[L] not in boundary:
                continue
            cand = scene_hash[:L]
            path = self._vipe_prompt_index.get(cand)
            if path is not None:
                return path
        return None

    def _lookup_prompt_paths(self, scene_hash):
        """Resolve all prompt JSON candidates for ``scene_hash`` in root order."""
        if not self._vipe_prompt_index:
            return []
        if self._vipe_prompt_path_index is None:
            self._vipe_prompt_index = self._build_vipe_prompt_index()
        candidates = list((self._vipe_prompt_path_index or {}).get(scene_hash) or [])
        if candidates:
            return candidates
        if self.loose_prompt_match:
            boundary = self._PROMPT_PREFIX_BOUNDARY_CHARS
            for L in range(len(scene_hash) - 1, 0, -1):
                if scene_hash[L] not in boundary:
                    continue
                cand = scene_hash[:L]
                candidates = list((self._vipe_prompt_path_index or {}).get(cand) or [])
                if candidates:
                    return candidates
        path = self._lookup_prompt_path(scene_hash)
        return [] if path is None else [path]

    def _load_prompt_segments(self, json_path):
        """Parse prompt JSON into either segments or per-frame cache entries."""
        cached = self._vipe_prompt_payload_cache.get(json_path)
        if cached is not None:
            return cached
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            result = {"format": "segments", "segments": []}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        detailed = payload.get("detailed", {}) or {}
        if not detailed:
            result = {"format": "segments", "segments": []}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        if isinstance(detailed, list):
            # List-form segment entries are auto-detected; the old
            # vipe_prompt_segment_format gate is gone (it used to silently
            # blank BOTH the segments and the prompt when off).
            segments = []
            for item in detailed:
                if not isinstance(item, dict):
                    continue
                prompt = item.get("prompt")
                if not isinstance(prompt, dict):
                    continue
                try:
                    start_frame = int(item["start"])
                except (KeyError, TypeError, ValueError):
                    continue
                start = (prompt.get("start") or "").strip()
                dyn = (prompt.get("dynamic") or "").strip()
                if start or dyn:
                    segments.append((start_frame, start, dyn))
            segments.sort(key=lambda item: item[0])
            result = {
                "format": "segments",
                "segments": segments,
                "source_format": "segment_entries_v1",
            }
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = segments
            return result
        if not isinstance(detailed, dict):
            result = {"format": "segments", "segments": []}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        try:
            keys = sorted(detailed.keys(), key=lambda k: int(k))
        except (TypeError, ValueError):
            result = {"format": "segments", "segments": []}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        if payload.get("prompt_cache_format") == "frame_entries_v1":
            frames = {}
            for k in keys:
                entry = detailed[k]
                if not isinstance(entry, dict):
                    continue
                text = self._compose_vipe_prompt(
                    entry.get("start"), entry.get("dynamic")
                )
                if text:
                    frames[int(k)] = text
            result = {"format": "frame_entries_v1", "frames": frames}
            self._vipe_prompt_payload_cache[json_path] = result
            self._vipe_prompt_segment_cache[json_path] = []
            return result
        segments = []
        prev_start = prev_dyn = None
        for k in keys:
            entry = detailed[k]
            if not isinstance(entry, dict):
                continue
            start = (entry.get("start") or "").strip()
            dyn = (entry.get("dynamic") or "").strip()
            if (start, dyn) != (prev_start, prev_dyn):
                segments.append((int(k), start, dyn))
                prev_start, prev_dyn = start, dyn
        result = {"format": "segments", "segments": segments}
        self._vipe_prompt_payload_cache[json_path] = result
        self._vipe_prompt_segment_cache[json_path] = segments
        return result

    def _resolve_prompt_from_path(self, json_path, frame_min, frame_max):
        prompt_payload = self._load_prompt_segments(json_path)
        if prompt_payload.get("format") == "frame_entries_v1":
            frames = prompt_payload.get("frames") or {}
            lo = min(int(frame_min), int(frame_max))
            hi = max(int(frame_min), int(frame_max))
            prompt_hashes = [
                frames.get(frame) for frame in range(lo, hi + 1) if frames.get(frame)
            ]
            if not prompt_hashes:
                return ""
            counter = Counter(prompt_hashes)
            text = max(counter.items(), key=lambda item: (item[1], item[0]))[0]
            return text.strip()

        segments = prompt_payload.get("segments") or []
        if not segments:
            return ""

        def _segment_index_for(frame):
            # Largest segment whose start_frame <= frame. ``segments`` is
            # already sorted ascending and starts at frame 0 in practice;
            # if frame predates segment 0 we still return 0.
            idx = 0
            for i, (s, _, _) in enumerate(segments):
                if s <= frame:
                    idx = i
                else:
                    break
            return idx

        lo = min(int(frame_min), int(frame_max))
        hi = max(int(frame_min), int(frame_max))
        first_idx = _segment_index_for(lo)
        last_idx = _segment_index_for(hi)
        best_idx = first_idx
        best_overlap = -1
        for idx in range(first_idx, last_idx + 1):
            start_frame, _, _ = segments[idx]
            next_start = segments[idx + 1][0] if idx + 1 < len(segments) else hi + 1
            overlap_lo = max(lo, start_frame)
            overlap_hi = min(hi + 1, next_start)
            overlap = max(0, overlap_hi - overlap_lo)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        _start_frame, start, dyn = segments[best_idx]
        return self._compose_vipe_prompt(start, dyn)

    def _resolve_prompt_for_scene(self, scene_hash, frame_min, frame_max):
        """Resolve the prompt for a video-frame window ``[frame_min, frame_max]``.

        Rules (matching the user spec):
          * Frame keys in ``detailed`` use *video* frame indices, not latent.
          * Pick the prompt segment that covers the most frames in the query
            window and return only its ``dynamic`` text.
          * If frame indices exceed the largest documented key, the
            algorithm naturally pins them to the last segment, which is
            equivalent to padding with the final prompt.

        Falls back to ``default_prompt`` when no prompt source is configured
        or when no caption is available for this scene.
        """
        if not self._vipe_prompt_roots and not self._vipe_prompt_index:
            return self.default_prompt
        if self._vipe_prompt_index is None:
            self._vipe_prompt_index = self._build_vipe_prompt_index()
        for json_path in self._lookup_prompt_paths(scene_hash):
            text = self._resolve_prompt_from_path(json_path, frame_min, frame_max)
            if text:
                return text
        return self.default_prompt

    def _load_extrinsics(self, path):
        """Load VIPE-style camera poses and return them in W2C convention.

        VIPE writes the per-frame camera-to-world transforms under the
        `"data"` key of pose/video.npz with shape (N, 4, 4). We invert
        each matrix to obtain W2C, which is what the rest of the mosaic
        pipeline (frustum handler, PRoPE camera unit, etc.) consumes.
        """
        with np.load(path, allow_pickle=True) as f:
            array = np.asarray(f["data"])
        if array.ndim != 3 or array.shape[1:] != (4, 4):
            raise ValueError(
                f"{path} extrinsics shape must be (N, 4, 4), but got {array.shape}"
            )
        w2c = np.linalg.inv(array.astype(np.float64)).astype(np.float32)
        return w2c

    def _load_intrinsics(self, path, num_frames):
        """Load VIPE-style intrinsics (shape (N, 4): [fx, fy, cx, cy] in pixels)
        and expand to a (N, 3, 3) K-matrix stack, broadcasting if needed.
        """
        with np.load(path, allow_pickle=True) as f:
            array = np.asarray(f["data"])
        array = self._intrinsics_packed_to_matrix(array)
        if array.shape == (3, 3):
            array = array[None, ...]
        if array.ndim != 3 or array.shape[1:] != (3, 3):
            raise ValueError(
                f"intrinsics shape must be (N, 3, 3), but got {array.shape}"
            )
        if array.shape[0] == 1 and num_frames > 1:
            array = np.repeat(array, repeats=num_frames, axis=0)
        return array.astype(np.float32)

    @staticmethod
    def _intrinsics_packed_to_matrix(array):
        """Accept VIPE's packed (N, 4) [fx, fy, cx, cy] OR an already-built
        (3, 3) / (N, 3, 3) K matrix and always return a K-matrix tensor."""
        if array.ndim == 2 and array.shape[-1] == 4:
            n = array.shape[0]
            ks = np.zeros((n, 3, 3), dtype=np.float32)
            ks[:, 0, 0] = array[:, 0]  # fx
            ks[:, 1, 1] = array[:, 1]  # fy
            ks[:, 0, 2] = array[:, 2]  # cx
            ks[:, 1, 2] = array[:, 3]  # cy
            ks[:, 2, 2] = 1.0
            return ks
        return array


try:
    import OpenEXR
    import Imath
except ModuleNotFoundError:
    OpenEXR = None
    Imath = None
from pathlib import Path
import zipfile, tempfile


def _load_depth_npz_lazy(
    depth_path_or_npz,
    frame_count,
    *,
    depth_format=None,
    depth_metadata=None,
    require_depth_format=False,
):
    """Load the first ``frame_count`` per-frame depths from a depth ``.npz``.

    Supports three on-disk layouts so older caches keep working:

    1. **Per-frame keys** (preferred, producer side now writes this):
       ``frame_{i:05d}`` -- one (H, W) array per frame. We only fetch the
       first ``frame_count`` keys so the rest of the archive is never
       decompressed.
    2. ``depth_{i:05d}`` -- legacy per-frame key naming used by older
       mosaic caches.
    3. Bulk ``data`` or ``depth`` key -- single (N, H, W) array; we
       still slice ``[:frame_count]``.

    Args:
        depth_path_or_npz: path to ``video.npz``, or an already-opened
            ``NpzFile``-like context manager (used by unit tests to spy on
            which keys are actually accessed).
        frame_count: number of frames to load from the start of the clip.

    Returns:
        ``np.ndarray`` of shape ``(frame_count, H, W)``, dtype ``float32``.
    """
    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError(f"frame_count must be >= 0, got {frame_count}")

    if hasattr(depth_path_or_npz, "files"):
        _validate_depth_format_available(
            depth_path_or_npz,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
            require_depth_format=require_depth_format,
        )
        return _load_depth_npz_from_handle(
            depth_path_or_npz,
            frame_count,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
        )

    depth_format, depth_metadata = _resolve_depth_metadata_for_npz(
        depth_path_or_npz, depth_format, depth_metadata
    )
    _validate_depth_format_available(
        depth_path_or_npz,
        depth_format=depth_format,
        depth_metadata=depth_metadata,
        require_depth_format=require_depth_format,
    )
    with np.load(depth_path_or_npz, allow_pickle=True) as f:
        return _load_depth_npz_from_handle(
            f,
            frame_count,
            source=depth_path_or_npz,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
        )


def _load_depth_npz_sparse(
    depth_path_or_npz,
    frame_count,
    frame_ids,
    *,
    depth_format=None,
    depth_metadata=None,
    require_depth_format=False,
):
    """Load only selected frame depths while preserving raw frame-id indexing.

    Returned arrays have first dimension ``frame_count`` so existing frustum
    code can keep indexing ``depths[raw_frame_id]``. Unselected frames are
    zero-filled and should not be consumed by the caller.
    """
    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError(f"frame_count must be >= 0, got {frame_count}")
    frame_ids_iter = [] if frame_ids is None else frame_ids
    selected_ids = sorted(
        {int(fid) for fid in frame_ids_iter if 0 <= int(fid) < frame_count}
    )
    if not selected_ids:
        return np.zeros((0,), dtype=np.float32)

    if hasattr(depth_path_or_npz, "files"):
        _validate_depth_format_available(
            depth_path_or_npz,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
            require_depth_format=require_depth_format,
        )
        return _load_depth_npz_sparse_from_handle(
            depth_path_or_npz,
            frame_count,
            selected_ids,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
        )

    depth_format, depth_metadata = _resolve_depth_metadata_for_npz(
        depth_path_or_npz, depth_format, depth_metadata
    )
    _validate_depth_format_available(
        depth_path_or_npz,
        depth_format=depth_format,
        depth_metadata=depth_metadata,
        require_depth_format=require_depth_format,
    )
    with np.load(depth_path_or_npz, allow_pickle=True) as f:
        return _load_depth_npz_sparse_from_handle(
            f,
            frame_count,
            selected_ids,
            source=depth_path_or_npz,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
        )


def _load_depth_npz_from_handle(
    f, frame_count, *, source=None, depth_format=None, depth_metadata=None
):
    keys = list(getattr(f, "files", []))
    source_str = "" if source is None else f"{source} "

    # Per-frame keys take priority because they let us read only the
    # frames we need without decompressing the rest of the archive.
    has_frame_keys = any(k.startswith("frame_") for k in keys)
    has_depth_xxxxx_keys = any(
        k.startswith("depth_") and k[len("depth_") :].isdigit() for k in keys
    )

    if has_frame_keys:
        prefix = "frame_"
    elif has_depth_xxxxx_keys:
        prefix = "depth_"
    elif "data" in keys:
        depths = _decode_depth_array(f["data"], depth_format, depth_metadata)
        return _validate_and_slice_bulk_depth(depths, frame_count, source_str, "data")
    elif "depth" in keys:
        depths = _decode_depth_array(f["depth"], depth_format, depth_metadata)
        return _validate_and_slice_bulk_depth(depths, frame_count, source_str, "depth")
    else:
        raise KeyError(
            f"{source_str}depth npz has no recognised schema; expected one of "
            "{'frame_xxxxx', 'depth_xxxxx', 'data', 'depth'} but got "
            f"{sorted(keys)[:8]}..."
        )

    frames = []
    for i in range(frame_count):
        key = f"{prefix}{i:05d}"
        if key not in keys:
            raise KeyError(
                f"{source_str}depth npz missing key {key!r} (requested "
                f"frame_count={frame_count}, available "
                f"{prefix}* count={sum(1 for k in keys if k.startswith(prefix))})"
            )
        frames.append(_decode_depth_array(f[key], depth_format, depth_metadata))

    if not frames:
        return np.zeros((0,), dtype=np.float32)
    return np.stack(frames, axis=0)


def _load_depth_npz_sparse_from_handle(
    f,
    frame_count,
    frame_ids,
    *,
    source=None,
    depth_format=None,
    depth_metadata=None,
):
    keys = list(getattr(f, "files", []))
    source_str = "" if source is None else f"{source} "
    has_frame_keys = any(k.startswith("frame_") for k in keys)
    has_depth_xxxxx_keys = any(
        k.startswith("depth_") and k[len("depth_") :].isdigit() for k in keys
    )

    if has_frame_keys:
        prefix = "frame_"
    elif has_depth_xxxxx_keys:
        prefix = "depth_"
    elif "data" in keys:
        depths = _decode_depth_array(f["data"], depth_format, depth_metadata)
        full = _validate_and_slice_bulk_depth(depths, frame_count, source_str, "data")
        return _zero_unselected_depth_frames(full, frame_ids)
    elif "depth" in keys:
        depths = _decode_depth_array(f["depth"], depth_format, depth_metadata)
        full = _validate_and_slice_bulk_depth(depths, frame_count, source_str, "depth")
        return _zero_unselected_depth_frames(full, frame_ids)
    else:
        raise KeyError(
            f"{source_str}depth npz has no recognised schema; expected one of "
            "{'frame_xxxxx', 'depth_xxxxx', 'data', 'depth'} but got "
            f"{sorted(keys)[:8]}..."
        )

    loaded = []
    for fid in frame_ids:
        key = f"{prefix}{int(fid):05d}"
        if key not in keys:
            raise KeyError(
                f"{source_str}depth npz missing key {key!r} (requested sparse "
                f"frame ids up to frame_count={frame_count}, available "
                f"{prefix}* count={sum(1 for k in keys if k.startswith(prefix))})"
            )
        loaded.append(
            (int(fid), _decode_depth_array(f[key], depth_format, depth_metadata))
        )
    if not loaded:
        return np.zeros((0,), dtype=np.float32)
    first = loaded[0][1]
    out = np.zeros((frame_count, *first.shape), dtype=np.float32)
    for fid, frame in loaded:
        if frame.shape != first.shape:
            raise ValueError(
                f"{source_str}{prefix}{fid:05d} depth has shape {frame.shape} "
                f"but first selected frame has shape {first.shape}; all frames "
                "must share the same resolution."
            )
        out[fid] = frame
    return out


def _zero_unselected_depth_frames(depths, frame_ids):
    out = np.zeros_like(depths, dtype=np.float32)
    valid_ids = [int(fid) for fid in frame_ids if 0 <= int(fid) < int(depths.shape[0])]
    if valid_ids:
        out[valid_ids] = depths[valid_ids]
    return out


def _resolve_depth_metadata_for_npz(depth_path, depth_format=None, depth_metadata=None):
    if depth_format or depth_metadata:
        return depth_format, depth_metadata
    try:
        metadata_path = os.path.join(
            os.path.dirname(os.fspath(depth_path)), "metadata.json"
        )
    except TypeError:
        return depth_format, depth_metadata
    if not os.path.isfile(metadata_path):
        return depth_format, depth_metadata
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except (OSError, ValueError):
        return depth_format, depth_metadata
    if not isinstance(payload, dict):
        return depth_format, depth_metadata
    return _depth_encoding_name(depth_metadata=payload)[0], payload


def _validate_depth_format_available(
    depth_path, *, depth_format=None, depth_metadata=None, require_depth_format=False
):
    if not require_depth_format:
        return
    encoding, _metadata = _depth_encoding_name(depth_format, depth_metadata)
    if encoding:
        return
    raise ValueError(
        "depth_format/depth_metadata is required for this depth npz. "
        f"Got none for {depth_path!r}. Rebuild the DA3 index with current "
        "build_da3_video_index.py, keep depth/metadata.json next to video.npz, "
        "or pass --dataset_compact_mode only for old float16/float NPZ datasets."
    )


def _normalize_depth_metadata(depth_metadata):
    if isinstance(depth_metadata, dict):
        return depth_metadata
    if isinstance(depth_metadata, str) and depth_metadata:
        try:
            payload = json.loads(depth_metadata)
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _depth_encoding_name(depth_format=None, depth_metadata=None):
    metadata = _normalize_depth_metadata(depth_metadata)
    candidates = (
        depth_format,
        metadata.get("depth_format"),
        metadata.get("format"),
        metadata.get("depth_encoding"),
        metadata.get("quantization"),
    )
    for value in candidates:
        name = str(value or "").strip().lower()
        if name in {"npz_log_u10", "log_u10", "logu10"}:
            return "npz_log_u10", metadata
        if name in {"npz_log_u16", "log_u16", "logu16"}:
            return "npz_log_u16", metadata
        if name in {"npz_float", "float", "float32"}:
            return "npz_float", metadata
    return "", metadata


def _decode_depth_array(array, depth_format=None, depth_metadata=None):
    encoding, metadata = _depth_encoding_name(depth_format, depth_metadata)
    if encoding == "npz_log_u10":
        return _decode_log_uint_depth(array, metadata, default_bit_depth=10)
    if encoding == "npz_log_u16":
        return _decode_log_uint_depth(array, metadata, default_bit_depth=16)
    return np.asarray(array, dtype=np.float32)


def _decode_log_uint_depth(array, metadata, *, default_bit_depth):
    bit_depth = int(metadata.get("bit_depth") or default_bit_depth)
    max_code = float((1 << bit_depth) - 1)
    depth_min = float(
        metadata.get("depth_min", metadata.get("configured_depth_min", 0.1))
    )
    depth_max = float(
        metadata.get("depth_max", metadata.get("configured_depth_max", 1000.0))
    )
    values = np.asarray(array, dtype=np.float32)
    values = np.clip(values, 0.0, max_code) / max_code
    log_min = np.log(np.float32(depth_min))
    log_max = np.log(np.float32(depth_max))
    return np.exp(values * (log_max - log_min) + log_min).astype(np.float32)


def _validate_and_slice_bulk_depth(depths, frame_count, source_str, key_name):
    if depths.ndim != 3:
        raise ValueError(
            f"{source_str}{key_name!r} depth data must have shape (N, H, W), "
            f"got {depths.shape}"
        )
    return depths[:frame_count]


def load_depth_zip_to_array(zip_path, frame_count=None):
    """Load a VIPE-style depth zip (per-frame EXR HALF, single ``Z`` channel)
    into a ``(T, H, W)`` float32 numpy array of metric depths.

    Args:
        zip_path: path to the depth ``video.zip`` (or any iterable of EXRs
            inside a zip, named ``00000.exr``, ``00001.exr``, ...).
        frame_count: optional cap on the number of frames to load. Useful
            when the caller only needs the first N frames (e.g. for
            ``_query_frustum`` which reads up to ``noisy_frame_indices[0]``).
            ``None`` means load every frame in the archive.

    Returns:
        ``np.ndarray`` of shape ``(T, H, W)`` and dtype ``float32``; empty
        ``(0,)`` array if the archive holds no EXRs (e.g. corrupt zip).
    """
    zip_path = Path(zip_path)
    frames = []
    with zipfile.ZipFile(zip_path, "r") as z:
        # Sort numerically so 00000.exr -> 0 ordering is preserved.
        names = sorted(
            (n for n in z.namelist() if n.lower().endswith(".exr")),
            key=lambda n: int(Path(n).stem) if Path(n).stem.isdigit() else 1 << 30,
        )
        if frame_count is not None:
            names = names[: int(frame_count)]

        H = W = None
        for name in names:
            stem = Path(name).stem
            if not stem.isdigit():
                # Skip non-numeric entries (rare, but the VIPE writer
                # never emits them so this is a safety net).
                continue
            with z.open(name, "r") as f:
                buf = f.read()
            # OpenEXR has no in-memory reader in the Python binding shipped
            # with the env, so we spool to a tempfile per frame. The cost
            # is ~10% of the EXR decode for typical 960x540 HALF Z frames,
            # which is negligible next to ``query_hits_mode_new``.
            if OpenEXR is None or Imath is None:
                raise ImportError(
                    "Reading EXR depth frames requires the optional OpenEXR and "
                    "Imath Python packages. NPZ depth input does not require them."
                )
            with tempfile.NamedTemporaryFile(suffix=".exr") as tmp:
                tmp.write(buf)
                tmp.flush()
                exr = OpenEXR.InputFile(tmp.name)
                dw = exr.header()["dataWindow"]
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                if H is None:
                    H, W = height, width
                elif (H, W) != (height, width):
                    raise ValueError(
                        f"{zip_path}:{name} depth has shape ({height}, "
                        f"{width}) but the first frame was ({H}, {W}); "
                        "all frames must share the same resolution."
                    )
                half = Imath.PixelType(Imath.PixelType.HALF)
                z_bytes = exr.channel("Z", half)
                exr.close()
            depth = (
                np.frombuffer(z_bytes, dtype=np.float16)
                .astype(np.float32)
                .reshape(H, W)
            )
            frames.append(depth)

    if not frames:
        return np.zeros((0,), dtype=np.float32)
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# DA3 video-flow dataset (frame-level random start, single-frame memory latents,
# joint clean+noisy encode with mode A/B). Dataset returns RAW RGB frames;
# the trainer's ``_materialize_data`` runs all VAE encodes on the GPU rank.
# ---------------------------------------------------------------------------


import warnings as _warnings_for_video


class _MosaicVideoDatasetBase(_MosaicVideoSharedDataset):
    """Shared DA3 video-flow dataset runtime.

    DA3 strict mode drives each scene from sqlite-indexed RGB, camera, prompt,
    and depth records. The trainer encodes all clean/noisy/memory frames on GPU.
    Key runtime properties:
        * Memory uses **single-frame** latents (1 source frame == 1 latent
          slot), so frustum candidate ``frame_id`` indexes the memory buffer
          directly.
        * Random start point is at the **frame** level (not latent level).
        * Joint clean+noisy encode supports mode A (1 clean frame) and
          mode B (4 clean frames + extra causal context); per-sample mode
          is sampled from ``--vae_clean_context_mode_ratio``.
        * Optional per-frame memory-latent cache via sha256 fanout
          (``--memory_latent_cache_dir`` / ``--memory_latent_cache_version``).
    """

    DEPTH_PREFERRED_FILENAMES = ("video.zip",)
    DEPTH_FILE_SUFFIXES = (".zip",)

    def __init__(
        self,
        *args,
        height,
        width,
        vae_clean_context_blocks_max=0,
        vae_clean_context_mode_ratio="3:7",
        memory_latent_cache_dir="",
        memory_latent_cache_version="wan2.2_ti2v_5b_vae",
        memory_neighbor_window=False,
        memory_neighbor_window_max_trans_m=0.2,
        memory_neighbor_window_max_rot_deg=5.0,
        memory_vae_encode_input_frames=1,
        train_clean_history_context_blocks_min=0,
        train_anchor_warmup_prepend_frames=0,
        validation_min_anchor_frame_idx=0,
        dynamic_object_mask_filter=False,
        dynamic_object_mask_subdir="objects",
        dynamic_object_mask_filename="dynamic_masks.jsonl",
        dynamic_object_mask_temporal_dilate_radius=0,
        dynamic_object_mask_dilate_latent=0,
        dynamic_object_mask_use_bbox_fallback=False,
        **kwargs,
    ):
        # Refuse random_start_latent_prob in video flow: the parent's
        # implementation rewrites latent indices in ways that don't map
        # to a frame-level random start. Pin to 0 with a warning so users
        # who copy the latent-flow CLI don't silently get unexpected
        # behaviour.
        rs_prob = float(kwargs.get("random_start_latent_prob", 0.0) or 0.0)
        if rs_prob != 0.0:
            _warnings_for_video.warn(
                "_MosaicVideoDatasetBase does not support "
                "random_start_latent_prob; pinning to 0.0."
            )
            kwargs["random_start_latent_prob"] = 0.0

        self.height = int(height)
        self.width = int(width)
        # Wan 2.2 TI2V-5B: VAE upsampling_factor=16 AND DIT patch_size=2,
        # so the pipeline's height_division_factor is 16*2 = 32. Heights
        # that are 16-aligned but not 32-aligned (e.g. 720) end up with
        # an odd latent row count (45) which the pipeline silently
        # rounds up via ``check_resize_height_width`` (-> 736 -> 46
        # latent rows), producing a noise tensor that no longer matches
        # our mosaic_latent. Enforce 32-alignment up front so the
        # operator picks a compatible resolution rather than getting a
        # cryptic "(1,48,20,45,80) vs (1,48,20,46,80)" error from
        # WanVideoUnit_MosaicLatent.
        if self.height % 32 != 0 or self.width % 32 != 0:

            def _suggest(v):
                return ((v + 31) // 32 * 32, max(32, v // 32 * 32))

            h_up, h_down = _suggest(self.height)
            w_up, w_down = _suggest(self.width)
            raise ValueError(
                f"height/width must be divisible by 32 (VAE upsampling x DIT patch_size); "
                f"got {self.height}x{self.width}. "
                f"Closest valid options: {h_down}x{w_down}, {h_up}x{w_up} "
                f"(720p flows usually use 704x1280 or 736x1280)."
            )
        self.vae_clean_context_blocks_max = int(vae_clean_context_blocks_max)
        if self.vae_clean_context_blocks_max < 0:
            raise ValueError("vae_clean_context_blocks_max must be >= 0")
        self.vae_clean_context_mode_ratio = str(vae_clean_context_mode_ratio)
        self._mode_p_a = self._parse_mode_ratio(self.vae_clean_context_mode_ratio)
        # v2 anchor plan: GT frames prepended into the cn read so the
        # trainer's block-anchor joint encode needs no second video decode.
        self.train_anchor_warmup_prepend_frames = int(
            train_anchor_warmup_prepend_frames or 0
        )
        if (
            self.train_anchor_warmup_prepend_frames < 0
            or self.train_anchor_warmup_prepend_frames % 4 != 0
        ):
            raise ValueError(
                "train_anchor_warmup_prepend_frames must be a non-negative "
                f"multiple of 4, got {self.train_anchor_warmup_prepend_frames}."
            )

        self.memory_latent_cache_dir = str(memory_latent_cache_dir or "")
        self.memory_latent_cache_version = str(memory_latent_cache_version)
        # Trainer-side encode style for memory latents; folded into the
        # cache key when != 1 (see _memory_cache_hash).
        self.memory_vae_encode_input_frames = max(
            1, int(memory_vae_encode_input_frames or 1)
        )
        # Dynamic-object memory filtering is opt-in. When disabled, the
        # dataset does not inspect object mask files and the trainer receives
        # no source_valid_masks field, preserving the original video flow.
        self.dynamic_object_mask_filter = bool(dynamic_object_mask_filter)
        self.dynamic_object_mask_subdir = str(dynamic_object_mask_subdir or "objects")
        self.dynamic_object_mask_filename = str(
            dynamic_object_mask_filename or "dynamic_masks.jsonl"
        )
        self.dynamic_object_mask_temporal_dilate_radius = max(
            0, int(dynamic_object_mask_temporal_dilate_radius)
        )
        self.dynamic_object_mask_dilate_latent = max(
            0, int(dynamic_object_mask_dilate_latent)
        )
        self.dynamic_object_mask_use_bbox_fallback = bool(
            dynamic_object_mask_use_bbox_fallback
        )
        self._dynamic_mask_records_cache = OrderedDict()
        try:
            self._dynamic_mask_records_cache_max = int(
                os.environ.get("DYNAMIC_OBJECT_MASK_RECORD_CACHE_MAX", "64")
            )
        except ValueError:
            self._dynamic_mask_records_cache_max = 64
        self._dynamic_mask_records_cache_max = max(
            1, int(self._dynamic_mask_records_cache_max)
        )

        # Neighbour-window memory encoding (independent of the trainer's
        # ``memory_vae_encode_input_frames`` replicate path). When enabled,
        # each candidate memory frame is encoded from a real 5-frame
        # temporal window [fid-2 .. fid+2] (clamped to the clip) unless the
        # camera moves too much across those frames, in which case it falls
        # back to single-frame sparse encoding. The per-frame decision is
        # folded into the latent cache hash so the two modes never collide.
        self.memory_neighbor_window = bool(memory_neighbor_window)
        self.memory_neighbor_window_max_trans_m = float(
            memory_neighbor_window_max_trans_m
        )
        self.memory_neighbor_window_max_rot_deg = float(
            memory_neighbor_window_max_rot_deg
        )
        self.train_clean_history_context_blocks_min = max(
            0, int(train_clean_history_context_blocks_min or 0)
        )
        self.validation_min_anchor_frame_idx = max(
            0, int(validation_min_anchor_frame_idx or 0)
        )

        # DA3 strict mode populates this directly from sqlite index records.
        self._vipe_video_index = None

        super().__init__(*args, **kwargs)

        if self.vae_clean_context_blocks_max == 0 and getattr(self, "rank", 0) == 0:
            print(
                "[_MosaicVideoDatasetBase] vae_clean_context_blocks_max=0; "
                "--vae_clean_context_mode_ratio is ignored (mode A only)."
            )

    @staticmethod
    def _parse_mode_ratio(spec):
        """Parse ``"<a>:<b>"`` into ``a/(a+b)`` (probability of mode A)."""
        try:
            a_str, b_str = str(spec).split(":")
            a = float(a_str.strip())
            b = float(b_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"vae_clean_context_mode_ratio must be 'A:B', got {spec!r}"
            ) from exc
        if a < 0 or b < 0:
            raise ValueError(
                "vae_clean_context_mode_ratio weights must be non-negative, "
                f"got {spec!r}"
            )
        total = a + b
        if total <= 0:
            raise ValueError(
                f"vae_clean_context_mode_ratio sum must be > 0, got {spec!r}"
            )
        return a / total

    def _default_dynamic_mask_path_for_scene_dir(self, scene_dir):
        return os.path.join(
            scene_dir,
            self.dynamic_object_mask_subdir,
            self.dynamic_object_mask_filename,
        )

    @staticmethod
    def _resize_bool_mask_for_training(mask, *, height, width):
        """Center-crop and resize a bool mask like video frame preprocessing."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape={mask.shape}.")
        src_h, src_w = int(mask.shape[0]), int(mask.shape[1])
        height = int(height)
        width = int(width)
        crop_h = min(src_h, height)
        crop_w = min(src_w, width)
        top = max((src_h - crop_h) // 2, 0)
        left = max((src_w - crop_w) // 2, 0)
        mask = mask[top : top + crop_h, left : left + crop_w]
        if mask.shape == (height, width):
            return mask.astype(bool, copy=False)
        import cv2

        if mask.shape[0] >= height and mask.shape[1] >= width:
            resized = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            return resized > 0.0
        resized = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized > 0

    def _load_dynamic_mask_records(self, mask_path):
        """Load dynamic mask jsonl lazily per worker/process, indexed by frame."""
        if not mask_path or not os.path.exists(mask_path):
            return {}
        cached = self._dynamic_mask_records_cache.get(mask_path)
        if cached is not None:
            self._dynamic_mask_records_cache.move_to_end(mask_path)
            return cached
        records_by_frame = {}
        with open(mask_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                try:
                    frame_idx = int(row.get("frame_idx"))
                except (TypeError, ValueError):
                    continue
                records_by_frame.setdefault(frame_idx, []).append(row)
        self._dynamic_mask_records_cache[mask_path] = records_by_frame
        self._dynamic_mask_records_cache.move_to_end(mask_path)
        while (
            len(self._dynamic_mask_records_cache) > self._dynamic_mask_records_cache_max
        ):
            self._dynamic_mask_records_cache.popitem(last=False)
        return records_by_frame

    def _build_source_valid_masks_from_dynamic_masks(
        self,
        mask_path,
        *,
        memory_frame_count,
        video_height,
        video_width,
        H_lat,
        W_lat,
        frame_ids=None,
    ):
        """Return optional ``(memory_frame_count,H_lat,W_lat)`` valid masks.

        ``True`` means a source latent cell may be fused; ``False`` excludes
        dynamic-object pixels from mosaic memory. ``None`` means the selected
        source frames are all valid.
        """
        memory_frame_count = int(memory_frame_count)
        if memory_frame_count <= 0:
            return None
        records_by_frame = self._load_dynamic_mask_records(mask_path)
        if not records_by_frame:
            return None
        frame_filter = None
        if frame_ids is not None:
            frame_filter = {
                int(fid) for fid in frame_ids if 0 <= int(fid) < memory_frame_count
            }
            radius_for_decode = max(
                0, int(self.dynamic_object_mask_temporal_dilate_radius)
            )
            if radius_for_decode > 0 and frame_filter:
                expanded = set(frame_filter)
                for fid in list(frame_filter):
                    for off in range(1, radius_for_decode + 1):
                        if fid - off >= 0:
                            expanded.add(fid - off)
                        if fid + off < memory_frame_count:
                            expanded.add(fid + off)
                frame_filter = expanded

        if frame_filter is None:
            iterable_frame_ids = sorted(
                fid for fid in records_by_frame if 0 <= int(fid) < memory_frame_count
            )
        else:
            iterable_frame_ids = sorted(
                fid for fid in frame_filter if int(fid) in records_by_frame
            )
        if not iterable_frame_ids:
            return None

        postprocess = _dynamic_mask_postprocess_module()
        dynamic_latent = np.zeros(
            (memory_frame_count, int(H_lat), int(W_lat)), dtype=bool
        )
        radius = max(0, int(self.dynamic_object_mask_temporal_dilate_radius))
        for frame_idx in iterable_frame_ids:
            frame_idx = int(frame_idx)
            if frame_idx < 0 or frame_idx >= memory_frame_count:
                continue
            for row in records_by_frame.get(frame_idx, ()):
                image_mask = postprocess.mask_from_record(
                    row,
                    height=int(video_height),
                    width=int(video_width),
                    use_bbox_fallback=bool(self.dynamic_object_mask_use_bbox_fallback),
                )
                if image_mask is None:
                    continue
                image_mask = self._resize_bool_mask_for_training(
                    image_mask, height=self.height, width=self.width
                )
                latent_mask = postprocess.image_mask_to_latent_mask(
                    image_mask, int(H_lat), int(W_lat)
                )
                dynamic_latent[frame_idx] |= latent_mask

        if not dynamic_latent.any():
            return None
        if radius > 0:
            dynamic_latent = postprocess.apply_temporal_dilation_to_frame_masks(
                dynamic_latent, radius=radius
            )
        if int(self.dynamic_object_mask_dilate_latent) > 0:
            dynamic_latent = np.stack(
                [
                    postprocess.dilate_bool_mask(
                        mask, int(self.dynamic_object_mask_dilate_latent)
                    )
                    for mask in dynamic_latent
                ],
                axis=0,
            )
        if not dynamic_latent.any():
            return None
        return np.logical_not(dynamic_latent).astype(bool, copy=False)

    def _dataset_cache_inputs_dict(self):
        base = super()._dataset_cache_inputs_dict()
        base.update(
            {
                "dataset_kind": "video",
                "height": self.height,
                "width": self.width,
                "vae_clean_context_blocks_max": self.vae_clean_context_blocks_max,
                "vae_clean_context_mode_ratio": self.vae_clean_context_mode_ratio,
                "train_clean_history_context_blocks_min": (
                    self.train_clean_history_context_blocks_min
                ),
                "memory_latent_cache_version": self.memory_latent_cache_version,
            }
        )
        if self.dynamic_object_mask_filter:
            base.update(
                {
                    "dynamic_object_mask_filter": True,
                    "dynamic_object_mask_subdir": self.dynamic_object_mask_subdir,
                    "dynamic_object_mask_filename": self.dynamic_object_mask_filename,
                    "dynamic_object_mask_temporal_dilate_radius": int(
                        self.dynamic_object_mask_temporal_dilate_radius
                    ),
                    "dynamic_object_mask_dilate_latent": int(
                        self.dynamic_object_mask_dilate_latent
                    ),
                    "dynamic_object_mask_use_bbox_fallback": bool(
                        self.dynamic_object_mask_use_bbox_fallback
                    ),
                }
            )
        return base

    # ------------------------------------------------------------------
    # Memory latent cache helpers (used by both dataset and trainer).
    # ------------------------------------------------------------------

    def _memory_cache_hash(self, clean_name, frame_id, mode_tag=None):
        """Stable per-frame hash; mirrored in trainer for atomic dump.

        ``mode_tag`` is appended only when set (neighbour-window mode) so
        legacy single-frame caches keep their exact pre-existing hashes
        while neighbour-window latents land in a disjoint namespace. The
        same back-compat rule applies to ``memory_vae_encode_input_frames``:
        the trainer's replicate-N encode (keep-last-latent) is numerically
        different from the single-frame encode, so non-default values get
        their own namespace while the default 1 keeps existing hashes.
        """
        text = (
            f"{clean_name}|{int(frame_id)}|"
            f"{self.height}|{self.width}|{self.memory_latent_cache_version}"
        )
        encode_input_frames = int(
            getattr(self, "memory_vae_encode_input_frames", 1) or 1
        )
        if encode_input_frames != 1:
            text = f"{text}|mvif{encode_input_frames}"
        if mode_tag:
            text = f"{text}|{mode_tag}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _memory_cache_path(self, clean_name, frame_id, mode_tag=None):
        """Two-level fanout dir layout to keep any single subdir <100k files."""
        if not self.memory_latent_cache_dir:
            return None
        h = self._memory_cache_hash(clean_name, frame_id, mode_tag=mode_tag)
        return os.path.join(
            self.memory_latent_cache_dir,
            h[:2],
            h[2:4],
            f"{clean_name}_{int(frame_id):06d}_{h}.pt",
        )

    # ------------------------------------------------------------------
    # Neighbour-window memory encoding helpers.
    # ------------------------------------------------------------------

    def _neighbor_window_motion(self, extrinsics, frame_id, frame_count):
        """Compute the camera-motion magnitude for one candidate frame's
        5-frame window.

        Returns ``(window_ids, max_trans_m, max_rot_deg)`` where:
          * ``window_ids`` are the (clamped) source-video frame indices
            ``[fid-2 .. fid+2]`` in temporal order,
          * ``max_trans_m`` is the max pairwise camera-center translation
            distance across the window, in meters,
          * ``max_rot_deg`` is the max pairwise geodesic rotation angle
            across the window, in degrees.

        The math mirrors ``frustum_handler._compute_group_motion_scores``
        (camera center ``C = -R^T t``, rotation ``arccos((tr(R_a R_b^T)-1)/2)``)
        and is vectorised over the <=5 frames, so it costs only a handful of
        small numpy ops.
        """
        fid = int(frame_id)
        hi = int(frame_count) - 1
        window_ids = [min(max(fid + off, 0), hi) for off in (-2, -1, 0, 1, 2)]

        win = np.asarray(extrinsics, dtype=np.float64)[window_ids]  # (W, 4, 4)
        R = win[:, :3, :3]  # (W, 3, 3)
        t = win[:, :3, 3]  # (W, 3)
        # Camera centers in world coords: C = -R^T t.
        centers = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)  # (W, 3)

        diff = centers[:, None, :] - centers[None, :, :]  # (W, W, 3)
        max_trans = float(np.sqrt((diff * diff).sum(-1)).max())

        # Pairwise geodesic rotation: R_rel = R_a @ R_b^T; angle from trace.
        rel = np.einsum("aij,bkj->abik", R, R)  # R_a @ R_b^T -> (W, W, 3, 3)
        trace = np.einsum("abii->ab", rel)
        cos_a = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        max_rot_deg = float(np.degrees(np.arccos(cos_a)).max())

        return window_ids, max_trans, max_rot_deg

    def _neighbor_window_plan(self, extrinsics, frame_id, frame_count):
        """Decide the VAE input window for a single candidate memory frame.

        Returns ``(window_ids, mode_tag)`` where ``window_ids`` is the list
        of (clamped) source-video frame indices to feed the VAE in temporal
        order and ``mode_tag`` namespaces the latent cache entry:

          * low motion  -> 5-frame window ``[fid-2 .. fid+2]``, tag ``"nbr_w5"``
          * large motion -> single frame ``[fid]``, tag ``"nbr_s1"``

        Either the max pairwise translation (meters) or the max pairwise
        rotation (degrees) exceeding its threshold triggers the single-frame
        fallback. See ``_neighbor_window_motion`` for the motion math.
        """
        fid = int(frame_id)
        window_ids, max_trans, max_rot_deg = self._neighbor_window_motion(
            extrinsics, fid, frame_count
        )

        # ``thresh_text`` keeps the cache hash sensitive to threshold changes
        # so re-tuning the judge cannot silently reuse stale decisions.
        thresh_text = (
            f"t{self.memory_neighbor_window_max_trans_m:g}"
            f"r{self.memory_neighbor_window_max_rot_deg:g}"
        )

        large_motion = (
            max_trans > self.memory_neighbor_window_max_trans_m
            or max_rot_deg > self.memory_neighbor_window_max_rot_deg
        )
        if large_motion:
            return [fid], f"nbr_s1_{thresh_text}"
        return window_ids, f"nbr_w5_{thresh_text}"

    # ------------------------------------------------------------------
    # __getitem__: produce raw frames + select_candidates plan.
    # ------------------------------------------------------------------

    def _sample_blocks(self):
        """Sample mode A/B per the configured ratio. Returns ``blocks``.

        ``blocks=0`` means mode A (1 clean frame, no extra context).
        ``blocks>=1`` means mode B (4 clean frames + ``blocks-1`` extra).

        Uses the per-sample rng stashed by ``__getitem__`` so the same
        ``(seed, dataset_pass_id, epoch, rank, index)`` tuple always
        picks the same mode + block count.
        """
        if self.vae_clean_context_blocks_max == 0:
            return 0
        rng = self._get_sample_rng()
        if rng.random() < self._mode_p_a or self.is_val_dataset:
            return 0
        # ``rng.integers`` upper bound is EXCLUSIVE.
        return int(rng.integers(1, self.vae_clean_context_blocks_max + 1))

    def _read_video_frames(
        self,
        video_path,
        frame_indices,
        *,
        return_metadata=False,
        normalize=True,
    ):
        """Read ``frame_indices`` from ``video_path`` (decord) and return
        a ``(C, T, H, W)`` tensor.

        ``frame_indices`` must be a sorted list of unique ints. Replicates
        ``batch_encode_videos.VideoLatentDataset.{load_video,center_crop_resize}``
        so memory and training-time encodes see byte-identical inputs.
        ``normalize=False`` keeps uint8 pixels for GPU-side normalization in the
        trainer materialization path.
        """
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_path), ctx=cpu())
        frames = reader.get_batch(list(frame_indices)).asnumpy()
        metadata = {
            "source_height": int(frames.shape[1]),
            "source_width": int(frames.shape[2]),
        }
        out = _video_frames_uint8_to_normalized_tensor(
            frames,
            height=self.height,
            width=self.width,
            normalize=normalize,
        )
        if return_metadata:
            return out, metadata
        return out

    def _prompt_aligned_generating_start_candidates(
        self,
        info,
        *,
        lo,
        hi,
        align_train_clean_history=False,
    ):
        if not getattr(self, "align_generating_to_prompt_segments", False):
            return []
        prompt_path = info.get("prompt_path")
        if not prompt_path:
            return []
        prompt_payload = self._load_prompt_segments(prompt_path)
        if prompt_payload.get("format") != "segments":
            return []
        candidates = []
        segments = list(prompt_payload.get("segments") or [])
        for index, segment in enumerate(segments):
            try:
                segment_start = int(segment[0])
            except (TypeError, ValueError, IndexError):
                continue
            # The first segment starts at frame 0, but G needs at least one
            # frame of clean context, so start the first prompt at start+1.
            candidate = segment_start + 1 if index == 0 else segment_start
            if align_train_clean_history and candidate % self.VAE_T_SCALING != 0:
                candidate = (
                    (candidate + self.VAE_T_SCALING - 1)
                    // self.VAE_T_SCALING
                    * self.VAE_T_SCALING
                )
            next_start = (
                int(segments[index + 1][0]) if index + 1 < len(segments) else int(hi)
            )
            if candidate < int(lo) or candidate >= int(hi) or candidate >= next_start:
                continue
            candidates.append(int(candidate))
        return sorted(dict.fromkeys(candidates))

    def _sample_generating_first_frame_idx(
        self,
        info,
        *,
        lo,
        hi,
        align_train_clean_history=False,
    ):
        candidates = self._prompt_aligned_generating_start_candidates(
            info,
            lo=lo,
            hi=hi,
            align_train_clean_history=align_train_clean_history,
        )
        if candidates:
            if self.is_val_dataset:
                return int(candidates[0])
            idx = int(self._get_sample_rng().integers(0, len(candidates)))
            return int(candidates[idx])

        if self.is_val_dataset:
            return int(lo)
        if align_train_clean_history:
            choice_count = ((hi - 1 - lo) // self.VAE_T_SCALING) + 1
            return int(
                lo
                + self.VAE_T_SCALING
                * int(self._get_sample_rng().integers(0, choice_count))
            )
        return int(self._get_sample_rng().integers(lo, hi))

    def _max_clean_context_for_generating_start(self):
        if getattr(self, "is_val_dataset", False) and (
            self.train_clean_history_context_blocks_min <= 0
        ):
            return 1
        max_clean_context_blocks = max(
            int(self.vae_clean_context_blocks_max),
            int(self.train_clean_history_context_blocks_min),
        )
        max_clean_context = 1 + 4 * max_clean_context_blocks
        if self.train_clean_history_context_blocks_min > 0:
            max_clean_context = (
                (max_clean_context + self.VAE_T_SCALING - 1)
                // self.VAE_T_SCALING
                * self.VAE_T_SCALING
            )
        return int(max_clean_context)

    def __getitem__(self, index):
        # Per-sample reproducible rng (see base class
        # ``_make_sample_rng`` for derivation rules); MUST be stashed
        # before ``_sample_blocks`` / G sampling so they pull from it.
        self._make_sample_rng(index)
        path = self._select_path(index)
        info = dict(self.dataset_info[path])

        frame_count = int(info["frame_count"])
        num_generating_frames = 4 * int(self.latent_window_size)
        align_train_clean_history = self.train_clean_history_context_blocks_min > 0
        max_clean_context = self._max_clean_context_for_generating_start()

        # Pick a per-sample mode and the matching number of clean blocks.
        blocks = self._sample_blocks()
        K = 4 * blocks
        clean_context = 1 + K  # frames to the left of G that VAE sees

        # Frame-level random start. The lower bound guarantees enough
        # context for the largest legacy clean-context draw or the largest
        # train clean-history draw, so later trainer-side sampling cannot
        # silently truncate old history.
        lo = max_clean_context
        if getattr(self, "is_val_dataset", False):
            lo = max(lo, int(self.validation_min_anchor_frame_idx) + 1)
        hi = frame_count - num_generating_frames
        if hi <= lo:
            # ``too_short`` filter should have dropped this scene; fail loud.
            raise RuntimeError(
                f"_MosaicVideoDatasetBase: scene {info['clean_name']} too short "
                f"(frame_count={frame_count}, num_generating_frames="
                f"{num_generating_frames}, max_clean_context="
                f"{max_clean_context})"
            )
        # Same testing escape hatch the parent class honours: a
        # ``SFgenerating_first_idx`` attribute pins G so unit/smoke
        # tests can assert deterministic candidate sets across calls.
        if hasattr(self, "SFgenerating_first_idx"):
            G = int(self.SFgenerating_first_idx)
            if not (lo <= G <= hi - 1):
                raise ValueError(
                    f"SFgenerating_first_idx={G} out of valid range "
                    f"[{lo}, {hi - 1}] for scene {info['clean_name']}."
                )
            if align_train_clean_history and G % self.VAE_T_SCALING != 0:
                raise ValueError(
                    f"SFgenerating_first_idx={G} must be divisible by "
                    f"{self.VAE_T_SCALING} when train clean-history is enabled."
                )
        elif self.is_val_dataset:
            G = self._sample_generating_first_frame_idx(
                info,
                lo=lo,
                hi=hi,
                align_train_clean_history=align_train_clean_history,
            )
        else:
            # ``random.randint(lo, hi - 1)`` is inclusive on both ends;
            # ``rng.integers(lo, hi)`` matches that exactly because its
            # upper bound is EXCLUSIVE.
            G = self._sample_generating_first_frame_idx(
                info,
                lo=lo,
                hi=hi,
                align_train_clean_history=align_train_clean_history,
            )

        latent_window_size = int(self.latent_window_size)
        H_lat = self.height // self.VAE_HW_SCALING
        W_lat = self.width // self.VAE_HW_SCALING

        # ----- Camera / depth (re-implemented from read_camera_params,
        # but keyed off frame indices instead of clean_latents.shape) -----
        extrinsics = self._load_extrinsics(self._get_extrinsic_path(info))
        intrinsics = self._load_intrinsics(
            info["intrinsic_path"], num_frames=extrinsics.shape[0]
        )
        # The dataset's mosaic_intrinsics_mode controls whether K remains
        # per-frame, collapses to a per-episode mean, or reuses frame 0.
        scaled_intrinsics = self.normalize_and_scale_intrinsics(
            intrinsics,
            H_img=self.height,
            W_img=self.width,
            temporal_mean=self._intrinsics_temporal_mean_enabled(),
        )

        if blocks == 0:
            clean_frame_indices = [G - 1] * self.VAE_T_SCALING
        else:
            clean_frame_indices = [G - 4, G - 3, G - 2, G - 1]
        noisy_frame_indices = list(range(G, G + num_generating_frames))

        clean_frame_extrinsics = extrinsics[clean_frame_indices]
        clean_frame_intrinsics = scaled_intrinsics[clean_frame_indices]
        noisy_frame_extrinsics = extrinsics[noisy_frame_indices]
        noisy_frame_intrinsics = scaled_intrinsics[noisy_frame_indices]

        # Memory frustum needs depths and w2c up to frame G (exclusive).
        # DA3 strict mode requires a depth file during dataset indexing.
        depth_path = info.get("depth_path")
        if not depth_path:
            raise RuntimeError(
                f"DA3 sample {info.get('clean_name')} has no depth_path; "
                "dataset indexing should have filtered it out."
            )
        sparse_depth_after_selection = self.mosaic_selection_mode in {
            "pose_nearest",
            "pose_pool_temporal_earliest",
        }
        if sparse_depth_after_selection:
            depths = np.zeros((0,), dtype=np.float32)
        else:
            depths = self._load_depths_for_memory(
                depth_path,
                frame_count=G,
                depth_format=info.get("depth_format"),
                depth_metadata=info.get("depth_metadata"),
            )
        w2c_memory = extrinsics[:G]

        scene_hash = (
            info.get("clean_name")
            or os.path.basename(info.get("video_path", "")).split(".")[0]
        )
        prompt_text = self._resolve_prompt_for_scene(
            scene_hash,
            frame_min=noisy_frame_indices[0],
            frame_max=noisy_frame_indices[-1],
        )

        # ----- Phase 1: select frustum candidates (CPU-only) -----
        # Per-frame K split:
        #   * memory_K: K for frames [0..G) that the frustum can pick from.
        #   * query_K : K for the noisy/query frames (== noisy_frame_intrinsics).
        # Inference paths (validation rollout) keep using the legacy single
        # K shape, so the handler accepts both modes.
        memory_K = scaled_intrinsics[:G].astype(np.float32, copy=False)
        query_K_arr = scaled_intrinsics[noisy_frame_indices].astype(
            np.float32, copy=False
        )
        handler = self._get_frustum_handler()
        depth_downsample_pool = {}
        try:
            # See note in _query_frustum: distance_threshold is now in METERS.
            cand_frame_ids = handler.select_candidates(
                query_extrinsics=noisy_frame_extrinsics,
                depths=depths,
                w2c=w2c_memory,
                memory_K=memory_K,
                query_K=query_K_arr,
                H_lat=H_lat,
                W_lat=W_lat,
                total_latents=G,
                latent_merge_4frames=False,
                candidates_per_query_group=self.candidates_per_query_group_train,
                angle_threshold=None,
                distance_threshold=None,
                temporal_threshold=None,
                geometry_rep_mode="second_frame",
                selection_mode=self.mosaic_selection_mode,
                query_reference_frame=self.mosaic_query_reference_frame,
                candidate_nms_mode=self.mosaic_candidate_nms_mode,
                candidate_nms_projection_iou_threshold=(
                    self.mosaic_candidate_nms_projection_iou_threshold
                ),
                candidate_nms_min_temporal_gap=(
                    self.mosaic_candidate_nms_min_temporal_gap
                ),
                candidate_nms_pose_distance_threshold=(
                    self.mosaic_candidate_nms_pose_distance_threshold
                ),
                candidate_nms_pool_multiplier=(
                    self.mosaic_candidate_nms_pool_multiplier
                ),
                coverage_grid_downsample=self.mosaic_coverage_grid_downsample,
                coverage_pool_stride=self.mosaic_coverage_pool_stride,
                depth_downsample_pool=depth_downsample_pool,
            )
        finally:
            # The pool can hold one latent-sized depth map per candidate
            # group. Keep it scoped to this sample only.
            depth_downsample_pool.clear()

        # ----- Build memory_set + cache lookup -----
        memory_set = set()
        for group in cand_frame_ids:
            for fid in group:
                fid = int(fid)
                if 0 <= fid < G:
                    memory_set.add(fid)

        # ----- Neighbour-window plan (optional) -----
        # When enabled, decide per candidate memory frame whether to encode a
        # real 5-frame temporal window or fall back to a single frame, based
        # on the camera motion across that window. The resolved window ids +
        # a cache mode tag are reused below for the cache lookup, the MP4
        # read, and the per-slot tensor assembly.
        neighbor_window_plan = {}
        if self.memory_neighbor_window:
            for fid in memory_set:
                neighbor_window_plan[fid] = self._neighbor_window_plan(
                    extrinsics, fid, frame_count
                )

        if sparse_depth_after_selection:
            depths = self._load_depths_for_memory(
                depth_path,
                frame_count=G,
                frame_ids=sorted(memory_set),
                depth_format=info.get("depth_format"),
                depth_metadata=info.get("depth_metadata"),
            )

        cached_paths = {}
        miss_set = set()
        target_paths = {}
        for fid in memory_set:
            mode_tag = (
                neighbor_window_plan[fid][1] if self.memory_neighbor_window else None
            )
            cache_path = self._memory_cache_path(
                info["clean_name"], fid, mode_tag=mode_tag
            )
            if cache_path is not None:
                target_paths[fid] = cache_path
                if os.path.exists(cache_path):
                    cached_paths[fid] = cache_path
                    continue
            miss_set.add(fid)
        memory_cache_debug = {
            "cache_enabled": bool(self.memory_latent_cache_dir),
            "cache_dir": self.memory_latent_cache_dir,
            "cache_version": self.memory_latent_cache_version,
            "selected_frame_ids": sorted(int(fid) for fid in memory_set),
            "hit_frame_ids": sorted(int(fid) for fid in cached_paths),
            "miss_frame_ids": sorted(int(fid) for fid in miss_set),
            "target_paths": {int(fid): str(path) for fid, path in target_paths.items()},
            "cached_paths": {int(fid): str(path) for fid, path in cached_paths.items()},
        }

        # ----- MP4 read: union(miss memory frames, cn_frames range) -----
        # The v2 block-anchor plan needs ``warmup_prepend`` GT frames
        # immediately before the cn window ([G-1-P .. G-2]) so the trainer
        # can jointly encode anchor warm-up + anchor + targets WITHOUT a
        # second video decode. They ride along in this same read; the
        # trainer slices them off again for single-anchor steps.
        warmup_prepend = int(
            getattr(self, "train_anchor_warmup_prepend_frames", 0) or 0
        )
        if getattr(self, "is_val_dataset", False):
            warmup_prepend = 0
        cn_start = G - 1 - K
        if warmup_prepend > cn_start:
            raise RuntimeError(
                f"_MosaicVideoDatasetBase: G={G} leaves no room for the "
                f"{warmup_prepend}-frame anchor warm-up prefix (scene "
                f"{info.get('clean_name')}); the G lower bound reservation "
                "(train_clean_history_context_blocks_min) is inconsistent."
            )
        cn_end = G + num_generating_frames
        cn_frame_indices = list(range(cn_start - warmup_prepend, cn_end))
        needs_recent_first_rgb = (
            self._mosaic_fuse_mode_is_recent_first_like
            and self.recent_first_photo_refine
        )
        # Miss memory frames contribute either a single frame id (legacy /
        # large-motion fallback) or the full 5-frame neighbour window.
        if self.memory_neighbor_window:
            memory_read_ids = set()
            for fid in miss_set:
                memory_read_ids.update(neighbor_window_plan[fid][0])
        else:
            memory_read_ids = set(miss_set)
        all_indices_to_read = sorted(
            memory_read_ids
            | set(cn_frame_indices)
            | (memory_set if needs_recent_first_rgb else set())
        )
        memory_cache_debug["read_frame_ids"] = [int(fid) for fid in all_indices_to_read]
        if all_indices_to_read:
            all_frames, video_meta = self._read_video_frames(
                info["video_path"],
                all_indices_to_read,
                return_metadata=True,
                normalize=False,
            )
            # all_frames: (C, T_all, H, W), T_all == len(all_indices_to_read)
            idx_to_pos = {fid: pos for pos, fid in enumerate(all_indices_to_read)}
        else:
            all_frames = None
            video_meta = {}
            idx_to_pos = {}

        source_valid_masks = None
        if self.dynamic_object_mask_filter:
            video_height = int(video_meta.get("source_height", 0) or 0)
            video_width = int(video_meta.get("source_width", 0) or 0)
            if video_height <= 0 or video_width <= 0:
                raise RuntimeError(
                    "dynamic_object_mask_filter requires source video dimensions "
                    f"from the RGB read for scene {info.get('clean_name')}; "
                    f"video_path={info.get('video_path')!r}"
                )
            source_valid_masks = self._build_source_valid_masks_from_dynamic_masks(
                info.get("dynamic_mask_path"),
                memory_frame_count=int(G),
                video_height=video_height,
                video_width=video_width,
                H_lat=int(H_lat),
                W_lat=int(W_lat),
                frame_ids=memory_set,
            )

        memory_frames_by_id = {}
        for fid in sorted(miss_set):
            if self.memory_neighbor_window:
                window_ids = neighbor_window_plan[fid][0]
                positions = [idx_to_pos[w] for w in window_ids]
                # (C, W, H, W) in temporal order (W == 5 for low motion, 1 for
                # the large-motion fallback). The trainer groups slots by this
                # time dim, encodes each group, and keeps the last latent step.
                memory_frames_by_id[fid] = all_frames[:, positions].contiguous()
            else:
                pos = idx_to_pos[fid]
                # Slice as (C, 1, H, W) so trainer can stack and call
                # pipe.vae.encode without an extra unsqueeze.
                memory_frames_by_id[fid] = all_frames[:, pos : pos + 1].contiguous()

        recent_first_memory_frames_by_id = {}
        if needs_recent_first_rgb:
            for fid in sorted(memory_set):
                pos = idx_to_pos[fid]
                recent_first_memory_frames_by_id[fid] = all_frames[
                    :, pos : pos + 1
                ].contiguous()

        # cn_frames must come back in frame-order (matches encode expectations).
        cn_positions = [idx_to_pos[fid] for fid in cn_frame_indices]
        cn_frames = all_frames[:, cn_positions].contiguous()
        assert (
            cn_frames.shape[1]
            == warmup_prepend + 1 + K + num_generating_frames
        )

        # ----- Index tensors for the DIT (positional, not frame-level) -----
        clean_latent_indices_start = torch.tensor([0], dtype=torch.long)
        clean_latent_indices = torch.tensor([1], dtype=torch.long)
        noisy_latent_indices = torch.arange(
            2, 2 + latent_window_size, dtype=torch.long
        )[None, :]

        per_call_info = {
            **info,
            "generating_first_frame_idx": int(G),
            "blocks": int(blocks),
            "cn_drop_count": int(blocks),
        }

        # Per-sample neighbour-window tally: how many of THIS sample's unique
        # candidate memory frames resolved to single-frame (S1) vs the 5-frame
        # window (W5). Counts every selected candidate (cache hit or miss),
        # not per query-group occurrence. Surfaced for the preview JSON.
        if self.memory_neighbor_window:
            num_s1 = sum(
                1 for fid in memory_set if "nbr_s1" in neighbor_window_plan[fid][1]
            )
            neighbor_window_stats = {
                "enabled": True,
                "max_trans_m": float(self.memory_neighbor_window_max_trans_m),
                "max_rot_deg": float(self.memory_neighbor_window_max_rot_deg),
                "num_candidates": int(len(memory_set)),
                "num_s1": int(num_s1),
                "num_w5": int(len(memory_set) - num_s1),
            }
        else:
            neighbor_window_stats = {"enabled": False}

        data = {
            "clean_latent_indices_start": clean_latent_indices_start,
            "clean_latent_indices": clean_latent_indices,
            "noisy_latent_indices": noisy_latent_indices,
            "clean_latent_indices_prope_intrinsic": torch.from_numpy(
                clean_frame_intrinsics
            ).float(),
            "clean_latent_indices_prope_extrinsic": torch.from_numpy(
                clean_frame_extrinsics
            ).float(),
            "noisy_latent_indices_prope_intrinsic": torch.from_numpy(
                noisy_frame_intrinsics
            ).float(),
            "noisy_latent_indices_prope_extrinsic": torch.from_numpy(
                noisy_frame_extrinsics
            ).float(),
            "memory_frames_by_id": memory_frames_by_id,
            "recent_first_memory_frames_by_id": recent_first_memory_frames_by_id,
            "cached_memory_paths_by_id": cached_paths,
            "memory_cache_hash_by_id": target_paths,
            "memory_cache_debug": memory_cache_debug,
            "memory_neighbor_window_stats": neighbor_window_stats,
            "frame_pixel_range": "uint8_or_255",
            "cn_frames": cn_frames,
            "cn_drop_count": int(blocks),
            "cn_anchor_warmup_frames": int(warmup_prepend),
            "fused_query_inputs": {
                "query_extrinsics": np.asarray(
                    noisy_frame_extrinsics, dtype=np.float32
                ),
                "candidate_frame_ids": cand_frame_ids,
                "w2c": w2c_memory,
                "depths": depths,
                # Legacy ``K`` (single 3x3) kept for backward-compat with
                # any consumer that still reads it; trainer prefers the
                # per-frame ``memory_K`` / ``query_K`` below.
                "K": np.asarray(query_K_arr[0], dtype=np.float32),
                "memory_K": memory_K,
                "query_K": query_K_arr,
                "query_frame_indices": noisy_frame_indices,
                "H_lat": int(H_lat),
                "W_lat": int(W_lat),
                "memory_total_frames": int(G),
                "query_reference_frame": int(self.mosaic_query_reference_frame),
            },
            "needs_vae_materialization": True,
            "prompt": prompt_text,
            "info": per_call_info,
            "lookup": {},
            "is_starting": False,
            "updatetqdm": ".",
        }
        if source_valid_masks is not None:
            data["fused_query_inputs"]["source_valid_masks"] = source_valid_masks
        return data

    def read_camera_params(self, data, info, lookup=None):
        """Per-section PRoPE camera recompute used by ``run_mosaic_validation``.

        The validation loop drifts ``clean_latent_indices`` /
        ``noisy_latent_indices`` by ``latent_stride`` per section and
        calls this method on each section to refresh PRoPE camera fields
        and ``prompt`` for the new window. Mirrors the parent's API but
        works in frame-level coordinates instead of latent-level
        ``_latent_indices_to_frame_indices`` math (which assumed 4-frame
        latent compression -- not applicable to single-frame memory).

        Frame indices that walk past the source video's last frame get
        clamped (the predicted-only sections genuinely have no GT
        camera; clamping to the final pose is the same fallback the
        DA3 validation uses).
        """
        extrinsics = self._load_extrinsics(self._get_extrinsic_path(info))
        intrinsics = self._load_intrinsics(
            info["intrinsic_path"], num_frames=extrinsics.shape[0]
        )
        # Validation rollout follows validation_mosaic_intrinsics_mode as
        # passed through this dataset's mosaic_intrinsics_mode.
        scaled_intrinsics = self.normalize_and_scale_intrinsics(
            intrinsics,
            H_img=self.height,
            W_img=self.width,
            temporal_mean=self._intrinsics_temporal_mean_enabled(),
        )

        # Original G this sample was drawn at; the validation drifts by
        # full-window strides so re-derive the drifted G from the
        # currently-running clean_latent_indices.
        g_orig = int(info.get("generating_first_frame_idx", 0))
        # Section-0 clean_latent_indices == [1]; per-section drift of
        # ``latent_stride`` (== latent_window_size) maps to a frame drift
        # of ``stride * 4`` since 1 latent step covers 4 noisy frames.
        clean_pos = int(data["clean_latent_indices"][0].item())
        drift_frames = (clean_pos - 1) * 4
        G = g_orig + drift_frames

        blocks = int(info.get("blocks", 0))
        is_starting = bool(data.get("is_starting", True))
        if blocks == 0 and is_starting:
            # Section-0 bootstrap anchor is a genuine single-frame
            # (image-mode) latent: replicate its one pose, matching the
            # trainer's single-anchor convention.
            clean_frame_indices = [G - 1] * self.VAE_T_SCALING
        else:
            # Sections >= 1 feed back the last GENERATED latent -- a dense
            # block covering frames [G-4..G-1]. PRoPE applies one camera
            # per sub-frame slot, so give it the 4 DISTINCT poses exactly
            # like the trainer's block-anchor branch
            # (anchor_cam_source='block_distinct_frames'). Replicating the
            # last pose here would assert zero ego-motion through a moving
            # anchor -- a (dense content, static camera) pairing no
            # training branch ever produces.
            clean_frame_indices = [G - 4, G - 3, G - 2, G - 1]
        num_generating_frames = 4 * int(self.latent_window_size)
        noisy_frame_indices = list(range(G, G + num_generating_frames))

        num_camera_frames = int(extrinsics.shape[0])
        max_idx = num_camera_frames - 1
        if bool(data.get("camera_pingpong", False)):
            # Ping-pong past the trajectory end instead of freezing on the
            # last pose (see _pingpong_reflect_index / --validation_camera_pingpong).
            clean_frame_indices = [
                _pingpong_reflect_index(fid, num_camera_frames)
                for fid in clean_frame_indices
            ]
            noisy_frame_indices = [
                _pingpong_reflect_index(fid, num_camera_frames)
                for fid in noisy_frame_indices
            ]
        else:
            clean_frame_indices = [
                min(max(0, fid), max_idx) for fid in clean_frame_indices
            ]
            noisy_frame_indices = [
                min(max(0, fid), max_idx) for fid in noisy_frame_indices
            ]

        scene_hash = (
            info.get("clean_name")
            or os.path.splitext(os.path.basename(info.get("video_path", "")))[0]
        )
        # min/max (not [0]/[-1]) so a ping-pong backward window, whose indices
        # descend, still resolves a sane frame range.
        data["prompt"] = self._resolve_prompt_for_scene(
            scene_hash,
            frame_min=min(noisy_frame_indices),
            frame_max=max(noisy_frame_indices),
        )
        # ``w2c`` / ``depths`` are not consumed by validation's downstream
        # flow (it runs its own DA3 + register_source_sequence), but a
        # stable shape keeps debug prints meaningful.
        data["w2c"] = extrinsics[: noisy_frame_indices[0]]
        data["depths"] = np.zeros((0,), dtype=np.float32)

        data["clean_latent_indices_prope_intrinsic"] = torch.from_numpy(
            scaled_intrinsics[clean_frame_indices]
        ).float()
        data["clean_latent_indices_prope_extrinsic"] = torch.from_numpy(
            extrinsics[clean_frame_indices]
        ).float()
        data["noisy_latent_indices_prope_intrinsic"] = torch.from_numpy(
            scaled_intrinsics[noisy_frame_indices]
        ).float()
        data["noisy_latent_indices_prope_extrinsic"] = torch.from_numpy(
            extrinsics[noisy_frame_indices]
        ).float()
        return data


class DA3MosaicVideoDataset(_MosaicVideoDatasetBase):
    """Video-flow dataset for DA3-batched outputs.

    Layout uses sqlite-indexed ``rgb/``, ``pose/`` and ``intrinsics/`` records.
    Depth lives at ``depth/video.npz``; the loader auto-
    detects the on-disk schema (see ``_load_depth_npz_lazy``):

    - **Preferred (current producer)**: per-frame keys ``frame_{i:05d}``,
      one (H, W) array each. Only the first ``frame_count`` keys are
      read so the rest of the archive stays compressed on disk.
    - Legacy bulk ``data`` (or ``depth``) key with a single (N, H, W)
      array; still supported for older caches.
    """

    DEPTH_PREFERRED_FILENAMES = ("video.npz",)
    DEPTH_FILE_SUFFIXES = (".npz",)

    def __init__(self, *args, dataset_compact_mode=False, **kwargs):
        self.dataset_compact_mode = bool(dataset_compact_mode)
        if not kwargs.get("dataset_index_path"):
            raise ValueError(
                "DA3MosaicVideoDataset strict mode requires --dataset_index_path. "
                "DA3 video data is read only from the sqlite index."
            )
        # Parent classes still carry legacy constructor requirements for
        # directory scans. In strict DA3 mode these roots are inert: sqlite
        # records provide every video/camera/depth/prompt path.
        args = list(args)
        for idx in range(min(3, len(args))):
            args[idx] = ()
        args = tuple(args)
        for legacy_key in ("base_path", "camera_params_path", "depth_path"):
            kwargs[legacy_key] = ()
        super().__init__(*args, dataset_compact_mode=dataset_compact_mode, **kwargs)

    def _da3_index_debug(self, message):
        if getattr(self, "rank", 0) == 0:
            print(f"[DA3MosaicVideoDataset:index] {message}")

    def _da3_index_meta(self):
        if not getattr(self, "dataset_index_path", None):
            raise ValueError(
                "DA3MosaicVideoDataset strict mode requires --dataset_index_path."
            )
        cached = getattr(self, "_da3_video_index_meta", None)
        if cached is not None:
            return cached
        try:
            from .da3_video_index import (
                DA3_VIDEO_INDEX_SCHEMA_VERSION,
                load_da3_video_index_meta,
            )

            meta = load_da3_video_index_meta(self.dataset_index_path)
            if str(meta.get("dataset_kind")) != "da3_video":
                raise ValueError("dataset_kind is not da3_video")
            schema_version = int(meta.get("schema_version", -1))
            supported_schema_versions = {2, DA3_VIDEO_INDEX_SCHEMA_VERSION}
            if schema_version not in supported_schema_versions and not bool(
                getattr(self, "dataset_compact_mode", False)
            ):
                raise ValueError(
                    f"schema_version={schema_version} "
                    f"expected one of {sorted(supported_schema_versions)}"
                )
            if schema_version != DA3_VIDEO_INDEX_SCHEMA_VERSION:
                self._da3_index_debug(
                    f"accepting legacy schema_version={schema_version}; "
                    f"current={DA3_VIDEO_INDEX_SCHEMA_VERSION}"
                )
        except Exception as exc:
            raise ValueError(
                "DA3MosaicVideoDataset strict mode cannot use "
                f"dataset_index_path={self.dataset_index_path!r}: {exc}"
            ) from exc
        self._da3_video_index_meta = meta
        self._da3_index_debug(
            f"loaded meta path={self.dataset_index_path} "
            f"records={meta.get('record_count')} kept={meta.get('kept_count')} "
            f"rejected={meta.get('rejected_count')} "
            f"content_sha1={meta.get('content_sha1')}"
        )
        return meta

    def _using_da3_video_index(self):
        self._da3_index_meta()
        return True

    def _load_da3_video_index_records(self):
        cached = getattr(self, "_da3_video_index_records", None)
        if cached is not None:
            return cached
        self._using_da3_video_index()
        from .da3_video_index import load_da3_video_index_records

        records = load_da3_video_index_records(
            self.dataset_index_path,
            only_complete=False,
            min_frame_count=None,
        )
        meta = self._da3_index_meta() or {}
        self._da3_index_debug(
            f"loaded records total={len(records)} "
            f"min_frame_count={self.min_frame_count} "
            f"sqlite_kept={meta.get('kept_count')}"
        )
        self._da3_video_index_records = records
        self._da3_video_index_records_by_path = {
            record["synthetic_path"]: record for record in records
        }
        self._vipe_camera_index = {
            record["clean_name"]: (
                record["extrinsic_path"],
                record["intrinsic_path"],
            )
            for record in records
        }
        self._vipe_video_index = {
            record["clean_name"]: record["video_path"] for record in records
        }
        self._vipe_depth_zip_index = {
            record["clean_name"]: record["depth_path"]
            for record in records
            if record.get("depth_path")
        }
        self._vipe_prompt_index = {
            record["clean_name"]: record["prompt_path"]
            for record in records
            if record.get("prompt_path")
        }
        self._vipe_prompt_path_index = {
            record["clean_name"]: [record["prompt_path"]]
            for record in records
            if record.get("prompt_path")
        }
        return records

    def _collect_latent_paths(self, filter_yaml):
        self._using_da3_video_index()
        records = self._load_da3_video_index_records()
        rawpath = sorted(record["synthetic_path"] for record in records)
        before_filter = len(rawpath)
        if filter_yaml is None:
            self._da3_index_debug(f"collected raw paths from index: {before_filter}")
            return rawpath

        import yaml

        with open(filter_yaml, "r") as f:
            filter_criteria = yaml.safe_load(f) or []
        filtered = [
            path
            for path in rawpath
            if not any(filter_token in path for filter_token in filter_criteria)
        ]
        self._da3_index_debug(
            f"filter_yaml applied: before={before_filter} after={len(filtered)} "
            f"path={filter_yaml}"
        )
        return filtered

    def _apply_include_yaml(self, rawpath, include_yaml):
        before = len(rawpath)
        result = super()._apply_include_yaml(rawpath, include_yaml)
        if self._using_da3_video_index() and include_yaml:
            self._da3_index_debug(
                f"include_yaml applied: before={before} after={len(result)} "
                f"path={include_yaml}"
            )
        return result

    def _split_train_val_paths(self, rawpath, val_assign_yaml=None, val_ratio=0.05):
        self._using_da3_video_index()
        if val_assign_yaml is not None:
            return super()._split_train_val_paths(
                rawpath,
                val_assign_yaml=val_assign_yaml,
                val_ratio=val_ratio,
            )
        val_ratio = 0.0 if val_ratio is None else float(val_ratio)
        val_ratio = max(0.0, min(1.0, val_ratio))
        split_seed = _derive_seed(self.seed, "split")
        val_paths = set()
        for path in sorted(rawpath):
            key = self._raw_path_to_dataset_key(path)
            token = f"{self.seed}|split|{key}"
            score = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:16], 16)
            score = score / float(16**16)
            if score < val_ratio:
                val_paths.add(path)
        train_paths = [path for path in sorted(rawpath) if path not in val_paths]
        matched = [path for path in sorted(rawpath) if path in val_paths]
        val_keys = [self._raw_path_to_dataset_key(path) for path in matched]
        train_keys = [self._raw_path_to_dataset_key(path) for path in train_paths]
        if set(train_keys) & set(val_keys):
            raise ValueError("DA3 index train/val split produced overlapping keys.")
        self._split_seed = int(split_seed)
        self._split_source_count = len(rawpath)
        self._split_train_keys = train_keys
        self._split_validation_keys = val_keys
        self.val_raw_paths.extend(val_keys)
        self._da3_index_debug(
            f"stable split complete: source={len(rawpath)} "
            f"train={len(train_paths)} val={len(matched)} "
            f"val_ratio={val_ratio} seed={self.seed}"
        )
        return (matched if self.is_val_dataset else train_paths), []

    def _da3_dynamic_min_frame_count(self):
        num_generating_frames = 4 * int(self.latent_window_size)
        max_clean_context = self._max_clean_context_for_generating_start()
        dynamic_min = num_generating_frames + max_clean_context + 1
        if self.min_frame_count is None:
            return int(dynamic_min)
        return max(int(self.min_frame_count), int(dynamic_min))

    def _dataset_cache_inputs_dict(self):
        base = super()._dataset_cache_inputs_dict()
        base["dataset_kind"] = "da3_video"
        base["dataset_compact_mode"] = bool(
            getattr(self, "dataset_compact_mode", False)
        )
        # Folding the RoPE table bound into the cache key forces a rescan
        # of manifests built before the ``rope_time_overflow`` filter
        # existed, so the filter applies to cached datasets too.
        base["rope_freqs_table_end"] = ROPE_FREQS_TABLE_END
        meta = self._da3_index_meta()
        if meta:
            base["dataset_index"] = {
                "path": self.dataset_index_path,
                "schema_version": meta.get("schema_version"),
                "content_sha1": meta.get("content_sha1"),
            }
        return base

    def _build_dataset_info(self, rawpath, val_assigns):
        self._using_da3_video_index()
        records_by_path = getattr(self, "_da3_video_index_records_by_path", None)
        if records_by_path is None:
            self._load_da3_video_index_records()
            records_by_path = self._da3_video_index_records_by_path

        dataset_info = {}
        min_frames = self._da3_dynamic_min_frame_count()
        self._da3_index_debug(
            f"building dataset_info from index: raw_paths={len(rawpath)} "
            f"dynamic_min_frame_count={min_frames} is_val={self.is_val_dataset}"
        )
        for synthetic_path in tqdm(
            rawpath,
            desc="Build DA3 dataset_info from index",
            disable=getattr(self, "rank", 0) != 0,
        ):
            record = records_by_path.get(synthetic_path)
            if record is None:
                continue
            clean_name = record["clean_name"]
            scan_error = record.get("scan_error")
            if scan_error:
                self.reason[clean_name] = scan_error
                continue
            if not record.get("prompt_path") and not self.allow_no_prompt:
                self.reason[clean_name] = "no_prompt"
                continue
            if not record.get("video_path"):
                self.reason[clean_name] = "no_video"
                continue
            if not record.get("extrinsic_path") or not record.get("intrinsic_path"):
                self.reason[clean_name] = "no_camera_params"
                continue
            if record.get("frame_count") is None:
                self.reason[clean_name] = "missing_frame_count"
                continue
            try:
                frame_count = self._cap_frame_count(int(record["frame_count"]))
            except (TypeError, ValueError) as exc:
                self.reason[clean_name] = f"invalid_frame_count:{exc}"
                continue
            if frame_count < min_frames:
                self.reason[clean_name] = "too_short"
                continue
            overflow_time = _rope_time_overflow(frame_count, self.latent_window_size)
            if overflow_time is not None:
                # Stable label (no per-scene suffix) so the reason counter
                # aggregates; the scan summary below reports the count.
                self.reason[clean_name] = "rope_time_overflow"
                continue
            if self.require_depth and not record.get("depth_path"):
                self.reason[clean_name] = "no_depth"
                continue
            depth_format = record.get("depth_format")
            depth_metadata = record.get("depth_metadata_json")
            if (
                getattr(self, "dataset_compact_mode", False)
                and record.get("depth_path")
                and not (depth_format or depth_metadata)
            ):
                depth_format = "npz_float"
            else:
                depth_format, depth_metadata = _resolve_depth_metadata_for_npz(
                    record.get("depth_path"),
                    depth_format,
                    depth_metadata,
                )
            if record.get("depth_path") and not (depth_format or depth_metadata):
                if getattr(self, "dataset_compact_mode", False):
                    depth_format = "npz_float"
                else:
                    self.reason[clean_name] = "missing_depth_format"
                    continue
            info = {
                "frame_count": int(frame_count),
                "clean_name": clean_name,
                "camera_root": record.get("camera_root"),
                "camera_root_order": record.get("camera_root_order"),
                "video_path": record["video_path"],
                "depth_path": record["depth_path"],
                "depth_format": depth_format,
                "depth_metadata_path": record.get("depth_metadata_path"),
                "depth_metadata": depth_metadata,
                "dirname": record["scene_dir"],
                "extrinsic_path": record["extrinsic_path"],
                "intrinsic_path": record["intrinsic_path"],
                "prompt_path": record.get("prompt_path"),
            }
            if self.dynamic_object_mask_filter:
                info["dynamic_mask_path"] = (
                    self._default_dynamic_mask_path_for_scene_dir(record["scene_dir"])
                )
            key = f"{info['dirname']}/{clean_name}"
            dataset_info[key] = info
            if record.get("prompt_path"):
                reason = "success" if record.get("depth_path") else "success_no_depth"
            else:
                reason = (
                    "success_no_prompt"
                    if record.get("depth_path")
                    else "success_no_prompt_no_depth"
                )
            self.reason[clean_name] = reason

        if self.is_val_dataset:
            info_keys = set(dataset_info)
            self.val_raw_paths = [
                key for key in dict.fromkeys(self.val_raw_paths) if key in info_keys
            ]
            self._split_validation_keys = sorted(info_keys)
            self._split_train_keys = []
        reason_counts = Counter(self.reason.values())
        self._da3_index_debug(
            f"dataset_info ready: kept={len(dataset_info)} "
            f"reasons={dict(reason_counts)}"
        )
        return dataset_info

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
        if require_depth_format is None:
            require_depth_format = not bool(
                getattr(self, "dataset_compact_mode", False)
            )
        if frame_ids is not None:
            return _load_depth_npz_sparse(
                depth_path,
                frame_count=frame_count,
                frame_ids=frame_ids,
                depth_format=depth_format,
                depth_metadata=depth_metadata,
                require_depth_format=require_depth_format,
            )
        return _load_depth_npz_lazy(
            depth_path,
            frame_count=frame_count,
            depth_format=depth_format,
            depth_metadata=depth_metadata,
            require_depth_format=require_depth_format,
        )
