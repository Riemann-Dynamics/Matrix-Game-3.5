import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.wanvideo.pipeline.mosaic import *
from examples.wanvideo.pipeline.mosaic import (
    config as _compat_config,
    datasets as _compat_datasets,
    history as _compat_history,
    prompting as _compat_prompting,
    qwen as _compat_qwen,
    timing as _compat_timing,
    runner as _compat_runner,
    video_io as _compat_video_io,
)
from examples.wanvideo.pipeline.mosaic.config import (
    _MosaicFuseModeAction,
    _apply_intrinsics_mode_aliases,
    _apply_legacy_intrinsics_mode_config,
    _apply_recent_first_zbuffer_alias,
    _intrinsics_mode_arg,
    _normalize_mosaic_fuse_mode,
    _normalize_mosaic_intrinsics_mode,
)
from examples.wanvideo.pipeline.mosaic.datasets import (
    _append_prompt_cache_dir_to_vipe_prompt_path,
    _resolve_dataset_cls_and_extra_kwargs,
)
from examples.wanvideo.pipeline.mosaic.history import *
from examples.wanvideo.pipeline.mosaic.inference import (
    run_mosaic_segment_inference,
)
from examples.wanvideo.pipeline.mosaic.prompting import *
from examples.wanvideo.pipeline.mosaic.qwen import *
from examples.wanvideo.pipeline.mosaic.timing import *
from examples.wanvideo.pipeline.mosaic.runner import (
    _maybe_resume_from_log_dir,
)
from examples.wanvideo.pipeline.mosaic.video_io import *

for _compat_module in (
    _compat_config,
    _compat_datasets,
    _compat_history,
    _compat_prompting,
    _compat_qwen,
    _compat_timing,
    _compat_runner,
    _compat_video_io,
):
    globals().update(
        {
            _name: _value
            for _name, _value in vars(_compat_module).items()
            if _name.startswith("_") and not _name.startswith("__")
        }
    )

del (
    _compat_config,
    _compat_datasets,
    _compat_history,
    _compat_module,
    _compat_prompting,
    _compat_qwen,
    _compat_timing,
    _compat_runner,
    _compat_video_io,
)


if __name__ == "__main__":
    main()
