from .common import *
from .history import (
    _parse_train_clean_history_ac_ratio,
    _validate_block_anchor_frames,
    _validate_train_context_selection,
)
from .qwen import VALIDATION_QWEN_MODES

RECENT_FIRST_ZBUFFER_MODE = "recent_first_zbuffer"
MOSAIC_INTRINSICS_MODES = ("per_frame", "episode_mean", "first_frame")
TRAIN_PROMPT_DROPOUT_PREVIEW_TEXT = "<no_prompt>"


def _normalize_mosaic_fuse_mode(mode):
    mode = str(mode or "")
    if mode == RECENT_FIRST_ZBUFFER_MODE:
        return "recent_first_smooth"
    return mode


def _is_recent_first_fuse_mode(mode):
    return _normalize_mosaic_fuse_mode(mode) == "recent_first_smooth"


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
    if mode not in MOSAIC_INTRINSICS_MODES:
        raise ValueError(
            "mosaic intrinsics mode must be one of "
            f"{MOSAIC_INTRINSICS_MODES}, got {mode!r}."
        )
    return mode


def _intrinsics_mode_arg(args, name, default):
    return _normalize_mosaic_intrinsics_mode(getattr(args, name, default))


def _apply_legacy_intrinsics_mode_config(config):
    config = dict(config)
    if (
        "mosaic_intrinsics_mode" in config
        and "train_mosaic_intrinsics_mode" not in config
    ):
        config["train_mosaic_intrinsics_mode"] = config["mosaic_intrinsics_mode"]
    if config.pop("mosaic_intrinsics_first_frame_only", False):
        config.setdefault("train_mosaic_intrinsics_mode", "first_frame")
        config.setdefault("validation_mosaic_intrinsics_mode", "first_frame")
    if config.pop("validation_frustum_fixed_first_frame_intrinsics", False):
        config["validation_mosaic_intrinsics_mode"] = "first_frame"
    config.pop("mosaic_intrinsics_mode", None)
    return config


def _apply_intrinsics_mode_aliases(args):
    if not hasattr(args, "train_mosaic_intrinsics_mode"):
        setattr(args, "train_mosaic_intrinsics_mode", "per_frame")
    if not hasattr(args, "validation_mosaic_intrinsics_mode"):
        setattr(args, "validation_mosaic_intrinsics_mode", "episode_mean")

    if hasattr(args, "mosaic_intrinsics_mode"):
        setattr(
            args,
            "train_mosaic_intrinsics_mode",
            getattr(args, "mosaic_intrinsics_mode"),
        )
    if bool(getattr(args, "mosaic_intrinsics_first_frame_only", False)):
        setattr(args, "train_mosaic_intrinsics_mode", "first_frame")
        setattr(args, "validation_mosaic_intrinsics_mode", "first_frame")
    if bool(getattr(args, "validation_frustum_fixed_first_frame_intrinsics", False)):
        setattr(args, "validation_mosaic_intrinsics_mode", "first_frame")

    setattr(
        args,
        "train_mosaic_intrinsics_mode",
        _normalize_mosaic_intrinsics_mode(args.train_mosaic_intrinsics_mode),
    )
    setattr(
        args,
        "validation_mosaic_intrinsics_mode",
        _normalize_mosaic_intrinsics_mode(args.validation_mosaic_intrinsics_mode),
    )
    return args


class _MosaicFuseModeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        if values == RECENT_FIRST_ZBUFFER_MODE:
            setattr(namespace, "recent_first_zbuffer", True)


def _apply_recent_first_zbuffer_alias(args):
    if (
        getattr(args, "mosaic_fuse_mode", None) == RECENT_FIRST_ZBUFFER_MODE
        or getattr(args, "mosaic_fuse_mode_train", None) == RECENT_FIRST_ZBUFFER_MODE
    ):
        setattr(args, "recent_first_zbuffer", True)
    return args


def _apply_context_alias(args):
    """Map the canonical ``validation_context_*`` front-door names onto the old
    internal ``validation_clean_latent_pool_history*`` attrs the code reads.

    Only fires when a new name was explicitly given (SUPPRESS default), so old
    args.yaml using the legacy names keeps working unchanged. The count alias
    converts vocabulary: the new ``validation_context_count`` EXCLUDES the
    anchor (like train_context_count_max), the old count INCLUDED it, so
    old = new + 1.
    """
    if hasattr(args, "validation_context_count"):
        setattr(
            args,
            "validation_clean_latent_history_count",
            int(getattr(args, "validation_context_count")) + 1,
        )
    if hasattr(args, "validation_context_pool"):
        setattr(
            args,
            "validation_clean_latent_pool_history",
            bool(getattr(args, "validation_context_pool")),
        )
    if hasattr(args, "validation_context_selection"):
        setattr(
            args,
            "validation_clean_latent_pool_history_selection_mode",
            str(getattr(args, "validation_context_selection")),
        )
    if hasattr(args, "validation_context_pool_bias"):
        setattr(
            args,
            "validation_clean_latent_pool_history_bias",
            int(getattr(args, "validation_context_pool_bias")),
        )
    if hasattr(args, "validation_context_pool_rope_original_indices"):
        setattr(
            args,
            "validation_clean_latent_pool_history_rope_original_indices",
            bool(getattr(args, "validation_context_pool_rope_original_indices")),
        )
    return args


def _parse_trans_scale_arg(value):
    """argparse type for --trans_scale: a float (legacy linear divide) or
    one of the compression modes 'log'/'logd4'/'tanh'. Also normalises values
    that arrive as YAML defaults via ``parser.set_defaults``."""
    if isinstance(value, (int, float)):
        return float(value)
    mode = str(value).strip().lower()
    if mode in ("log", "logd4", "tanh"):
        return mode
    try:
        return float(mode)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"trans_scale must be a number, 'log', 'logd4' or 'tanh', "
            f"got {value!r}."
        )


def wan_mosaic_parser():
    parser = argparse.ArgumentParser(
        description="Mosaic training script for Wan TI2V-5B."
    )
    parser.add_argument(
        "--omegaconf_config",
        "--config",
        dest="omegaconf_config",
        type=str,
        default=None,
        help=(
            "Optional OmegaConf YAML config. Config keys use argparse dest "
            "names and override parser defaults; explicitly provided CLI "
            "arguments override the config."
        ),
    )
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.set_defaults(
        dataset_base_path="",
        height=704,
        width=1280,
        num_frames=None,
        num_epochs=10,
        output_path="./logs",
    )
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="Path to tokenizer."
    )
    parser.add_argument(
        "--audio_processor_path",
        type=str,
        default=None,
        help="Path to the audio processor.",
    )
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help=(
            "Global gradient-norm clip threshold; <=0 disables clipping. "
            "Under DeepSpeed it is wired into the engine's "
            "`gradient_clipping` config (unless the ds_config already sets "
            "a value); under plain DDP it is applied via "
            "accelerator.clip_grad_norm_ at each update boundary."
        ),
    )
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument(
        "--trained_dit",
        type=str,
        default=None,
        help="Path to a trained DIT checkpoint used before training/validation.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help=(
            "Enable timing/debug instrumentation from train_mosaic_timing.py. "
            "Default off for normal training."
        ),
    )
    parser.add_argument(
        "--timing_enabled",
        dest="timing_enabled",
        default=True,
        action="store_true",
        help="Allow lightweight timing reports when --debug is set.",
    )
    parser.add_argument(
        "--timing_disabled",
        dest="timing_enabled",
        action="store_false",
        help="Disable timing reports even when --debug is set.",
    )
    parser.add_argument(
        "--timing_interval",
        type=int,
        default=20,
        help="Debug timing: print rolling timing summaries every N train steps.",
    )
    parser.add_argument(
        "--timing_warmup_steps",
        type=int,
        default=1,
        help="Debug timing: skip writing the first N train step timing records.",
    )
    parser.add_argument(
        "--timing_dirname",
        type=str,
        default="timing",
        help="Debug timing: subdirectory under the run log dir for timing reports.",
    )
    parser.add_argument(
        "--log_dir_name",
        "--run_name",
        dest="log_dir_name",
        type=str,
        default=None,
        help=(
            "Override the auto timestamped run directory name under "
            "--output_path. Use this for multi-node launches where the "
            "wrapper script provides one shared name to every node."
        ),
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="auto",
        choices=["auto", "online", "offline", "disabled"],
        help=(
            "Metrics backend mode. 'auto' (default) probes wandb status at "
            "startup: remote logging only when an API key is configured AND "
            "api.wandb.ai is reachable, otherwise the run degrades to a "
            "LOCAL wandb offline run stored under <log_dir>/wandb (upload "
            "later with `wandb sync`). 'disabled' skips wandb entirely and "
            "logs to tensorboard under <log_dir>/tensorboard instead. "
            "Logged scalars: train/loss, train/lr, train/epoch, "
            "train/nonfinite plus the per-plan buckets "
            "train/loss_mode_{single,block,single_ctx,block_ctx}, "
            "train/history_count and train/anchor_is_single. (Pre-v2 runs "
            "logged train/loss_mode_{C,A,B,off} instead.)"
        ),
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="diffsynth-mosaic",
        help="wandb project name for the metrics logger.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help=(
            "wandb run name; defaults to the run log dir basename "
            "(i.e. --log_dir_name / the auto timestamped name)."
        ),
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Optional wandb entity (team/user); empty uses the default.",
    )
    parser.add_argument(
        "--timing_no_cuda_sync",
        action="store_true",
        help="Debug timing: do not synchronize CUDA around timing scopes.",
    )
    parser.add_argument(
        "--timing_print_top_k",
        type=int,
        default=5,
        help="Debug timing: number of slowest rolling scopes to print.",
    )
    parser.add_argument(
        "--dataset_timing_enabled",
        dest="dataset_timing_enabled",
        default=True,
        action="store_true",
        help="Allow dataset-worker JSON timing reports when --debug is set.",
    )
    parser.add_argument(
        "--dataset_timing_disabled",
        dest="dataset_timing_enabled",
        action="store_false",
        help="Disable dataset-worker timing reports even when --debug is set.",
    )
    parser.add_argument(
        "--dataset_timing_sample_every",
        type=int,
        default=1,
        help="Debug timing: record one dataset timing sample every N __getitem__ calls.",
    )
    parser.add_argument(
        "--loose_model_loading",
        default=False,
        action="store_true",
        help=(
            "Allow base model loading to ignore checkpoint keys that do not "
            "exist in the detected target model, while still requiring all "
            "target keys and shapes to match. Default off. Does not affect "
            "--trained_dit, which remains strict."
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Path to a previous run's log directory (e.g. "
            "/datasets_genie/runjia.qian/logs_vipe_video/20260513_140536). "
            "When set, the trainer reads <dir>/train_state.json to recover "
            "(start_epoch, dataset_pass_id) and auto-injects the latest "
            "epoch-*.safetensors as --trained_dit (unless --trained_dit is "
            "explicitly given). dataset_pass_id is incremented by 1 so the "
            "post-resume data sequence is reproducible BUT disjoint from "
            "the original run -- different scenes per step, different G, "
            "different mode A/B draws."
        ),
    )
    parser.add_argument(
        "--dataset_pass_id",
        type=int,
        default=None,
        help=(
            "Folded into every per-sample seed derivation. Same (--seed, "
            "--dataset_pass_id) -> identical data sequence; bump it by 1 "
            "to get a disjoint-but-reproducible sequence. Default: 0 for "
            "fresh runs, auto-bumped to N+1 by --resume_from when the "
            "previous train_state.json had N. Pass an explicit value to "
            "override the auto-bump."
        ),
    )
    parser.add_argument("--framewise_decoding", default=False, action="store_true")
    parser.add_argument(
        "--camera_params_path",
        type=str,
        default=None,
        help="Path to DA3 camera parameter npy files.",
    )
    parser.add_argument(
        "--force_override_extrinsic",
        type=str,
        default=None,
        help=(
            "Validation-only extrinsic override. If set to a file, every "
            "validation sample reads that file; if set to a directory, each "
            "validation sample randomly reads one .npz inside it."
        ),
    )
    parser.add_argument(
        "--val_image_path",
        type=str,
        default=None,
        help=(
            "Validation-only segment-0 initial-frame override. If set to a "
            "single png/jpg/jpeg file, every rank starts its rollout from that "
            "image; if set to a directory of png/jpg/jpeg files, the images are "
            "distributed across global ranks in sorted order (no overlap until "
            "the pool is exhausted, then wraps). The image is cover-resized + "
            "centre-cropped (no stretch) to --width x --height and VAE-encoded "
            "like a dataset frame, replacing only the segment-0 initial frame; "
            "later segments roll out normally and camera params are unchanged. "
            "Pairs with --validation_qwen_prompt self_rollout / "
            "self_rollout_event (the segment-0 prompter then sees this image)."
        ),
    )
    parser.add_argument(
        "--validation_camera_pingpong",
        default=False,
        action="store_true",
        help=(
            "When a validation episode's camera trajectory is shorter than the "
            "rollout (its frame count < num_validation_blocks * frames_per_"
            "segment + 1), traverse the trajectory ping-pong / boustrophedon "
            "instead of clamping to the final pose: read forward to the last "
            "frame, then backward to the first, then forward again, repeating. "
            "Off (default) keeps the legacy clamp-to-last-pose behaviour."
        ),
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        default=None,
        help="Path to precomputed depth npz files.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Number of dataloader steps per epoch.",
    )
    parser.add_argument(
        "--latent_window_size",
        type=int,
        default=21,
        help="Number of latent frames sampled per item.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="da3_video",
        choices=["da3_video", "da3_video_subject_ref"],
        help=(
            "Strict mode supports 'da3_video' and 'da3_video_subject_ref'. "
            "Data, camera, depth, and prompt paths are read exclusively from "
            "--dataset_index_path or --val_dataset_index_path sqlite records."
        ),
    )
    parser.add_argument(
        "--vipe_default_prompt",
        type=str,
        default="",
        help=(
            "Legacy VIPE fallback prompt. Ignored in strict da3_video sqlite "
            "mode; sqlite records must provide prompt_path unless "
            "--allow_no_prompt is explicitly set."
        ),
    )
    parser.add_argument(
        "--vipe_prompt_path",
        type=str,
        default="",
        help=(
            "Legacy VIPE caption roots. Ignored in strict da3_video sqlite "
            "mode; prompt JSON paths come only from sqlite prompt_path."
        ),
    )
    parser.add_argument(
        "--vipe_prompt_type",
        type=int,
        default=2,
        choices=(1, 2),
        help=(
            "Prompt composition for dataset captions. 1 keeps the existing "
            "detailed.dynamic-only prompt. 2 prepends the same segment's "
            "detailed.start text before detailed.dynamic and emits the joined "
            "string as one prompt."
        ),
    )
    parser.add_argument(
        "--vipe_prompt_segment_format",
        action="store_true",
        help=(
            "DEPRECATED no-op for parsing: list-form ({start,prompt} segment "
            "entries) and legacy per-frame dict prompt JSONs are both "
            "auto-detected now. Kept only so old args.yaml files still parse. "
            "NOTE: this flag NO LONGER implicitly enables "
            "--align_generating_to_prompt_segments; pass that flag explicitly."
        ),
    )
    parser.add_argument(
        "--align_generating_to_prompt_segments",
        action="store_true",
        help=(
            "Sample each video's generating_first_frame_idx from prompt "
            "segment starts. Works with BOTH prompt JSON forms: list-form "
            "segment entries and legacy per-frame dicts (consecutive equal "
            "prompts are folded into segments automatically). The first "
            "segment uses start+1 so clean context remains available."
        ),
    )
    parser.add_argument(
        "--allow_no_prompt",
        action="store_true",
        help=(
            "If set, scenes with no usable caption JSON are still kept "
            "during dataset scan. In strict da3_video sqlite mode this is "
            "the only way to keep records whose sqlite prompt_path is empty; "
            "otherwise they are rejected with reason='no_prompt'."
        ),
    )
    parser.add_argument(
        "--loose_prompt_match",
        action="store_true",
        help=(
            "Legacy VIPE prompt-root fuzzy matching. Ignored in strict "
            "da3_video sqlite mode because prompt paths come only from "
            "sqlite records."
        ),
    )
    # -- DA3 video flow ---------------------------------------------------
    parser.add_argument(
        "--vae_clean_context_blocks_max",
        type=int,
        default=0,
        help=(
            "Video flow: max number of clean-context latent blocks (1 block "
            "= 4 source frames) to feed the VAE before the noisy window. "
            "0 (default) -> always mode A (1 clean frame); >0 -> per-sample "
            "mode A vs B sampled by --vae_clean_context_mode_ratio, with "
            "B drawing blocks uniformly from [1, max]."
        ),
    )
    parser.add_argument(
        "--vae_clean_context_mode_ratio",
        type=str,
        default="2:8",
        help=(
            "Video flow: '<single>:<block>' weight for the clean-anchor "
            "form. Without history it drives the dataset's mode A "
            "(single clean frame) vs mode B (4-frame block) draw. When "
            "--train_clean_latent_history_count_max > 0 the trainer-side "
            "plan samples the FINAL anchor form with the same meaning: "
            "P(single)=a/(a+b) -> mode C (no history, segment-0-like); "
            "otherwise block anchor with history_count ~ U[1,max] "
            "(mode A/B). Pass --train_clean_history_legacy_plan to restore "
            "the deprecated count-first sampler."
        ),
    )
    parser.add_argument(
        "--train_clean_latent_history_count_max",
        type=int,
        default=0,
        help=(
            "Video-flow training clean-history max. 0 keeps legacy "
            "--vae_clean_context_blocks_max behaviour. >0 enables the "
            "per-step plan: single anchor (mode C) with probability from "
            "--vae_clean_context_mode_ratio, else block anchor with "
            "history_count uniform in [1,max] (1 -> mode A latest-block "
            "re-encode, >1 -> mode B oldest_spread pool + latest block)."
        ),
    )
    parser.add_argument(
        "--train_clean_history_legacy_plan",
        default=False,
        action="store_true",
        help=(
            "REMOVED (warn-and-ignore). The pre-v2 count-first sampler no "
            "longer exists; the v2 anchor/context plan (--train_anchor_* / "
            "--train_context_*) is the single source of truth."
        ),
    )
    # -- v2 anchor/context plan (single source of truth) -------------------
    parser.add_argument(
        "--train_anchor_block_prob",
        type=float,
        default=None,
        help=(
            "v2 plan: per-step probability that the clean anchor is a "
            "4-frame BLOCK latent (else a single-frame latent). The anchor "
            "is ALWAYS jointly encoded with the noisy targets; block "
            "anchors get (train_block_anchor_frames-1)/4 - 1 warm-up "
            "blocks prepended and dropped. Default: derived from "
            "--vae_clean_context_mode_ratio as b/(a+b) ('2:8' -> 0.8)."
        ),
    )
    parser.add_argument(
        "--train_block_anchor_frames",
        type=int,
        default=13,
        help=(
            "v2 plan: total frames of the block-anchor encode group "
            "(warm-up + anchor block), jointly encoded with the targets. "
            "Must be 4k+1 and >= 5; anchor chain depth = (n-1)/4. Default "
            "13 (depth 3) -- where the encode-transient channel statistics "
            "saturate against a deep-chain reference."
        ),
    )
    parser.add_argument(
        "--train_context_count_max",
        type=int,
        default=None,
        help=(
            "v2 plan: max number of discrete CONTEXT blocks (far-past "
            "history latents appended with their own relative RoPE times "
            "and cameras, always separately encoded). Per step the count "
            "is uniform in [0, max]; 0 disables context. Default: derived "
            "as max(0, --train_clean_latent_history_count_max - 1)."
        ),
    )
    parser.add_argument(
        "--train_context_selection",
        type=str,
        default="oldest_spread",
        choices=["oldest_spread", "recent", "segment_coverage", "per_3latentframe"],
        help=(
            "v2 plan: how context blocks are picked from the candidate pool. "
            "Shares the validation pool-history selector; segment_coverage "
            "ranks blocks by camera-pose coverage of the generating segment "
            "(needs extrinsics, threaded from fused_query_inputs['w2c']). "
            "per_3latentframe IGNORES --train_context_count_max for the emitted "
            "count and instead picks exactly n_gen//3 context blocks -- one per "
            "consecutive group of 3 generating latents, each chosen to best "
            "cover that group's camera poses (also needs w2c extrinsics). NOTE: "
            "--train_context_count_max still governs the dataset's candidate-pool "
            "G reservation, so keep it >= n_gen//3 or per_3 falls back to recent "
            "blocks for the shortfall."
        ),
    )
    parser.add_argument(
        "--memory_latent_cache_dir",
        type=str,
        default="./.cache/memory_latents",
        help=(
            "Video flow: root directory for per-frame memory-latent cache. "
            "Empty (default) disables the cache. When set, dataset workers "
            "skip both the MP4 read and VAE encode for cache hits, and the "
            "trainer atomically dumps newly encoded frames so they're "
            "reusable across epochs / runs."
        ),
    )
    parser.add_argument(
        "--memory_latent_cache_version",
        type=str,
        default="wan2.2_ti2v_5b_vae",
        help=(
            "Version string folded into the per-frame cache hash. Bump "
            "(or change) this whenever the VAE weights or scaling rules "
            "change to invalidate every entry under "
            "--memory_latent_cache_dir without manual deletion."
        ),
    )
    parser.add_argument("--val_assign_yaml", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--filter_yaml", type=str, default=None)
    parser.add_argument(
        "--include_yaml",
        type=str,
        default=None,
        help=(
            "Optional whitelist YAML. When set, keep only samples whose path "
            "contains one of the listed hashes/tokens. Applies to vipe, video "
            "and da3_video datasets before train/val split."
        ),
    )
    parser.add_argument("--random_start_latent_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_data_items", type=int, default=None)
    parser.add_argument("--max_scan_items", type=int, default=None)
    parser.add_argument(
        "--dataset_index_path",
        type=str,
        default=None,
        help=(
            "Optional pre-scanned SQLite index for --dataset_type=da3_video. "
            "When set, dataset initialization reads the index instead of "
            "scanning DA3/prompt folders."
        ),
    )
    parser.add_argument(
        "--val_dataset_index_path",
        type=str,
        default=None,
        help=(
            "Optional pre-scanned SQLite index used only for validation/"
            "sanity-check when --dataset_type=da3_video. When set, the "
            "validation dataset is built entirely from this index instead "
            "of the train dataset split or validation manifest."
        ),
    )
    parser.add_argument(
        "--min_frame_count",
        type=int,
        default=None,
        help=(
            "Drop DA3 video scenes shorter than this frame count when reading "
            "a pre-scanned index. The dataset also applies its dynamic "
            "minimum based on latent_window_size and clean context blocks."
        ),
    )
    parser.add_argument(
        "--dataset_compact_mode",
        action="store_true",
        help=(
            "Compatibility mode for old DA3 dataset indexes/manifests that do "
            "not carry depth_format/depth_metadata. Only in this mode, missing "
            "depth metadata is treated as legacy float/float16 NPZ depth."
        ),
    )
    parser.add_argument(
        "--dataset_balance_mode",
        type=str,
        default="global",
        choices=["global", "camera_root", "camera_root_manual"],
        help=(
            "Training sample selection policy. 'global' keeps the existing "
            "flat shuffled sample order. 'camera_root' first balances across "
            "the comma-separated --camera_params_path roots. "
            "'camera_root_manual' uses --dataset_balance_root_probs to assign "
            "explicit root sampling percentages. Validation keeps its existing "
            "deterministic selection order."
        ),
    )
    parser.add_argument(
        "--dataset_balance_root_probs",
        type=str,
        default="",
        help=(
            "Comma-separated '<root-pattern>:<percent>' entries for "
            "--dataset_balance_mode=camera_root_manual, e.g. "
            "'dl3dv:70,objaverse:30'. Root patterns match camera_root paths "
            "case-insensitively by substring. Percentages must be non-negative "
            "ints and must sum exactly to 100; each actual root must match "
            "exactly one entry."
        ),
    )
    parser.add_argument(
        "--max_frames_per_scene",
        type=int,
        default=None,
        help=(
            "Test / debug knob: cap each scene to its first N frames "
            "(applies to extrinsics / intrinsics / depth / RGB and to "
            "the G upper bound). Use to overfit on a short prefix or "
            "smoke-test long clips without paying for full footage. "
            "None / <=0 = no cap. Folded into the dataset cache "
            "fingerprint so changing the cap re-runs the scan."
        ),
    )
    # Dataset-info cache + manifest. The same JSON payload is written to
    # ``--dataset_cache_dir/<fingerprint>.json`` (persistent, reused across
    # runs to skip re-scanning) AND to ``<log_dir>/dataset_manifest_*.json``
    # (per-run snapshot living next to the rest of the run artifacts so
    # you can grep through the log dir for exactly which scenes a run
    # consumed). Caching is keyed on a fingerprint that incorporates the
    # exact list of latent paths plus every scan-affecting input, so
    # adding / removing data automatically invalidates stale entries.
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default="./.cache/dataset_cache",
        help=(
            "Directory for persistent dataset-info cache. Set to an "
            "empty string to disable. Default: ./.cache/dataset_info"
        ),
    )
    parser.add_argument(
        "--no_dataset_cache",
        default=False,
        action="store_true",
        help="Disable the dataset-info cache (equivalent to --dataset_cache_dir '').",
    )
    parser.add_argument(
        "--dataset_prefetch_factor",
        type=int,
        default=None,
        help=(
            "Training DataLoader prefetch_factor when --dataset_num_workers > 0. "
            "Default: same value as --dataset_num_workers. Ignored when "
            "--dataset_num_workers=0."
        ),
    )
    parser.add_argument("--preview_dataset", default=True, action="store_true")
    parser.add_argument("--num_preview", type=int, default=1)
    parser.add_argument("--run_validation", default=True, action="store_true")
    parser.add_argument("--run_sanity_check", default=False, action="store_true")
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--num_val_batches", type=int, default=1)
    parser.add_argument(
        "--validation_sample_offset",
        type=int,
        default=0,
        help=(
            "Start validation sampling from this offset inside the fixed val "
            "set. This changes which held-out samples are evaluated first "
            "without changing the train/val split."
        ),
    )
    parser.add_argument(
        "--validation_rotate_samples",
        default=False,
        action="store_true",
        help=(
            "Rotate validation sample offset by epoch so repeated validation "
            "runs cover different held-out samples while keeping the val set "
            "itself fixed. Effective offset is "
            "validation_sample_offset + validation_call_index * "
            "num_val_batches * world_size."
        ),
    )
    parser.add_argument(
        "--validation_manifest_path",
        "--val_manifest_path",
        dest="validation_manifest_path",
        type=str,
        default=None,
        help=(
            "Optional dataset_manifest_*.json to pin validation/sanity-check "
            "samples. New manifests use explicit validation_keys/sample_keys; "
            "older val manifests are made val-only by subtracting the sibling "
            "dataset_manifest_train.json when it exists."
        ),
    )
    parser.add_argument(
        "--validation_manifest_exclude_path",
        "--val_manifest_exclude_path",
        dest="validation_manifest_exclude_path",
        type=str,
        default=None,
        help=(
            "Optional manifest whose sample keys are excluded from "
            "--validation_manifest_path. Defaults to a sibling "
            "dataset_manifest_train.json when present."
        ),
    )
    parser.add_argument("--validation_seed", type=int, default=3407)
    parser.add_argument(
        "--val_dataset_path_shuffle_seed",
        type=int,
        default=42,
        help=(
            "Seed for validation dataset path order after scan. Default 42 "
            "preserves the previous fixed validation ordering."
        ),
    )
    parser.add_argument("--validation_num_inference_steps", type=int, default=25)
    parser.add_argument("--validation_cfg_scale", type=float, default=5.0)
    parser.add_argument(
        "--validation_negative_no_prope",
        default=False,
        action="store_true",
        help=(
            "During validation CFG, run the negative branch with PRoPE disabled "
            "so its self-attention uses the standard SDPA path. The positive "
            "branch keeps the validation PRoPE settings."
        ),
    )
    parser.add_argument(
        "--validation_negative_no_context",
        default=False,
        action="store_true",
        help=(
            "During validation CFG, run the negative branch with only the "
            "latest clean latent context while the positive branch keeps the "
            "configured clean latent history. Requires unmerged CFG."
        ),
    )
    parser.add_argument("--num_validation_blocks", type=int, default=2)
    parser.add_argument(
        "--validation_t2v_no_ref",
        default=False,
        action="store_true",
        help=(
            "Diagnostic validation/sample mode: section 0 is pure T2V with "
            "no input_image and no first_frame_latents, so the pipeline denoises "
            "latent_window_size noisy latents directly. Later sections continue "
            "from generated history. Pair with --no_mosaic for no-memory T2V."
        ),
    )
    parser.add_argument(
        "--validation_negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    )
    parser.add_argument(
        "--validation_use_decoded_first_image",
        default=False,
        action="store_true",
        help=(
            "If set, validation feeds the first frame to the pipeline by VAE-"
            "decoding the previous segment's last clean latent into a PIL "
            "image (round-trip), matching the legacy code path. Default off: "
            "pass the clean latent directly as `first_frame_latents`, which "
            "is faster and avoids VAE encode/decode losses."
        ),
    )
    parser.add_argument(
        "--validation_clean_latent_history_count",
        type=int,
        default=1,
        help=(
            "Validation-only clean latent prefix length for sections after "
            "section 0. Section 0 always keeps the legacy single clean latent. "
            "By default, section 1 onward uses the latest N history latents "
            "in temporal order. With --validation_clean_latent_pool_history, "
            "it instead uses N-1 candidate-pool history latents plus the "
            "latest latent."
        ),
    )
    parser.add_argument(
        "--validation_clean_latent_pool_history",
        default=False,
        action="store_true",
        help=(
            "Opt in to selecting the first N-1 validation clean history "
            "latents from the frustum candidate pool after candidate "
            "selection. Default off keeps the original temporal recent-window "
            "selection for --validation_clean_latent_history_count."
        ),
    )
    parser.add_argument(
        "--validation_clean_latent_pool_history_selection_mode",
        type=str,
        default="recent",
        choices=["recent", "oldest_spread", "segment_coverage", "per_3latentframe"],
        help=(
            "Selection strategy used only when "
            "--validation_clean_latent_pool_history is enabled. "
            "'recent' preserves the previous pool behaviour. "
            "'oldest_spread' anchors older pool latents and spreads the "
            "remaining picks across the timeline toward the latest clean latent. "
            "'segment_coverage' uses target segment camera poses to pick "
            "diverse history latents that cover the segment without expensive "
            "per-candidate projection IoU. "
            "'per_3latentframe' IGNORES --validation_context_count and instead "
            "emits exactly latent_window_size//3 context latents -- one per "
            "consecutive group of 3 generating latents, each covering its "
            "group's poses; it requires --validation_clean_latent_pool_history "
            "and that latent_window_size be divisible by 3 (both enforced)."
        ),
    )
    parser.add_argument(
        "--validation_clean_latent_pool_history_bias",
        type=int,
        default=1,
        help=(
            "Validation-only RoPE time gap between the candidate-pool history "
            "clean latents and the latest clean latent when "
            "--validation_clean_latent_history_count > 1. Must be >= 1; "
            "default 1 keeps the latest clean immediately after the history "
            "pool, and noisy/mosaic RoPE times follow after it."
        ),
    )
    parser.add_argument(
        "--validation_clean_latent_pool_history_rope_original_indices",
        default=False,
        action="store_true",
        help=(
            "Validation-only, default off. When clean latents are selected "
            "from the validation candidate pool, use their original latent "
            "positions in the rollout history as temporal RoPE indices. "
            "When off, validation keeps the legacy compact history-bias "
            "mapping controlled by --validation_clean_latent_pool_history_bias."
        ),
    )
    # --- Canonical "context" vocabulary: front-door aliases for the five
    # validation_clean_latent_pool_history* params above (back-compat). Each
    # has default=SUPPRESS so it only overrides the old attr when explicitly
    # given (CLI or YAML). validation_context_count EXCLUDES the anchor (aligns
    # with train_context_count_max's vocabulary); the old
    # validation_clean_latent_history_count INCLUDED it, so the alias adds +1.
    parser.add_argument(
        "--validation_context_count",
        type=int,
        default=SUPPRESS,
        help=(
            "Validation context budget EXCLUDING the always-present anchor "
            "(0 = anchor only, no context), mirroring train_context_count_max "
            "(train uses it as a random max; val uses it directly). Canonical "
            "alias for --validation_clean_latent_history_count (old = new + 1)."
        ),
    )
    parser.add_argument(
        "--validation_context_pool",
        default=SUPPRESS,
        action="store_true",
        help="Canonical alias for --validation_clean_latent_pool_history.",
    )
    parser.add_argument(
        "--validation_context_selection",
        type=str,
        default=SUPPRESS,
        choices=["recent", "oldest_spread", "segment_coverage", "per_3latentframe"],
        help=(
            "Canonical alias for "
            "--validation_clean_latent_pool_history_selection_mode; shares the "
            "same selector as train_context_selection (incl. per_3latentframe)."
        ),
    )
    parser.add_argument(
        "--validation_context_pool_bias",
        type=int,
        default=SUPPRESS,
        help="Canonical alias for --validation_clean_latent_pool_history_bias.",
    )
    parser.add_argument(
        "--validation_context_pool_rope_original_indices",
        default=SUPPRESS,
        action="store_true",
        help=(
            "Canonical alias for "
            "--validation_clean_latent_pool_history_rope_original_indices."
        ),
    )
    parser.add_argument(
        "--validation_min_anchor_frame_idx",
        type=int,
        default=0,
        help=(
            "Validation-only lower bound for the clean anchor RGB frame index "
            "(anchor = generating_first_frame_idx - 1). Use 200 to keep at "
            "least frames [0,199] before the anchor for context-pool sampling."
        ),
    )
    parser.add_argument(
        "--eval_pool_variants",
        "--eval-pool-variants",
        dest="eval_pool_variants",
        type=str,
        default="correct_pool",
        help=(
            "Standalone subject-ref validation pool-context axis as a comma "
            "list. Choices: no_pool, correct_pool, wrong_pool."
        ),
    )
    parser.add_argument(
        "--eval_ref_variants",
        "--eval-ref-variants",
        dest="eval_ref_variants",
        type=str,
        default="correct_ref",
        help=(
            "Standalone subject-ref validation ref-token axis as a comma "
            "list. Choices: no_ref, correct_ref, wrong_ref."
        ),
    )
    parser.add_argument(
        "--eval_grid_preset",
        "--eval-grid-preset",
        dest="eval_grid_preset",
        type=str,
        default="compact",
        choices=["compact", "full", "custom"],
        help=(
            "Standalone subject-ref validation grid preset. compact runs the "
            "key pool/ref ablations; full runs eval_pool_variants x "
            "eval_ref_variants; custom runs eval_variants exactly."
        ),
    )
    parser.add_argument(
        "--eval_variants",
        "--eval-variants",
        dest="eval_variants",
        type=str,
        default="",
        help=(
            "Comma list of exact standalone validation variants when "
            "eval_grid_preset=custom, e.g. "
            "no_pool_no_ref,correct_pool_correct_ref."
        ),
    )
    parser.add_argument(
        "--eval_text_cfg_scale",
        "--eval-text-cfg-scale",
        dest="eval_text_cfg_scale",
        type=float,
        default=1.0,
        help="Standalone subject-ref validation text CFG scale.",
    )
    parser.add_argument(
        "--eval_num_inference_steps",
        "--eval-num-inference-steps",
        dest="eval_num_inference_steps",
        type=int,
        default=25,
        help="Standalone subject-ref validation denoising step count.",
    )
    parser.add_argument(
        "--eval_num_rollout_blocks",
        "--eval-num-rollout-blocks",
        dest="eval_num_rollout_blocks",
        type=int,
        default=1,
        help=(
            "Standalone subject-ref validation rollout sections. Values >1 "
            "reuse the training-time validation rollout path and currently "
            "support correct_pool variants only."
        ),
    )
    parser.add_argument(
        "--eval_seed",
        "--eval-seed",
        dest="eval_seed",
        type=int,
        default=42,
        help="Standalone subject-ref validation sampler seed.",
    )
    parser.add_argument(
        "--eval_context_patch_mask_prob",
        "--eval-context-patch-mask-prob",
        dest="eval_context_patch_mask_prob",
        type=float,
        default=None,
        help=(
            "Standalone subject-ref validation override for "
            "subject_context_patch_mask_prob. Leave unset to use "
            "subject_context_patch_mask_prob from YAML/CLI."
        ),
    )
    parser.add_argument(
        "--eval_require_subject_ref_keys",
        "--eval-require-subject-ref-keys",
        dest="eval_require_subject_ref_keys",
        default=True,
        action="store_true",
        help=(
            "Standalone subject-ref validation safety check: require "
            "--trained_dit to contain subject_ref_ keys. Disable for "
            "pool-context-only checkpoints with "
            "--eval-no-require-subject-ref-keys or "
            "eval_require_subject_ref_keys: false in YAML."
        ),
    )
    parser.add_argument(
        "--eval-no-require-subject-ref-keys",
        dest="eval_require_subject_ref_keys",
        action="store_false",
        help=(
            "Allow standalone validation with checkpoints that do not contain "
            "subject_ref_ keys. Only use for no_ref / pool-context-only "
            "ablations."
        ),
    )
    parser.add_argument(
        "--validation_qwen_prompt",
        default=False,
        action="store_true",
        help=(
            "During validation, generate section prompts with a vision-language "
            "prompt model. See --validation_qwen_mode for frame source semantics."
        ),
    )
    parser.add_argument(
        "--validation_qwen_mode",
        type=str,
        default="oracle_section_gt",
        choices=VALIDATION_QWEN_MODES,
        help=(
            "Validation prompt generation mode: oracle_section_gt reads future "
            "GT frames per segment; oracle_full_gt_static reads the full GT "
            "rollout once and reuses one prompt; self_rollout uses only current "
            "or past generated/context frames; self_rollout_event also binds one "
            "event per validation sample."
        ),
    )
    parser.add_argument(
        "--validation_qwen_event_yaml",
        type=str,
        default="",
        help=(
            "YAML event set for --validation_qwen_mode self_rollout_event. "
            "Accepts a list or an {events: [...]} mapping; entries may be strings "
            "or mappings with id/text/tags/phase_hint. Empty uses built-in events."
        ),
    )
    parser.add_argument(
        "--validation_qwen_full_video_sample_frames",
        type=int,
        default=0,
        help=(
            "Frame count for oracle_full_gt_static full-rollout prompt sampling. "
            "0 means validation_qwen_sample_frames * num_validation_blocks."
        ),
    )
    parser.add_argument(
        "--validation_qwen_provider",
        type=str,
        default="auto",
        choices=("auto", "qwen", "gemma4"),
        help=(
            "Prompt model provider. 'auto' infers qwen/gemma4 from local model "
            "path/config. gemma4 currently supports gemma-4-E2B-it/E4B-it."
        ),
    )
    parser.add_argument(
        "--validation_qwen_model_path",
        "--validation_qwen_model_id",
        dest="validation_qwen_model_path",
        type=str,
        default="runjia.qian/models/Qwen/Qwen3.5-4B",
        help=(
            "Local path for the validation Qwen prompt model. "
            "--validation_qwen_model_id is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--validation_qwen_device_map",
        type=str,
        default="transient_cuda",
        help=(
            "device_map passed to AutoModelForImageTextToText.from_pretrained. "
            "Use 'none' to omit device_map, 'cpu' for CPU-only inference, or "
            "'transient_cuda' to keep Qwen on CPU except while generating a prompt."
        ),
    )
    parser.add_argument(
        "--validation_qwen_torch_dtype",
        type=str,
        default="bfloat16",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        help="torch_dtype for the validation Qwen prompt model.",
    )
    parser.add_argument(
        "--validation_qwen_max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens generated by the validation Qwen prompt model.",
    )
    parser.add_argument(
        "--validation_qwen_word_limit",
        type=int,
        default=200,
        help="Maximum English words kept in each generated validation prompt.",
    )
    parser.add_argument(
        "--validation_qwen_sample_frames",
        type=int,
        default=5,
        help=(
            "Number of section-local GT video frames uniformly sampled for "
            "validation Qwen prompt generation."
        ),
    )
    parser.add_argument(
        "--validation_qwen_frame_max_side",
        type=int,
        default=300,
        help=(
            "Downsample each validation Qwen input frame so its longer side "
            "is at most this value."
        ),
    )
    parser.add_argument(
        "--validation_qwen_trust_remote_code",
        default=False,
        action="store_true",
        help="Pass trust_remote_code=True when loading the validation Qwen model.",
    )
    parser.add_argument(
        "--train_qwen_prompt",
        default=False,
        action="store_true",
        help=(
            "During training, override each dataset prompt with a Qwen prompt "
            "generated from uniformly sampled GT video frames before preview "
            "and model forward."
        ),
    )
    parser.add_argument(
        "--train_qwen_model_path",
        "--train_qwen_model_id",
        dest="train_qwen_model_path",
        type=str,
        default="",
        help=(
            "Local path for the training Qwen prompt model. Empty means reuse "
            "--validation_qwen_model_path."
        ),
    )
    parser.add_argument(
        "--train_qwen_provider",
        type=str,
        default="auto",
        choices=("auto", "qwen", "gemma4"),
        help=(
            "Training prompt model provider. 'auto' infers qwen/gemma4 from "
            "the local train model path/config. gemma4 currently supports "
            "gemma-4-E2B-it/E4B-it."
        ),
    )
    parser.add_argument(
        "--train_qwen_device_map",
        type=str,
        default="auto",
        help=(
            "device_map passed to AutoModelForImageTextToText.from_pretrained "
            "for training prompt generation. Use 'none' to omit device_map."
        ),
    )
    parser.add_argument(
        "--train_qwen_torch_dtype",
        type=str,
        default="bfloat16",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        help="torch_dtype for the training Qwen prompt model.",
    )
    parser.add_argument(
        "--train_qwen_max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens generated by the training Qwen prompt model.",
    )
    parser.add_argument(
        "--train_qwen_word_limit",
        type=int,
        default=200,
        help="Maximum English words kept in each generated training prompt.",
    )
    parser.add_argument(
        "--train_qwen_sample_frames",
        type=int,
        default=8,
        help="Number of GT video frames uniformly sampled for training Qwen prompt.",
    )
    parser.add_argument(
        "--train_qwen_frame_max_side",
        type=int,
        default=448,
        help="Downsample each training Qwen input frame so its longer side is at most this value.",
    )
    parser.add_argument(
        "--train_qwen_prompt_missing_only",
        default=False,
        action="store_true",
        help=(
            "With --train_qwen_prompt, generate a Qwen prompt only when the "
            "dataset prompt is empty. Samples that already have a prompt keep "
            "their dataset prompt."
        ),
    )
    parser.add_argument(
        "--train_qwen_prompt_cache",
        default=False,
        action="store_true",
        help=(
            "With --train_qwen_prompt --train_qwen_prompt_missing_only, dump "
            "new Qwen prompts into --train_qwen_prompt_cache_dir as "
            "<scene_id>.json files compatible with --vipe_prompt_path. "
            "The cache is read by subsequent runs, not by the current in-memory "
            "dataset instance."
        ),
    )
    parser.add_argument(
        "--train_qwen_prompt_cache_dir",
        type=str,
        default="",
        help=(
            "Directory for train Qwen prompt cache dumps. Each generated "
            "missing-only prompt appends/overwrites frame entries in "
            "<scene_id>.json under this directory."
        ),
    )
    parser.add_argument(
        "--train_qwen_trust_remote_code",
        default=False,
        action="store_true",
        help="Pass trust_remote_code=True when loading the training Qwen model.",
    )
    parser.add_argument("--use_prope", default=True, action="store_true")
    parser.add_argument(
        "--prope_attention_interval",
        type=int,
        default=1,
        help=(
            "Stride for enabling PRoPE self-attention across DiT blocks. "
            "Only effective when --use_prope (or --only_prope) is set. "
            "Default 1 = every block uses PRoPE (legacy behaviour). "
            "N>1 enables PRoPE only on blocks whose 0-based index satisfies "
            "(block_id %% N) == 0 (e.g. N=2 -> blocks 0,2,4,...); all other "
            "blocks fall back to plain SDPA. If the total block count is not "
            "divisible by N, the rule is applied up to the actual block count "
            "with no further adjustment."
        ),
    )
    parser.add_argument(
        "--prope_disable_native_rope",
        default=False,
        action="store_true",
        help=(
            "When sparse PRoPE is enabled, skip Wan's native 3D RoPE inside "
            "the blocks that run PRoPE attention. Requires "
            "--prope_attention_interval > 1. Default off preserves legacy "
            "PRoPE+3D-RoPE behaviour."
        ),
    )
    parser.add_argument(
        "--prope_disable_t_rope",
        default=False,
        action="store_true",
        help=(
            "When sparse PRoPE is enabled, skip only the TEMPORAL RoPE band "
            "inside the blocks that run PRoPE attention, keeping x/y RoPE "
            "for intra-frame spatial structure; cross-frame addressing in "
            "those blocks is delegated to the camera matrices. Requires "
            "--prope_attention_interval > 1 and is mutually exclusive with "
            "--prope_disable_native_rope. Default off."
        ),
    )
    parser.add_argument(
        "--prope_camera_layout",
        type=str,
        default="full",
        choices=["full", "sf13"],
        help=(
            "Head_dim camera tiling for PRoPE attention. 'full' (default) "
            "tiles every sub-frame matrix across all head_dim (legacy, "
            "overlaps x/y-RoPE). 'sf13' subsamples the 4 VAE sub-frame "
            "cameras to the 2nd and 4th ({1,3}) and writes each as a "
            "contiguous 16-dim block (4 four-vectors) into the front of "
            "head_dim, which is rope-free under --prope_disable_t_rope, so "
            "the camera lives in a clean subspace with no x/y-RoPE "
            "entanglement. 'sf13' requires --prope_disable_t_rope."
        ),
    )
    parser.add_argument(
        "--only_prope",
        default=False,
        action="store_true",
        help="Enable camera/PRoPE control only: disable mosaic latent conditioning.",
    )
    parser.add_argument(
        "--trans_scale",
        type=_parse_trans_scale_arg,
        default=50.0,
        help=(
            "Translation normalisation for PRoPE camera extrinsics, applied "
            "to the recentered ``w2c[..., :3, 3]`` inside "
            "``WanVideoUnit_PropeCamera``. A number keeps the legacy linear "
            "divide ``t / trans_scale`` and must match the dataset's "
            "extrinsic units (e.g. ~50 for the VIPE/DL3DV layouts these "
            "checkpoints were trained on). The strings 'log'/'logd4'/'tanh' "
            "select direction-preserving radial compression of the magnitude "
            "(``||t|| -> log1p(||t||)`` resp. ``log1p(||t||) / 4`` resp. "
            "``tanh(||t||)``), which keeps small window-local motions "
            "near-identity instead of crushing them by a fixed divisor. "
            "'logd4' is 'log' with the compressed magnitude divided by 4 "
            "(log base e**4): same dynamic range as 'log' but the far history "
            "baseline is re-anchored near ~1 instead of ~4, so large pose "
            "gaps stop inflating the attention logits. Only effective when "
            "--use_prope (or --only_prope) is set; default 50.0 preserves the "
            "legacy hardcoded value."
        ),
    )
    # Mosaic-specific
    parser.add_argument(
        "--enable_mosaic",
        default=True,
        action="store_true",
        help="Enable mosaic-latent conditioning during training.",
    )
    # VAE decode tiling for preview / validation. Tiled decode existed
    # historically because the legacy 832x448 latent (28x52) fit in one
    # 30x52 tile -> tiling was a no-op. At video-flow resolutions
    # (e.g. 704x1280 -> 44x80 latent) the same tile_size triggers
    # 3x4=12 overlapping decodes per timestep, dominating preview /
    # validation wall time. Default False (full-image decode) since it
    # fits comfortably in modern VRAM at single-batch decode; flip on
    # only when running on a small GPU.
    parser.add_argument(
        "--vae_decode_tiled",
        dest="vae_decode_tiled",
        default=False,
        action="store_true",
        help=(
            "Use tiled VAE decode for preview / validation visuals "
            "(slower but lower VRAM). Default off."
        ),
    )
    parser.add_argument(
        "--no_vae_decode_tiled",
        dest="vae_decode_tiled",
        action="store_false",
        help="Force full-image VAE decode (default).",
    )
    parser.add_argument(
        "--no_mosaic",
        dest="enable_mosaic",
        action="store_false",
        help="Disable mosaic-latent conditioning (fallback to vanilla TI2V-5B training).",
    )
    parser.add_argument(
        "--mosaic_use_revgrid_rope",
        dest="mosaic_use_revgrid_rope",
        default=False,
        action="store_true",
        help=(
            "Use revgrid-based fractional spatial RoPE for mosaic frames "
            "during training (looks up `dit.freqs[3]/[4]` at 1/16-latent "
            "precision via the canvas-space float coords stored in "
            "`data['ql_rope_indice_pef']`). Default off, which reuses the "
            "noisy-frame integer RoPE -- the legacy behaviour. Has no "
            "effect when --no_mosaic or --only_prope is set."
        ),
    )
    parser.add_argument(
        "--mosaic_view_change_prope",
        default=False,
        action="store_true",
        help=(
            "Enable optional [da/db,phi,theta] PRoPE for mosaic/source tokens. "
            "Requires --use_prope and mosaic conditioning; first/noisy target "
            "tokens stay canonical (1,0,0)."
        ),
    )
    parser.add_argument(
        "--mosaic_interval",
        type=int,
        default=1,
        help=(
            "Keep one mosaic conditioning latent per interval chunk, using the "
            "last noisy latent in each chunk. Default 1 preserves dense mosaic."
        ),
    )
    parser.add_argument(
        "--no_mosaic_use_revgrid_rope",
        dest="mosaic_use_revgrid_rope",
        action="store_false",
        help="Force standard integer-grid RoPE for mosaic frames (default).",
    )
    parser.add_argument(
        "--mosaic_no_mosaic_sync_mode",
        type=str,
        default="none",
        choices=["none", "rank0_broadcast"],
        help=(
            "Forward-compatible knob for synchronising a future "
            "`no_mosaic_prob_train` 'this step skips mosaic' decision "
            "across DDP ranks. Today the main branch has no "
            "`no_mosaic_prob_train` feature, so the default 'none' is a "
            "no-op; 'rank0_broadcast' will (when that feature lands) "
            "broadcast rank 0's decision to all ranks so faster ranks "
            "don't sit idle waiting on slower mosaic ranks."
        ),
    )
    parser.add_argument(
        "--mosaic_sequence_balance_mode",
        type=str,
        default="profile_only",
        choices=["profile_only", "pad_to_world_max"],
        help=(
            "How to handle variable-length mosaic sequences once "
            "--mosaic_drop_holes shrinks the per-rank token count. "
            "'profile_only' (default) just logs the per-rank kept "
            "mosaic token count without altering the sequence; "
            "'pad_to_world_max' will, in a future patch, pad every "
            "rank's mosaic slice up to the world-max so DDP stays "
            "balanced. Reserved for a follow-up that wires the "
            "all_gather into the trainer."
        ),
    )
    parser.add_argument(
        "--mosaic_train_random_schedule",
        default=False,
        action="store_true",
        help=(
            "Enable the per-step DDP-synchronised mosaic mode schedule. When "
            "set, each training step samples one mode from "
            "--mosaic_train_schedule (default 3:4:3 over interval3 / "
            "interval1 / nomosaic) identically on every rank, and forces "
            "drop_holes on for both training and validation. Validation is "
            "pinned to interval1. When unset, the legacy fixed --mosaic_interval "
            "+ per-rank --no_mosaic_prob_train behaviour is preserved."
        ),
    )
    parser.add_argument(
        "--mosaic_train_schedule",
        type=str,
        default="interval3:4,interval1:4,nomosaic:2",
        help=(
            "Comma-separated '<mode>:<weight>' entries for the per-step "
            "training schedule, e.g. 'interval3:3,interval1:4,nomosaic:3'. "
            "Mode is 'interval<N>' or 'nomosaic'; weights are non-negative "
            "ints. Only consulted when --mosaic_train_random_schedule is set."
        ),
    )
    parser.add_argument(
        "--mosaic_schedule_sync_mode",
        type=str,
        default="seed",
        choices=["seed", "rank0_broadcast"],
        help=(
            "How the per-step mode is synchronised across DDP ranks. 'seed' "
            "(default) derives the choice deterministically from "
            "(args.seed, step) so every rank computes the same mode with zero "
            "communication. 'rank0_broadcast' samples on rank 0 and broadcasts "
            "the choice; falls back to local sampling when torch.distributed "
            "is not initialised."
        ),
    )
    parser.add_argument(
        "--mosaic_drop_holes",
        default=False,
        action="store_true",
        help=(
            "Actually drop (via index_select) the hole tokens identified by "
            "the patch-level hole mask, instead of just zeroing them in "
            "place. Cuts the transformer sequence length, freqs, t_mod and "
            "PROPE viewmats (re-broadcast to per-token) to match. Default "
            "False keeps the legacy 'mask but keep token count' behaviour."
        ),
    )
    parser.add_argument(
        "--mosaic_skip_cross_attn",
        default=False,
        action="store_true",
        help=(
            "DEPRECATED no-op, scheduled for removal (see README). The flag "
            "was reserved for a cross-attn shortcut that never landed; the "
            "mosaic cross_attn_keep_mask already zeroes mosaic-token "
            "cross-attn output unconditionally. Kept only so old YAML "
            "configs / dumped args.yaml files keep parsing."
        ),
    )
    parser.add_argument(
        "--subject_ref_dir_name",
        type=str,
        default="protagonist_refs",
        help=(
            "Scene objects subdirectory containing protagonist reference "
            "candidates for --dataset_type=da3_video_subject_ref."
        ),
    )
    parser.add_argument(
        "--subject_num_refs_max",
        type=int,
        default=2,
        help="Maximum protagonist reference images to use per sample.",
    )
    parser.add_argument(
        "--subject_dissimilar_top_k",
        type=int,
        default=8,
        help="Top-k most dissimilar refs to sample from after each selected ref.",
    )
    parser.add_argument(
        "--subject_max_similarity",
        type=float,
        default=0.94,
        help="Preferred max pairwise similarity between selected refs.",
    )
    parser.add_argument(
        "--subject_no_shuffle_refs",
        dest="subject_shuffle_refs",
        default=True,
        action="store_false",
        help="Disable random ordering of selected reference images.",
    )
    parser.add_argument(
        "--subject_ref_dropout_prob",
        type=float,
        default=0.0,
        help=(
            "Training probability of dropping all subject refs for a sample. "
            "Dropped samples also skip anchor/pool subject erase because no "
            "ref identity anchor is available."
        ),
    )
    parser.add_argument(
        "--subject_ref_canvas_slot_ratio",
        type=float,
        default=0.5,
        help=(
            "Side length of the lower-right square reference slot as a "
            "fraction of min(height, width)."
        ),
    )
    parser.add_argument(
        "--subject_ref_background",
        type=str,
        default="imagenet_mean",
        choices=["imagenet_mean", "zero", "black"],
        help="Background fill for reference canvas.",
    )
    parser.add_argument(
        "--subject_pool_context_dynamic_erase",
        default=True,
        action="store_true",
        help=(
            "Erase dynamic-object/protagonist regions from subject-ref pool "
            "context latents before concatenating them with the anchor. This "
            "prevents pool context from leaking the same subject appearance "
            "that ref memory is supposed to provide."
        ),
    )
    parser.add_argument(
        "--no_subject_pool_context_dynamic_erase",
        dest="subject_pool_context_dynamic_erase",
        action="store_false",
        help=(
            "Disable dynamic-object/protagonist erase for subject-ref pool "
            "context latents. Useful only for ablations."
        ),
    )
    parser.add_argument(
        "--subject_pool_context_dynamic_erase_prob",
        type=float,
        default=1.0,
        help=(
            "Probability of applying dynamic-object/protagonist erase to pool "
            "context latents when subject refs are present."
        ),
    )
    parser.add_argument(
        "--subject_pool_context_dynamic_erase_ratio_min",
        type=float,
        default=1.0,
        help=(
            "Minimum fraction of pool dynamic/protagonist patches to replace "
            "when pool context erase is enabled for the sample."
        ),
    )
    parser.add_argument(
        "--subject_pool_context_dynamic_erase_ratio_max",
        type=float,
        default=1.0,
        help=(
            "Maximum fraction of pool dynamic/protagonist patches to replace "
            "when pool context erase is enabled for the sample."
        ),
    )
    parser.add_argument(
        "--subject_pool_context_dynamic_erase_fill",
        type=str,
        default="background_patch",
        choices=["background_patch", "background_mean"],
        help="Fill strategy for erased pool context dynamic/protagonist patches.",
    )
    parser.add_argument(
        "--subject_context_patch_mask_prob",
        type=float,
        default=0.5,
        help=(
            "Training probability of replacing visible protagonist context "
            "patches with same-frame background patches."
        ),
    )
    parser.add_argument(
        "--subject_context_patch_mask_ratio_min",
        type=float,
        default=0.25,
        help="Minimum fraction of visible protagonist patches to replace.",
    )
    parser.add_argument(
        "--subject_context_patch_mask_ratio_max",
        type=float,
        default=0.75,
        help="Maximum fraction of visible protagonist patches to replace.",
    )
    parser.add_argument(
        "--subject_context_patch_mask_fill",
        type=str,
        default="background_patch",
        choices=["background_patch", "background_mean"],
        help="Fill strategy for masked context protagonist latent patches.",
    )
    parser.add_argument(
        "--subject_context_patch_mask_temporal_dilate_radius",
        type=int,
        default=0,
        help="Temporal dilation radius for context protagonist masks.",
    )
    parser.add_argument(
        "--subject_object_loss_alpha",
        type=float,
        default=0.0,
        help=(
            "Extra protagonist-region diffusion loss weight. The total loss is "
            "L_full + alpha * L_obj when a target protagonist mask is available."
        ),
    )
    parser.add_argument(
        "--subject_object_loss_min_mask_ratio",
        type=float,
        default=0.0,
        help=(
            "Disable the extra protagonist-region loss when the object mask "
            "covers less than this fraction of the latent target. Full-frame "
            "diffusion loss is still used."
        ),
    )
    parser.add_argument(
        "--subject_object_loss_temporal_dilate_radius",
        type=int,
        default=0,
        help="Temporal dilation radius for target protagonist loss masks.",
    )
    parser.add_argument(
        "--subject_ref_memory",
        default=False,
        action="store_true",
        help=(
            "Enable VAE protagonist reference memory tokens in DiT self-attention. "
            "Only used with subject_ref_latents from da3_video_subject_ref."
        ),
    )
    parser.add_argument(
        "--subject_ref_time_gap",
        type=int,
        default=1,
        help=(
            "Temporal RoPE gap for subject reference memory. The video "
            "timeline is not shifted."
        ),
    )
    parser.add_argument(
        "--subject_ref_prope_mode",
        type=str,
        default="identity",
        choices=["identity", "clean_anchor"],
        help=(
            "PRoPE camera transform assigned to subject ref prefix tokens. "
            "'identity' keeps the current canonical-camera behavior. "
            "'clean_anchor' copies the current clean-context anchor camera pose "
            "to all ref tokens."
        ),
    )
    parser.add_argument(
        "--subject_ref_learning_rate",
        type=float,
        default=None,
        help=(
            "Optional AdamW learning rate for parameters whose names contain "
            "'subject_ref_'. When unset, all trainable parameters use "
            "--learning_rate in one optimizer group."
        ),
    )
    parser.add_argument(
        "--subject_ref_weight_decay",
        type=float,
        default=None,
        help=(
            "Optional weight decay for the subject_ref_ optimizer group. "
            "Defaults to --weight_decay when unset."
        ),
    )
    parser.add_argument(
        "--mosaic_fuse_block_size",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "Spatial block size used by frustum mosaic fuse. 1 (default) is "
            "the per-L-cell forward-warp; 2 is the patch_embedding-aligned "
            "2x2-block fuse. Default is 1 to match the validation path "
            "(query_hits_mode_new fuses at block-1), keeping train/inference "
            "mosaic construction consistent. Block-2 mode runs the intra- and "
            "cross-candidate z-buffer at P-patch granularity and copies the "
            "winning source 2x2 slice as a single atomic unit (patch-atomic, "
            "so it aligns exactly with the patch-level hole detection/drop), "
            "removing the intra-token jitter that block-1 can introduce when "
            "two adjacent latent cells of the same target patch came from "
            "different source frames."
        ),
    )
    parser.add_argument(
        "--mosaic_return_source_frame_ids",
        default=False,
        action="store_true",
        help=(
            "Ask the frustum mosaic fuse pipeline to surface a per-P-patch "
            "(H_lat//2, W_lat//2) int64 source frame id grid as diagnostic "
            "metadata under data['ql_source_frame_ids']. This does not "
            "alter PROPE camera lookup (mosaic[i] always reuses noisy[i]'s "
            "camera) and is intended for offline analysis and as the "
            "ingredient that an opt-in per-token PROPE experiment would "
            "consume if turned on later."
        ),
    )
    parser.add_argument(
        "--mosaic_fuse_mode",
        type=str,
        default="fill_stop_zbuffer",
        action=_MosaicFuseModeAction,
        choices=[
            "masked_mean",
            "fill_stop",
            "fill_stop_zbuffer",
            "recent_first_smooth",
            RECENT_FIRST_ZBUFFER_MODE,
        ],
        help=(
            "Frustum mosaic fuse mode used at VALIDATION time only. Default "
            "fill_stop preserves existing validation behaviour; "
            "recent_first_smooth enables v40. Training-time fuse mode is "
            "controlled separately by --mosaic_fuse_mode_train so val can "
            "stay deterministic while train uses the random sampler."
        ),
    )
    parser.add_argument(
        "--mosaic_fuse_mode_train",
        type=str,
        default="random",
        action=_MosaicFuseModeAction,
        choices=[
            "masked_mean",
            "fill_stop",
            "fill_stop_zbuffer",
            "recent_first_smooth",
            RECENT_FIRST_ZBUFFER_MODE,
            "random",
        ],
        help=(
            "Frustum mosaic fuse mode used at TRAINING time (both dataset "
            "preparation and trainer-side materialize). Default 'random' "
            "picks uniformly between fill_stop_zbuffer and "
            "recent_first_smooth per materialize call; the dataset prepares "
            "the superset of inputs (recent_first_smooth candidate budget "
            "and RGB memory frames when --recent_first_photo_refine is "
            "set) so either resolved mode is servable. Validation is not "
            "affected and continues to use --mosaic_fuse_mode."
        ),
    )
    parser.add_argument(
        "--recent_first_photo_refine",
        default=False,
        action="store_true",
        help="Enable v40 photometric border refinement (requires RGB memory frames).",
    )
    parser.add_argument(
        "--no_recent_first_dedupe_tiles",
        dest="recent_first_dedupe_tiles",
        default=True,
        action="store_false",
        help="Disable v40 recent-frame source tile dedupe.",
    )
    parser.add_argument(
        "--candidates_per_query_group_val",
        type=int,
        default=5,
        help=(
            "Candidate budget per query group used at VALIDATION time "
            "(validation rollout's `query_hits_mode_new` and its inline "
            "`recent_first_cfg.warn_min_candidates`). Renamed from the "
            "previous --recent_first_candidates_per_query_group; "
            "validation-side semantics are unchanged. Training-time budget "
            "is controlled separately by --candidates_per_query_group_train."
        ),
    )
    parser.add_argument(
        "--candidates_per_query_group_train",
        type=int,
        default=5,
        help=(
            "Candidate budget per query group used at TRAINING time. This "
            "is the SOLE training-side budget: it is applied unconditionally "
            "in the dataset's `select_candidates` (regardless of the "
            "resolved fuse mode) and in the trainer's "
            "`recent_first_cfg.warn_min_candidates`. Default 20 matches the "
            "previous recent_first default so the more conditioning-heavy "
            "mode is preserved out of the box; lower it (e.g. 5) to save "
            "IO / memory in fill_stop training. Does not affect validation."
        ),
    )
    parser.add_argument(
        "--validation_dynamic_object_filter",
        default=False,
        action="store_true",
        help=(
            "Validation-only: run YOLO segmentation on rollout memory frames "
            "and exclude dynamic-object source latent pixels before frustum fuse."
        ),
    )
    parser.add_argument(
        "--validation_dynamic_object_yolo_model",
        type=str,
        default="",
        help="YOLO segmentation model path used by --validation_dynamic_object_filter.",
    )
    parser.add_argument(
        "--validation_dynamic_object_classes",
        type=str,
        default="person,bicycle,car,motorcycle,bus,truck,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe",
        help="Comma-separated YOLO class names treated as dynamic objects.",
    )
    parser.add_argument(
        "--validation_dynamic_object_conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold for validation dynamic-object filtering.",
    )
    parser.add_argument(
        "--validation_dynamic_object_imgsz",
        type=int,
        default=960,
        help="YOLO inference image size for validation dynamic-object filtering.",
    )
    parser.add_argument(
        "--validation_dynamic_object_device",
        type=str,
        default="",
        help=(
            "YOLO device for validation dynamic-object filtering. Use 'gpu' or leave "
            "empty to run YOLO in a rank-local masked subprocess on worker cuda:0; "
            "use 'cpu' to keep YOLO on CPU."
        ),
    )
    parser.add_argument(
        "--validation_dynamic_object_debug_device",
        default=False,
        action="store_true",
        help=(
            "Print YOLO requested/current/model device before and after "
            "Ultralytics track() calls for multi-GPU debugging."
        ),
    )
    parser.add_argument(
        "--validation_dynamic_object_mask_dilate_latent",
        type=int,
        default=0,
        help="Dilate dynamic-object masks by this radius on the latent grid.",
    )
    parser.add_argument(
        "--validation_dynamic_object_temporal_dilate_radius",
        type=int,
        default=0,
        help=(
            "Temporally OR dynamic-object masks into neighboring frames by this "
            "radius before latent downsampling. 1 applies t masks to t-1/t/t+1."
        ),
    )
    parser.add_argument(
        "--validation_dynamic_object_track_gap_fill_max_gap",
        type=int,
        default=0,
        help=(
            "Use YOLO tracking ids to fill same-track missing spans up to this "
            "many frames with interpolated bbox masks before temporal dilation."
        ),
    )
    parser.add_argument(
        "--validation_dynamic_object_tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config used by validation dynamic-object filtering.",
    )
    parser.add_argument(
        "--validation_dynamic_object_save_preview",
        default=False,
        action="store_true",
        help="Save per-section RGB overlays of dynamic-object latent masks.",
    )
    parser.add_argument(
        "--dynamic_object_mask_filter",
        default=False,
        action="store_true",
        help=(
            "Training video-flow only: read <scene>/objects/dynamic_masks.jsonl "
            "and exclude dynamic-object source latent cells from mosaic memory. "
            "Disabled by default, preserving datasets without object masks."
        ),
    )
    parser.add_argument(
        "--dynamic_object_mask_subdir",
        type=str,
        default="objects",
        help="Scene-relative directory containing dynamic object mask jsonl.",
    )
    parser.add_argument(
        "--dynamic_object_mask_filename",
        type=str,
        default="dynamic_masks.jsonl",
        help="Filename under --dynamic_object_mask_subdir used by training.",
    )
    parser.add_argument(
        "--dynamic_object_mask_temporal_dilate_radius",
        type=int,
        default=0,
        help="Training mask temporal dilation radius in frame units.",
    )
    parser.add_argument(
        "--dynamic_object_mask_dilate_latent",
        type=int,
        default=0,
        help="Training mask dilation radius on the latent grid.",
    )
    parser.add_argument(
        "--dynamic_object_mask_bbox_fallback",
        dest="dynamic_object_mask_use_bbox_fallback",
        default=False,
        action="store_true",
        help=(
            "Use bbox rectangles when dynamic mask rows are missing segmentation "
            "RLE. Off by default to avoid over-masking static background."
        ),
    )
    parser.add_argument(
        "--dynamic_object_mask_no_bbox_fallback",
        dest="dynamic_object_mask_use_bbox_fallback",
        action="store_false",
        help=(
            "Keep bbox fallback disabled for dynamic mask rows missing "
            "segmentation RLE. This is the default."
        ),
    )
    parser.add_argument(
        "--recent_first_geometry_device",
        choices=["cpu", "cuda", "gpu", "auto"],
        default="cpu",
        help=(
            "Device for v40 recent_first_smooth candidate geometry. "
            "Use 'cuda' or 'auto' to enable the optional GPU path."
        ),
    )
    parser.add_argument(
        "--recent_first_backend",
        choices=["legacy", "gpu_fast"],
        default="legacy",
        help=(
            "Backend for v40 recent_first_smooth. Default legacy preserves "
            "existing behaviour; gpu_fast enables the v40.5 batch path."
        ),
    )
    parser.add_argument(
        "--recent_first_warp_dtype",
        choices=["none", "float16", "bfloat16"],
        default="none",
        help="Optional matmul dtype override for the gpu_fast forward-warp stage.",
    )
    parser.add_argument(
        "--recent_first_enable_tf32",
        default=False,
        action="store_true",
        help="Allow TF32 matmul inside the opt-in gpu_fast recent_first backend.",
    )
    parser.add_argument(
        "--recent_first_zbuffer",
        default=False,
        action="store_true",
        help=(
            "Allow non-recent recent_first_smooth candidates to overwrite "
            "recent-frame pixels when their query-space depth is nearer. "
            "Default keeps legacy recent-first fill semantics."
        ),
    )
    parser.add_argument(
        "--recent_first_n_candidates_keyframe",
        type=int,
        default=0,
        help=(
            "Opt-in number of pose-motion keyframe candidates to add for "
            "recent_first_smooth. Default 0 keeps existing candidate semantics."
        ),
    )
    parser.add_argument(
        "--recent_first_keyframe_neighbour_window",
        type=int,
        default=2,
        help="Temporal latent-neighbour window added around opt-in keyframe candidates.",
    )
    parser.add_argument(
        "--mosaic_selection_mode",
        choices=["projection_iou", "pose_nearest", "pose_pool_temporal_earliest"],
        default="pose_nearest",
        help=(
            "Dataset-side FrustumHandler.select_candidates mode. "
            "Use pose_nearest to skip projection-IoU scoring; "
            "pose_pool_temporal_earliest keeps a near-pose pool then picks "
            "earlier frame ids."
        ),
    )
    parser.add_argument(
        "--mosaic_candidate_nms_mode",
        choices=["none", "temporal", "pose", "projection_iou", "coverage"],
        default="none",
        help=(
            "Optional dataset-side candidate NMS / picker. projection_iou and "
            "coverage both require mosaic_selection_mode=projection_iou; "
            "pose_nearest supports temporal/pose. 'coverage' is a greedy "
            "UNION-coverage picker: it adds the candidate contributing the "
            "most new query-canvas cells each step (fixes the redundant, "
            "mutually-overlapping picks of pose/individual-IoU selectors)."
        ),
    )
    parser.add_argument(
        "--mosaic_candidate_nms_projection_iou_threshold",
        type=float,
        default=0.7,
        help="Projection-bbox IoU threshold for mosaic_candidate_nms_mode=projection_iou.",
    )
    parser.add_argument(
        "--mosaic_candidate_nms_min_temporal_gap",
        type=int,
        default=0,
        help="Minimum frame-id gap for mosaic_candidate_nms_mode=temporal.",
    )
    parser.add_argument(
        "--mosaic_candidate_nms_pose_distance_threshold",
        type=float,
        default=0.0,
        help="Camera-centre distance threshold for mosaic_candidate_nms_mode=pose.",
    )
    parser.add_argument(
        "--mosaic_candidate_nms_pool_multiplier",
        type=float,
        default=2.0,
        help="Coarse candidate pool multiplier used when candidate NMS is enabled.",
    )
    parser.add_argument(
        "--mosaic_coverage_grid_downsample",
        type=int,
        default=4,
        help=(
            "Coarse-grid / depth downsample factor for the vectorised "
            "'coverage' picker (selection only; the fuse still runs at full "
            "latent resolution). Coverage masks are rasterised at "
            "(H_lat//d, W_lat//d). Larger d = faster, slightly lower coverage. "
            "Only used by mosaic_candidate_nms_mode=coverage."
        ),
    )
    parser.add_argument(
        "--mosaic_coverage_pool_stride",
        type=int,
        default=2,
        help=(
            "Memory-pool stride for the vectorised 'coverage' picker: the "
            "greedy considers every Nth memory frame as a candidate. 1 = full "
            "pool (best coverage, slowest); 2-4 trade a little coverage for "
            "speed. Only used by mosaic_candidate_nms_mode=coverage."
        ),
    )
    parser.add_argument(
        "--train_mosaic_intrinsics_mode",
        choices=MOSAIC_INTRINSICS_MODES,
        default="per_frame",
        help=(
            "Training frustum intrinsics mode. per_frame uses each frame's K; "
            "episode_mean averages K over the episode and reuses it for all "
            "frames; first_frame reuses frame 0's K for every frame."
        ),
    )
    parser.add_argument(
        "--validation_mosaic_intrinsics_mode",
        choices=MOSAIC_INTRINSICS_MODES,
        default="episode_mean",
        help=(
            "Validation frustum intrinsics mode. episode_mean preserves the "
            "legacy validation single-K rollout; first_frame reuses frame 0's "
            "K for the validation dataset and FrustumHandler registration."
        ),
    )
    parser.add_argument(
        "--mosaic_intrinsics_mode",
        dest="mosaic_intrinsics_mode",
        choices=MOSAIC_INTRINSICS_MODES,
        default=SUPPRESS,
        help=SUPPRESS,
    )
    parser.add_argument(
        "--mosaic_intrinsics_first_frame_only",
        dest="mosaic_intrinsics_first_frame_only",
        default=SUPPRESS,
        action="store_true",
        help=SUPPRESS,
    )
    parser.add_argument(
        "--validation_frustum_fixed_first_frame_intrinsics",
        dest="validation_frustum_fixed_first_frame_intrinsics",
        default=SUPPRESS,
        action="store_true",
        help=SUPPRESS,
    )
    parser.add_argument(
        "--mosaic_query_reference_frame",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help=(
            "1-based frame within each 4-frame query group used as the representative "
            "pose/K for frustum candidate selection and reproject. Default 2."
        ),
    )
    parser.add_argument(
        "--memory_vae_encode_input_frames",
        type=int,
        default=1,
        help=(
            "Number of frames fed to the VAE per cache-miss memory slot. "
            "Must follow the 1+4k rule (1, 5, 9, ...). Default 1 preserves "
            "the legacy single-frame -> single-latent behaviour. When >1, "
            "the single RGB source frame is replicated along the time axis "
            "before encoding and only the last latent step is kept, so the "
            "per-slot memory latent tensor stays (1, C, 1, H_lat, W_lat)."
        ),
    )
    parser.add_argument(
        "--memory_neighbor_window_train",
        dest="memory_neighbor_window_train",
        default=False,
        action="store_true",
        help=(
            "Training-only: for every candidate memory frame, encode a real "
            "5-frame temporal window [fid-2, fid-1, fid, fid+1, fid+2] (clamped "
            "to the clip range) so the VAE yields a denser last latent step, "
            "instead of replicating a single frame. A fast camera-motion judge "
            "over the 5 frames' extrinsics decides per frame: if the max "
            "pairwise translation (meters) or max pairwise rotation (degrees) "
            "exceeds the thresholds below, it falls back to single-frame sparse "
            "encoding. This is an independent channel from "
            "--memory_vae_encode_input_frames (which is left untouched). The "
            "neighbour-window vs single-frame decision is folded into the "
            "memory-latent cache hash so cached entries never collide across "
            "modes. Validation/inference are unaffected."
        ),
    )
    parser.add_argument(
        "--memory_neighbor_window_val",
        dest="memory_neighbor_window_val",
        default=False,
        action="store_true",
        help=(
            "Validation-only: when --force_using_input_extrinics is also enabled, "
            "reuse each generated dense latent for its 4 decoded RGB slots when "
            "those 4 slots have low camera motion; high-motion slots keep the "
            "legacy per-frame VAE re-encode path. Every slot still uses its own "
            "registered camera/depth, so validation source_latents stay 1:1 with "
            "FrustumHandler registered frames."
        ),
    )
    parser.add_argument(
        "--memory_neighbor_window_max_trans_m",
        type=float,
        default=0.5,
        help=(
            "Used with --memory_neighbor_window_train/val. Max allowed pairwise "
            "camera-center translation (in meters) across the train 5-frame "
            "window or validation 4-frame generated-latent group; above this "
            "the frame/group falls back to single-frame sparse encoding. "
            "Tune to your dataset's metric scale (default 0.5 is a placeholder)."
        ),
    )
    parser.add_argument(
        "--memory_neighbor_window_max_rot_deg",
        type=float,
        default=10.0,
        help=(
            "Used with --memory_neighbor_window_train/val. Max allowed pairwise "
            "geodesic rotation (in degrees) across the train 5-frame window or "
            "validation 4-frame generated-latent group; above this the "
            "frame/group falls back to single-frame sparse encoding. "
            "Default 10.0 is a placeholder; tune to your dataset."
        ),
    )
    parser.add_argument(
        "--no_mosaic_prob_train",
        type=float,
        default=0.0,
        help=(
            "Training-only probability in [0, 1] of running a step with NO "
            "mosaic conditioning: queried_latent is zeroed, mosaic pipeline "
            "kwargs are dropped, and memory VAE encode / frustum fuse are "
            "skipped for that step. Default 0 keeps the legacy always-mosaic "
            "behaviour. Validation and inference are unaffected."
        ),
    )
    parser.add_argument(
        "--clean_latent_noise_train",
        default=False,
        action="store_true",
        help=(
            "Training-only switch (default off). When set, each training "
            "step independently perturbs ONLY the most recent clean latent "
            "frame (the last frame of first_frame_latents; mosaic latents "
            "and older clean-history/context latents are untouched) with "
            "probability --clean_latent_noise_prob_train, adding Gaussian "
            "noise N(0, magnitude^2) where magnitude is "
            "--clean_latent_noise_magnitude_train. Validation and inference "
            "are unaffected."
        ),
    )
    parser.add_argument(
        "--clean_latent_noise_prob_train",
        type=float,
        default=0.2,
        help=(
            "Probability in [0, 1] that a training step adds Gaussian noise "
            "to the most recent clean latent frame. Only consulted when "
            "--clean_latent_noise_train is set. Default 0.2."
        ),
    )
    parser.add_argument(
        "--clean_latent_noise_magnitude_train",
        type=float,
        default=0.03,
        help=(
            "Std of the Gaussian noise added to the most recent clean "
            "latent frame: noise ~ N(0, 1) * magnitude. Only consulted "
            "when --clean_latent_noise_train is set. Default 0.03."
        ),
    )
    parser.add_argument(
        "--svi_num_grids",
        type=int,
        default=50,
        help="SVI two-bank timestep bucket count. Only used by task=sft:svi.",
    )
    parser.add_argument(
        "--svi_error_buffer_k",
        type=int,
        default=4,
        help="Max samples kept per SVI bank bucket. Only used by task=sft:svi.",
    )
    parser.add_argument(
        "--svi_buffer_replacement_strategy",
        type=str,
        default="random",
        choices=["random", "fifo", "l2_batch"],
        help="SVI buffer replacement strategy once a bucket is full.",
    )
    parser.add_argument(
        "--svi_save_error_buffer",
        dest="svi_save_error_buffer",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--svi_no_save_error_buffer",
        dest="svi_save_error_buffer",
        action="store_false",
    )
    parser.add_argument(
        "--svi_inject_noise",
        dest="svi_inject_noise",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--svi_no_inject_noise", dest="svi_inject_noise", action="store_false"
    )
    parser.add_argument("--svi_noise_prob", type=float, default=0.5)
    parser.add_argument("--svi_noise_scale", type=float, default=1.0)
    parser.add_argument("--svi_buffer_warmup_steps", type=int, default=50)
    parser.add_argument(
        "--svi_inject_latent",
        dest="svi_inject_latent",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--svi_no_inject_latent", dest="svi_inject_latent", action="store_false"
    )
    parser.add_argument("--svi_latent_prob", type=float, default=0.5)
    parser.add_argument("--svi_latent_scale", type=float, default=1.0)
    parser.add_argument(
        "--svi_inject_clean_context",
        dest="svi_inject_clean_context",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--svi_no_inject_clean_context",
        dest="svi_inject_clean_context",
        action="store_false",
    )
    parser.add_argument("--svi_clean_context_prob", type=float, default=0.5)
    parser.add_argument("--svi_clean_context_scale", type=float, default=1.0)
    parser.add_argument("--svi_clean_context_frames", type=int, default=1)
    parser.add_argument("--svi_clean_prob", type=float, default=0.5)
    parser.add_argument("--svi_clean_buffer_update_prob", type=float, default=0.1)
    parser.add_argument(
        "--mosaic_latent_noise_train",
        default=False,
        action="store_true",
        help=(
            "Training-only switch (default off), parallel to "
            "--clean_latent_noise_train but for the mosaic/queried latent. "
            "When set, each step independently perturbs ONLY the valid "
            "(non-hole) mosaic latents with probability "
            "--mosaic_latent_noise_prob_train, adding Gaussian noise "
            "N(0, magnitude^2) where magnitude is "
            "--mosaic_latent_noise_magnitude_train. Holes stay exactly zero "
            "(patch-level hole detection unaffected). This narrows the "
            "train/inference gap on the mosaic conditioning channel, where "
            "training warps from GT frames but inference warps from "
            "model-generated frames. Validation and inference are unaffected."
        ),
    )
    parser.add_argument(
        "--mosaic_latent_noise_prob_train",
        type=float,
        default=0.2,
        help=(
            "Probability in [0, 1] that a training step adds Gaussian noise "
            "to the valid mosaic latents. Only consulted when "
            "--mosaic_latent_noise_train is set. Default 0.2."
        ),
    )
    parser.add_argument(
        "--mosaic_latent_noise_magnitude_train",
        type=float,
        default=0.03,
        help=(
            "Std of the Gaussian noise added to the valid mosaic latents: "
            "noise ~ N(0, 1) * magnitude. Only consulted when "
            "--mosaic_latent_noise_train is set. Default 0.03."
        ),
    )
    parser.add_argument(
        "--context_latent_noise_train",
        default=False,
        action="store_true",
        help=(
            "Training-only switch (default off), parallel to "
            "--clean_latent_noise_train but for the CONTEXT latents (every "
            "clean history latent BEFORE the anchor). Only fires when context "
            "blocks are present this step (train_context_count_max > 0 and the "
            "sampled context_count > 0). With probability "
            "--context_latent_noise_prob_train, adds Gaussian noise "
            "N(0, magnitude^2) where magnitude is "
            "--context_latent_noise_magnitude_train. The anchor (latest clean "
            "latent) is left to --clean_latent_noise_train. Validation and "
            "inference are unaffected."
        ),
    )
    parser.add_argument(
        "--context_latent_noise_prob_train",
        type=float,
        default=0.2,
        help=(
            "Probability in [0, 1] that a training step adds Gaussian noise "
            "to the context latents. Only consulted when "
            "--context_latent_noise_train is set. Default 0.2."
        ),
    )
    parser.add_argument(
        "--context_latent_noise_magnitude_train",
        type=float,
        default=0.03,
        help=(
            "Std of the Gaussian noise added to the context latents: "
            "noise ~ N(0, 1) * magnitude. Only consulted when "
            "--context_latent_noise_train is set. Default 0.03."
        ),
    )
    parser.add_argument(
        "--train_prompt_dropout_prob",
        type=float,
        default=0.0,
        help=(
            "Training-only probability in [0, 1] of replacing the current "
            "sample prompt with an empty string before text encoding. Default "
            "0 disables prompt dropout. Preview renders dropped prompts as "
            f"{TRAIN_PROMPT_DROPOUT_PREVIEW_TEXT!r}."
        ),
    )
    parser.add_argument(
        "--force_using_input_extrinics",
        dest="force_using_input_extrinics",
        default=False,
        action="store_true",
        help=(
            "Validation only: at each FrustumHandler.register_source_sequence "
            "call, hand the handler the dataset-supplied (intrinsics, "
            "extrinsics) that drove the previous generation step instead of "
            "letting DA3 re-recover them from the rendered RGB. Keeps the "
            "memory trajectory aligned with the model's expected camera "
            "frame and avoids drift from DA3's per-section re-inference. "
            "Default off, which preserves the legacy DA3-recover-everything "
            "behaviour."
        ),
    )
    parser.add_argument(
        "--registration_estimator",
        dest="registration_estimator",
        type=str,
        default="da3",
        choices=["da3", "vggt_omega_metric"],
        help=(
            "Validation only: which depth/pose estimator the FrustumHandler "
            "uses to register newly generated frames. 'da3' (default) uses the "
            "DepthAnything-3 nested metric model. 'vggt_omega_metric' uses "
            "VGGT-Omega depth/pose anchored to DA3's metric branch for the "
            "global scale (loads VGGT-Omega + the DA3 metric ViT-L branch). "
            "With --force_using_input_extrinics only the estimated metric depth "
            "is consumed; otherwise its pose+intrinsics are used too."
        ),
    )
    parser.add_argument(
        "--vggt_omega_ckpt",
        dest="vggt_omega_ckpt",
        type=str,
        default="runjia.qian/huggingface_models/VGGT-Omega/vggt_omega_1b_512.pt",
        help=(
            "Checkpoint path for --registration_estimator vggt_omega_metric. "
            "Overridable via $VGGT_OMEGA_CKPT. The DA3 metric branch reuses the "
            "same weights as the standalone DA3 path ($DA3_MODEL_PATH / "
            "$DA3_MODEL_ID)."
        ),
    )
    parser.add_argument(
        "--vggt_omega_image_resolution",
        dest="vggt_omega_image_resolution",
        type=int,
        default=512,
        help=(
            "VGGT-Omega preprocessing resolution; must match the checkpoint "
            "(the default vggt_omega_1b_512.pt expects 512)."
        ),
    )
    parser.add_argument(
        "--vggt_omega_da3_process_res",
        dest="vggt_omega_da3_process_res",
        type=int,
        default=504,
        help=(
            "DA3 metric-branch processing resolution used when estimating the "
            "global metric scale for --registration_estimator vggt_omega_metric."
        ),
    )
    return parser


def _argparse_dest_names(parser):
    return {
        action.dest
        for action in parser._actions
        if action.dest and action.dest != argparse.SUPPRESS
    }


def _omegaconf_to_container(path):
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "--omegaconf_config requires omegaconf. Install omegaconf or use "
            "plain CLI arguments."
        ) from exc
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def _flatten_train_mosaic_config(payload):
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("OmegaConf config must be a mapping at the top level.")
    merged = {}
    for key, value in payload.items():
        if key in {"train_mosaic", "args"}:
            if not isinstance(value, dict):
                raise ValueError(f"Config section {key!r} must be a mapping.")
            merged.update(value)
        else:
            merged[key] = value
    return merged


def _apply_omegaconf_defaults(parser, config_path):
    payload = _omegaconf_to_container(config_path)
    config = _flatten_train_mosaic_config(payload)
    config = _apply_legacy_intrinsics_mode_config(config)
    valid_dests = _argparse_dest_names(parser)
    unknown = sorted(str(key) for key in config if str(key) not in valid_dests)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in OmegaConf config {config_path}: {unknown}. "
            "Use argparse dest names, e.g. dataset_base_path or run_validation."
        )
    parser.set_defaults(**config)
    return config


def _sanitize_run_arg_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_sanitize_run_arg_value(item) for item in value]
    if isinstance(value, set):
        return [_sanitize_run_arg_value(item) for item in sorted(value, key=str)]
    if isinstance(value, dict):
        return {
            str(key): _sanitize_run_arg_value(value[key])
            for key in sorted(value.keys(), key=str)
        }
    return str(value)


def _run_args_to_yaml_payload(args):
    return {
        str(key): _sanitize_run_arg_value(value)
        for key, value in sorted(vars(args).items(), key=lambda item: str(item[0]))
    }


def _dump_run_args_yaml(args, path):
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = _run_args_to_yaml_payload(args)
    try:
        from omegaconf import OmegaConf

        OmegaConf.save(config=OmegaConf.create(payload), f=str(path))
    except ImportError:
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=True, allow_unicode=True)
    return path


def parse_pipeline_args(argv=None):
    from .prompting import _normalize_train_prompt_dropout_prob

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--omegaconf_config",
        "--config",
        dest="omegaconf_config",
        type=str,
        default=None,
    )
    config_args, _ = config_parser.parse_known_args(argv)
    parser = wan_mosaic_parser()
    if config_args.omegaconf_config:
        config_defaults = _apply_omegaconf_defaults(
            parser,
            config_args.omegaconf_config,
        )
        for action in parser._actions:
            if action.dest in config_defaults and action.dest != "omegaconf_config":
                action.required = False
    cli_parser = argparse.ArgumentParser(add_help=False)
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if not action.option_strings:
            continue
        kwargs = {
            "dest": action.dest,
            "default": SUPPRESS,
        }
        if isinstance(action, argparse._StoreTrueAction):
            kwargs["action"] = "store_true"
        elif isinstance(action, argparse._StoreFalseAction):
            kwargs["action"] = "store_false"
        elif isinstance(action, argparse._StoreConstAction):
            kwargs["action"] = "store_const"
            kwargs["const"] = action.const
        elif isinstance(action, argparse._AppendAction):
            kwargs["action"] = "append"
            kwargs["type"] = action.type
            kwargs["choices"] = action.choices
        else:
            kwargs["type"] = action.type
            kwargs["choices"] = action.choices
            kwargs["nargs"] = action.nargs
            kwargs["const"] = action.const
        cli_parser.add_argument(*action.option_strings, **kwargs)
    cli_args, _ = cli_parser.parse_known_args(argv)
    explicit_cli = vars(cli_args)
    args = parser.parse_args(argv)
    for key, value in explicit_cli.items():
        setattr(args, key, value)
    args.train_prompt_dropout_prob = _normalize_train_prompt_dropout_prob(
        args.train_prompt_dropout_prob
    )
    if bool(getattr(args, "prope_disable_native_rope", False)):
        assert (
            int(getattr(args, "prope_attention_interval", 1)) > 1
        ), "prope_disable_native_rope requires prope_attention_interval > 1."
    if bool(getattr(args, "prope_disable_t_rope", False)):
        assert (
            int(getattr(args, "prope_attention_interval", 1)) > 1
        ), "prope_disable_t_rope requires prope_attention_interval > 1."
        assert not bool(getattr(args, "prope_disable_native_rope", False)), (
            "prope_disable_t_rope and prope_disable_native_rope are "
            "mutually exclusive."
        )
    if str(getattr(args, "prope_camera_layout", "full")) != "full":
        assert bool(getattr(args, "prope_disable_t_rope", False)), (
            "prope_camera_layout="
            f"{getattr(args, 'prope_camera_layout')!r} requires "
            "prope_disable_t_rope so the camera block lands in the "
            "rope-free t-band."
        )
    args = _resolve_train_anchor_context_args(args)
    args = _apply_context_alias(
        _apply_recent_first_zbuffer_alias(_apply_intrinsics_mode_aliases(args))
    )
    # per_3latentframe (train and/or validation context selection) is only valid
    # when n_gen (== latent_window_size) is divisible by 3, and -- on the
    # validation side -- when the pool-history path is on with a single decoded
    # anchor. Enforce ALL of this at config time so a bad window or a
    # validation-only misconfig fails FAST instead of training a full epoch and
    # then hard-erroring at the first validation boundary (run_mosaic_validation
    # has no except, so that crash kills the job). Runs AFTER _apply_context_alias
    # so the resolved validation mode is visible.
    _train_per3 = (
        str(getattr(args, "train_context_selection", "")) == "per_3latentframe"
    )
    _val_per3 = (
        str(getattr(args, "validation_clean_latent_pool_history_selection_mode", ""))
        == "per_3latentframe"
    )
    if _train_per3 or _val_per3:
        _lws = max(0, int(getattr(args, "latent_window_size", 0) or 0))
        if _lws % 3 != 0:
            raise ValueError(
                "context selection 'per_3latentframe' requires the generating-"
                f"latent count (n_gen == latent_window_size == {_lws}) to be "
                "divisible by 3; set --latent_window_size to a multiple of 3."
            )
    if _val_per3:
        if not bool(getattr(args, "validation_clean_latent_pool_history", False)):
            raise ValueError(
                "validation selection_mode='per_3latentframe' requires "
                "--validation_clean_latent_pool_history to be enabled."
            )
        if bool(getattr(args, "validation_use_decoded_first_image", False)):
            raise ValueError(
                "validation selection_mode='per_3latentframe' is incompatible "
                "with --validation_use_decoded_first_image: it emits "
                "latent_window_size//3 context latents, but the decoded PIL image "
                "path provides only one clean anchor latent."
            )
    # When the per-step random schedule is on (real usage), the legacy
    # interval / no-mosaic knobs are dead: the schedule sets the per-step
    # interval and the nomosaic decision, and mosaic_no_mosaic_sync_mode is
    # never read. Warn loudly if any is left at a non-default so configs do
    # not pretend to control something they don't.
    if bool(getattr(args, "mosaic_train_random_schedule", False)):
        ignored = []
        if float(getattr(args, "no_mosaic_prob_train", 0.0) or 0.0) > 0.0:
            ignored.append("no_mosaic_prob_train")
        if int(getattr(args, "mosaic_interval", 1) or 1) != 1:
            ignored.append("mosaic_interval")
        if str(getattr(args, "mosaic_no_mosaic_sync_mode", "none") or "none") != "none":
            ignored.append("mosaic_no_mosaic_sync_mode")
        if ignored:
            _config_deprecation_note(
                "mosaic_train_random_schedule=true governs all mosaic "
                "scheduling; ignoring " + ", ".join(ignored) + " (interval and "
                "the no-mosaic ratio come from mosaic_train_schedule).",
            )
    return args


def _config_deprecation_note(message, *, loud=False):
    if loud:
        bar = "!" * 78
        print(f"\n{bar}\n[config] WARNING: {message}\n{bar}\n", flush=True)
    else:
        print(f"[config] NOTE: {message}", flush=True)


def _resolve_train_anchor_context_args(args):
    """Fill the v2 anchor/context plan args from deprecated ones when unset.

    Canonical fields after this call (always present and valid):
    ``train_anchor_block_prob`` (float), ``train_block_anchor_frames``
    (4k+1 int), ``train_context_count_max`` (int >= 0),
    ``train_context_selection`` (str). Mapping from deprecated args:
    block_prob = 1 - a/(a+b) of --vae_clean_context_mode_ratio;
    context_max = max(0, --train_clean_latent_history_count_max - 1).
    --train_clean_history_legacy_plan is REMOVED: warn and ignore.
    """
    legacy_count_max = int(
        getattr(args, "train_clean_latent_history_count_max", 0) or 0
    )
    if getattr(args, "train_anchor_block_prob", None) is None:
        ratio = str(getattr(args, "vae_clean_context_mode_ratio", "2:8") or "2:8")
        p_single = _parse_train_clean_history_ac_ratio(ratio)
        args.train_anchor_block_prob = 1.0 - p_single
        if legacy_count_max <= 0:
            # The dangerous silent flip: old count_max==0 args meant "plan
            # OFF" (dataset-side anchor draw, or pure single anchors). That
            # state no longer exists -- the v2 plan is always on and would
            # now train block anchors with the derived probability.
            _config_deprecation_note(
                "this args set predates the v2 anchor/context plan "
                "(train_clean_latent_history_count_max=0, no "
                "--train_anchor_block_prob). The old 'plan OFF' semantics "
                "no longer exist: the v2 plan is ALWAYS on, and "
                f"vae_clean_context_mode_ratio={ratio!r} derives "
                f"P(block anchor)={args.train_anchor_block_prob:.2f} with "
                "trainer-side joint-encode warm-up (NOT the old dataset-"
                "side draw). Anchor and G distributions DIFFER from the "
                "original run. Pass --train_anchor_block_prob (and 0.0 to "
                "train pure single anchors) to silence this warning.",
                loud=True,
            )
        else:
            _config_deprecation_note(
                "train_anchor_block_prob derived from deprecated "
                f"vae_clean_context_mode_ratio={ratio!r} -> "
                f"{args.train_anchor_block_prob:.3f} (prefer setting "
                "--train_anchor_block_prob directly)."
            )
    args.train_anchor_block_prob = min(
        1.0, max(0.0, float(args.train_anchor_block_prob))
    )
    if getattr(args, "train_context_count_max", None) is None:
        args.train_context_count_max = max(0, legacy_count_max - 1)
        if legacy_count_max > 1:
            _config_deprecation_note(
                "train_context_count_max derived from deprecated "
                f"train_clean_latent_history_count_max={legacy_count_max} -> "
                f"{args.train_context_count_max} (prefer setting "
                "--train_context_count_max directly)."
            )
    args.train_context_count_max = max(0, int(args.train_context_count_max))
    raw_frames = getattr(args, "train_block_anchor_frames", None)
    args.train_block_anchor_frames = _validate_block_anchor_frames(
        13 if raw_frames is None else raw_frames
    )
    args.train_context_selection = _validate_train_context_selection(
        getattr(args, "train_context_selection", "oldest_spread")
    )
    if args.train_context_selection == "per_3latentframe":
        # per_3 derives the emitted context count from n_gen//3 at runtime (n_gen
        # is the generating-latent count == latent_window_size for BOTH train and
        # validation: the joint clean+noisy encode of (1 + 4*latent_window_size)
        # frames yields latent_window_size+1 latents -- slot 0 the clean anchor,
        # slots 1: the noisy targets), so the random count knob is superseded --
        # but train_context_count_max STILL controls the dataset's candidate-pool
        # G reservation, so warn if it is too small to supply n_gen//3 covering
        # blocks. (Divisibility of n_gen by 3 is enforced as a HARD error for
        # train and validation together below, once the context aliases have
        # been resolved.)
        n_gen = max(0, int(getattr(args, "latent_window_size", 0) or 0))
        need = n_gen // 3
        if int(args.train_context_count_max) < need:
            _config_deprecation_note(
                f"train_context_selection='per_3latentframe' emits n_gen//3={need}"
                f" context blocks but train_context_count_max="
                f"{args.train_context_count_max} reserves fewer candidate blocks; "
                f"raise --train_context_count_max to >= {need} or per_3 will "
                "fall back to recent blocks for the shortfall."
            )
    if bool(getattr(args, "train_clean_history_legacy_plan", False)):
        _config_deprecation_note(
            "--train_clean_history_legacy_plan is removed and IGNORED; the "
            "v2 anchor/context plan is always used.",
            loud=True,
        )
        args.train_clean_history_legacy_plan = False
    if bool(getattr(args, "vipe_prompt_segment_format", False)) and not bool(
        getattr(args, "align_generating_to_prompt_segments", False)
    ):
        _config_deprecation_note(
            "--vipe_prompt_segment_format no longer implies "
            "--align_generating_to_prompt_segments; prompt-segment window "
            "alignment stays OFF unless you pass the align flag."
        )
    return args


_DATASET_TYPE_REGISTRY = {
    "da3_video": DA3MosaicVideoDataset,
    "da3_video_subject_ref": SubjectRefMemoryDA3MosaicVideoDataset,
}
