#!/usr/bin/env python3
"""Distilled causal inference: explicit inputs + checkpoint -> result.mp4."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import infer as base_infer
from distilled_config import load_inference_config


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs/infer_distilled.yaml"
DEFAULT_ACCELERATE_CONFIG = REPO_ROOT / "configs/accelerate_inference_1gpu.yaml"


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", default="")
    prompt.add_argument("--prompt-file", type=Path)
    prompt.add_argument("--caption", type=Path)
    parser.add_argument(
        "--camera-convention", choices=("c2w", "w2c"), default="c2w"
    )
    parser.add_argument("--wan-dir", default="")
    parser.add_argument("--tokenizer-dir", default="")
    parser.add_argument("--da3-dir", default="")
    parser.add_argument("--output", type=Path, default=Path("result.mp4"))
    parser.add_argument(
        "--accelerate-config", type=Path, default=DEFAULT_ACCELERATE_CONFIG
    )
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_cli(args):
    for label, path in (
        ("config", args.config),
        ("checkpoint", args.checkpoint),
        ("image", args.image),
        ("camera", args.camera),
        ("accelerate config", args.accelerate_config),
        ("prompt file", args.prompt_file),
        ("caption", args.caption),
    ):
        if path is not None and not path.is_file():
            raise SystemExit(f"ERROR: {label} does not exist: {path}")
    if args.output.suffix.lower() != ".mp4":
        raise SystemExit("ERROR: --output must be an .mp4 file path")
    load_inference_config(args.config)


def _workspace_args(args, num_blocks):
    return SimpleNamespace(
        person="first",
        image=str(args.image),
        camera=str(args.camera),
        caption=str(args.caption or ""),
        prompt=args.prompt,
        prompt_file=str(args.prompt_file or ""),
        refs="",
        num_blocks=num_blocks,
        camera_convention=args.camera_convention,
    )


def build_command(
    args,
    *,
    workspace,
    dataset_index,
    wan_dir,
    tokenizer_dir,
    tmp_root,
    output,
):
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(args.accelerate_config.resolve()),
        str(REPO_ROOT / "tools/run_distilled_inference.py"),
        "--config",
        str(args.config.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--dataset-index",
        str(dataset_index),
        "--workspace",
        str(workspace),
        "--wan-dir",
        str(wan_dir),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--memory-cache-dir",
        str(tmp_root / "memory_latents"),
        "--output",
        str(output),
    ]


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_cli(args)
    config = load_inference_config(args.config)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["DA3_MODEL_PATH"] = base_infer.resolve_weight("da3", args.da3_dir)
    env.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    wan_dir = Path(base_infer.resolve_weight("wan", args.wan_dir))
    tokenizer_dir = Path(base_infer.resolve_weight("tokenizer", args.tokenizer_dir))

    run_id = f"distilled_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    tmp_root = REPO_ROOT / ".cache/infer_runs" / run_id
    workspace = Path(
        base_infer.build_input_workspace(
            _workspace_args(args, config.num_blocks),
            str(tmp_root),
        )
    )
    dataset_index = Path(base_infer.build_index(str(workspace), sys.executable, env))
    command = build_command(
        args,
        workspace=workspace,
        dataset_index=dataset_index,
        wan_dir=wan_dir,
        tokenizer_dir=tokenizer_dir,
        tmp_root=tmp_root,
        output=output,
    )
    print("[infer-distilled] " + " ".join(command), flush=True)
    if args.dry_run:
        print(f"[infer-distilled] prepared workspace: {workspace}")
        return 0

    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode:
        print(
            f"[infer-distilled] failed; workspace kept at {tmp_root}",
            file=sys.stderr,
        )
        return completed.returncode
    if not output.is_file() or output.stat().st_size == 0:
        print(
            f"[infer-distilled] no result video; workspace kept at {tmp_root}",
            file=sys.stderr,
        )
        return 1
    if not args.keep_workspace:
        shutil.rmtree(tmp_root)
    print(f"[infer-distilled] done: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
