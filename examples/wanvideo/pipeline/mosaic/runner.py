"""Inference runner for Matrix-Game-3.5.

This release ships inference only: the runner prepares the model on the
accelerator and drives the generation/validation rollout. The original
training loop (optimizer, epochs, checkpointing) is not part of this
repository.
"""

import glob
import json
import os
import re

from .validation import run_mosaic_validation

_EPOCH_CKPT_RE = re.compile(r"^epoch-(\d+)\.safetensors$")


def _latest_epoch_checkpoint(log_dir):
    best = None
    for path in glob.glob(os.path.join(log_dir, "epoch-*.safetensors")):
        match = _EPOCH_CKPT_RE.match(os.path.basename(path))
        if match is None:
            continue
        epoch_id = int(match.group(1))
        if best is None or epoch_id > best[0]:
            best = (epoch_id, path)
    return None if best is None else best[1]


def _maybe_resume_from_log_dir(args):
    """Resolve ``args.resume_from`` into a concrete ``trained_dit`` weight.

    Kept from the original runner so a previous run directory can be used
    as the checkpoint source. Mutates ``args``; returns
    ``(start_epoch, dataset_pass_id)``.
    """
    start_epoch = 0
    pass_id = int(args.dataset_pass_id) if args.dataset_pass_id is not None else 0
    if args.resume_from:
        state_path = os.path.join(args.resume_from, "train_state.json")
        if os.path.isfile(state_path):
            with open(state_path, "r") as f:
                ts = json.load(f) or {}
            start_epoch = int(ts.get("next_epoch", int(ts.get("epoch_id", -1)) + 1))
            if args.dataset_pass_id is None:
                pass_id = int(ts.get("dataset_pass_id", 0)) + 1
            args._resume_step_count = int(ts.get("global_step", 0) or 0)
        if not args.trained_dit:
            latest_ckpt = _latest_epoch_checkpoint(args.resume_from)
            if latest_ckpt:
                args.trained_dit = latest_ckpt
                print(f"[resume] trained_dit auto-set to {args.trained_dit}")
    args.start_epoch = int(start_epoch)
    args.dataset_pass_id = int(pass_id)
    return start_epoch, pass_id


def run_mosaic_inference_task(
    accelerator,
    dataset,
    model,
    model_logger,
    log_dir,
    args=None,
    val_dataset=None,
):
    """Prepare the model and run one generation/validation rollout."""
    if int(getattr(args, "num_epochs", 0) or 0) > 0:
        raise NotImplementedError(
            "Training is not included in this inference release "
            "(num_epochs must be 0)."
        )
    model = accelerator.prepare(model)
    if getattr(args, "run_sanity_check", False) or getattr(
        args, "run_validation", False
    ):
        run_mosaic_validation(
            accelerator, val_dataset, model, log_dir, args=args, epoch_id=-1
        )
    else:
        print(
            "[runner] nothing to do: enable run_sanity_check/run_validation "
            "to generate."
        )
