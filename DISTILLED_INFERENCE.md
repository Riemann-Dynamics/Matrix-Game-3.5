# Distilled causal inference

The public contract is deliberately small:

```text
config + image + camera + prompt + base-model paths + checkpoint -> result.mp4
```

There is no training-run replay mode. The inference path never reads
`args_resolved.json`, validation videos, manifests, `_sections.json`, epochs,
or experiment directories. A checkpoint is only a weight file loaded for the
current request.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python infer_distilled.py \
  --config configs/infer_distilled.yaml \
  --checkpoint /path/to/distilled.safetensors \
  --image /path/to/input.png \
  --camera /path/to/camera.npz \
  --prompt-file /path/to/prompt.txt \
  --wan-dir /path/to/Wan2.2-TI2V-5B \
  --tokenizer-dir /path/to/umt5-xxl \
  --da3-dir /path/to/DA3NESTED-GIANT-LARGE-1.1 \
  --output result.mp4
```

`--prompt`, `--prompt-file`, and `--caption` are mutually exclusive. The
camera file has the same format as base inference: `extrinsics_c2w` (or
`extrinsics`) plus `intrinsics`. `--camera-convention w2c` converts world-to-
camera input before generation.

The three model directories follow the base-inference resolution order:
explicit CLI path, environment variable, then the documented `checkpoints/`
layout. `--checkpoint`, input data, and output are always explicit.

## Hyperparameters

Only inference hyperparameters belong in `configs/infer_distilled.yaml`:

- output size, block count, seed, and few-step schedule;
- `standard`, `hiar-sde`, or `sink-anchor-context` profile;
- causal chunk/context window;
- online-memory publication and context selection;
- student CFG scale;
- Mosaic selection/fusion settings;
- DA3 inference resolution and autocast dtype.

Unknown config keys fail immediately. Training keys and experiment paths are
not accepted. HiAR scales must contain one value per denoising step and are
valid only for the `hiar-sde` profile.

The released first-person checkpoint uses `student_cfg_scale: 3.0`. Set it to
`1.0` only for an explicit no-CFG ablation.

## Context and nonlocal retrieval

The default is:

```yaml
dynamic_context: true
dynamic_context_selection: oldest
nonlocal_memory_context: false
context_pose_pool_size: 5
```

`dynamic_context_selection: oldest` does not choose a globally unrelated old
frame. It first builds a pose-nearest candidate pool of
`context_pose_pool_size` entries, then chooses the oldest source-time entry in
that pool. This favors long-term recall while retaining geometric relevance.

`nonlocal_memory_context: false` preserves normal Mosaic Memory retrieval and
allows dynamic context to use any published entry. Enabling it activates the
`nonlocal_oldest` policy from the research implementation: both Mosaic Memory
and dynamic context exclude sources that are still visible in the bounded
rolling KV prefix. Dynamic context then applies the same pose-near/oldest
ranking to the remaining entries.

## Output

The only user-facing artifact is the exact file passed to `--output`, normally
`result.mp4`. Intermediate one-scene dataset/index files live under
`.cache/infer_runs/` and are removed after success. Use `--keep-workspace` only
for debugging a failed input conversion or model run.

## Inference-only boundary

This entrypoint constructs no loss, optimizer, teacher/fake scorer, EMA,
backward pass, epoch loop, or training checkpoint writer. The causal KV and
online-memory kernels reuse the existing public Matrix model and input
pipeline; they do not copy the Stage3 training stack.

## Verified direct run

On 2026-07-19 the explicit-input interface was run end to end on an A800 with
the bundled first-person case and a standard distilled epoch-10 checkpoint.
The checkpoint matched all DiT keys, all seven causal chunks completed with
online DA3 registration, and the process exited 0. The only output was
`test_outputs/direct-interface-standard/result.mp4`: H.264, 1280x704, 85
frames, 16 fps, 5.3125 seconds. Full decode and a 12-frame visual audit found
coherent camera motion, stable car/building identity, and no black-frame or
geometry collapse.

## Validation status

Final-commit acceptance was rerun on 2026-07-19 after the interface was
streamlined:

- 23/23 unit tests passed;
- full-repository `compileall` passed;
- CLI help, whitespace, inference-only, and replay-boundary checks passed;
- the tracked-file credential-pattern scan was clean;
- the base and distilled `result.mp4` files both passed full-frame FFmpeg
  decode and exact frame counting;
- the distilled output directory contains only `result.mp4`.

Implementation, direct GPU inference, visual inspection, automated acceptance,
and documentation are complete. Publishing the single commit to GitHub remains
an explicit separate operation.
