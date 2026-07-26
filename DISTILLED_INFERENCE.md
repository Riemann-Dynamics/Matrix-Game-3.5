# Distilled causal inference

`infer_distilled.py` runs the standard three-step first-person causal model from
explicit user inputs:

```text
config + checkpoint + image + camera + prompt -> result.mp4
```

It does not require a training directory, validation artifact, manifest, epoch,
or experiment metadata.

## Run

Download the distilled checkpoint and shared Wan2.2/Depth-Anything-3
dependencies as described in the main [README](README.md), then run:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_distilled.py \
  --config configs/infer_distilled_6blocks.yaml \
  --checkpoint checkpoints/distilled-first-person.safetensors \
  --image samples/distilled/suburban_street_6blocks/input.png \
  --camera samples/distilled/suburban_street_6blocks/camera.npz \
  --caption samples/distilled/suburban_street_6blocks/caption.json \
  --output result.mp4
```

The six-block example uses `configs/infer_distilled_6blocks.yaml`. It has the
same model settings as `configs/infer_distilled.yaml`, with only
`num_blocks: 6` changed. Use the latter for a one-block smoke test. Both contain
the checkpoint's resolution, three-step denoising schedule, causal window,
memory settings, CFG scale, and depth configuration. Unknown configuration
keys fail immediately.

## Inputs

- `--image`: anchor RGB image.
- `--camera`: `.npz` camera trajectory containing `extrinsics_c2w` (or
  `extrinsics`) and `intrinsics`.
- `--prompt`, `--prompt-file`, or `--caption`: exactly one text-conditioning
  source.
- `--checkpoint`: distilled `first-person.safetensors` weight file.
- `--output`: destination `.mp4` path.

Use `--camera-convention w2c` when the supplied extrinsics are
world-to-camera. Shared model directories can be overridden with `--wan-dir`,
`--tokenizer-dir`, and `--da3-dir`; otherwise the entrypoint uses the
environment variables and `checkpoints/` layout documented in the main README.

## Output

The generated video is written to the exact path passed to `--output`.
Intermediate files under `.cache/infer_runs/` are removed after a successful
run. Add `--keep-workspace` only when debugging input conversion or inference.
