<div align="center">
<h1 align="center">Matrix-Game 3.5</h1>
<h3 align="center">Enhancing Real-Time Streaming Interactive World Models with Patch Memory</h3>
</div>

<font size=7><div align='center' >  [[🤗 HuggingFace](https://huggingface.co/RiemannDynamics/Matrix-Game-3.5-Base)] [[📖 Technical Report](assets/Matrix_Game_3_5.pdf)] [[🚀 Project Website](https://matrix-game-v3-5.github.io/)] </div></font>

> 🎬 Teaser: [`assets/teaser.mp4`](assets/teaser.mp4)

## 📝 Overview
**Matrix-Game-3.5** is a memory-augmented interactive world model for 720p long-horizon camera-controllable video generation, in both **first-person** and **third-person** modes.

- **Geometry-aware camera control**: PRoPE-based camera conditioning is integrated into the spatiotemporal RoPE — each latent frame is associated with a full world-to-image projection matrix, yielding best-in-class camera accuracy across pose metrics.
- **Patch-level Mosaic Memory**: past observations are reprojected into the current view using metric depth, camera poses and intrinsics, giving long-horizon scene recall with visibility-aware fusion and stable revisit consistency.
- **Static–dynamic decoupling**: motion-aware object filtering separates the dynamic subject from the static scene; the third-person model conditions on **protagonist reference tokens** for identity-consistent character generation.
- **Real-time few-step distillation**: a distillation pipeline combining Distribution Matching Distillation on autoregressive rollouts with a consistency objective enables real-time streaming generation without future-information leakage.

## 🤗 Matrix-Game-3.5 Models
We currently provide two pretrained **5B base (bidirectional) models**, built on Wan2.2-TI2V-5B:

| Model | Mode | Extra conditioning |
|---|---|---|
| `first-person.safetensors` | first-person (egocentric) | — |
| `third-person.safetensors` | third-person | protagonist reference images (0–4 crops) |

Both are available in the [Matrix-Game-3.5-Base](https://huggingface.co/RiemannDynamics/Matrix-Game-3.5-Base) HuggingFace repository.

The **distilled real-time autoregressive models** will be released soon! 🚀🚀

## Requirements
* One NVIDIA GPU with ≥ 40 GB VRAM (A/H series tested); 704×1280 generation peaks around 40 GB.
* Linux operating system, ≥ 64 GB RAM.
* Python 3.10.

Note: this repo carries several third-party components — the DiffSynth-based
model/pipeline library (`diffsynth/`), the frustum reprojection engine for
Mosaic Memory (`frustum/`), and the Depth-Anything-3 source
(`third_party/depth-anything-3/`). All Python dependencies are pinned in
`requirements.txt`; no extra manual builds are required (no flash-attention
compilation needed).

## ⚙️ Quick Start
### Installation
```bash
git clone <this-repo> Matrix-Game-3.5
cd Matrix-Game-3.5
conda create -n matrix-game-3.5 python=3.10 -y
conda activate matrix-game-3.5

# 1) PyTorch matching your CUDA version, e.g. CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) remaining dependencies:
pip install -r requirements.txt
```

The bundled third-party sources — `diffsynth/` (DiffSynth-based pipeline),
`frustum/` (Mosaic Memory reprojection engine) and
`third_party/depth-anything-3/` — are vendored in this repo and imported
directly from source: no extra installation or compilation step is needed.

### Model Download
Three sets of weights are required. Place (or symlink) them into
`checkpoints/` under these exact names — no further configuration needed:

```bash
pip install "huggingface_hub[cli]"

# 1. Matrix-Game-3.5 base models (ours)
huggingface-cli download RiemannDynamics/Matrix-Game-3.5-Base --local-dir checkpoints/Matrix-Game-3.5-Base
ln -s Matrix-Game-3.5-Base/first-person.safetensors checkpoints/first-person.safetensors
ln -s Matrix-Game-3.5-Base/third-person.safetensors checkpoints/third-person.safetensors

# 2. Wan2.2-TI2V-5B — provides the T5 text encoder, VAE, DiT scaffold and the
#    umt5-xxl tokenizer (bundled under google/umt5-xxl); our checkpoints are
#    DiT weights loaded on top of it
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --exclude "assets/*" "examples/*" --local-dir checkpoints/Wan2.2-TI2V-5B

# 3. Depth-Anything-3 (metric depth for Mosaic Memory)
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir checkpoints/DA3NESTED-GIANT-LARGE-1.1
```

```
checkpoints/
├── Wan2.2-TI2V-5B/              DiT shards + T5 encoder + VAE + tokenizer
├── DA3NESTED-GIANT-LARGE-1.1/   depth estimator
├── first-person.safetensors     Matrix-Game-3.5 first-person model
└── third-person.safetensors     Matrix-Game-3.5 third-person model
```

Custom locations are supported via `--wan-dir / --tokenizer-dir / --da3-dir /
--ckpt` or the env vars `WAN22_TI2V_5B_DIR / UMT5_TOKENIZER_DIR /
DA3_MODEL_PATH / CKPT_FIRST_PERSON / CKPT_THIRD_PERSON`.

### Inference
`infer.py` is the single entry point. Inputs: an **anchor image**, a **camera
trajectory**, a **text prompt** — and optionally, for third person,
**protagonist reference crops**.

```bash
# first person — bundled sample (SANA-WM-Bench scene)
python infer.py --person first \
    --image  samples/first_person/case_7/input.png \
    --camera samples/first_person/case_7/camera.npz \
    --prompt-file samples/first_person/case_7/prompt.txt

# third person — --refs is OPTIONAL: without it the model generates the
# protagonist freely; with it the protagonist identity is locked to your crops
python infer.py --person third \
    --image  samples/third_person/case_1/input.png \
    --camera samples/third_person/case_1/camera.npz \
    --prompt-file samples/third_person/case_1/prompt.txt

python infer.py --person third \
    --image  samples/third_person/case_1/input.png \
    --camera samples/third_person/case_1/camera.npz \
    --prompt-file samples/third_person/case_1/prompt.txt \
    --refs   samples/third_person/case_1/refs

# your own data
python infer.py --person first \
    --image my_scene.png --camera my_camera.npz \
    --prompt "A slow walk along a rainy street at dusk."
```

Results land in `outputs/{first_person,third_person}/<timestamp>/`:
- `result.mp4` — the generated video
- `memory_visualization.mp4` — diagnostic two-row panel (generation | mosaic memory)
- `subject_ref_preview.jpg` — protagonist reference canvas (third person)

| Option | Default | Meaning |
|---|---|---|
| `--num-blocks` | 1 | blocks to generate; each block = 80 frames and consumes 84 camera poses |
| `--steps` | 25 | denoising steps |
| `--cfg-scale` | 5.0 | classifier-free guidance scale |
| `--seed` | 3407 | generation seed |
| `--camera-convention` | `c2w` | pass `w2c` if your extrinsics are world-to-camera |
| `--refs` | — | (third person) directory of protagonist crops; masks (`*_mask.png`) optional — full-white masks assumed otherwise |
| `--caption` | — | segment caption json instead of a single prompt (multi-block runs) |
| `--keep-workspace` | off | keep intermediate artifacts in `.cache/infer_runs/` for debugging |

### Distilled causal inference

Use `infer_distilled.py` for released few-step causal checkpoints. It follows
the same explicit image/camera/prompt interface as the base model. Inference
hyperparameters live in `configs/infer_distilled.yaml`; no training run,
validation video, manifest, or sidecar metadata is read.

- `standard`: clean rolling prefix, online mosaic memory, dynamic context.
- `hiar-sde`: HiAR next-timestep prefix/context corruption for a HiAR-trained checkpoint.
- `sink-anchor-context`: online memory with the original C0 anchor used as context.

```bash
python infer_distilled.py \
    --config configs/infer_distilled.yaml \
    --checkpoint checkpoints/distilled-stage3.safetensors \
    --image samples/first_person/case_0/input.png \
    --camera samples/first_person/case_0/camera.npz \
    --prompt-file samples/first_person/case_0/prompt.txt \
    --output result.mp4
```

Set `profile` in the config to `standard`, `hiar-sde`, or
`sink-anchor-context`. See [`DISTILLED_INFERENCE.md`](DISTILLED_INFERENCE.md)
for the complete interface and configuration fields.

**Camera format** (`--camera`): a `.npz` with
`extrinsics_c2w` — `(N,4,4)` camera-to-world matrices, metric translation —
and `intrinsics` — `(N,4) [fx,fy,cx,cy]` (or `(4,)` / `(3,3)` / `(N,3,3)`)
in pixels of the anchor image. A trajectory shorter than
`1 + 84 × num_blocks` poses is padded by holding the last pose.

## ⭐ Acknowledgements
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) for the strong base model
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for the diffusion framework this codebase builds on
- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) for metric depth estimation
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work on autoregressive distillation

## 📜 License
This project is licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
Bundled first-person sample assets are from SANA-WM-Bench (CC BY 4.0).

## 📖 Citation
If you find this work useful for your research, please kindly cite:

```
  @misc{2026matrixgame35,
    title={Matrix-Game 3.5: Enhancing Real-Time Streaming Interactive World Models with Patch Memory},
    author={{Skywork AI Matrix-Game Team}},
    year={2026},
    howpublished={Project page},
    url={https://matrix-game-v3-5.github.io/}
  }
```
