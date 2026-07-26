

<div align="center">
<h1 align="center">Matrix-Game 3.5</h1>
<h3 align="center">Enhancing Real-Time Streaming Interactive World Models with Patch Memory</h3>
</div>

<font size=7><div align='center' >  [[🤗 Base Model](https://huggingface.co/RiemannDynamics/Matrix-Game-3.5-Base)] [[🤗 Distilled Model](https://huggingface.co/RiemannDynamics/Matrix-Game-3.5-Distilled)] [[📖 Technical Report](https://matrix-game-v3-5.github.io/paper/Matrix-Game-3.5.pdf)] [[🚀 Project Website](https://matrix-game-v3-5.github.io/)] </div></font>


https://github.com/user-attachments/assets/26d45554-4964-4a71-8e2e-43cb70c28a4c

## 📝 Overview
**Matrix-Game-3.5** is a memory-augmented interactive world model for 720p long-horizon camera-controllable video generation, in both **first-person** and **third-person** modes.

- **Patch Memory + Warped PRoPE**: a parameter-free long-term geometric memory framework — Patch Memory lifts past observations into a 3D memory with geometric alignment and visibility-aware retrieval for cross-view scene recall, while Warped PRoPE folds camera projection matrices into the spatiotemporal RoPE to jointly model temporal relations and view geometry, delivering long-horizon consistency and precise camera control without modifying the backbone.
- **Static–dynamic decoupled world representation**: the static scene keeps long-term geometric memory through Patch Memory, while dynamic subjects are maintained by lightweight multi-view **reference tokens** carrying identity and appearance; combined with motion-aware filtering and leakage-free subject training, this unifies geometric and subject consistency and mitigates ghosting and identity drift in long-horizon generation.
- **Long-horizon real-time distillation**: from a bidirectional diffusion model to a few-step causal generator — Flow Matching in a perceptual feature space jointly learns autoregressive denoising and few-step generation as a strong initialization, then curriculum-style self-rollout DMD progressively distills CFG, camera control and memory conditioning, enabling minute-long, few-step, real-time interactive generation.

## 🤗 Matrix-Game-3.5 Models
We currently provide two pretrained **5B base (bidirectional) models**, built on
Wan2.2-TI2V-5B:

| Model | Mode | Extra conditioning |
|---|---|---|
| `first-person.safetensors` | first-person (egocentric) | — |
| `third-person.safetensors` | third-person | protagonist reference images (0–4 crops) |

Both are available in the
[Matrix-Game-3.5-Base](https://huggingface.co/RiemannDynamics/Matrix-Game-3.5-Base)
Hugging Face repository. The standard three-step first-person causal checkpoint
is available in
[Matrix-Game-3.5-Distilled](https://huggingface.co/RiemannDynamics/Matrix-Game-3.5-Distilled).

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
Choose either the base or distilled Matrix-Game checkpoint. Wan2.2-TI2V-5B and
Depth-Anything-3 are shared dependencies for both inference paths. Place or
symlink the downloaded files under `checkpoints/` as shown below:

```bash
pip install -U huggingface_hub

# 1a. Matrix-Game-3.5 base models
hf download RiemannDynamics/Matrix-Game-3.5-Base --local-dir checkpoints/Matrix-Game-3.5-Base
ln -s Matrix-Game-3.5-Base/first-person.safetensors checkpoints/first-person.safetensors
ln -s Matrix-Game-3.5-Base/third-person.safetensors checkpoints/third-person.safetensors

# 1b. Distilled first-person model (for infer_distilled.py)
hf download RiemannDynamics/Matrix-Game-3.5-Distilled --local-dir checkpoints/Matrix-Game-3.5-Distilled
ln -s Matrix-Game-3.5-Distilled/first-person.safetensors checkpoints/distilled-first-person.safetensors

# 2. Wan2.2-TI2V-5B — provides the T5 text encoder, VAE, DiT scaffold and the
#    umt5-xxl tokenizer (bundled under google/umt5-xxl); our checkpoints are
#    DiT weights loaded on top of it
hf download Wan-AI/Wan2.2-TI2V-5B --exclude "assets/*" "examples/*" --local-dir checkpoints/Wan2.2-TI2V-5B

# 3. Depth-Anything-3 (metric depth for Mosaic Memory)
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir checkpoints/DA3NESTED-GIANT-LARGE-1.1
```

```
checkpoints/
├── Wan2.2-TI2V-5B/              DiT shards + T5 encoder + VAE + tokenizer
├── DA3NESTED-GIANT-LARGE-1.1/   depth estimator
├── first-person.safetensors     Matrix-Game-3.5 first-person model
├── third-person.safetensors     Matrix-Game-3.5 third-person model
└── distilled-first-person.safetensors  distilled three-step first-person model
```

For custom locations, both entrypoints accept `--wan-dir`, `--tokenizer-dir`,
and `--da3-dir`. Base inference accepts `--ckpt`; distilled inference requires
an explicit `--checkpoint`. The shared dependency paths can also be set through
`WAN22_TI2V_5B_DIR`, `UMT5_TOKENIZER_DIR`, and `DA3_MODEL_PATH`.

### Base model inference
Use `infer.py` with an **anchor image**, a **camera trajectory**, and a **text
prompt**. Third-person inference optionally accepts **protagonist reference
crops**.

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

Use `infer_distilled.py` with the standard three-step first-person causal
checkpoint. It follows the same explicit image/camera/prompt interface as the
base model. Use the paired `configs/infer_distilled.yaml` configuration
unchanged; no training run, validation artifact, manifest, or sidecar metadata
is required.

```bash
python infer_distilled.py \
    --config configs/infer_distilled.yaml \
    --checkpoint checkpoints/distilled-first-person.safetensors \
    --image samples/first_person/case_0/input.png \
    --camera samples/first_person/case_0/camera.npz \
    --prompt-file samples/first_person/case_0/prompt.txt \
    --output result.mp4
```

See [`DISTILLED_INFERENCE.md`](DISTILLED_INFERENCE.md) for the input contract
and complete command-line interface.

**Camera format** (`--camera`): a `.npz` with
`extrinsics_c2w` — `(N,4,4)` camera-to-world matrices, metric translation —
and `intrinsics` — `(N,4) [fx,fy,cx,cy]` (or `(4,)` / `(3,3)` / `(N,3,3)`)
in pixels of the anchor image. A trajectory shorter than
`1 + 84 × num_blocks` poses is padded by holding the last pose.

## 🔗 Related Links
**Matrix-Game Series**
- [Matrix-Game 3.0](https://matrix-game-v3.github.io/) — Real-time and streaming interactive world model with long-horizon memory
- [Matrix-Game 2.0](https://matrix-game-v2.github.io/) — Real-time, streaming interactive world model

**Acknowledgements**
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
    author={{Riemann Dynamics}},
    year={2026},
    howpublished={Project page},
    url={https://matrix-game-v3-5.github.io/}
  }
```
