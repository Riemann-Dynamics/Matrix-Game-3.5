from .common import *
from .prompting import _require_nonempty_prompt

# Scheduler attributes mutated by ``FlowMatchScheduler.set_timesteps``. The
# pipeline's ``__call__`` invokes ``set_timesteps`` with *inference* settings
# (e.g. 25 steps, ``training=False``), which rebinds ``sigmas``/``timesteps``
# but leaves the 1000-entry ``linear_timesteps_weights`` table untouched.
# If training resumes on that polluted state, ``FlowMatchSFTLoss`` samples
# timesteps from the 25-step inference grid and looks up training weights at
# ``timestep_id`` 0..24 of the stale 1000-entry table -- collapsing the loss
# weight to ~0 and silently killing training after the first validation.
_SCHEDULER_STATE_ATTRS = (
    "sigmas",
    "timesteps",
    "training",
    "linear_timesteps_weights",
)


@contextlib.contextmanager
def _preserve_scheduler_state(scheduler):
    """Snapshot/restore scheduler state across an inference pipe call.

    ``set_timesteps`` rebinds (not mutates) these attributes, so holding
    references is enough for an exact restore; no clone needed.
    """
    snapshot = {
        name: getattr(scheduler, name)
        for name in _SCHEDULER_STATE_ATTRS
        if hasattr(scheduler, name)
    }
    try:
        yield
    finally:
        for name, value in snapshot.items():
            setattr(scheduler, name, value)


def run_mosaic_segment_inference(
    pipe,
    prompt,
    negative_prompt,
    input_image,
    height,
    width,
    num_frames,
    seed,
    *,
    num_inference_steps=25,
    cfg_scale=5.0,
    negative_no_prope=False,
    negative_no_context=False,
    enable_mosaic=True,
    only_prope=False,
    mosaic_latent=None,
    mosaic_revgrid=None,
    mosaic_use_revgrid_rope=False,
    mosaic_view_change=None,
    mosaic_view_change_prope=False,
    mosaic_mask_holes=True,
    mosaic_drop_holes=False,
    mosaic_frame_indices=None,
    tiled=True,
    prope_camera_kwargs=None,
    return_latent=True,
    first_frame_latents=None,
    latent_rope_time_indices=None,
    subject_ref_latents=None,
    subject_ref_slot_ratio=0.5,
    subject_ref_time_gap=1,
    subject_ref_prope_mode="identity",
    allow_empty_prompt=False,
):
    """Run one TI2V mosaic inference pass.

    Builds the pipeline kwargs (mosaic conditioning, PRoPE camera, etc.) and
    delegates a single call to `pipe`. The return value is the pipeline's raw
    output: a latent tensor when `return_latent=True`, otherwise the decoded
    video frames. Multi-segment rollout (chaining sections, registering
    generated frames into a frustum handler, etc.) is the caller's
    responsibility.

    `first_frame_latents` and `input_image` are mutually exclusive condition
    sources for the first-frame slot. Passing `first_frame_latents` directly
    skips the VAE decode->encode round-trip that happens when an `input_image`
    is provided.
    """
    prompt = _require_nonempty_prompt(
        prompt,
        phase="validation",
        allow_empty=allow_empty_prompt,
    )
    use_mosaic = enable_mosaic and not only_prope
    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "tiled": tiled,
        "height": height,
        "width": width,
        "input_image": input_image,
        "first_frame_latents": first_frame_latents,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "cfg_scale": cfg_scale,
        "negative_no_prope": negative_no_prope,
        "negative_no_context": negative_no_context,
        "return_latent": return_latent,
        "latent_rope_time_indices": latent_rope_time_indices,
        "subject_ref_latents": subject_ref_latents,
        "subject_ref_slot_ratio": subject_ref_slot_ratio,
        "subject_ref_time_gap": subject_ref_time_gap,
        "subject_ref_prope_mode": subject_ref_prope_mode,
    }
    if use_mosaic:
        pipe_kwargs.update(
            {
                "mosaic_latent": mosaic_latent,
                "mosaic_timestep_zero": True,
                "mosaic_revgrid": mosaic_revgrid,
                "mosaic_use_revgrid_rope": mosaic_use_revgrid_rope,
                "mosaic_view_change": mosaic_view_change,
                "mosaic_view_change_prope": mosaic_view_change_prope,
                "mosaic_mask_holes": mosaic_mask_holes,
                "mosaic_drop_holes": mosaic_drop_holes,
                "mosaic_frame_indices": mosaic_frame_indices,
            }
        )
    if prope_camera_kwargs:
        pipe_kwargs.update(prope_camera_kwargs)

    # Restore the (training) scheduler state after the call so mid-training
    # validation can never leak inference timesteps into the SFT loss.
    with _preserve_scheduler_state(pipe.scheduler):
        return pipe(**pipe_kwargs)
