from diffsynth.pipelines.wan_video import WAN_VIDEO_PROPE_CAMERA_KEYS

from .common import *
from .inference import run_mosaic_segment_inference
from .prompting import _maybe_apply_train_prompt_dropout
from .timing import get_timing_reporter

class _ForwardMixin:
    def forward(self, data, inputs=None):
        timing = get_timing_reporter(self)
        if (
            isinstance(data, dict)
            and self.training
            and "_no_mosaic_this_step" not in data
            and getattr(self, "mosaic_train_random_schedule", False)
        ):
            # Per-step DDP-synchronised schedule (default 3:4:3 over
            # interval3 / interval1 / nomosaic). ``self.step_count`` is the
            # current pre-increment step index and advances in lockstep on
            # every rank, so all ranks pick the SAME mode this step -- a
            # nomosaic step is a nomosaic step everywhere (no fast-rank-waits-
            # for-slow-rank skew). All interval modes run with drop_holes on.
            kind, interval = self._sample_mosaic_step_mode(self.step_count)
            if kind == "nomosaic":
                data["_no_mosaic_this_step"] = True
            else:
                data["_no_mosaic_this_step"] = False
                data["_mosaic_step_interval"] = int(interval)
                data["_mosaic_drop_holes_effective"] = True
        elif (
            isinstance(data, dict)
            and self.training
            and self.no_mosaic_prob_train > 0.0
            and "_no_mosaic_this_step" not in data
        ):
            # Legacy path: one independent Bernoulli draw per training step.
            # We honour any pre-set value (e.g. a unit test or a future
            # feature could pin it) but otherwise sample here -- doing it
            # before materialize lets the no-mosaic path skip memory VAE
            # encode and frustum fuse.
            data["_no_mosaic_this_step"] = bool(
                random.random() < self.no_mosaic_prob_train
            )
        if isinstance(data, dict) and data.get("needs_vae_materialization"):
            with timing.scope("forward.materialize"):
                data = self._materialize_data(data)
        with timing.scope("forward.train_qwen_prompt"):
            data = self._maybe_apply_train_qwen_prompt(data)
        if isinstance(data, dict) and self.training:
            with timing.scope("forward.train_prompt_dropout"):
                data = _maybe_apply_train_prompt_dropout(
                    data,
                    probability=self.train_prompt_dropout_prob,
                )
        with timing.scope("forward.preview_dataset"):
            self.maybe_preview_dataset(data)
        if inputs is None:
            with timing.scope("forward.get_pipeline_inputs"):
                inputs = self.get_pipeline_inputs(data)
        inputs_shared, inputs_posi, inputs_nega = inputs

        self.step_count += 1
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        # PRoPE camera matrices must reach WanVideoUnit_PropeCamera in fp32:
        # bf16-quantizing absolute extrinsics here destroys the relative
        # translations recovered later by the fp32 composition (catastrophic
        # cancellation on large world coordinates). Keep the originals and
        # re-assert them after the blanket dtype transfer.
        prope_fp32 = {
            key: inputs_shared[key]
            for key in WAN_VIDEO_PROPE_CAMERA_KEYS
            if isinstance(inputs_shared.get(key), torch.Tensor)
        }
        with timing.scope("forward.transfer_data_to_device"):
            inputs = self.transfer_data_to_device(
                inputs, self.pipe.device, self.pipe.torch_dtype
            )
        inputs_shared, inputs_posi, inputs_nega = inputs
        for key, value in prope_fp32.items():
            inputs_shared[key] = value.to(
                device=self.pipe.device, dtype=torch.float32
            )
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit_idx, unit in enumerate(self.pipe.units):
            unit_name = unit.__class__.__name__
            with timing.scope(f"forward.unit.{unit_idx}.{unit_name}"):
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        with timing.scope("forward.loss"):
            loss = self.task_to_loss[self.task](self.pipe, *inputs)
        loss_metrics = getattr(self.pipe, "_last_loss_metrics", None)
        if isinstance(data, dict) and isinstance(loss_metrics, dict):
            data["loss_metrics"] = dict(loss_metrics)
            if loss_metrics:
                timing.set_metadata(loss_metrics=dict(loss_metrics))
        mosaic_sequence_debug = None
        if inputs and isinstance(inputs[0], dict):
            mosaic_sequence_debug = inputs[0].get("mosaic_sequence_debug")
        if isinstance(mosaic_sequence_debug, dict) and mosaic_sequence_debug:
            timing.set_metadata(mosaic_sequence_stats=dict(mosaic_sequence_debug))
        return loss

    @torch.no_grad()
    def inference_step(self, **kwargs):
        kwargs.setdefault("enable_mosaic", self.enable_mosaic)
        kwargs.setdefault("only_prope", self.only_prope)
        kwargs.setdefault("mosaic_view_change_prope", self.mosaic_view_change_prope)
        kwargs.setdefault("allow_empty_prompt", self.allow_no_prompt)
        return run_mosaic_segment_inference(pipe=self.pipe, **kwargs)
