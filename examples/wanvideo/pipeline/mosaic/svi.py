import os
import random

import torch


SVI_BUFFER_FILENAME = "svi_error_buffer.pt"


def _prob(value, name):
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def _rms(tensor):
    return float(tensor.detach().float().pow(2).mean().sqrt().item())


def _is_finite_tensor(tensor):
    return torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all().item())


class SVIErrorBank:
    BANKS = ("noise", "latent")
    REPLACEMENT_STRATEGIES = ("random", "fifo", "l2_batch")

    def __init__(self, num_grids=50, capacity=4, replacement_strategy="random"):
        self.num_grids = max(1, int(num_grids))
        self.capacity = max(1, int(capacity))
        self.replacement_strategy = str(replacement_strategy or "random")
        if self.replacement_strategy not in self.REPLACEMENT_STRATEGIES:
            raise ValueError(
                "svi_buffer_replacement_strategy must be one of "
                f"{self.REPLACEMENT_STRATEGIES}, got {self.replacement_strategy!r}."
            )
        self.buffers = {
            bank: {grid_id: [] for grid_id in range(self.num_grids)}
            for bank in self.BANKS
        }

    def grid_id(self, scheduler, timestep):
        value = (
            float(timestep.detach().flatten()[0].cpu().item())
            if isinstance(timestep, torch.Tensor)
            else float(timestep)
        )
        timesteps = scheduler.timesteps.detach().float().cpu()
        nearest = int(torch.argmin((timesteps - value).abs()).item())
        if len(timesteps) <= 1 or self.num_grids <= 1:
            return 0
        return int(round(nearest * (self.num_grids - 1) / (len(timesteps) - 1)))

    def add(self, bank, grid_id, tensor):
        if not _is_finite_tensor(tensor):
            return False
        bucket = self.buffers[bank][int(grid_id)]
        stored = tensor.detach().cpu().to(dtype=torch.float16).contiguous()
        if len(bucket) < self.capacity:
            bucket.append(stored)
        elif self.replacement_strategy == "random":
            bucket[random.randrange(self.capacity)] = stored
        elif self.replacement_strategy == "fifo":
            bucket.pop(0)
            bucket.append(stored)
        elif self.replacement_strategy == "l2_batch":
            stored_flat = torch.stack(bucket).flatten(start_dim=1).float()
            new_flat = stored.flatten().float()
            distances = torch.norm(stored_flat - new_flat.unsqueeze(0), p=2, dim=1)
            bucket[int(torch.argmin(distances).item())] = stored
        return True

    def sample(self, bank, grid_id, like_tensor):
        bucket = self.buffers[bank][int(grid_id)]
        if not bucket:
            return None
        return random.choice(bucket).to(device=like_tensor.device, dtype=like_tensor.dtype)

    def occupancy(self):
        return {
            bank: sum(len(bucket) for bucket in grid.values())
            for bank, grid in self.buffers.items()
        }

    def bucket_occupancy(self, grid_id):
        return {
            bank: len(grid[int(grid_id)])
            for bank, grid in self.buffers.items()
        }

    def state_dict(self):
        return {
            "version": 1,
            "num_grids": self.num_grids,
            "capacity": self.capacity,
            "replacement_strategy": self.replacement_strategy,
            "buffers": self.buffers,
            "occupancy": self.occupancy(),
        }

    def load_state_dict(self, state):
        if int(state.get("version", 0)) != 1:
            raise ValueError("Unsupported SVI buffer version.")
        if int(state.get("num_grids", -1)) != self.num_grids:
            raise ValueError("SVI num_grids mismatch.")
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("SVI buffer capacity mismatch.")
        loaded_strategy = str(state.get("replacement_strategy", self.replacement_strategy))
        if loaded_strategy != self.replacement_strategy:
            raise ValueError("SVI buffer replacement strategy mismatch.")
        loaded = state.get("buffers", {})
        loaded = {
            bank: {int(grid_id): bucket for grid_id, bucket in loaded.get(bank, {}).items()}
            for bank in self.BANKS
        }
        self.buffers = {
            bank: {
                grid_id: [
                    item.detach().cpu().to(dtype=torch.float16).contiguous()
                    for item in loaded.get(bank, {}).get(grid_id, [])
                    if _is_finite_tensor(item)
                ][: self.capacity]
                for grid_id in range(self.num_grids)
            }
            for bank in self.BANKS
        }


class SVIState:
    def __init__(self, owner, args):
        self.owner = owner
        self.bank = SVIErrorBank(
            num_grids=getattr(args, "svi_num_grids", 50),
            capacity=getattr(args, "svi_error_buffer_k", 4),
            replacement_strategy=getattr(
                args, "svi_buffer_replacement_strategy", "random"
            ),
        )
        self.buffer_warmup_steps = max(
            0, int(getattr(args, "svi_buffer_warmup_steps", 50))
        )
        self.save_error_buffer = bool(getattr(args, "svi_save_error_buffer", True))
        self.inject_noise = bool(getattr(args, "svi_inject_noise", True))
        self.noise_prob = _prob(getattr(args, "svi_noise_prob", 0.5), "svi_noise_prob")
        self.noise_scale = float(getattr(args, "svi_noise_scale", 1.0))
        self.inject_latent = bool(getattr(args, "svi_inject_latent", True))
        self.latent_prob = _prob(getattr(args, "svi_latent_prob", 0.5), "svi_latent_prob")
        self.latent_scale = float(getattr(args, "svi_latent_scale", 1.0))
        self.inject_clean_context = bool(getattr(args, "svi_inject_clean_context", True))
        self.clean_context_prob = _prob(
            getattr(args, "svi_clean_context_prob", 0.5),
            "svi_clean_context_prob",
        )
        self.clean_context_scale = float(
            getattr(args, "svi_clean_context_scale", 1.0)
        )
        self.clean_context_frames = max(
            1, int(getattr(args, "svi_clean_context_frames", 1))
        )
        self.clean_prob = _prob(getattr(args, "svi_clean_prob", 0.5), "svi_clean_prob")
        self.clean_buffer_update_prob = _prob(
            getattr(args, "svi_clean_buffer_update_prob", 0.1),
            "svi_clean_buffer_update_prob",
        )
        self.debug = bool(getattr(args, "debug", False))
        self.step = 0
        self.last_metrics = {}

    def buffer_path(self, log_dir=None):
        root = log_dir or getattr(self.owner, "log_dir", None)
        return None if not root else os.path.join(root, SVI_BUFFER_FILENAME)

    def maybe_load_from_resume(self, resume_from):
        if not resume_from:
            return
        path = os.path.join(resume_from, SVI_BUFFER_FILENAME)
        if not os.path.isfile(path):
            print(f"[svi] WARNING: no error buffer at {path}; starting empty.", flush=True)
            return
        try:
            state = torch.load(path, map_location="cpu")
            self.bank.load_state_dict(state)
        except Exception as exc:
            print(
                f"[svi] WARNING: failed to load error buffer from {path}: {exc}; "
                "starting empty.",
                flush=True,
            )
            return
        print(f"[svi] loaded error buffer from {path}", flush=True)

    def save(self, log_dir=None):
        if not self.save_error_buffer:
            return None
        path = self.buffer_path(log_dir)
        if path is None:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.bank.state_dict(), path)
        return path

    def state_metadata(self, log_dir=None):
        return {
            "svi_error_buffer_path": self.buffer_path(log_dir),
            "svi_error_buffer_version": 1,
            "svi_error_buffer_occupancy": self.bank.occupancy(),
        }

    def prepare(self, *, pipe, inputs, clean_latents, noise, timestep):
        self.step += 1
        grid_id = self.bank.grid_id(pipe.scheduler, timestep)
        debug = {
            "grid_id": grid_id,
            "bucket_occupancy": self.bank.bucket_occupancy(grid_id),
            "injected": {
                "noise_to_sampled_noise": False,
                "latent_to_input_latents": False,
                "latent_to_first_frame_tail": False,
            },
            "skip_reason": {},
            "use_clean_input": False,
            "update_buffer": True,
            "shapes": {
                "input_latents": tuple(clean_latents.shape),
                "noise": tuple(noise.shape),
                "first_frame_latents": (
                    tuple(inputs["first_frame_latents"].shape)
                    if torch.is_tensor(inputs.get("first_frame_latents"))
                    else None
                ),
            },
            "config": {
                "noise_prob": self.noise_prob,
                "noise_scale": self.noise_scale,
                "latent_prob": self.latent_prob,
                "latent_scale": self.latent_scale,
                "clean_context_prob": self.clean_context_prob,
                "clean_context_scale": self.clean_context_scale,
                "clean_context_frames": self.clean_context_frames,
                "clean_prob": self.clean_prob,
                "clean_buffer_update_prob": self.clean_buffer_update_prob,
                "buffer_warmup_steps": self.buffer_warmup_steps,
            },
            "rms": {},
        }
        if random.random() < self.clean_prob:
            debug["use_clean_input"] = True
            for key in debug["injected"]:
                debug["skip_reason"][key] = "clean_input"
            return {
                "model_latents": clean_latents,
                "target_noise": noise,
                "context": debug,
            }
        target_noise = self._maybe_add(
            "noise",
            grid_id,
            noise,
            enabled=self.inject_noise,
            prob=self.noise_prob,
            scale=self.noise_scale,
            debug=debug,
            name="sampled_noise",
        )
        model_latents = self._maybe_add(
            "latent",
            grid_id,
            clean_latents,
            enabled=self.inject_latent,
            prob=self.latent_prob,
            scale=self.latent_scale,
            debug=debug,
            name="input_latents",
        )
        self._inject_first_frame_tail(inputs, grid_id, debug)
        return {
            "model_latents": model_latents,
            "target_noise": target_noise,
            "context": debug,
        }

    def _maybe_add(self, bank, grid_id, target, *, enabled, prob, scale, debug, name):
        key = f"{bank}_to_{name}"
        if not enabled:
            debug["skip_reason"][key] = "disabled"
            return target
        if scale == 0.0:
            debug["skip_reason"][key] = "scale_zero"
            return target
        if random.random() >= prob:
            debug["skip_reason"][key] = "prob_gate"
            return target
        error = self.bank.sample(bank, grid_id, target)
        if error is None:
            debug["skip_reason"][key] = "empty_bucket"
            return target
        if error.shape != target.shape:
            debug["skip_reason"][key] = (
                f"shape_mismatch error={tuple(error.shape)} target={tuple(target.shape)}"
            )
            return target
        applied = error * float(scale)
        debug["injected"][key] = True
        debug["skip_reason"][key] = None
        debug["rms"][f"{name}_injection"] = _rms(applied)
        return target + applied

    def _inject_first_frame_tail(self, inputs, grid_id, debug):
        key = "latent_to_first_frame_tail"
        target = inputs.get("first_frame_latents")
        reference = inputs.get("input_latents")
        if target is None:
            debug["skip_reason"][key] = "missing_first_frame_latents"
            return
        if reference is None:
            debug["skip_reason"][key] = "missing_input_latents"
            return
        if not self.inject_clean_context:
            debug["skip_reason"][key] = "disabled"
            return
        if self.clean_context_scale == 0.0:
            debug["skip_reason"][key] = "scale_zero"
            return
        if random.random() >= self.clean_context_prob:
            debug["skip_reason"][key] = "prob_gate"
            return
        error = self.bank.sample("latent", grid_id, reference)
        if error is None:
            debug["skip_reason"][key] = "empty_bucket"
            return
        if (
            error.shape[:2] != target.shape[:2]
            or error.shape[-2:] != target.shape[-2:]
        ):
            debug["skip_reason"][key] = (
                f"shape_mismatch error={tuple(error.shape)} target={tuple(target.shape)}"
            )
            return
        frames = min(self.clean_context_frames, int(target.shape[2]))
        applied = error[:, :, -frames:] * self.clean_context_scale
        if int(applied.shape[2]) < frames:
            applied = applied[:, :, -1:].expand(-1, -1, frames, -1, -1)
        first_frame_latents = target.clone()
        first_frame_latents[:, :, -frames:] = target[:, :, -frames:] + applied
        inputs["first_frame_latents"] = first_frame_latents
        debug["injected"][key] = True
        debug["skip_reason"][key] = None
        debug["rms"]["first_frame_tail_injection"] = _rms(applied)

    def update(self, *, pipe, inputs, result):
        timestep = result["timestep"]
        noisy_latents = result["noisy_latents"]
        training_target = result["training_target"]
        noise_pred = result["noise_pred"]
        grid_id = int(result["context"]["grid_id"])
        update_buffer = True
        if bool(result["context"].get("use_clean_input", False)):
            update_buffer = random.random() < self.clean_buffer_update_prob
        result["context"]["update_buffer"] = update_buffer
        with torch.no_grad():
            noise_error = self._endpoint_error(
                pipe.scheduler,
                noise_pred,
                training_target,
                timestep,
                noisy_latents,
                sigma_to=1.0,
            )
            latent_error = self._endpoint_error(
                pipe.scheduler,
                noise_pred,
                training_target,
                timestep,
                noisy_latents,
                sigma_to=0.0,
            )
            mode = self._update_bank_with_warmup_gather(
                pipe.scheduler,
                noise_error,
                latent_error,
                timestep,
                fallback_grid_id=grid_id,
                update_buffer=update_buffer,
            )
        result["context"]["buffer_update_mode"] = mode
        result["context"]["noise_error_rms"] = _rms(noise_error)
        result["context"]["latent_error_rms"] = _rms(latent_error)
        result["context"]["shapes"]["noise_error"] = tuple(noise_error.shape)
        result["context"]["shapes"]["latent_error"] = tuple(latent_error.shape)
        self.last_metrics = self._metrics_from_context(result["context"])
        if self.debug:
            self._print_debug(result)

    def _metrics_from_context(self, ctx):
        rms = ctx.get("rms", {})
        occupancy = self.bank.occupancy()
        return {
            "occupancy/noise": float(occupancy.get("noise", 0)),
            "occupancy/latent": float(occupancy.get("latent", 0)),
            "rms/noise_error": float(ctx.get("noise_error_rms", 0.0)),
            "rms/latent_error": float(ctx.get("latent_error_rms", 0.0)),
            "rms/sampled_noise_injection": float(
                rms.get("sampled_noise_injection", 0.0)
            ),
            "rms/input_latents_injection": float(
                rms.get("input_latents_injection", 0.0)
            ),
            "rms/first_frame_tail_injection": float(
                rms.get("first_frame_tail_injection", 0.0)
            ),
        }

    def _endpoint_error(
        self, scheduler, noise_pred, training_target, timestep, noisy_latents, sigma_to
    ):
        pred = self._scheduler_endpoint(
            scheduler, noise_pred, timestep, noisy_latents, sigma_to=sigma_to
        )
        target = self._scheduler_endpoint(
            scheduler, training_target, timestep, noisy_latents, sigma_to=sigma_to
        )
        return pred - target

    def _scheduler_endpoint(self, scheduler, model_output, timestep, sample, sigma_to):
        timestep_cpu = timestep.detach().cpu() if isinstance(timestep, torch.Tensor) else timestep
        timestep_id = torch.argmin((scheduler.timesteps - timestep_cpu).abs())
        sigma = scheduler.sigmas[timestep_id].to(device=sample.device, dtype=sample.dtype)
        return sample + model_output * (float(sigma_to) - sigma)

    def _update_bank_with_warmup_gather(
        self,
        scheduler,
        noise_error,
        latent_error,
        timestep,
        fallback_grid_id,
        update_buffer,
    ):
        if self.step > self.buffer_warmup_steps:
            if not update_buffer:
                return "skipped"
            self._add_error_pair(fallback_grid_id, noise_error, latent_error)
            return "local"

        gathered = self._gather_error_triplet(noise_error, latent_error, timestep)
        if not update_buffer:
            return "warmup_gather_skipped" if gathered is not None else "skipped"
        if gathered is None:
            self._add_error_pair(fallback_grid_id, noise_error, latent_error)
            return "warmup_local"

        gathered_noise, gathered_latent, gathered_timesteps = gathered
        for item_noise, item_latent, item_timestep in zip(
            gathered_noise, gathered_latent, gathered_timesteps
        ):
            grid_id = self.bank.grid_id(scheduler, item_timestep)
            self._add_error_pair(grid_id, item_noise, item_latent)
        return "warmup_gather"

    def _add_error_pair(self, grid_id, noise_error, latent_error):
        if not _is_finite_tensor(noise_error) or not _is_finite_tensor(latent_error):
            return False
        self.bank.add("noise", grid_id, noise_error)
        self.bank.add("latent", grid_id, latent_error)
        return True

    def _gather_error_triplet(self, noise_error, latent_error, timestep):
        try:
            import torch.distributed as dist
        except (ImportError, AttributeError):
            return None

        if (
            not dist.is_available()
            or not dist.is_initialized()
            or int(dist.get_world_size()) <= 1
        ):
            return None
        gathered_noise = [torch.empty_like(noise_error) for _ in range(dist.get_world_size())]
        gathered_latent = [
            torch.empty_like(latent_error) for _ in range(dist.get_world_size())
        ]
        timestep_tensor = timestep.detach().to(device=noise_error.device)
        gathered_timesteps = [
            torch.empty_like(timestep_tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_noise, noise_error.detach())
        dist.all_gather(gathered_latent, latent_error.detach())
        dist.all_gather(gathered_timesteps, timestep_tensor)
        return gathered_noise, gathered_latent, gathered_timesteps

    def _print_debug(self, result):
        timestep = result["timestep"]
        timestep_value = (
            float(timestep.detach().flatten()[0].cpu().item())
            if isinstance(timestep, torch.Tensor)
            else float(timestep)
        )
        ctx = result["context"]
        print(
            "[svi][debug] "
            f"step={self.step} "
            f"grid={ctx['grid_id']} "
            f"timestep={timestep_value} "
            f"loss={float(result['loss'].detach().float().item())} "
            f"bank_occupancy={self.bank.occupancy()} "
            f"bucket_occupancy={ctx['bucket_occupancy']} "
            f"use_clean_input={ctx['use_clean_input']} "
            f"update_buffer={ctx['update_buffer']} "
            f"buffer_update_mode={ctx['buffer_update_mode']} "
            f"injected={ctx['injected']} "
            f"skip_reason={ctx['skip_reason']} "
            f"shapes={ctx['shapes']} "
            f"config={ctx['config']} "
            f"rms={ctx['rms']} "
            f"noise_error_rms={ctx['noise_error_rms']} "
            f"latent_error_rms={ctx['latent_error_rms']}",
            flush=True,
        )
