from .base_pipeline import BasePipeline
import torch


def _subject_object_loss_mask(inputs, target):
    mask = inputs.get("subject_object_loss_mask")
    alpha = float(inputs.get("subject_object_loss_alpha", 0.0) or 0.0)
    if alpha <= 0.0 or mask is None:
        return None, 0.0, 0.0
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=target.device, dtype=torch.bool)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 5:
        raise ValueError(
            "subject_object_loss_mask must have shape (T,H,W), (B,T,H,W), "
            f"or (B,1,T,H,W), got {tuple(mask.shape)}."
        )
    if int(mask.shape[0]) == 1 and int(target.shape[0]) > 1:
        mask = mask.expand(int(target.shape[0]), -1, -1, -1, -1)
    if int(mask.shape[1]) == 1 and int(target.shape[1]) > 1:
        mask = mask.expand(-1, int(target.shape[1]), -1, -1, -1)
    if tuple(mask.shape) != tuple(target.shape):
        raise ValueError(
            "subject_object_loss_mask must broadcast to training target shape, "
            f"got mask={tuple(mask.shape)} target={tuple(target.shape)}."
        )
    min_mask_ratio = float(
        inputs.get("subject_object_loss_min_mask_ratio", 0.0) or 0.0
    )
    return mask, alpha, max(0.0, min_mask_ratio)


def _flow_match_mse_loss_with_optional_subject_object(
    pipe, inputs, noise_pred, target, timestep
):
    diff2 = (noise_pred.float() - target.float()).pow(2)
    full_loss = diff2.mean()
    mask, alpha, min_mask_ratio = _subject_object_loss_mask(inputs, target)
    obj_loss = None
    bg_loss = None
    mask_ratio = 0.0
    mask_filtered = 0.0
    if mask is not None and bool(mask.any()):
        mask_f = mask.to(dtype=diff2.dtype)
        mask_ratio = float(mask_f.mean().detach().item())
        if min_mask_ratio > 0.0 and mask_ratio < min_mask_ratio:
            mask_filtered = 1.0
            loss_unweighted = full_loss
        else:
            denom = mask_f.sum().clamp_min(1.0)
            obj_loss = (diff2 * mask_f).sum() / denom
            inv_mask_f = 1.0 - mask_f
            bg_denom = inv_mask_f.sum().clamp_min(1.0)
            bg_loss = (diff2 * inv_mask_f).sum() / bg_denom
            loss_unweighted = full_loss + float(alpha) * obj_loss
    else:
        loss_unweighted = full_loss

    training_weight = pipe.scheduler.training_weight(timestep)
    loss = loss_unweighted * training_weight
    weight_value = float(training_weight.detach().float().reshape(-1)[0].item())
    pipe._last_loss_metrics = {
        "loss_full_unweighted": float(full_loss.detach().item()),
        "loss_total_unweighted": float(loss_unweighted.detach().item()),
        "subject_object_loss_alpha": float(alpha),
        "subject_object_loss_min_mask_ratio": float(min_mask_ratio),
        "subject_object_mask_ratio": float(mask_ratio),
        "subject_object_loss_active": 1.0 if obj_loss is not None else 0.0,
        "subject_object_loss_filtered": float(mask_filtered),
        "training_weight": weight_value,
    }
    if obj_loss is not None:
        pipe._last_loss_metrics["loss_subject_object_unweighted"] = float(
            obj_loss.detach().item()
        )
    if bg_loss is not None:
        pipe._last_loss_metrics["loss_background_unweighted"] = float(
            bg_loss.detach().item()
        )
    return loss


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    pipe._last_loss_metrics = {}
    return flow_match_sft_forward(pipe, inputs)["loss"]


def flow_match_sft_forward(
    pipe: BasePipeline,
    inputs: dict,
    prepare_hook=None,
    update_hook=None,
):
    pipe._last_loss_metrics = {}
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    clean_latents = inputs["input_latents"]
    noise = torch.randn_like(clean_latents)
    model_latents = clean_latents
    target_noise = noise
    hook_context = {}
    if prepare_hook is not None:
        prepared = prepare_hook(
            pipe=pipe,
            inputs=inputs,
            clean_latents=clean_latents,
            noise=noise,
            timestep=timestep,
        )
        if prepared is not None:
            model_latents = prepared.get("model_latents", model_latents)
            target_noise = prepared.get("target_noise", target_noise)
            hook_context = prepared.get("context", hook_context)
    inputs["latents"] = pipe.scheduler.add_noise(model_latents, target_noise, timestep)
    training_target = pipe.scheduler.training_target(clean_latents, target_noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = _flow_match_mse_loss_with_optional_subject_object(
        pipe, inputs, noise_pred, training_target, timestep
    )
    result = {
        "loss": loss,
        "timestep": timestep,
        "noise": noise,
        "target_noise": target_noise,
        "noise_pred": noise_pred,
        "training_target": training_target,
        "noisy_latents": inputs["latents"],
        "clean_latents": clean_latents,
        "model_latents": model_latents,
        "context": hook_context,
    }
    if update_hook is not None:
        update_hook(pipe=pipe, inputs=inputs, result=result)
    return result


def FlowMatchSVILoss(pipe: BasePipeline, svi_state, **inputs):
    return flow_match_sft_forward(
        pipe,
        inputs,
        prepare_hook=svi_state.prepare,
        update_hook=svi_state.update,
    )["loss"]


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
