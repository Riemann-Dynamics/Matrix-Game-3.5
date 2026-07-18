import gc
import os

import torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    def on_step_end(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        save_steps=None,
        **kwargs,
    ):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(
        self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None
    ):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        # In DeepSpeed ZeRO-3 the params are sharded across ranks, so
        # `accelerator.get_state_dict` MUST be called collectively on every rank
        # to gather the full weights. In every other backend (DDP, ZeRO-2,
        # single-GPU) each rank already holds a full replica of the params, and
        # `get_state_dict` internally calls `clone_tensors_for_torch_save`,
        # which clones the whole model to CPU. Letting all ranks do that on
        # the same host multiplies host RAM by `num_processes` (e.g. 8x for an
        # 8-GPU node) and is the dominant source of end-of-epoch CPU OOMs.
        if self._needs_collective_state_dict(accelerator):
            state_dict = accelerator.get_state_dict(model)
        elif accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
        else:
            return

        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt
            )
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _needs_collective_state_dict(accelerator: Accelerator) -> bool:
        """Return True only for backends that require an all-rank gather to
        materialize a full state_dict (currently: DeepSpeed ZeRO-3)."""
        try:
            from accelerate.utils import DistributedType
        except ImportError:
            return False
        if accelerator.distributed_type != DistributedType.DEEPSPEED:
            return False
        plugin = getattr(accelerator, "deepspeed_plugin", None)
        if plugin is None:
            return False
        stage = getattr(plugin, "zero_stage", None)
        if stage is None:
            ds_config = getattr(plugin, "hf_ds_config", None)
            cfg = getattr(ds_config, "config", None) if ds_config is not None else None
            if isinstance(cfg, dict):
                stage = cfg.get("zero_optimization", {}).get("stage")
        return stage == 3
