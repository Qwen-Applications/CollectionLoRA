import torch
import os
import time
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from causvid.lora_ckpt_utils import normalize_lora_state_dict_for_peft_transformer



class FastFrozenLoraManager:
    def __init__(self, model, fsdp_module=None):
        self.model = model
        self.fsdp_module = fsdp_module  
        self.cpu_cache = {} 
        self.active_lora = None

    def register_lora(self, name, path):
        print(f"[Rank {dist.get_rank()}] Mapping LoRA {name} (Shared RAM)")

        if os.path.isdir(path):
            import glob

            files = glob.glob(os.path.join(path, "*.bin")) + glob.glob(os.path.join(path, "*.safetensors"))
            file_path = files[0]
        else:
            file_path = path

        state_dict = load_file(file_path, device='cpu')

        if any(str(k).startswith("lora_unet_") for k in state_dict):
            state_dict = normalize_lora_state_dict_for_peft_transformer(state_dict)
        else:
            keys = list(state_dict.keys())[0]

            if 'diffusion_model.' in keys:
                state_dict = {k.replace("diffusion_model.", "base_model.model."): v for k, v in state_dict.items() if "lora_" in k}
                keys = keys.replace("diffusion_model.", "base_model.model.")

            if 'model.base_model.model.' in keys:
                state_dict = {k.replace("model.base_model.model.", "base_model.model."): v for k, v in state_dict.items() if "lora_" in k}
                keys = keys.replace("model.base_model.model.", "base_model.model.")
            if 'default' in keys:
                state_dict = {k.replace("default.", ""): v for k, v in state_dict.items() if "lora_" in k}
                keys = keys.replace("default.", "")
            if 'base_model.model' not in keys:
                state_dict = {'base_model.model.'+k: v for k, v in state_dict.items() if "lora_" in k}
                keys = 'base_model.model.'+keys

        # insert lora_name to keys
        state_dict = {k.replace(".weight", ".current_lora.weight"): v for k, v in state_dict.items() if "lora_" in k}


        cached_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k or "adapter_" in k:
                cached_dict[k] = v  
        
        self.cpu_cache[name] = cached_dict
        

    def switch_lora(self, target_name):
        if self.active_lora == target_name:
            return
        
        if target_name not in self.cpu_cache:
            print(f"Warning: LoRA {target_name} not found! Keeping current state.")
            return


        target_dict = self.cpu_cache[target_name]
        num=0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                clean_name = name.replace('_fsdp_wrapped_module.','')
                if clean_name in target_dict:
                    target_tensor = target_dict[clean_name]
                    if param.shape == target_tensor.shape:
                        param.copy_(target_tensor)
                        num += 1
                    else:
                        raise ValueError(f"something wrong in fsdp wrapper")
        if num<1000:
            print(f"Copy {num} parameters from {target_name} to model")
            raise ValueError(f"Not enough parameters to copy from {target_name} to model")
        else:
            print(f"Copy {num} parameters from {target_name} to model")
        self.active_lora = target_name

    def register_identity_lora(self, name="base"):
        print(f"Creating Identity LoRA (Zero Weights) for '{name}'...")
        
        zero_dict = {}
        if self.fsdp_module is not None:
            with FSDP.summon_full_params(self.fsdp_module, writeback=False):
                for n, p in self.model.named_parameters():
                    if "lora_" in n or "adapter_" in n:
                        n = n.replace('_fsdp_wrapped_module.','')
                        zero_dict[n] = torch.zeros_like(p, device='cpu', dtype=p.dtype)
        else:
            for n, p in self.model.named_parameters():
                if "lora_" in n or "adapter_" in n:
                    n = n.replace('_fsdp_wrapped_module.','')
                    zero_dict[n] = torch.zeros_like(p, device='cpu', dtype=p.dtype)
        
        self.cpu_cache[name] = zero_dict