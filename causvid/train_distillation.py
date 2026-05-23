import sys
from pathlib import Path

# Repo root (directory that contains the `causvid` package). Needed when the entrypoint is
# `torchrun ... causvid/train_distillation.py` without PYTHONPATH or editable install.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import os
# from causvid.data import ODERegressionLMDBDataset
from causvid.data import MultiLoraDataset
from causvid.models import get_block_class
from causvid.data import TextDataset
from causvid.data import QwenImageEditPlusDataset
from causvid.util import (
    launch_distributed_job,
    prepare_for_saving,
    set_seed, init_logging_folder,
    fsdp_wrap, cycle,
    fsdp_state_dict,
    barrier
)
import torch.distributed as dist
from omegaconf import OmegaConf, ListConfig, DictConfig
from causvid.dmd import DMD
from peft import LoraConfig, get_peft_model, TaskType, set_peft_model_state_dict
from peft.tuners.lora import LoraLayer
import argparse
import torch
import wandb
import time
import shutil
import json
import math
from safetensors.torch import load_file
import random
from torchvision import transforms
from lora_manager import FastFrozenLoraManager
from causvid.lora_ckpt_utils import normalize_lora_state_dict_for_peft_transformer


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


class Trainer:
    def __init__(self, config):
        self.config = config

        self.multi_lora_teacher = getattr(config, "multi_lora_teacher", False)
        self.if_use_lora_data = False

        self.sync_all_rank = getattr(config, "sync_all_rank", False)
        self.target_prior_loss_weight = getattr(config, "target_prior_loss_weight", None)
        self.recaption_prompt = getattr(config, "recaption_prompt", None)
        self.fake_real_sync_init = getattr(config,'fake_real_sync_init',False)
        self.generator_denoising_loss = getattr(config,'generator_denoising_loss',False)

        self.reload_lora_and_retrain_path = getattr(config,'reload_lora_and_retrain_path',None)

        self.merge_lighting = getattr(config, "merge_lighting", False)
        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process:
            self.output_path, self.wandb_folder = init_logging_folder(config)

        # Step 2: Initialize the model and optimizer
        if config.distillation_loss == "dmd":
            self.distillation_model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distillation loss type")

        self.use_lora = getattr(config, "use_lora", False)
        self.lora_path = getattr(config, "lora_path", None)

        self.lora_type_path = getattr(config, "lora_type_path", None)  
       
        if self.reload_lora_and_retrain_path is not None:
            print(f"loading lora from {self.reload_lora_and_retrain_path}...")
            print(f"and reload the generator and fake score model..., this will retrain the lora")
            base_model = self.distillation_model.generator.model
            self.distillation_model.generator.model = self.load_lora(base_model, self.reload_lora_and_retrain_path, if_merge=False,static_key='generator')

            base_model = self.distillation_model.fake_score.model         
            self.distillation_model.fake_score.model = self.load_lora(base_model, self.reload_lora_and_retrain_path, if_merge=False,static_key='critic') 

            # init the student lora config
            lora_r = getattr(self.config, "lora_r", 128)
            lora_alpha = getattr(self.config, "lora_alpha", 128)
            lora_dropout = getattr(self.config, "lora_dropout", 0.0)
            lora_target_modules = getattr(self.config, "lora_target_modules", None)
            self.student_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=None,
            )
        else:
            # 3. load a new lora for generator, and this need to init the self.student_lora_config
            print("loading a new lora for generator...")
            base_model = self.distillation_model.generator.model
            self.distillation_model.generator.model = self.load_lora(base_model,lora_root_path=None, if_merge=False)

            print("loading a new lora for fake...")
            base_model = self.distillation_model.fake_score.model
            self.distillation_model.fake_score.model = self.load_lora(base_model,lora_root_path=None, if_merge=False)
                
            
            # 4. load the teacher lora for real model
            # 4.1 load lora jsonl file
            if self.lora_type_path is not None:
                base_model = self.distillation_model.real_score.model
                with open(self.lora_type_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        lora_name = 'current_lora'
                        lora_path = item['lora_path']
                        base_model = self.load_lora(base_model,lora_root_path=lora_path, if_merge=False,adapter_name=lora_name)
                        break
                self.distillation_model.real_score.model = base_model
                print(f"only one lora loaded for teacher model, just init")

                


        for name, param in self.distillation_model.generator.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            if 'lora_' in name and 'old' in name:
                param.requires_grad = False

        for name, param in self.distillation_model.fake_score.named_parameters():
            if getattr(self.config, "fake_full_finetune", False):
                param.requires_grad = True
            else:   
                if "lora_" not in name:
                    param.requires_grad = False
                if 'lora_' in name and 'ghost' in name:
                    param.requires_grad = False

        for name, param in self.distillation_model.real_score.named_parameters():
            param.requires_grad = False

    
        print("start init fsdp")
        self.distillation_model.generator = fsdp_wrap(
            self.distillation_model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.generator_fsdp_transformer_module),
                                ) if config.generator_fsdp_wrap_strategy == "transformer" else None
        )

        self.distillation_model.fake_score = fsdp_wrap(
            self.distillation_model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.fake_score_fsdp_transformer_module),
                                ) if config.fake_score_fsdp_wrap_strategy == "transformer" else None
        )

        self.distillation_model.text_encoder = fsdp_wrap(
            self.distillation_model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.text_encoder_fsdp_transformer_module),
                                ) if config.text_encoder_fsdp_wrap_strategy == "transformer" else None
        )


        if self.multi_lora_teacher:
            real_score_ignored_modules = []
            modules_to_move_to_gpu = []

            for m in self.distillation_model.real_score.modules():
                if isinstance(m, LoraLayer):
                    if hasattr(m, 'lora_A'):
                        real_score_ignored_modules.extend(m.lora_A.values())
                        modules_to_move_to_gpu.extend(m.lora_A.values())
                    
                    if hasattr(m, 'lora_B'):
                        real_score_ignored_modules.extend(m.lora_B.values())
                        modules_to_move_to_gpu.extend(m.lora_B.values())
                    
                    if hasattr(m, 'lora_embedding_A'):
                        real_score_ignored_modules.extend(m.lora_embedding_A.values())
                        modules_to_move_to_gpu.extend(m.lora_embedding_A.values())
                    if hasattr(m, 'lora_embedding_B'):
                        real_score_ignored_modules.extend(m.lora_embedding_B.values())
                        modules_to_move_to_gpu.extend(m.lora_embedding_B.values())

            print(f"Real Score: Identified {len(real_score_ignored_modules)} LoRA adapter modules to ignore.")

            current_device = torch.cuda.current_device()
            print(f"Moving ignored LoRA modules to {current_device}...")
            
            for mod in modules_to_move_to_gpu:
                mod.to(current_device)
            
            for mod in modules_to_move_to_gpu:
                for p in mod.parameters():
                    p.requires_grad = False

        else:
            real_score_ignored_modules=None

        self.distillation_model.real_score = fsdp_wrap(
            self.distillation_model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.real_score_fsdp_transformer_module),
                                ) if config.real_score_fsdp_wrap_strategy == "transformer" else None,
            ignored_modules=real_score_ignored_modules
        )


        if self.multi_lora_teacher:
            self.real_score_lora_manager = FastFrozenLoraManager(
                self.distillation_model.real_score.model,
                fsdp_module=self.distillation_model.real_score
            )
            self.real_score_lora_manager.register_identity_lora(name="base")
            print("Loading Other Teacher Real Score LoRA weights to CPU RAM...")
            lora_num = 0
            self.lora_name_to_jsonline = {}
            with open(self.lora_type_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    lora_name = item['lora_name']
                    lora_path = item['lora_path']
                    self.lora_name_to_jsonline[lora_name] = item
                    
                    self.real_score_lora_manager.register_lora(lora_name, lora_path)
                    lora_num += 1
            
            print(f"Real Score: {lora_num} LoRAs loaded to CPU Cache.")


        if not config.no_visualize:
            self.distillation_model.vae = self.distillation_model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.fake_score.parameters()
             if param.requires_grad],
            lr=config.fake_lr,
            betas=(config.beta1, config.beta2)
        )

        # Step 3: Initialize the dataloader

        self.backward_simulation = getattr(config, "backward_simulation", True)

        if self.backward_simulation:
            if self.config.model_name == "qwen-image-edit-plus" or self.config.model_name == "qwen-image-edit-2511":
                dataset = QwenImageEditPlusDataset(config.data_path,crop=getattr(config, "crop", "crop_and_resize_to_target_size")) if config.switch_lora_ratio != 1.0 else None
            else:
                dataset = TextDataset(config.data_path)
            self.lora_data_path = getattr(config, "lora_data_path", None)
            self.switch_lora_ratio = getattr(config, "switch_lora_ratio", 0.5)
            self.multi_lora_warmup_steps = getattr(config, "multi_lora_warmup_steps", -1)
            self.multi_lora_dataset_shuffle = getattr(config, "multi_lora_dataset_shuffle", False)
            if self.lora_data_path is not None and self.multi_lora_teacher:
                lora_dataset = MultiLoraDataset(self.lora_data_path)
                lora_sampler = torch.utils.data.distributed.DistributedSampler(
                    lora_dataset, shuffle=self.multi_lora_dataset_shuffle, drop_last=True)
                lora_dataloader = torch.utils.data.DataLoader(
                    lora_dataset, batch_size=config.batch_size, sampler=lora_sampler)
                self.lora_dataloader = cycle(lora_dataloader)

        if dataset is not None:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, drop_last=True)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.batch_size, sampler=sampler)
            self.dataloader = cycle(dataloader)
        else:
            print('switch_lora_ratio is 1.0, so no general dataset will be used')

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None
        # print(f"Trainer initialized on device: {self.device}")

        if self.is_main_process:
            print(f"Trainer initialized on device: {self.device}")
            print(self.config)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.is_main_process:
            generator_params = count_parameters(self.distillation_model.generator)
            critic_params = count_parameters(self.distillation_model.fake_score)
            text_encoder_params = count_parameters(self.distillation_model.text_encoder)
            vae_params = count_parameters(self.distillation_model.vae) if hasattr(self.distillation_model, "vae") else 0
            reward_params = count_parameters(self.distillation_model.reward_score_edit.model) if hasattr(self.distillation_model, "reward_score_edit") else 0



            total_trainable_params = (
                generator_params + critic_params + text_encoder_params + vae_params
            )
            print(f"Trainable parameters:")
            print(f"  Generator: {generator_params}")
            print(f"  Critic: {critic_params}")
            print(f"  Text Encoder: {text_encoder_params}")
            print(f"  VAE: {vae_params}")
            print(f"  Total: {total_trainable_params}")

    def load_lora(self,base_model,lora_root_path=None,if_merge=False,adapter_name=None,add_adapter=False,static_key=None):

        assert base_model is not None, "model must be set"

        try:
            base_dtype = next(base_model.parameters()).dtype
        except StopIteration:
            base_dtype = torch.bfloat16

        if lora_root_path is not None:
            # lora_root_path = _resolve_repo_path(lora_root_path)
            yaml_path = os.path.join(lora_root_path, "lora_config", "adapter_config.yaml")
            json_path = os.path.join(lora_root_path, "lora_config", "adapter_config.json")
            state_dict_path = os.path.join(lora_root_path, "model.pt")
            safetensors_path = os.path.join(lora_root_path, "model.safetensors")
            print(f"load a well-trained lora from {lora_root_path}")
        else:
            print("init a new lora to train, and this will init the self.student_lora_config")
            lora_r = getattr(self.config, "lora_r", 128)
            lora_alpha = getattr(self.config, "lora_alpha", 128)
            lora_dropout = getattr(self.config, "lora_dropout", 0.0)
            lora_target_modules = getattr(self.config, "lora_target_modules", None)
            self.student_lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=None,
            )
            peft_model = get_peft_model(base_model, self.student_lora_config)
            peft_model = peft_model.to(dtype=base_dtype)
            return peft_model
        
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                config = OmegaConf.load(f)
                r = config.r
                alpha = config.alpha
                dropout = config.dropout
                target_modules = config.target_modules
                self.lora_config = LoraConfig(
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    target_modules=target_modules,
                    bias="none",
                    task_type=None,
                )
        elif os.path.exists(json_path):
            with open(json_path, 'r') as f:
                lora_config_dict = json.load(f)
                self.lora_config = LoraConfig(**lora_config_dict)
        else:
            raise ValueError(f"Lora config not found in {lora_root_path}")
            
        
        
        if add_adapter:
            peft_model = base_model
            peft_model.add_adapter(adapter_name, self.lora_config)
        else:
            peft_model = get_peft_model(base_model, self.lora_config,adapter_name=adapter_name) if adapter_name is not None else get_peft_model(base_model, self.lora_config)

        peft_model = peft_model.to(dtype=base_dtype)

        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            state_dict = state_dict[static_key] if static_key is not None else state_dict
        elif os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            raise ValueError(f"Lora state dict not found in {lora_root_path}")

        # ComfyUI / Kohya flat keys vs legacy PEFT keys (see normalize_lora_state_dict_for_peft_transformer).
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

        mismatch_keys = set_peft_model_state_dict(peft_model, state_dict, adapter_name=adapter_name) if adapter_name is not None else set_peft_model_state_dict(peft_model, state_dict)
        lora_missing_key = 0
        for key in mismatch_keys.missing_keys:
            if 'lora_' in key:
                print(f"Missing key: {key}")
                lora_missing_key+=1
        

        print(f"Unexpected keys for lora: {len(mismatch_keys.unexpected_keys)}")
        print(f"Lora missing key: {lora_missing_key}")
        
        if len(mismatch_keys.unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys for lora: {len(mismatch_keys.unexpected_keys)}")
        if lora_missing_key > 0:
            raise ValueError(f"Lora missing key: {lora_missing_key}")


        if if_merge:
            peft_model = peft_model.merge_and_unload()

        return peft_model

    def save(self):
        print("Start gathering distributed model states...")
        
        full_generator_state_dict = fsdp_state_dict(self.distillation_model.generator)
        full_critic_state_dict = fsdp_state_dict(self.distillation_model.fake_score)
        
        if self.use_lora:
            generator_state_dict = {k: v for k, v in full_generator_state_dict.items() if "lora_" in k or "modules_to_save" in k}
            critic_state_dict = {k: v for k, v in full_critic_state_dict.items() if "lora_" in k or "modules_to_save" in k}
            if self.is_main_process:
                print(f"Saving LoRA weights only. Generator keys: {len(generator_state_dict)}, Critic keys: {len(critic_state_dict)}")
        else:
            generator_state_dict = full_generator_state_dict
            critic_state_dict = full_critic_state_dict


        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            
            if self.use_lora:
                config_dict = self.student_lora_config.to_dict()
                def make_json_serializable(obj):
                    if isinstance(obj, set):
                        return list(obj)
                    elif isinstance(obj, (list, ListConfig)):
                        return [make_json_serializable(item) for item in obj]
                    elif isinstance(obj, (dict, DictConfig)):
                        return {k: make_json_serializable(v) for k, v in obj.items()}
                    else:
                        return obj
                
                config_dict = make_json_serializable(config_dict)
                
                lora_config_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "lora_config")
                os.makedirs(lora_config_path, exist_ok=True)
                with open(os.path.join(lora_config_path, "adapter_config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2)

            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))


    def train_one_step(self):
        self.distillation_model.eval()  # prevent any randomness (e.g. dropout)

        TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
        VISUALIZE = self.step % (self.config.log_iters//10 )== 0 and not self.config.no_visualize

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        if 'qwen-image-edit' in self.config.model_name:
            if self.multi_lora_teacher and self.step%20 == 0 and self.step > self.multi_lora_warmup_steps:
                if not self.sync_all_rank:
                    self.if_use_lora_data = random.random() < self.switch_lora_ratio
                else:
                    decision_tensor = torch.tensor([0.0], device=self.device) 
                    if dist.get_rank() == 0:
                        if random.random() < self.switch_lora_ratio:
                            decision_tensor[0] = 1.0
                    dist.broadcast(decision_tensor, src=0)
                    self.if_use_lora_data = (decision_tensor.item() > 0.5)
                if dist.get_rank() in [0,1,2,3,4,5,6,7]: 
                    print(f"Rank {dist.get_rank()} use_lora: {self.if_use_lora_data}")
            if self.if_use_lora_data:
                batch = next(self.lora_dataloader)
                self.current_data_source = "lora_data"
            else:
                batch = next(self.dataloader)
                self.current_data_source = "base"


            if self.if_use_lora_data:
                current_lora_name = batch["lora_name"]
                if isinstance(current_lora_name, list):
                    current_lora_name = current_lora_name[0]
                # self.distillation_model.real_score.model.set_adapter(lora_name)
                self.real_score_lora_manager.switch_lora(current_lora_name)

                if self.target_prior_loss_weight is not None or self.generator_denoising_loss:
                    base_pred_image = batch['base_pred_image']

                if self.recaption_prompt is not None:
                    recaption_prompts = [self.lora_name_to_jsonline[current_lora_name][self.recaption_prompt]]
            elif self.multi_lora_teacher:
                self.real_score_lora_manager.switch_lora("base")
                if self.recaption_prompt is not None:
                    recaption_prompts = None

            source_image = batch["source_image"]
            text_prompts = batch["instruction"]
            clean_latent=None

        else:
            text_prompts = next(self.dataloader)
            clean_latent = None

        # support dynamic shape for qwen image edit, but not support multiple source images for now TODO
        if 'qwen-image-edit' in self.config.model_name:
            batch_size = len(source_image) # always 1 for qwen image edit
            assert batch_size == 1, "batch_size should be 1 for qwen image edit"
            if not isinstance(source_image, list):
                _,_,c,h, w = self.config.image_or_video_shape
                origin_h, origin_w = source_image.shape[2:]
                vae_scaling_factor = self.distillation_model.vae.vae_scale_factor
                latent_h , latent_w = origin_h // vae_scaling_factor , origin_w // vae_scaling_factor
                image_or_video_shape = [batch_size, 1, c, latent_h, latent_w]
            else:
                raise ValueError(f"Not supported multiple source images: {source_image}")
        else:
            batch_size = len(text_prompts)
            image_or_video_shape = list(self.config.image_or_video_shape)
            image_or_video_shape[0] = batch_size



        # Step 2: Extract the conditional infos
        with torch.no_grad():
            if self.config.model_name == "qwen-image" or self.config.model_name == "wan" or self.config.model_name == "sdxl":
                conditional_dict = self.distillation_model.text_encoder(
                    text_prompts=text_prompts)

                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.distillation_model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach()
                                        for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict
            elif 'qwen-image-edit' in self.config.model_name:
                condition_images, condition_image_sizes, vae_images, vae_image_sizes = self.distillation_model.vae.preprocess_image_for_conditioning(source_image)
                conditional_dict = self.distillation_model.text_encoder(
                    text_prompts=text_prompts, condition_images=condition_images)

                width, height = vae_image_sizes[0]
                _, image_latents = self.distillation_model.vae.prepare_latents(
                    images=vae_images,
                    batch_size=batch_size,
                    num_channels_latents=self.distillation_model.vae.latent_channels,
                    height=height,
                    width=width,
                    dtype=self.dtype,
                    device=self.device,
                    generator=None,
                    latents=None,
                )
                conditional_dict["image_latents"] = image_latents
                conditional_dict["vae_image_sizes"] = vae_image_sizes

                if self.recaption_prompt is not None:
                    if self.if_use_lora_data and recaption_prompts is not None:
                        recaption_conditional_dict = self.distillation_model.text_encoder(text_prompts=recaption_prompts, condition_images=condition_images)
                        conditional_dict["recaption_prompt_embeds"] = recaption_conditional_dict["prompt_embeds"]
                        conditional_dict["recaption_prompt_embeds_mask"] = recaption_conditional_dict["prompt_embeds_mask"]
                    elif not self.sync_all_rank:
                        forward_text_prompts = [' ']
                        _ = self.distillation_model.text_encoder(text_prompts=forward_text_prompts, condition_images=condition_images)
                        # conditional_dict["recaption_prompt_embeds"] = recaption_conditional_dict["prompt_embeds"]
                        # conditional_dict["recaption_prompt_embeds_mask"] = recaption_conditional_dict["prompt_embeds_mask"]

                if self.if_use_lora_data and self.multi_lora_teacher:
                    if self.target_prior_loss_weight is not None or self.generator_denoising_loss:
                        _, _, vae_images, vae_image_sizes = self.distillation_model.vae.preprocess_image_for_conditioning(base_pred_image)
                        width, height = vae_image_sizes[0]
                        _, image_latents_reg = self.distillation_model.vae.prepare_latents(
                            images=vae_images,
                            batch_size=batch_size,
                            num_channels_latents=self.distillation_model.vae.latent_channels,
                            height=height,
                            width=width,
                            dtype=self.dtype,
                            device=self.device,
                            generator=None,
                            latents=None,
                        )

                        base_pred_latent = self.distillation_model.generator._unpack_latents(image_latents_reg, height, width, self.distillation_model.generator.vae_scale_factor)
                        base_pred_latent = base_pred_latent.permute(0, 2, 1, 3, 4)
                        conditional_dict["base_pred_latent"] = base_pred_latent

                unconditional_dict = self.distillation_model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size, condition_images=condition_images)

                unconditional_dict["image_latents"] = image_latents
                unconditional_dict["vae_image_sizes"] = vae_image_sizes

            else:
                raise ValueError(f"Invalid model name: {self.config.model_name}")
                    
        # Step 3: Train the generator
        if TRAIN_GENERATOR:

            generator_loss, generator_log_dict = self.distillation_model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
            ) 

            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_grad_norm = self.distillation_model.generator.clip_grad_norm_(
                self.max_grad_norm)
            self.generator_optimizer.step()
        else:
            generator_log_dict = {}

        # Step 4: Train the critic
        critic_loss, critic_log_dict = self.distillation_model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self.distillation_model.fake_score.clip_grad_norm_(
            self.max_grad_norm)
        self.critic_optimizer.step()

        # Step 5: Logging
        if self.is_main_process:
            wandb_loss_dict = {
                "critic_loss": critic_loss.item(),
                "critic_grad_norm": critic_grad_norm.item()
            }

            if TRAIN_GENERATOR:
                wandb_loss_dict.update(
                    {
                        "generator_loss": generator_loss.item(),
                        "generator_grad_norm": generator_grad_norm.item(),
                        "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].item(), 
                    }
                )
                generator_log_dict['dmd_loss'] = generator_loss.item()

                if 'target_prior_loss' in generator_log_dict and self.target_prior_loss_weight is not None:
                    wandb_loss_dict.update(
                        {
                            "target_prior_loss": generator_log_dict['target_prior_loss'],
                        }
                    )
                
                if 'generator_denoising_loss' in generator_log_dict and self.generator_denoising_loss:
                    wandb_loss_dict.update(
                        {
                            "generator_denoising_loss": generator_log_dict['generator_denoising_loss'],
                        }
                    )


            if VISUALIZE:        
                if dist.get_rank() == 0:
                    self.add_visualization(generator_log_dict, critic_log_dict, wandb_loss_dict)

                    # save the source image and instruction
                    visual_path = getattr(self.config, "visual_path", '/primus_xpfs_workspace_T04/fangtai/CausVid/debug/save_img')
                    visual_path = os.path.join(visual_path, f'step_{self.step:06d}')

                    os.makedirs(visual_path, exist_ok=True)
                    
                    instruction_file = os.path.join(visual_path, f'instruction.txt')
                    with open(instruction_file, 'w', encoding='utf-8') as f:
                        if isinstance(text_prompts, list):
                            for i, prompt in enumerate(text_prompts):
                                f.write(f"Sample {i}:\n{prompt}\n\n")
                        else:
                            f.write(str(text_prompts))

                    if self.recaption_prompt is not None and self.if_use_lora_data and self.multi_lora_teacher:
                        recaption_prompt_file = os.path.join(visual_path, f'recaption_prompt.txt')
                        with open(recaption_prompt_file, 'w', encoding='utf-8') as f:
                            if isinstance(recaption_prompts, list):
                                for i, prompt in enumerate(recaption_prompts):
                                    f.write(f"Sample {i}:\n{prompt}\n\n")
                            else:
                                f.write(str(recaption_prompts))
                    
                    if (self.config.model_name == "qwen-image-edit-2509" or self.config.model_name == "qwen-image-edit-2511") and source_image is not None:
                        from PIL import Image
                        
                        def tensor_to_pil(img_tensor):
                            img_to_save = img_tensor.clone()
                            if img_to_save.max() <= 1.0:
                                img_to_save = (img_to_save * 255).clamp(0, 255)
                            img_np = img_to_save.cpu().numpy()
                            
                            if img_np.ndim == 4:
                                img_np = img_np.squeeze(0) 
                        
                            if img_np.ndim == 3:
                                if img_np.shape[0] == 1:
                                    img_np = img_np.squeeze(0)
                                elif img_np.shape[0] == 3:
                                    img_np = img_np.transpose(1, 2, 0)
                                else:
                                    pass
                            img_np = img_np.astype('uint8')

                            if img_np.ndim == 2:
                                return Image.fromarray(img_np, mode='L')
                            elif img_np.ndim == 3:
                                return Image.fromarray(img_np, mode='RGB')
                            else:
                                raise ValueError(f"Wrong shape to visualize: {img_np.shape}")
                        
                        if isinstance(source_image, list):
                            for i, img in enumerate(source_image):
                                if hasattr(img, 'save'):
                                    img.save(os.path.join(visual_path, f'source_image_sample_{i}.png'))
                                elif isinstance(img, torch.Tensor):
                                    img_pil = tensor_to_pil(img)
                                    img_pil.save(os.path.join(visual_path, f'source_image_sample_{i}.png'))
                        else:
                            if hasattr(source_image, 'save'):
                                source_image.save(os.path.join(visual_path, f'source_image.png'))
                            elif isinstance(source_image, torch.Tensor):
                                img_pil = tensor_to_pil(source_image)
                                img_pil.save(os.path.join(visual_path, f'source_image.png'))
            wandb.log(wandb_loss_dict, step=self.step)


    def add_visualization(self, generator_log_dict, critic_log_dict, wandb_loss_dict):  
        with torch.no_grad():
            visual_path = getattr(self.config, "visual_path", '/primus_xpfs_workspace_T04/fangtai/CausVid/debug/save_img')
            visual_path = os.path.join(visual_path, f'step_{self.step:06d}')

            os.makedirs(visual_path, exist_ok=True)

            critictrain_latent, critictrain_noisy_latent, critictrain_pred_image = map(
                lambda x: self.distillation_model.vae.decode_to_pixel(
                    x)[0],
                [critic_log_dict['critictrain_latent'], critic_log_dict['critictrain_noisy_latent'],
                    critic_log_dict['critictrain_pred_image']]
            )

            critictrain_latent.save(os.path.join(visual_path, 'critictrain_latent.jpg'))
            critictrain_noisy_latent.save(os.path.join(visual_path, 'critictrain_noisy_latent.jpg'))
            critictrain_pred_image.save(os.path.join(visual_path, 'critictrain_pred_image.jpg'))


            if "dmdtrain_clean_latent" in generator_log_dict:
                (dmdtrain_clean_latent, dmdtrain_noisy_latent, dmdtrain_pred_real_image, dmdtrain_pred_fake_image) = map(
                    lambda x: self.distillation_model.vae.decode_to_pixel(
                        x)[0],
                    [generator_log_dict['dmdtrain_clean_latent'], generator_log_dict['dmdtrain_noisy_latent'],
                        generator_log_dict['dmdtrain_pred_real_image'], generator_log_dict['dmdtrain_pred_fake_image']]
                )
                dmdtrain_clean_latent.save(os.path.join(visual_path, 'dmdtrain_clean_latent.jpg'))
                dmdtrain_noisy_latent.save(os.path.join(visual_path, 'dmdtrain_noisy_latent.jpg'))
                dmdtrain_pred_real_image.save(os.path.join(visual_path, 'dmdtrain_pred_real_image.jpg'))
                dmdtrain_pred_fake_image.save(os.path.join(visual_path, 'dmdtrain_pred_fake_image.jpg'))
            
            if 'input_noisy_image' in generator_log_dict:
                input_noisy_image = generator_log_dict['input_noisy_image']
                input_timestep = generator_log_dict['input_timestep']
                input_noisy_image = self.distillation_model.vae.decode_to_pixel(input_noisy_image)[0]
                
                input_noisy_image.save(os.path.join(visual_path, 'input_noisy_image.jpg'))
                # timestep save to txt
                with open(os.path.join(visual_path, 'input_timestep.txt'), 'w') as f:
                    f.write(str(input_timestep))


            if 'target_prior_loss' in generator_log_dict and self.target_prior_loss_weight is not None:
                target_prior_visual_path = os.path.join(visual_path, 'target_prior')
                os.makedirs(target_prior_visual_path, exist_ok=True)
                # save loss
                with open(os.path.join(target_prior_visual_path, 'target_prior_loss.txt'), 'w') as f:
                    f.write(str(generator_log_dict['target_prior_loss']))
                with open(os.path.join(target_prior_visual_path, 'dmd_loss.txt'), 'w') as f:
                    f.write(str(generator_log_dict['dmd_loss']))

                for key, value in generator_log_dict.items():
                    if key == 'timestep':
                        with open(os.path.join(target_prior_visual_path, 'dmd_timestep.txt'), 'w') as f:
                                f.write(str(value))
                    if 'target_prior_' in key:
                        if 'clean_latent' in key:
                            clean_latent_visual = self.distillation_model.vae.decode_to_pixel(value)[0]
                            clean_latent_visual.save(os.path.join(target_prior_visual_path, f'{key}.jpg'))
                        elif 'noisy_latent' in key:
                            noisy_latent_visual = self.distillation_model.vae.decode_to_pixel(value)[0]
                            noisy_latent_visual.save(os.path.join(target_prior_visual_path, f'{key}.jpg'))
                        elif 'pred_real_image' in key:
                            pred_real_image_visual = self.distillation_model.vae.decode_to_pixel(value)[0]
                            pred_real_image_visual.save(os.path.join(target_prior_visual_path, f'{key}.jpg'))
                        elif 'pred_fake_image' in key:
                            pred_fake_image_visual = self.distillation_model.vae.decode_to_pixel(value)[0]
                            pred_fake_image_visual.save(os.path.join(target_prior_visual_path, f'{key}.jpg'))
                        # save the timestep
                        elif 'timestep' in key:
                            with open(os.path.join(target_prior_visual_path, f'{key}.txt'), 'w') as f:
                                f.write(str(value))

            if 'generator_denoising_loss' in generator_log_dict and self.generator_denoising_loss:
                generator_denoising_visual_path = os.path.join(visual_path, 'generator_denoising')
                os.makedirs(generator_denoising_visual_path, exist_ok=True)
                # save loss
                with open(os.path.join(generator_denoising_visual_path, 'generator_denoising_loss.txt'), 'w') as f:
                    f.write(str(generator_log_dict['generator_denoising_loss']))
                for key, value in generator_log_dict.items():
                    if 'generator_denoising_' in key:
                        if 'clean_latent' in key:
                            clean_latent_visual = self.distillation_model.vae.decode_to_pixel(value)[0]
                            clean_latent_visual.save(os.path.join(generator_denoising_visual_path, f'{key}.jpg'))
                        elif 'noisy_latent' in key:
                            noisy_latent_visual = self.distillation_model.vae.decode_to_pixel(value)[0]
                            noisy_latent_visual.save(os.path.join(generator_denoising_visual_path, f'{key}.jpg'))
                        elif 'pred_latent' in key:
                            pred_real_image_visual = self.distillation_model.vae.decode_to_pixel(value)[0]
                            pred_real_image_visual.save(os.path.join(generator_denoising_visual_path, f'{key}.jpg'))
                    
    def train(self):
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time -
                              self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = Trainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
