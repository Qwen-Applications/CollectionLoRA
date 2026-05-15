import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from causvid.models.qwen_image_edit.bidirectional_inference import BidirectionalInferencePipeline
from omegaconf import OmegaConf
import torch
import json
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model

from PIL import Image


import torchvision.transforms as transforms
import argparse
from causvid.data import QwenImageEditPlusDataset
import math


class QwenImageEditInference:
    def __init__(self, config_path, checkpoint_folder, merge_init_path=None, merge_lighting=False, height=1024, width=1024, seed=42, stochastic_sampling=True):

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
        
        self.height = height
        self.width = width
        self.seed = seed
        self.stochastic_sampling = stochastic_sampling
        

        config = OmegaConf.load(config_path)
        

        self.pipe = BidirectionalInferencePipeline(config, device="cuda")


        generator_ckpt = getattr(config, "generator_ckpt", None)
        if generator_ckpt is not None:
            state_dict = torch.load(generator_ckpt, map_location="cpu")['generator']
            self.pipe.generator.load_state_dict(state_dict, strict=True)



        lora_config_path = os.path.join(checkpoint_folder, "lora_config", "adapter_config.json")
        use_lora = os.path.exists(lora_config_path)
        
        if use_lora:

            with open(lora_config_path, 'r') as f:
                lora_config_dict = json.load(f)
            
            lora_config = LoraConfig(**lora_config_dict)
            

            base_model = getattr(self.pipe.generator, "model", None)
            if base_model is None:
                raise ValueError("无法找到generator.model，无法应用LoRA")
            

            try:
                base_dtype = next(base_model.parameters()).dtype
            except StopIteration:
                base_dtype = torch.bfloat16
            
            peft_model = get_peft_model(base_model, lora_config)
            peft_model = peft_model.to(dtype=base_dtype)
            self.pipe.generator.model = peft_model
            

        
        state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")['generator']
        
        if use_lora:
            lora_state_dict = {k: v for k, v in state_dict.items() if "lora_" in k or "modules_to_save" in k}
            missing_keys = self.pipe.generator.load_state_dict(lora_state_dict, strict=False)
            if len(missing_keys.unexpected_keys) > 0:
                raise ValueError(f"加载LoRA权重时缺失了 {len(missing_keys)} 个参数")
        else:
            self.pipe.generator.load_state_dict(state_dict, strict=True)
        
        self.pipe = self.pipe.to(device="cuda", dtype=torch.bfloat16)
        
        self.latent_height = self.height // 8
        self.latent_width = self.width // 8
        
        self.generator = torch.Generator(device="cuda")
        if self.seed is not None:
            self.generator.manual_seed(self.seed)
        
    
    def preprocess_image(self, image_path):

        image = Image.open(image_path)
        w, h = image.size
        target_size = self.height

        long_side = max(w, h)
        short_side = min(w, h)
        is_width_long = w >= h


        scale = target_size / long_side


        short_scaled = int(math.floor((short_side * scale) / 32) * 32)
        if short_scaled <= 0:
            short_scaled = 32


        desired_short_in_src = short_scaled / scale
        if desired_short_in_src < short_side:
            delta = short_side - desired_short_in_src
            crop_margin = int(math.floor(delta / 2))
            if is_width_long:
                top = crop_margin
                bottom = h - (delta - crop_margin)
                image = image.crop((0, top, w, bottom))
                h = bottom - top
            else:
                left = crop_margin
                right = w - (delta - crop_margin)
                image = image.crop((left, 0, right, h))
                w = right - left
            short_side = min(w, h)

        if is_width_long:
            resized_size = (target_size, short_scaled)
        else:
            resized_size = (short_scaled, target_size)
        image = image.resize(resized_size, Image.BICUBIC)
        return image
    
    def inference(self, source_image_path, instruction, output_dir, output_filename=None):
        
        file_name = os.path.basename(source_image_path)
        source_image = self.preprocess_image(source_image_path)

        output_path = os.path.join(output_dir, f"{file_name}.png")
        if os.path.exists(output_path):
            return
        source_image = transforms.ToTensor()(source_image)
        source_image = source_image.unsqueeze(0) if source_image.dim() == 3 else source_image
        source_image = source_image.to(device="cuda")
        height, width = source_image.shape[-2:]
        latent_height, latent_width = height // 8, width // 8
        print(f"source_image shape: {source_image.shape}, height: {height}, width: {width}")
        
        noise = torch.randn(
            1, 1, 16, latent_height, latent_width,
            generator=self.generator,
            dtype=torch.bfloat16,
            device="cuda"
        )
        
        image = self.pipe.inference(
            noise=noise,
            text_prompts=instruction,
            source_image=source_image,
            stochastic_sampling=self.stochastic_sampling
        )[0]
        
        if output_filename is None:
            output_filename = os.path.splitext(file_name)[0]
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        image.save(output_path)

        source_img_filename = f"{file_name}_source_img.png"
        source_img_path = os.path.join(output_dir, source_img_filename)
        source_image_pil = transforms.ToPILImage()(source_image.squeeze(0).cpu())
        source_image_pil.save(source_img_path)
        instruction_filename = f"{file_name}.txt"
        instruction_path = os.path.join(output_dir, instruction_filename)
        with open(instruction_path, 'w', encoding='utf-8') as f:
            f.write(str(instruction))


        
        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/multi_lora/collectionlora_50_in_1.yaml")
    parser.add_argument("--checkpoint_folder", type=str, default="ckpt/50_in_1")
    parser.add_argument("--output_dir", type=str, default="outputs/batch_infer")

    args = parser.parse_args()
    config_path = args.config_path
    checkpoint_folder = args.checkpoint_folder
    config = OmegaConf.load(config_path)
    # extract effect lora name
    lora_type_path = config.lora_type_path
    recaption_prompt = getattr(config,'recaption_prompt',None)
    merge_init_path = getattr(config,'generator_lora_path',None)
    merge_lighting= getattr(config,'merge_lighting',False)
    # test1: for effect lora
    eval_bench_mapping = {
        '宠物':'minimal_inference/collectionlora/test_images/pet',
        '动物':'minimal_inference/collectionlora/test_images/animal',
        '单人人像':'minimal_inference/collectionlora/test_images/portrait',
    }
    with torch.inference_mode():
        inference = QwenImageEditInference(config_path, checkpoint_folder, merge_init_path=merge_init_path, merge_lighting=merge_lighting)
        with open(lora_type_path, 'r', encoding='utf-8') as f:
            for line in f:
                lora_type = json.loads(line)
                current_lora_name = lora_type['lora_name']
                current_prompt = lora_type['prompt']
                current_data_type = lora_type['data_type']
                current_eval_bench_path = eval_bench_mapping[current_data_type]
                if recaption_prompt is not None:
                    current_prompt = lora_type[recaption_prompt]

                image_path_list = [os.path.join(current_eval_bench_path, f) for f in os.listdir(current_eval_bench_path) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                image_path_list = [f for f in image_path_list if os.path.exists(f)]
                output_dir = os.path.join(args.output_dir, "1-effect")
                output_dir = os.path.join(output_dir, current_lora_name)
                for index, image_path in enumerate(image_path_list):
                    if index > 1:
                        break
                    inference.inference(
                        source_image_path=image_path,
                        instruction=current_prompt,
                        output_dir=output_dir)
