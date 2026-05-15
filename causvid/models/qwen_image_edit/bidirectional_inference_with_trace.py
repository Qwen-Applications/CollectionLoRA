import os

from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List
import torch
from omegaconf import OmegaConf
import argparse

from tqdm import tqdm

import torchvision.transforms as transforms
import math
from PIL import Image



class BidirectionalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, lora_root_path=None):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        # Step 2: Initialize all bidirectional SDXL hyperparameters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)

        self.scheduler = self.generator.get_scheduler()
        if getattr(args, "warp_denoising_step", False):  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(device)
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
        self.negative_prompt = ' '

    def inference(self, noise: torch.Tensor, text_prompts: List[str], source_image: torch.Tensor, stochastic_sampling: bool = True, true_cfg_scale: float = 4.0,return_all_steps: bool = False) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            image (torch.Tensor): The generated image tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        batch_size, num_frames, num_channels, height, width = noise.shape
        condition_images, condition_image_sizes, vae_images, vae_image_sizes = self.vae.preprocess_image_for_conditioning(source_image)
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts, condition_images=condition_images)

        height,width = vae_image_sizes[0]
        _, image_latents = self.vae.prepare_latents(
            images=vae_images,
            batch_size=batch_size,
            num_channels_latents=self.vae.latent_channels,
            height=height,
            width=width,
            dtype=torch.bfloat16,
            device="cuda",
            generator=None,
            latents=None,
        )
        conditional_dict["image_latents"] = image_latents
        conditional_dict["vae_image_sizes"] = vae_image_sizes

        unconditional_dict = self.text_encoder(
        text_prompts=[self.negative_prompt] * batch_size, condition_images=condition_images)
        unconditional_dict["image_latents"] = image_latents
        unconditional_dict["vae_image_sizes"] = vae_image_sizes

        # initial point
        noisy_image_or_video = noise
        stochastic_sampling = True
        # print(stochastic_sampling)
        if stochastic_sampling:
            x_list = []
            x_nosie_list = []
            for index, current_timestep in tqdm(enumerate(self.denoising_step_list)):
                print(f"Current timestep: {current_timestep}")
                pred_image_or_video = self.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
                )  # [B, F, C, H, W]
                if true_cfg_scale >1.0:
                    pred_real_image_uncond = self.generator(
                        noisy_image_or_video=noisy_image_or_video,
                        conditional_dict=unconditional_dict,
                        timestep=torch.ones(
                            noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
                    )
                    pred_image_or_video = pred_real_image_uncond + (pred_image_or_video - pred_real_image_uncond) * true_cfg_scale
                x_list.append(pred_image_or_video.detach().clone())
                x_nosie_list.append(noisy_image_or_video.detach().clone())



                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device)

                    noisy_image_or_video = self.scheduler.add_noise(
                        pred_image_or_video.flatten(0, 1),
                        torch.randn_like(pred_image_or_video.flatten(0, 1)),
                        next_timestep.flatten(0, 1)
                    ).unflatten(0, noise.shape[:2])

            image = self.vae.decode_to_pixel(pred_image_or_video)
        if return_all_steps:
            return image, x_list, x_nosie_list
        else:
            return image