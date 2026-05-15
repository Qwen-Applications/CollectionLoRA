import os
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image

from causvid.models.model_interface import (
    DiffusionModelInterface,
    TextEncoderInterface,
    VAEInterface,
)
import pdb
from diffusers.utils.torch_utils import randn_tensor
import math
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    calculate_shift,
    retrieve_timesteps,
)

import torch.distributed as dist
import time
import PIL
from PIL import Image

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024
MODEL_PATH = "/primus_xpfs_workspace_T04/fangtai/MODELS/qwen-image-edit-2509"

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
class QwenImageEditPlusScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self):
        super().__init__()

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, C, H, W]
            - noise: the noise with shape [B, C, H, W]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, C, H, W]
        """

        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

class QwenImageEditPlusTextEncoder(TextEncoderInterface):
    def __init__(self):
        super().__init__()
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(MODEL_PATH, "text_encoder"),
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
        )
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            os.path.join(MODEL_PATH, "tokenizer")
        )
        self.processor = Qwen2VLProcessor.from_pretrained(
            os.path.join(MODEL_PATH, "processor")
        )
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.tokenizer_max_length = 1024
        self.prompt_template_encode_start_idx = 64




    @property
    def device(self):
        return next(self.parameters()).device


    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._extract_masked_hidden
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result
    
    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, img in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.prompt_template_encode

        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(base_img_prompt + e) for e in prompt]

        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask
    
    def forward(self, text_prompts: List[str], condition_images) -> dict:
        # condition_images, condition_image_sizes = self.preprocess_image_for_conditioning(image)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt=text_prompts, image=condition_images)
        return {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_embeds_mask}

class QwenImageEditPlusVAE(VAEInterface):
    def __init__(self):
        super().__init__()
        self.mean: List[float] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
        self.std: List[float] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
        self.z_dim = 16
        self.model = AutoencoderKLQwenImage.from_pretrained(
            os.path.join(MODEL_PATH, "vae"),
            torch_dtype=torch.bfloat16, # 建议保持 bf16
            low_cpu_mem_usage=True
        )
        self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024
        self.latent_channels = 16
        self.model.enable_slicing()

    def preprocess_image_for_conditioning(self, image) -> torch.Tensor:
        if image is not None and not (isinstance(image, torch.Tensor) and image.shape[1] == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            condition_image_sizes = []
            condition_images = []
            vae_image_sizes = []
            vae_images = []
            for img in image:
                if isinstance(img, Image.Image):
                    image_width, image_height = img.size[0], img.size[1]
                else:
                    image_height, image_width = img.shape[2],img.shape[3]
                condition_width, condition_height = calculate_dimensions(
                    CONDITION_IMAGE_SIZE, image_width / image_height
                )
                vae_width, vae_height = image_width, image_height
                condition_image_sizes.append((condition_width, condition_height))
                vae_image_sizes.append((vae_width, vae_height))
                condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
                vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

        return condition_images, condition_image_sizes, vae_images, vae_image_sizes

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.model.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.model.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.mean)
            .view(1, self.z_dim, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.std)
            .view(1, self.z_dim, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents
    
    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.latent_channels:
                    image_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    image_latents = image
                if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                    # expand init_latents for batch_size
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
                elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([image_latents], dim=0)

                image_latent_height, image_latent_width = image_latents.shape[3:]
                image_latents = QwenImageEditPlusDiffusionWrapper._pack_latents(
                    image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
                )
                all_image_latents.append(image_latents)
            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = QwenImageEditPlusDiffusionWrapper._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents
    
    def decode_to_pixel(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)

        latents = latents.to(self.model.dtype)
        

        latents_mean = (
            torch.tensor(self.mean)
            .view(1, self.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.std).view(1, self.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        

        image = self.model.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image)
        return image

    
class QwenImageEditPlusDiffusionWrapper(DiffusionModelInterface):
    def __init__(self, current_model_name=None):        
        super().__init__()
        self.default_sample_size = 128
        self.vae_scale_factor = 8
        self.model = QwenImageTransformer2DModel.from_pretrained(
            os.path.join(MODEL_PATH, "transformer"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        self.scheduler = QwenImageEditPlusScheduler.from_pretrained(
            os.path.join(MODEL_PATH, "scheduler")
        )
        self.current_model_name = None

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, 1, C, H, W]
        xt: the input noisy data with shape [B, 1, C, H, W]
        timestep: the timestep with shape [B]
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )


        
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1,1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, 1, C, H, W]
        xt: the input noisy data with shape [B, 1, C, H, W]
        timestep: the timestep with shape [B]
        """

        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)

        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def forward(self, 
                noisy_image_or_video: torch.Tensor, 
                conditional_dict: dict, 
                timestep: torch.Tensor, 
                kv_cache: Optional[List[dict]] = None,
                crossattn_cache: Optional[List[dict]] = None,
                current_start: Optional[int] = None,
                current_end: Optional[int] = None,
    ) -> torch.Tensor:

        # CausVid：B, F, C, H, W
        # QwenImage: B, C, F, H, W
        noisy_image_or_video = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        batch_size, num_channels_latents,_, latent_height, latent_width = noisy_image_or_video.shape
        height, width = latent_height * self.vae_scale_factor, latent_width * self.vae_scale_factor

        if (self.current_model_name == 'generator' or self.current_model_name == 'fake') and 'recaption_prompt_embeds' in conditional_dict:
            prompt_embeds = conditional_dict["recaption_prompt_embeds"]
            prompt_embeds_mask = conditional_dict["recaption_prompt_embeds_mask"]
        else:
            prompt_embeds = conditional_dict["prompt_embeds"]
            prompt_embeds_mask = conditional_dict["prompt_embeds_mask"]


        image_latents = conditional_dict["image_latents"]
        vae_image_sizes = conditional_dict["vae_image_sizes"]


        img_shapes = [
            [
                (1, latent_height // 2, latent_width // 2),
                *[
                    (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                    for vae_width, vae_height in vae_image_sizes
                ],
            ]
        ] * batch_size
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        
        noisy_image_or_video = self._pack_latents(noisy_image_or_video, batch_size, num_channels_latents, latent_height, latent_width)

        latent_model_input = torch.cat([noisy_image_or_video, image_latents], dim=1)

        timestep = timestep.squeeze(1)
        noise_pred = self.model(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]
        noise_pred = noise_pred[:, : noisy_image_or_video.size(1)]

        noise_pred = self._unpack_latents(noise_pred, height, width, self.vae_scale_factor)
        noisy_image_or_video = self._unpack_latents(noisy_image_or_video, height, width, self.vae_scale_factor)

        x0_pred  = self._convert_flow_pred_to_x0(noise_pred, noisy_image_or_video, timestep)


        x0_pred = x0_pred.permute(0, 2, 1, 3, 4)
        return x0_pred
