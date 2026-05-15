import os
from causvid.models.model_interface import InferencePipelineInterface
from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper,
    get_inference_pipeline_wrapper
)
from causvid.loss import get_denoising_loss
import torch.nn.functional as F
from typing import Tuple
from torch import nn
import torch
import torch.distributed as dist
import time
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
import torch.distributed as dist
from torchvision import transforms
from PIL import Image

import math
class DMD(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__()

        self.sync_timestep = getattr(args, "sync_timestep", False)
        self.target_prior_loss_weight = getattr(args, "target_prior_loss_weight", None)
        self.limited_timestep_num = getattr(args, "limited_timestep_num", -1)
        self.timestep_lower_bound = getattr(args, "timestep_lower_bound", 0)
        self.timestep_upper_bound = getattr(args, "timestep_upper_bound", 1000)
        self.sync_all_rank = getattr(args, "sync_all_rank", False)
        self.limited_timestep_num_for_dmd = getattr(args, "limited_timestep_num_for_dmd", None)
        self.generator_denoising_loss = getattr(args, "generator_denoising_loss", False)
        # Step 1: Initialize all models

        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.real_model_name = getattr(args, "real_name", args.model_name)
        self.fake_model_name = getattr(args, "fake_name", args.model_name)

        self.generator_task_type = getattr(
            args, "generator_task_type", args.generator_task)
        self.real_task_type = getattr(
            args, "real_task_type", args.generator_task)
        self.fake_task_type = getattr(
            args, "fake_task_type", args.generator_task)

        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.generator.set_module_grad(
            module_grad=args.generator_grad
        )
        
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")[
                'generator']
            self.generator.load_state_dict(
                state_dict, strict=True
            )

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.real_score = get_diffusion_wrapper(
            model_name=self.real_model_name)()
        self.real_score.set_module_grad(
            module_grad=args.real_score_grad
        )

        self.fake_score = get_diffusion_wrapper(
            model_name=self.fake_model_name)()
        self.fake_score.set_module_grad(
            module_grad=args.fake_score_grad
        )

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.text_encoder.requires_grad_(False)

        self.vae = get_vae_wrapper(model_name=args.model_name)()
        self.vae.requires_grad_(False)

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: InferencePipelineInterface = None

        # Step 2: Initialize all dmd hyperparameters

        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        self.num_train_timestep = args.num_train_timestep
        # self.min_step = int(0.02 * self.num_train_timestep)
        # self.max_step = int(0.98 * self.num_train_timestep)
        self.min_step = int(0.0 * self.num_train_timestep)
        self.max_step = int(1.0 * self.num_train_timestep)
        self.real_guidance_scale = args.real_guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
       
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.scheduler = self.generator.get_scheduler()
        self.denoising_loss_func = get_denoising_loss(
            args.denoising_loss_type)()

        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda().cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(
                device)
        else:
            self.scheduler.alphas_cumprod = None


        self.qwen_image_cfg_norm = getattr(args, "qwen_image_cfg_norm", False)

        # set generator real fake name
        self.generator.current_model_name = 'generator'
        self.real_score.current_model_name = 'real'
        self.fake_score.current_model_name = 'fake'
        self.step=0
    def _process_timestep(self, timestep: torch.Tensor, type: str) -> torch.Tensor:
        """
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.
            - type: a string indicating the type of the current model (image, bidirectional_video, or causal_video).
        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if type == "image":
            assert timestep.shape[1] == 1
            return timestep
        elif type == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif type == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError("Unsupported model type {}".format(type))

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        batch_size, num_frame, num_channels_latents, latent_height, latent_width = noisy_image_or_video.shape
        # Step 1: Compute the fake score
        pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale



        if self.args.qwen_image_cfg_norm:
            noise_pred = pred_real_image_cond.clone().permute(0, 2, 1, 3, 4)
            comb_pred = pred_real_image.clone().permute(0, 2, 1, 3, 4)

            noise_pred = self.generator._pack_latents(noise_pred, batch_size, num_channels_latents, latent_height, latent_width)
            comb_pred = self.generator._pack_latents(comb_pred, batch_size, num_channels_latents, latent_height, latent_width)

            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

            noise_pred = self.generator._unpack_latents(noise_pred, latent_height*8, latent_width*8, 8)
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4)
            pred_real_image = noise_pred

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        if normalization:
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
            "dmdtrain_noisy_latent": noisy_image_or_video.detach(),
            "dmdtrain_pred_real_image": pred_real_image.detach(),
            "dmdtrain_pred_fake_image": pred_fake_image.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self, image_or_video: torch.Tensor, conditional_dict: dict,
        unconditional_dict: dict, gradient_mask: torch.Tensor = None,timestep:torch.Tensor = None, another_image_or_video: torch.Tensor = None,timestep_upper_bound:int = 1000,timestep_lower_bound:int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            if timestep is None:
                timestep = torch.randint(
                    timestep_lower_bound,
                    timestep_upper_bound,
                    [batch_size, num_frame],
                    device=self.device,
                    dtype=torch.long
                )

                timestep = self._process_timestep(
                    timestep, type=self.real_task_type)

                # TODO: Add timestep warping
                if self.timestep_shift > 1:
                    timestep = self.timestep_shift * \
                        (timestep / 1000) / \
                        (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
                timestep = timestep.clamp(self.min_step, self.max_step)
            else:
                timestep = timestep

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            if another_image_or_video is not None:
                dmd_loss = 0.5 * F.mse_loss(another_image_or_video.double(
                ), (another_image_or_video.double() - grad.double()).detach(), reduction="mean")
            else:
                dmd_loss = 0.5 * F.mse_loss(original_latent.double(
                ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = get_inference_pipeline_wrapper(
            self.generator_model_name,
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block
        )

    @torch.no_grad()
    def _consistency_backward_simulation(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(noise=noise, conditional_dict=conditional_dict)

    def _run_generator(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.tensor,visual=False, return_clean_pred=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
        """
        # Step 1: Sample noise and backward simulate the generator's input
        if getattr(self.args, "backward_simulation", True):
            simulated_noisy_input = self._consistency_backward_simulation(
                noise=torch.randn(image_or_video_shape,
                                  device=self.device, dtype=self.dtype),
                conditional_dict=conditional_dict
            )
        else:
            simulated_noisy_input = []
            for timestep in self.denoising_step_list:
                noise = torch.randn(
                    image_or_video_shape, device=self.device, dtype=self.dtype)

                noisy_timestep = timestep * torch.ones(
                    image_or_video_shape[:2], device=self.device, dtype=torch.long)

                if timestep != 0:
                    noisy_image = self.scheduler.add_noise(
                        clean_latent.flatten(0, 1),
                        noise.flatten(0, 1),
                        noisy_timestep.flatten(0, 1)
                    ).unflatten(0, image_or_video_shape[:2])
                else:
                    noisy_image = clean_latent

                simulated_noisy_input.append(noisy_image)

            simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)



        # Step 2: Randomly sample a timestep and pick the corresponding input
        if self.limited_timestep_num_for_dmd is not None:
            limited_denoising_step_list_for_dmd = int(self.limited_timestep_num_for_dmd)
        else:
            limited_denoising_step_list_for_dmd = 0
        index = torch.randint(limited_denoising_step_list_for_dmd, len(self.denoising_step_list), [
                              image_or_video_shape[0], image_or_video_shape[1]], device=self.device, dtype=torch.long)
        index = self._process_timestep(index, type=self.generator_task_type)


        # select the corresponding timestep's noisy input from the stacked tensor [B, T, F, C, H, W]
        noisy_input = torch.gather(
            simulated_noisy_input, dim=1,
            index=index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1).expand(
                -1, -1, -1, *image_or_video_shape[2:])
        ).squeeze(1)

        timestep = self.denoising_step_list[index]

        pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        gradient_mask = None  # timestep != 0

        # pred_image_or_video = noisy_input * \
        #     (1-gradient_mask.float()).reshape(*gradient_mask.shape, 1, 1, 1) + \
        #     pred_image_or_video * gradient_mask.float().reshape(*gradient_mask.shape, 1, 1, 1)
        pred_image_or_video = pred_image_or_video.type_as(noisy_input)

        if return_clean_pred:

            clean_index = torch.randint(len(self.denoising_step_list)-1, len(self.denoising_step_list), [
                              image_or_video_shape[0], image_or_video_shape[1]], device=self.device, dtype=torch.long)
            clean_index = self._process_timestep(clean_index, type=self.generator_task_type)
            clean_noisy_input = torch.gather(
                simulated_noisy_input, dim=1,
                index=clean_index.reshape(clean_index.shape[0], 1, clean_index.shape[1], 1, 1, 1).expand(
                    -1, -1, -1, *image_or_video_shape[2:])
            ).squeeze(1)
            clean_timestep = self.denoising_step_list[clean_index]
            clean_pred_image_or_video = self.generator(
                noisy_image_or_video=clean_noisy_input,
                conditional_dict=conditional_dict,
                timestep=clean_timestep
            )
            clean_pred_image_or_video = clean_pred_image_or_video.type_as(noisy_input)


        if return_clean_pred:
            if visual:
                return pred_image_or_video, gradient_mask, noisy_input,timestep, clean_pred_image_or_video
            else:
                return pred_image_or_video, gradient_mask, clean_pred_image_or_video

        if visual:
            return pred_image_or_video, gradient_mask, noisy_input,timestep
        else:
            return pred_image_or_video, gradient_mask

        
        # return pred_image_or_video, gradient_mask

    def generator_loss(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor) -> Tuple[torch.Tensor, dict,]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        visual = True
        pred_image, gradient_mask, noisy_input,timestep = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                visual=visual,
                return_clean_pred=False
            )


        # Step 2: Compute the DMD loss

        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask
        )


        if self.sync_timestep:
            timestep_sync = dmd_log_dict['timestep'].clone()
        else:
            timestep_sync = None


        if not self.sync_all_rank:
            if 'base_pred_latent' in conditional_dict:
                base_pred_latent = conditional_dict['base_pred_latent']
                valid_mask = 1.0
            else:
                base_pred_latent = torch.zeros_like(pred_image) 
                valid_mask = 0.0
        
            if self.target_prior_loss_weight is not None:                
                target_prior_dmd_loss,target_prior_log_dict  = self.compute_distribution_matching_loss_target_prior(
                    image_or_video=base_pred_latent,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    gradient_mask=gradient_mask,
                    timestep_sync=timestep_sync,
                    timestep_upper_bound=self.timestep_upper_bound,
                    timestep_lower_bound=self.timestep_lower_bound
                )
                
                dmd_loss = dmd_loss + valid_mask*target_prior_dmd_loss * self.target_prior_loss_weight
                if valid_mask > 0.0:
                    for key, value in target_prior_log_dict.items():
                        dmd_log_dict[f'target_prior_{key}'] = value
                    dmd_log_dict['target_prior_loss'] = target_prior_dmd_loss.item()
                    dmd_log_dict[f'target_prior_valid_mask'] = valid_mask
        else:
            if 'base_pred_latent' in conditional_dict:
                base_pred_latent = conditional_dict['base_pred_latent']
                if self.target_prior_loss_weight is not None:                    
                    target_prior_dmd_loss,target_prior_log_dict  = self.compute_distribution_matching_loss_target_prior(
                        image_or_video=base_pred_latent,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        gradient_mask=gradient_mask,
                        timestep_sync=timestep_sync,
                        timestep_upper_bound=self.timestep_upper_bound,
                        timestep_lower_bound=self.timestep_lower_bound
                    )
                    dmd_loss = dmd_loss + target_prior_dmd_loss * self.target_prior_loss_weight
                    for key, value in target_prior_log_dict.items():
                        dmd_log_dict[f'target_prior_{key}'] = value
                    dmd_log_dict['target_prior_loss'] = target_prior_dmd_loss.item()
            

        if self.generator_denoising_loss:
            if 'base_pred_latent' in conditional_dict:
                base_pred_latent = conditional_dict['base_pred_latent']
                valid_mask = 1.0
            else:
                base_pred_latent = torch.zeros_like(pred_image)
                valid_mask = 0.0
            generator_denoising_loss, generator_denoising_log_dict = self.denoising_loss_for_generator(image_or_video_shape, conditional_dict,base_pred_latent,generator_timestep=dmd_log_dict['timestep'].clone())

            dmd_loss = dmd_loss + generator_denoising_loss * valid_mask
            if valid_mask > 0.0:
                dmd_log_dict.update(generator_denoising_log_dict)
                dmd_log_dict['generator_denoising_loss'] = generator_denoising_loss.item()
        if visual:
            dmd_log_dict['input_noisy_image'] = noisy_input
            dmd_log_dict['input_timestep'] = timestep
        self.step += 1
        return dmd_loss, dmd_log_dict

    def compute_distribution_matching_loss_target_prior(self,image_or_video: torch.Tensor, conditional_dict: dict,
        unconditional_dict: dict, gradient_mask: torch.Tensor = None,timestep_sync: torch.Tensor = None,
        timestep_upper_bound:int = 1000,timestep_lower_bound:int = 0
    ) -> Tuple[torch.Tensor, dict]:
        # 1. use a target img to add a denoising list noise, and then generator generate a image, denoted as image_or_video_target_prior
        image_or_video_shape = image_or_video.shape
        if self.limited_timestep_num > 0:
            limited_denoising_step_list = self.denoising_step_list[-self.limited_timestep_num:]
            timestep_sync = None
        else:
            limited_denoising_step_list = self.denoising_step_list

        index = torch.randint(0, len(limited_denoising_step_list), [
                              image_or_video_shape[0], image_or_video_shape[1]], device=self.device, dtype=torch.long)
        index = self._process_timestep(index, type=self.generator_task_type)
        timestep = limited_denoising_step_list[index]* torch.ones(
                image_or_video.shape[:2], dtype=torch.long, device=image_or_video.device)
        noisy_image_or_video = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                torch.randn_like(image_or_video.flatten(0, 1)),
                timestep.flatten(0, 1)
            ).unflatten(0, image_or_video.shape[:2])
        pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )  # [B, F, C, H, W] 
        pred_image_or_video = pred_image_or_video.type_as(noisy_image_or_video)

        # 2. compute the distribution matching loss between real and fake with the image_or_video_target_prior
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
                image_or_video=pred_image_or_video,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                gradient_mask=gradient_mask,
                timestep=timestep_sync,
                timestep_upper_bound=timestep_upper_bound,
                timestep_lower_bound=timestep_lower_bound
        )

        return dmd_loss,dmd_log_dict
    

    def denoising_loss_for_generator(self, image_or_video_shape, conditional_dict: dict,clean_latent: torch.Tensor,generator_timestep: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute the denoising loss for the generator.
        """
        # limit the timestep it greater than generator loss timestep
        if generator_timestep is not None:
            min_timestep = int(generator_timestep.item())
            if min_timestep > self.num_train_timestep:
                min_timestep = self.num_train_timestep
            if min_timestep < 0:
                min_timestep = 0
        else:
            min_timestep = 0
        critic_timestep = torch.randint(
            min_timestep,
            self.num_train_timestep,
            image_or_video_shape[:2],
            device=self.device,
            dtype=torch.long
        )
        critic_timestep = self._process_timestep(
            critic_timestep, type=self.fake_task_type)

        # TODO: Add timestep warping
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(clean_latent)
        noisy_generated_image = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        pred_generated_image = self.generator(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the generator
        if self.args.denoising_loss_type == "flow":
            from causvid.models.qwen_image_edit.qwen_image_edit_wrapper import QwenImageEditPlusDiffusionWrapper
            flow_pred = QwenImageEditPlusDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_generated_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_generated_noise = None
        else:
            flow_pred = None
            pred_generated_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_generated_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        generator_denoising_loss = self.denoising_loss_func(
            x=clean_latent.flatten(0, 1),
            x_pred=pred_generated_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_generated_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )


        # Step 5: Debugging Log
        generator_denoising_log_dict = {
            "generator_denoising_clean_latent": clean_latent.detach(),
            "generator_denoising_noisy_latent": noisy_generated_image.detach(),
            "generator_denoising_pred_latent": pred_generated_image.detach(),
            "generator_denoising_timestep": critic_timestep.detach()
        }

        return generator_denoising_loss, generator_denoising_log_dict

    
    def critic_loss(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent
            )

        # Step 2: Compute the fake prediction
        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            image_or_video_shape[:2],
            device=self.device,
            dtype=torch.long
        )
        critic_timestep = self._process_timestep(
            critic_timestep, type=self.fake_task_type)

        # TODO: Add timestep warping
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from causvid.models.qwen_image_edit.qwen_image_edit_wrapper import QwenImageEditPlusDiffusionWrapper
            flow_pred = QwenImageEditPlusDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        critic_log_dict = {
            "critictrain_latent": generated_image.detach(),
            "critictrain_noisy_latent": noisy_generated_image.detach(), 
            "critictrain_pred_image": pred_fake_image.detach(), 
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
