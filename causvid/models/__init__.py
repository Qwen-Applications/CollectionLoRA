
from causvid.bidirectional_trajectory_pipeline import BidirectionalInferenceWrapper

from transformers.models.t5.modeling_t5 import T5Block

from .qwen_image_edit.qwen_image_edit_wrapper import QwenImageEditPlusDiffusionWrapper, QwenImageEditPlusTextEncoder, QwenImageEditPlusVAE



DIFFUSION_NAME_TO_CLASS = {
    "qwen-image-edit-plus": QwenImageEditPlusDiffusionWrapper,
}


def get_diffusion_wrapper(model_name):
    return DIFFUSION_NAME_TO_CLASS[model_name]


TEXTENCODER_NAME_TO_CLASS = {
    "qwen-image-edit-plus": QwenImageEditPlusTextEncoder,
}


def get_text_encoder_wrapper(model_name):
    return TEXTENCODER_NAME_TO_CLASS[model_name]


VAE_NAME_TO_CLASS = {
    "qwen-image-edit-plus": QwenImageEditPlusVAE,
}


def get_vae_wrapper(model_name):
    return VAE_NAME_TO_CLASS[model_name]


PIPELINE_NAME_TO_CLASS = {
    "qwen-image-edit-plus": BidirectionalInferenceWrapper,
}


def get_inference_pipeline_wrapper(model_name, **kwargs):
    return PIPELINE_NAME_TO_CLASS[model_name](**kwargs)


BLOCK_NAME_TO_BLOCK_CLASS = {
    "T5Block": T5Block
}


def get_block_class(model_name):
    return BLOCK_NAME_TO_BLOCK_CLASS[model_name]
