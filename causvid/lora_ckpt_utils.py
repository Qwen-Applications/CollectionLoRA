"""Shared LoRA checkpoint key normalization for PEFT Qwen image transformer."""

import re

# ComfyUI / Kohya-style flat keys: lora_unet_transformer_blocks_{i}_attn_..._{lora_down|lora_up}.weight
_UNET_TAIL_TO_MODULE = {
    "attn_add_k_proj": "attn.add_k_proj",
    "attn_add_q_proj": "attn.add_q_proj",
    "attn_add_v_proj": "attn.add_v_proj",
    "attn_to_add_out": "attn.to_add_out",
    "attn_to_k": "attn.to_k",
    "attn_to_q": "attn.to_q",
    "attn_to_v": "attn.to_v",
    "attn_to_out_0": "attn.to_out.0",
    "img_mlp_net_0_proj": "img_mlp.net.0.proj",
    "img_mlp_net_2": "img_mlp.net.2",
    "img_mod_1": "img_mod.1",
    "txt_mlp_net_0_proj": "txt_mlp.net.0.proj",
    "txt_mlp_net_2": "txt_mlp.net.2",
    "txt_mod_1": "txt_mod.1",
}


def _convert_lora_unet_flat_state_dict_to_peft(lora_state_dict):
    """Map lora_unet_transformer_blocks_* + lora_down/lora_up -> PEFT base_model.model.* + lora_A/B.weight.

    Keys must not include a ``.default.`` segment: ``set_peft_model_state_dict(..., adapter_name=...)`` inserts
    the adapter name (e.g. ``current_lora``) before the final parameter suffix and would mis-parse ``.default.``.
    """
    out = {}
    pat = re.compile(r"^transformer_blocks_(\d+)_(.+)$")
    for k, v in lora_state_dict.items():
        if k.endswith(".alpha"):
            continue
        if ".lora_down.weight" not in k and ".lora_up.weight" not in k:
            continue
        if not k.startswith("lora_unet_transformer_blocks_"):
            continue
        rest = k[len("lora_unet_") :]
        base, lor_part, w = rest.rsplit(".", 2)
        if w != "weight" or lor_part not in ("lora_down", "lora_up"):
            continue
        m = pat.match(base)
        if not m:
            raise ValueError(f"Unexpected lora_unet key (block): {k}")
        block_id, tail_us = m.group(1), m.group(2)
        mod_rel = _UNET_TAIL_TO_MODULE.get(tail_us)
        if mod_rel is None:
            raise ValueError(f"Unexpected lora_unet key (tail {tail_us!r}): {k}")
        ab = "lora_A" if lor_part == "lora_down" else "lora_B"
        peft_k = f"base_model.model.transformer_blocks.{block_id}.{mod_rel}.{ab}.weight"
        out[peft_k] = v
    return out


def normalize_lora_state_dict_for_peft_transformer(lora_state_dict):
    """
    Normalize various LoRA exports to keys expected by peft.get_peft_model(transformer, ...).

    Handles:
    - Comfy / A1111-style lora_unet_transformer_blocks_* (lora_down / lora_up)
    - Legacy PEFT-ish prefixes (diffusion_model., optional base_model.model., default adapter segment)
    """
    probe = next(
        (
            k
            for k in lora_state_dict
            if (".lora_down.weight" in k or ".lora_up.weight" in k or "lora_A" in k or "lora_B" in k)
        ),
        next(iter(lora_state_dict.keys()), None),
    )
    if probe is None:
        return {}

    if probe.startswith("lora_unet_transformer_blocks_"):
        converted = _convert_lora_unet_flat_state_dict_to_peft(lora_state_dict)
        if len(converted) == 0:
            raise ValueError("lora_unet_transformer_blocks_* seen but no lora_down/lora_up tensors converted")
        return converted

    keys = probe
    out = dict(lora_state_dict)
    if "diffusion_model." in keys:
        out = {k.replace("diffusion_model.", "base_model.model."): v for k, v in out.items() if "lora_" in k}
        keys = keys.replace("diffusion_model.", "base_model.model.")
    if "default.weight" not in keys:
        out = {k.replace(".weight", ".default.weight"): v for k, v in out.items() if "lora_" in k}
        keys = keys.replace(".weight", ".default.weight")
    if "base_model.model" not in keys:
        out = {"base_model.model." + k: v for k, v in out.items() if "lora_" in k}
        keys = keys.replace("base_model.model.", "")
    if "lora_unet_" in keys:
        out = {k.replace("lora_unet_", ""): v for k, v in out.items() if "lora_" in k}
        keys = keys.replace("lora_unet_", "")
    return out
