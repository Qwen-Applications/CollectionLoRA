<div align="center">

# CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher On-Policy Distillation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2605.25378>)
[![Project Page](https://img.shields.io/badge/Project-Page-159957.svg)](<https://collectionlora.github.io/>)
<!-- [![Comming Soon Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/<INSERT_MODEL_LINK_HERE>) -->

<br>

**Fangtai Wu**<sup>1,2</sup> · **Hailong Guo**<sup>2</sup> · **Shijie Huang**<sup>2</sup> · **Jiayi Song**<sup>2,3</sup> · **Yubo Huang**<sup>2</sup> · **[Mushui Liu](https://xiaobul.github.io/)**<sup>1</sup> · **Zhao Wang**<sup>1,\*</sup> · **Yunlong Yu**<sup>1,\*</sup> · **Jiaming Liu**<sup>2,\*‡</sup> · **Ruihua Huang**<sup>2</sup>

<sup>1</sup> Zhejiang University  ·  <sup>2</sup> Qwen Applications Business Group, Alibaba  ·  <sup>3</sup> Xi'an Jiaotong University

<sup>*</sup> Corresponding authors  ·  <sup>‡</sup> Project leader

[wft@zju.edu.cn](mailto:wft@zju.edu.cn)  ·  [liujiaming.ljl@alibaba-inc.com](mailto:liujiaming.ljl@alibaba-inc.com)

</div>

<br>

> **TL;DR:** **CollectionLoRA** is a **multi-teacher on-policy distillation** framework that distills the concepts from many effect LoRAs (e.g., **50~180**) **and** few-step generation capability into **a single LoRA**, cutting deployment cost and reducing interference from cascading many adapters with acceleration modules.

---

## 🌟 Highlights

| Key | Summary |
| :--- | :--- |
| **`50→1`** | **Beats the Teachers** — Higher **VSA** & lower **BCR** vs. per-effect Base on **EffectBench** @ **8 NFE** (Base: **40×2**). |
| **`8 NFE`** | **Two Abilities, One LoRA** — Many experts, few steps in **one LoRA**. |
| **`A ⊕ B`** | **Emergent Composition** — Zero-shot **A ⊕ B**: pair any two trained teachers at inference; **no extra training**. |
| **`180→1`**| **Mega Distillation** — One LoRA fits **180** effect teachers, competitive with single-effect baselines. |

---

### 📄 Todo List

#### 🌟 Milestones & Core Release

* ✅ **Release the paper**
* ✅ **Open-source training and inference code**
* [ ] **Open-source model weights**


## 🛠️ Installation

```bash
conda create -n collectionlora python=3.10 -y
conda activate collectionlora

pip install -r requirements.txt
pip install -e .

```

## 🚀 Inference

Download the weights and follow this `ckpt/` layout (paths relative to the repo root). Use `50_in_1/` for inference; `data/manga_tone/` holds the starter teacher LoRA and toy training assets referenced by the training demo.

```text
ckpt/
├── 50_in_1/
│   ├── model.pt
│   └── lora_config/
│       └── adapter_config.json
└── data/
    ├── general_data_demo.jsonl
    ├── 50_in_1_lora_type.jsonl
    ├── 180_in_1_lora_type.jsonl
    └── manga_tone/
        ├── models/
        │   ├── model.safetensors
        │   └── lora_config/
        │       └── adapter_config.yaml
        └── data/
            ├── lora_type.jsonl
            ├── train_data_with_base_pred.jsonl
            └── …

```

The script defaults to `ckpt/50_in_1` and the same config as training:

```bash
cd /path/to/CollectionLoRA
python minimal_inference/collectionlora/qwen_image_edit_inference.py

```

The `__main__` block runs several **zero-shot composition** demos (effect A, effect B, then A→B in one instruction). Replace the hard-coded prompts and image paths for your own data.

### Batch Inference

For scripted runs over the built-in eval subsets (`test_images/` per data type), use:

```bash
cd /path/to/CollectionLoRA
python minimal_inference/collectionlora/batch_infer.py

```

By default, the script samples up to **two** images per effect from `minimal_inference/collectionlora/test_images/{pet,animal,portrait}` according to each line in `lora_type_path`.

## ⚙️ Training

We provide a teacher LoRA to start the training demo. You can directly run the following command:

```bash
bash train_qwen_image_edit_plus_multi_lora_teacher.sh

```

**NOTE:** The template config `configs/multi_lora/train_demo.yaml` sets **`switch_lora_ratio: 1.0`**, so **no general dataset** is used (`data_path` is ignored in that mode).

> **Optional (PDSR):** To use **PDSR**, build your own general-data JSONL in the same schema as `ckpt/data/general_data_demo.jsonl`. Then point **`data_path`** in **`train_demo.yaml`** to that file and set **`switch_lora_ratio`** below **`1.0`** so those samples are sampled. Also, verify **`lora_data_path`** / **`lora_type_path`** match your effect JSONLs (defaults assume paths under `data/manga_tone/…`; adjust if you keep assets under `ckpt/data/manga_tone/…`).

## ✒️ Citation

If you find CollectionLoRA useful or relevant to your research, please kindly cite our papers:

```bibtex
@misc{wu2026collectionloracollecting50effects,
      title={CollectionLoRA: Collecting 50 Effects in 1 LoRA via Multi-Teacher On-Policy Distillation}, 
      author={Fangtai Wu and Hailong Guo and Shijie Huang and Jiayi Song and Yubo Huang and Mushui Liu and Zhao Wang and Yunlong Yu and Jiaming Liu and Ruihua Huang},
      year={2026},
      eprint={2605.25378},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.25378}, 
}

```

## 🙏 Acknowledgements

We would like to express our gratitude to the following projects:

*   [CausVid](https://github.com/tianweiy/CausVid)
*   [DMD/DMD2](https://github.com/tianweiy/DMD2)

