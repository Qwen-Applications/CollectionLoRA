import os
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as transforms
import lmdb
import json

from PIL import Image

import math
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
import time
# import jsonlines
class TextDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class QwenImageEditPlusDataset(Dataset):
    def __init__(self, data_path, crop='crop_and_resize_to_target_size', general_data=False):
        # will load form jsonl
        self.crop = crop
        self.data_dicts = []
        # if data_path is jsonl, load from jsonl
        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                for line in f:
                    self.data_dicts.append(json.loads(line))
        elif os.path.isdir(data_path):
            for file in os.listdir(data_path):
                if file.endswith(".png"):
                    file_path = os.path.join(data_path, file.replace(".png", ".txt"))
                    with open(file_path, "r") as f:
                        instruction = f.read()
                    self.data_dicts.append({
                        'source_image': os.path.join(data_path, file),
                        'instruction': instruction
                    })
        else:
            raise ValueError(f"Invalid data path: {data_path}")


    def __len__(self):
        return len(self.data_dicts)

    @staticmethod
    def preprocess_image(image_path, eq_wh=1024):
        
        image = Image.open(image_path)
        w, h = image.size
        
        if w > h:
            left = (w - h) // 2
            right = left + h
            image = image.crop((left, 0, right, h))
        elif w < h:
            top = (h - w) // 2
            bottom = top + w
            image = image.crop((0, top, w, bottom))
        
        image = image.resize((eq_wh, eq_wh))
        return image
    
    def preprocess_image_crop_and_resize_to_target_size(self,image_path,target_size=1024):
        image = Image.open(image_path)
        w, h = image.size
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
        
    def load_from_oss_save_to_local(self,save_local_path):
        os.makedirs(save_local_path, exist_ok=True)
        for i, item in enumerate(self.data_dicts):
            image = QwenImageEditPlusDataset.preprocess_image(item['source_image'])
            image.save(f"{save_local_path}/image_{i}.png")
            with open(f"{save_local_path}/image_{i}.txt", "w") as f:
                f.write(item['instruction'])

    def get_image_path(self,idx):
        image = self.data_dicts[idx]['source_image']
        instruction = self.data_dicts[idx]['instruction']
        return {
            'source_image': image,
            'instruction': instruction
        }


    def __getitem__(self, idx):
        crop = self.crop
        # crop = None
        if crop=='eq_wh':
            image = QwenImageEditPlusDataset.preprocess_image(self.data_dicts[idx]['source_image'],eq_wh=1024)
        elif crop=='crop_and_resize_to_target_size':
            image = self.preprocess_image_crop_and_resize_to_target_size(self.data_dicts[idx]['source_image'],target_size=1024)
        else:
            image = Image.open(self.data_dicts[idx]['source_image'])
        image = transforms.ToTensor()(image)
        instruction = self.data_dicts[idx]['instruction']
        if 'edit_prompt' in self.data_dicts[idx]:
            edit_prompt = self.data_dicts[idx]['edit_prompt']
            return {
                "source_image": image,
                "instruction": instruction,
                "edit_prompt": edit_prompt
            }
        return {
            "source_image": image,
            "instruction": instruction
        }

class MultiLoraDataset(QwenImageEditPlusDataset):
    def __init__(self, data_path):
        super().__init__(data_path)
        item_0 = self.data_dicts[0]
        assert 'lora_name' in item_0, "lora_name is not in the data"
        assert 'source_image' in item_0, "source_image is not in the data"
        assert 'instruction' in item_0, "instruction is not in the data"
        # assert 'base_pred_image' in item_0, "base_pred_image is not in the data"


    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        """
            return image,instruction,lora_name
        """
        image_path = self.data_dicts[idx]['source_image']
        instruction = self.data_dicts[idx]['instruction']
        lora_name = self.data_dicts[idx]['lora_name']

        image = self.preprocess_image_crop_and_resize_to_target_size(image_path,target_size=1024)
        image = transforms.ToTensor()(image)
        return_dict = {}
        return_dict["source_image"] = image
        return_dict["instruction"] = instruction
        return_dict["lora_name"] = lora_name
        if 'base_pred_image' in self.data_dicts[idx]:
            target_image_path = self.data_dicts[idx]['base_pred_image']
            h,w = image.shape[1:]
            target_image = self.preprocess_image_crop_and_resize_to_target_shape(target_image_path,h,w)
            target_image = transforms.ToTensor()(target_image)
            return_dict["base_pred_image"] = target_image

        for k,v in self.data_dicts[idx].items():
            if 'recaption_prompt' in k:
                return_dict[k] = v

        return return_dict


    def preprocess_image_crop_and_resize_to_target_shape(self, target_image_path, height, width):
        image = Image.open(target_image_path)
        w, h = image.size
        target_ratio = width / height
        src_ratio = w / h
        
        if abs(src_ratio - target_ratio) > 1e-6:
            if src_ratio > target_ratio:
                new_w = int(h * target_ratio)
                left = (w - new_w) // 2
                right = left + new_w
                image = image.crop((left, 0, right, h))
                w = new_w
            else:
                new_h = int(w / target_ratio)
                top = (h - new_h) // 2
                bottom = top + new_h
                image = image.crop((0, top, w, bottom))
                h = new_h
        
        image = image.resize((width, height), Image.BICUBIC)
        return image