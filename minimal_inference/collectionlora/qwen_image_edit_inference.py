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
    def __init__(self, config_path, checkpoint_folder, height=1024, width=1024, seed=42, stochastic_sampling=True):

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
            
            # 记录原始dtype
            try:
                base_dtype = next(base_model.parameters()).dtype
            except StopIteration:
                base_dtype = torch.bfloat16
            
            # 应用LoRA
            peft_model = get_peft_model(base_model, lora_config)
            peft_model = peft_model.to(dtype=base_dtype)
            self.pipe.generator.model = peft_model
            
        
        state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")['generator']
        
        if use_lora:
            lora_state_dict = {k: v for k, v in state_dict.items() if "lora_" in k or "modules_to_save" in k}
            print(f"✓ 加载LoRA权重，共 {len(lora_state_dict)} 个参数")
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
        
        print("✓ 模型加载完成")
    
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
        
        output_path = os.path.join(output_dir, f"{file_name}-{output_filename}.png")
        image.save(output_path)
        print(f"✓ 图像已保存到: {output_path}")

        source_img_filename = f"{file_name}-{output_filename}_source_img.png"
        source_img_path = os.path.join(output_dir, source_img_filename)
        source_image_pil = transforms.ToPILImage()(source_image.squeeze(0).cpu())
        source_image_pil.save(source_img_path)
        instruction_filename = f"{file_name}-{output_filename}.txt"
        instruction_path = os.path.join(output_dir, instruction_filename)
        with open(instruction_path, 'w', encoding='utf-8') as f:
            f.write(str(instruction))


        
        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/multi_lora/collectionlora_50_in_1.yaml")
    parser.add_argument("--checkpoint_folder", type=str, default="ckpt/50_in_1")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    config_path = args.config_path
    checkpoint_folder = args.checkpoint_folder
    base_output_dir = args.output_dir

    with torch.inference_mode():
        inference = QwenImageEditInference(config_path, checkpoint_folder)


        my_ins_1 = '<1117_mengchongxianfengdui_1D2gT0>为宠物穿上带有红色领巾的制服，制服上饰有金色徽章，整体风格庄重可爱，保持真实毛发质感与自然神态，视角保持不变，突出宠物的正面形象。'
        my_ins_2 = '<1117_moebiusjeangiraudfenggechongwuchahua_jvVMKs4B>改变风格为Moebius，改变背景为花海。保持宠物外貌不变。改变姿势为看蝴蝶。"'
        my_ins_3 = f"将输入图像先变为'{my_ins_1}'，然后执行'{my_ins_2}'。"
        source_image_path = 'minimal_inference/collectionlora/test_images/zero-shot_combination/black_cat_emerald_eyes.png'
        output_dir = os.path.join(base_output_dir, 'A_B_C_test1')
        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_1,
            output_dir=os.path.join(output_dir, 'effectA'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_2,
            output_dir=os.path.join(output_dir, 'effectB'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_3,
            output_dir=os.path.join(output_dir, 'effectA_+_B'))


        my_ins_1 = '<1117_youyamengchong_APUnp>将画面中的宠物转化为哑光PVC材质的精致雕像，头戴华丽的金色王冠，身披富丽堂皇的贵族披风，披风边缘饰有珍珠与金色刺绣，整体姿态、表情、服饰细节及背景环境保持不变。'
        my_ins_2 = '<1120_geichongwuhuanshanghuangdifuzhuang_MX3V9ZdqU1>将宠物穿着华丽的金色龙纹帝王服饰，头戴精致皇冠，服饰细节丰富，绣有祥云与龙纹图案，整体风格庄重典雅，保持原有姿态与背景不变，增强真实感与视觉层次。'
        my_ins_3 = f"将输入图像先变为'{my_ins_1}'，然后执行'{my_ins_2}'。"

        source_image_path = 'minimal_inference/collectionlora/test_images/zero-shot_combination/british_shorthair_sofa.png'
        output_dir = os.path.join(base_output_dir, 'A_B_C_test2')

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_1,
            output_dir=os.path.join(output_dir, 'effectA'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_2,
            output_dir=os.path.join(output_dir, 'effectB'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_3,
            output_dir=os.path.join(output_dir, 'effectA_+_B'))



        my_ins_1 = '<chongwubianshenguiguiyouling_1112_a4obtur8>将宠物转化为幽灵般的半透明形态，保留原有姿态与细节，身体轮廓散发柔和蓝白色光芒，呈现轻盈漂浮的视觉效果，保持真实感与环境背景不变。'
        my_ins_2 = '<monaihuazhongchong_1112_9N3v43Peo3>改变风格为莫奈风格'
        my_ins_3 = f"将输入图像先变为'{my_ins_1}'，然后执行'{my_ins_2}'。"

        source_image_path = 'minimal_inference/collectionlora/test_images/zero-shot_combination/shiba_inu_cherry_blossom.png'
        output_dir = os.path.join(base_output_dir, 'A_B_C_test3')

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_1,
            output_dir=os.path.join(output_dir, 'effectA'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_2,
            output_dir=os.path.join(output_dir, 'effectB'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_3,
            output_dir=os.path.join(output_dir, 'effectA_+_B'))
            

        source_image_path = 'minimal_inference/collectionlora/test_images/zero-shot_combination/asian_female_office_worker.png'
        output_dir = os.path.join(base_output_dir, 'A_B_C_test4')

        my_ins_1 = '<1118_xiariwuhoujietou_voRIC>将画面变为在夏日午后，画面中的人物在街头行走，保留主体原有服饰与姿态细节，整体风格真实自然。'
        my_ins_2 = '<1119_shuijingdx_8XtwPWp1r>变身水晶雕像'
        my_ins_3 = f"将输入图像先变为'{my_ins_1}'，然后执行'{my_ins_2}'。"

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_1,
            output_dir=os.path.join(output_dir, 'effectA'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_2,
            output_dir=os.path.join(output_dir, 'effectB'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_3,
            output_dir=os.path.join(output_dir, 'effectA_+_B'))

        source_image_path = 'minimal_inference/collectionlora/test_images/zero-shot_combination/western_male_tuxedo_gala.png'
        output_dir = os.path.join(base_output_dir, 'A_B_C_test5')
        my_ins_1 = '<1117_shenshengwangzuorenxiangsheying_0T8rn6>保持人物面部特征不变，调整姿势为放松姿态，更换为白色飘逸长袍，背景替换为神圣庄严的白色帷幕与大理石王座，王座饰以金色雕花，周围环绕衔着绿枝的白鸽。'
        my_ins_2 = '<1117_tanxian_eP4oL>将人物转化为简约卡通风格，保留服饰与配饰细节，面部特征简化为柔和线条与圆润轮廓，眼神微闭或微笑，脸颊带有淡淡红晕，整体风格温暖可爱，背景保持原有构图但适度风格化处理，视角维持原图正面或微侧角度。'
        my_ins_3 = f"将输入图像先变为'{my_ins_1}'，然后执行'{my_ins_2}'。"

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_1,
            output_dir=os.path.join(output_dir, 'effectA'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_2,
            output_dir=os.path.join(output_dir, 'effectB'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_3,
            output_dir=os.path.join(output_dir, 'effectA_+_B'))

        source_image_path = 'minimal_inference/collectionlora/test_images/zero-shot_combination/asian_male_skater_street.png'
        output_dir = os.path.join(base_output_dir, 'A_B_C_test6')
        my_ins_1 = '<1117_hudietuya_PReer0f>在人物的背部添加风格化涂鸦蝴蝶翅膀，翅膀采用鲜艳流畅的色彩笔触，融合抽象艺术与自然形态，保持真实光影与服装细节，增强梦幻感但不破坏场景真实感，高清细节，保留原始视角与姿态。'
        my_ins_2 = '<1117_menghuanyuhangyuanchahua_qiEIczG>将人物转换为梦幻动画风格，背景变为明亮的橙色抽象画风，穿着未来感十足的宇航员服装，保留面部特征和姿态，整体呈现高饱和度色彩与柔和光影效果。'
        my_ins_3 = f"将输入图像先变为'{my_ins_1}'，然后执行'{my_ins_2}'。"

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_1,
            output_dir=os.path.join(output_dir, 'effectA'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_2,
            output_dir=os.path.join(output_dir, 'effectB'))

        inference.inference(
            source_image_path=source_image_path,
            instruction=my_ins_3,
            output_dir=os.path.join(output_dir, 'effectA_+_B'))



