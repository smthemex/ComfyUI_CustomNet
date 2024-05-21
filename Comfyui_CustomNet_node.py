# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path,"custom_net"))
path_dir = os.path.dirname(path)
file_path = os.path.dirname(path_dir)
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import copy
from .custom_net.ddim import DDIMSampler
from einops import rearrange
import math
from PIL import Image, ImageDraw
from .customnet_util import instantiate_from_config, img2tensor
from .gradio_utils import load_preprocess_model, preprocess_image


def get_instance_path(path):
    os_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        os_path = os_path.replace('\\', "/")
    return os_path


class CustomNet_Plus:

    def __init__(self):
        pass

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "on the seaside"}),
                "neg_prompt": ("STRING", {"multiline": True,"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "x0": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "display": "number"}),
                "y0": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "display": "number"}),
                "x1": ("INT", {"default": 256, "min": 0, "max": 256, "step": 1, "display": "number"}),
                "y1": ("INT", {"default": 256, "min": 0, "max": 256, "step": 1, "display": "number"}),
                "polar": ("FLOAT", {"default": 0, "min": -30, "max": 30, "step": -0.5, "display": "number"}),
                "azimuth": ("FLOAT", {"default": 0, "min": -60, "max": 30, "step": -0.5, "display": "number"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image", )
    FUNCTION = "customnet_main"
    CATEGORY = "CustomNet_Plus"

    def preprocess_input(self,preprocess_model, input_image): # reg background
        # global input_img
        processed_image = preprocess_image(preprocess_model, input_image)
        # input_img = (processed_image / 255.0).astype(np.float32)
        return processed_image
        # return processed_image, processed_image

    def prepare_data(self,device, input_image, x0, y0, x1, y1, polar, azimuth, prompt):
        if input_image.size[0] != 256 or input_image.size[1] != 256:
            input_image = input_image.resize((256, 256))
        input_image = np.array(input_image)

        img_cond = img2tensor(input_image, bgr2rgb=False, float32=True).unsqueeze(0) / 255.
        img_cond = img_cond * 2 - 1

        img_location = copy.deepcopy(img_cond)
        input_im_padding = torch.ones_like(img_location)

        x_0 = min(x0, x1)
        x_1 = max(x0, x1)
        y_0 = min(y0, y1)
        y_1 = max(y0, y1)
        # print(x0, y0, x1, y1)
        # print(x_0, y_0, x_1, y_1)
        img_location = torch.nn.functional.interpolate(img_location, (y_1 - y_0, x_1 - x_0), mode="bilinear")
        input_im_padding[:, :, y_0:y_1, x_0:x_1] = img_location
        img_location = input_im_padding

        T = torch.tensor(
            [[math.radians(polar), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), 0.0]]).unsqueeze(1)
        batch = {
            "image_cond": img_cond.to(device),
            "image_location": img_location.to(device),
            'T': T.to(device),
            'text': [prompt],
        }
        return batch

    @torch.no_grad()
    def customnet_main(self,input_image,neg_prompt,x0, y0, x1, y1, polar, azimuth, prompt, seed):
        # load model
        device = torch.device("cuda")
        preprocess_model = load_preprocess_model()

        path_yaml = os.path.join(path,"configs","config_customnet.yaml")
        path_yaml = get_instance_path(path_yaml)
        config = OmegaConf.load(path_yaml)
        config = config.model
        model = instantiate_from_config(config)
        path_ckpt = os.path.join(path,"pretrain","customnet_v1.pt")
        path_ckpt = get_instance_path(path_ckpt)
        ckpt = torch.load(path_ckpt, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        del ckpt
        model = model.to(device)
        sampler = DDIMSampler(model, device=device)

        input_image = self.tensor_to_image(input_image) # comfy img2pil_img
        input_image = self.preprocess_input(preprocess_model, input_image)  # reg img

        seed_everything(seed)
        batch = self.prepare_data(device, input_image, x0, y0, x1, y1, polar, azimuth, prompt)
        c = model.get_learned_conditioning(batch["image_cond"])
        c = torch.cat([c, batch["T"]], dim=-1)
        c = model.cc_projection(c)

        ## condition
        cond = {}
        cond['c_concat'] = [model.encode_first_stage((batch["image_location"])).mode().detach()]
        cond['c_crossattn'] = [c]
        text_embedding = model.text_encoder(batch["text"])
        cond["c_crossattn"].append(text_embedding)

        ## null-condition
        uc = {}

        uc['c_concat'] = [torch.zeros(1, 4, 32, 32).to(c.device)]
        uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
        uc_text_embedding = model.text_encoder([neg_prompt])
        uc['c_crossattn'].append(uc_text_embedding)

        ## sample
        shape = [4, 32, 32]
        samples_latents, _ = sampler.sample(
            S=50,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=999,  # useless
            conditioning=cond,
            unconditional_conditioning=uc,
            cfg_type=0,
            cfg_scale_dict={"img": 0., "text": 0., "all": 3.0}
        )

        x_samples = model.decode_first_stage(samples_latents)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu().numpy()
        x_samples = rearrange(255.0 * x_samples[0], 'c h w -> h w c').astype(np.uint8)
        image = Image.fromarray(x_samples)
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "CustomNet_Plus": CustomNet_Plus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomNet_Plus": "CustomNet_Plus"
}
