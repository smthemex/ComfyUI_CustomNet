# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import copy
from .custom_net.ddim import DDIMSampler
from einops import rearrange
import math
from PIL import Image, ImageDraw
from .custom_net.customnet_util import instantiate_from_config, img2tensor
from .gradio_utils import load_preprocess_model, preprocess_image
from comfy.utils import common_upscale
import folder_paths
from folder_paths import base_path
sys.path.append(os.path.join(base_path,"custom_nodes","ComfyUI_CustomNet","custom_net"))
cur_path = os.path.dirname(os.path.abspath(__file__))

def get_instance_path(path):
    os_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        os_path = os_path.replace('\\', "/")
    return os_path

def tensor_to_image( tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def upscale_to_pil(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_image(samples)
    return img_pil

def preprocess_input(preprocess_model, input_image): # reg background
    processed_image = preprocess_image(preprocess_model, input_image)
    # input_img = (processed_image / 255.0).astype(np.float32)
    return processed_image
    # return processed_image, processed_image


def prepare_data(device, input_image, x0, y0, x1, y1, polar, azimuth, prompt,bg_image,use_inpaint):
    
    # if input_image.size[0] != 256 or input_image.size[1] != 256:
    #     input_image = input_image.resize((256, 256))
    # input_image = np.array(input_image)
    # img_cond = img2tensor(input_image, bgr2rgb=False, float32=True).unsqueeze(0) / 255.
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
    
    if use_inpaint:
        bg_image = np.array(bg_image)
        bg_cond=img2tensor( bg_image, bgr2rgb=False, float32=True) / 255.
        bg_cond =bg_cond* 2 - 1
        bg_cond =bg_cond.unsqueeze(0)
        bg_cond[:, :, y_0:y_1, x_0:x_1] = 1
    else:
        bg_cond=img_cond #no use
        
    T = torch.tensor(
        [[math.radians(polar), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), 0.0]]).unsqueeze(1)
    
    
    batch = {
        "image_cond": img_cond.to(device),
        "image_location": img_location.to(device),
        "bg_cond":bg_cond.to(device),
        'T': T.to(device),
        'text': [prompt],
    }
    return batch


device = torch.device("cuda")
class CustomNet_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (["none"] + folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL","DICT",)
    RETURN_NAMES = ("model","info")
    FUNCTION = "main_loader"
    CATEGORY = "CustomNet_Plus"

    def main_loader(self,ckpt_name,):
        # load model
        ckpt = folder_paths.get_full_path("checkpoints", ckpt_name)
        if "inpaint"in ckpt_name:
            path_yaml = os.path.join(cur_path, "configs", "config_customnet_inpaint.yaml")
            use_inpaint=True
        else:
            path_yaml = os.path.join(cur_path, "configs", "config_customnet.yaml")
            use_inpaint = False
        path_yaml = get_instance_path(path_yaml)
        config = OmegaConf.load(path_yaml)
        model = instantiate_from_config(config.model)
        ckpt_load = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(ckpt_load, strict=False)
        del ckpt_load
        model = model.to(device)
        info={"inpaint":use_inpaint,}
        return (model,info,)
    
class CustomNet_Sampler:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":("MODEL",),
                "info":("DICT",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "on the seaside"}),
                "neg_prompt": ("STRING", {"multiline": True,"default": ""}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1024,"step": 1,"display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 256, "min": 128, "max": 512, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 256, "min": 128, "max": 512, "step": 64, "display": "number"}),
                "obj_x": ("INT", {"default": 50, "min": 0, "max": 256, "step": 1, "display": "number"}),
                "obj_y": ("INT", {"default": 50, "min": 0, "max": 256, "step": 1, "display": "number"}),
                "bg_x": ("INT", {"default": 200, "min": 0, "max": 256, "step": 1, "display": "number"}),
                "bg_y": ("INT", {"default": 200, "min": 0, "max": 256, "step": 1, "display": "number"}),
                "polar": ("FLOAT", {"default": 0, "min": -30, "max": 30, "step": -0.5, "display": "number"}),
                "azimuth": ("FLOAT", {"default": 0, "min": -60, "max": 30, "step": -0.5, "display": "number"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1, "display": "number"}), },
            "optional": {
                "bg_image": ("IMAGE",), }
           
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image", )
    FUNCTION = "customnet_main"
    CATEGORY = "CustomNet_Plus"


    @torch.no_grad()
    def customnet_main(self,model,info,image, prompt,neg_prompt,steps,seed,width,height,obj_x, obj_y, bg_x, bg_y, polar, azimuth,batch_size,**kwargs):
        preprocess_model = load_preprocess_model()
        sampler = DDIMSampler(model, device=device)
        use_inpaint=info["inpaint"]
        print(f"inpaint is {use_inpaint}")
        input_image = upscale_to_pil(image, width, height) # comfy upscale tensor2pil
        if use_inpaint:
            bg_image=kwargs.get("bg_image")
            bg_image=upscale_to_pil(bg_image, width, height)
        else:
            bg_image=input_image
        
        input_image = preprocess_input(preprocess_model, input_image)  # using interface reg img
        seed_everything(seed)
        
        batch = prepare_data(device, input_image, obj_x, obj_y, bg_x, bg_y, polar, azimuth, prompt,bg_image,use_inpaint)
        
        c = model.get_learned_conditioning(batch["image_cond"])
        c = torch.cat([c, batch["T"]], dim=-1)
        c = model.cc_projection(c)
        if use_inpaint:
            bg_concat = model.encode_first_stage(batch["bg_cond"]).mode().detach()

        ## condition
        cond = {}
        cond['c_concat'] = [model.encode_first_stage((batch["image_location"])).mode().detach()]
        cond['c_crossattn'] = [c]
        text_embedding = model.text_encoder(batch["text"])
        cond["c_crossattn"].append(text_embedding)
        if use_inpaint:
            cond['c_concat'].append(bg_concat)

        ## null-condition
        uc = {}
        uc['c_concat'] = [torch.zeros(1, 4, 32, 32).to(c.device)]
        uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
        uc_text_embedding = model.text_encoder([neg_prompt])
        uc['c_crossattn'].append(uc_text_embedding)
        if use_inpaint:
            uc["c_concat"].append(bg_concat)
        

        ## sample
        shape = [4, 32, 32]
        samples_latents, _ = sampler.sample(
            S=steps,
            batch_size=batch_size,
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
    "CustomNet_LoadModel":CustomNet_LoadModel,
    "CustomNet_Sampler": CustomNet_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomNet_LoadModel":"CustomNet_LoadModel",
    "CustomNet_Sampler": "CustomNet_Sampler"
}
