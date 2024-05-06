import json
import os
from os.path import join as osp

import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from parts2whole.models import UNet2DConditionModel
from parts2whole.models.custom_attention_processor import (
    DecoupledCrossAttnProcessor2_0,
    set_unet_2d_condition_attn_processor,
)
from parts2whole.pipelines.pipeline_refs2image import Refs2ImagePipeline

### Define configurations ###
device = "cuda"
torch_dtype = torch.float16
seed = 42
model_dir = "pretrained_weights/parts2whole"
use_decoupled_cross_attn = True
decoupled_cross_attn_path = osp(model_dir, "decoupled_attn.pth")

### Load model ###
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
pipe = Refs2ImagePipeline.from_pretrained(model_dir, vae=vae)

if use_decoupled_cross_attn:
    print("Loading decoupled cross attention...")

    from safetensors.torch import load_file

    unet = pipe.unet
    state_dict = unet.state_dict()
    set_unet_2d_condition_attn_processor(
        unet,
        set_cross_attn_proc_func=lambda n, hs, cd, ori: DecoupledCrossAttnProcessor2_0(
            hidden_size=hs, cross_attention_dim=cd, max_image_length=6
        ),
    )
    dc_state_dict = load_file(decoupled_cross_attn_path, device="cpu")
    state_dict.update(dc_state_dict)
    unet.load_state_dict(state_dict)
    pipe.unet = unet

pipe = pipe.to(device, dtype=torch_dtype)

generator = torch.Generator(device=device)
generator.manual_seed(seed)


def load_images_from_dict(data_dict):
    images = {}
    image_list = []

    def recursive_load(path, key, container, image_list):
        if isinstance(path, dict):
            container[key] = {}
            for k, v in path.items():
                recursive_load(v, k, container[key], image_list)
        else:
            # Load and store the image
            image = Image.open(path)
            container[key] = image
            image_list.append(image)

    recursive_load(data_dict, "root", images, image_list)
    return images["root"], image_list


### Define input data ###
height, width = 768, 512
prompt = "a girl, high quality, realistic"
input_dict = {
    "appearance": {
        "face": "testset/face_4.jpg",
        "whole body clothes": "testset/whole_body_clothes_2.jpg",
    },
    "mask": {
        "face": "testset/face_4_mask.jpg",
        "whole body clothes": "testset/whole_body_clothes_2_mask.jpg",
    },
    "structure": {"densepose": "testset/densepose_2.png"},
}
image_dict, image_list = load_images_from_dict(input_dict)

### Inference ###
images = pipe(
    **image_dict,
    prompt=prompt,
    generator=generator,
    height=height,
    width=width,
    num_inference_steps=50,
    use_decoupled_cross_attn=use_decoupled_cross_attn,
).images
for j, image in enumerate(images):
    image.save(f"output_{j}.jpg")

# Concat ref image, pose images and generated images
images = [image.resize((width, height)).convert("RGB") for image in image_list] + images
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
total_image = Image.new("RGB", (total_width, max_height))
x_offset = 0
for im in images:
    total_image.paste(im, (x_offset, 0))
    x_offset += im.size[0]
total_image.save("visualize.jpg")
