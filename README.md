# Parts2Whole

[Arxiv 2024] From Parts to Whole: A Unified Reference Framework for Controllable Human Image Generation

- [x] Inference code and pretrained models.
- [x] Evaluation code.
- [ ] Training code.
- [ ] Training data.
- [ ] New model based on [Stable Diffusion 2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1).

## 🔥 Updates

[2024-05-06] 🔥🔥🔥 Code is released. Enjoy the human parts composition!

## 🏠 <a href="https://huanngzh.github.io/Parts2Whole/" target="_blank">Project Page</a> | <a href="https://arxiv.org/abs/2404.15267" target="_blank">Paper</a> | <a href="https://huggingface.co/huanngzh/Parts2Whole" target="_blank">Model</a>

![img:teaser](assets/teaser_mini.png)

Abstract: _We propose Parts2Whole, a novel framework designed for generating customized portraits from multiple reference images, including pose images and various aspects of human appearance. We first develop a semantic-aware appearance encoder to retain details of different human parts, which processes each image based on its textual label to a series of multi-scale feature maps rather than one image token, preserving the image dimension. Second, our framework supports multi-image conditioned generation through a shared self-attention mechanism that operates across reference and target features during the diffusion process. We enhance the vanilla attention mechanism by incorporating mask information from the reference human images, allowing for precise selection of any part._

## 🔨 Method Overview

![img:pipeline](assets/overview.png)

## ⚒️ Installation

Clone our repo, and install packages in `requirements.txt`. We test our model on a 80G A800 GPU with 11.8 CUDA and 2.0.1 PyTorch. But inference on smaller GPUs is possible.

```bash
conda create -n parts2whole
conda activate parts2whole
pip install -r requirements.txt
```

Download checkpoints <a href="https://huggingface.co/huanngzh/Parts2Whole" target="_blank">here</a> into `pretrained_weights/parts2whole` dir. We also provide a simple download script, using:

```python
python download_weights.py
```

## 🎨 Inference

Check `inference.py`. Modify the checkpoint path and input as you need, and run command:
```bash
python inference.py
```

You may need to modify the following code in the `inference.py` script:
```python
### Define configurations ###
device = "cuda"
torch_dtype = torch.float16
seed = 42
model_dir = "pretrained_weights/parts2whole"  # checkpoint path in your local machine
use_decoupled_cross_attn = True
decoupled_cross_attn_path = "pretrained_weights/parts2whole/decoupled_attn.pth" # include in the model_dir
```

```python
### Define input data ###
height, width = 768, 512
prompt = "This person is wearing a short-sleeve shirt." # input prompt
input_dict = {
    "appearance": {
        "face": "testset/face_man1.jpg",
        "whole body clothes": "testset/clothes_man1.jpg",
    },
    "mask": {
        "face": "testset/face_man1_mask.jpg",
        "whole body clothes": "testset/clothes_man1_mask.jpg",
    },
    "structure": {"densepose": "testset/densepose_man1.jpg"},
}
```

⭐️⭐️⭐️ Notably, the `input_dict` should contain keys `appearance`, `mask`, and `structure`. The first two mean specifying the appearance of parts of multiple reference images, and structure means postures such as densepose.

⭐️⭐️⭐️ The keys in these three parts also have explanations. Keys in `appearance` and `mask` should be the same. The choices include "upper body clothes", "lower body clothes", "whole body clothes", "hair or headwear", "face", "shoes". Key of `structure` should be "densepose". (The openpose model has not been release.)

🔨🔨🔨 In order to conveniently obtain the mask of each reference image, we also provide corresponding tools and explain how to use them in [Tools](#-tools). First, you can use Real-ESRGAN to increase the resolution of the reference image, and use segformer to obtain the masks of various parts of the human body.

## 😊 Evaluation

For evaluation, please install additional packages firstly:
```bash
pip install git+https://github.com/openai/CLIP.git # for clip
pip install dreamsim # for dreamsim
pip install lpips # for lpips
```

We provide easy-to-use evaluation scripts in `scripts/evals` folder. The scripts receive a unified formated data, which is organize as two lists of images as input. Modify the code for loading images as you need. Check our scripts for more details.

## 🔨 Tools

### Real-ESRGAN

To use [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to restore images, please download [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) into `./pretrained_weights/Real-ESRGAN` firstly. Then run command:

```bash
python -m scripts.real_esrgan -n RealESRGAN_x4plus -i /path/to/dir -o /path/to/dir --face_enhance
```

### SegFormer

To use [segformer](https://huggingface.co/mattmdjaga/segformer_b2_clothes) to segment human images and obtain hat, hair, face, clothes parts, please run command:

```bash
python scripts/segformer_b2_clothes.py --image-path /path/to/image --output-dir /path/to/dir
```

> Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"

## 😭 Limitations

At present, the generalization of the training data is average, and the number of women is relatively large, so the generalization of the model needs to be improved, such as stylization, etc. We are working hard to improve the robustness and capabilities of the model, and we also look forward to and welcome contributions/pull requests from the community.

## 🤝 Acknowledgement

We appreciate the open source of the following projects:

[diffusers](https://github.com/huggingface/diffusers) &#8194;
[magic-animate](https://github.com/magic-research/magic-animate) &#8194;
[Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) &#8194;
[DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) &#8194;
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## 📎 Citation

If you find this repository useful, please consider citing:

```
@misc{huang2024parts2whole,
  title={From Parts to Whole: A Unified Reference Framework for Controllable Human Image Generation},
  author={Huang, Zehuan and Fan, Hongxing and Wang, Lipeng and Sheng, Lu},
  journal={arXiv preprint arXiv:2404.15267},
  year={2024}
}
```
