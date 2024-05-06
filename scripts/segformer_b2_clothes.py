# https://huggingface.co/mattmdjaga/segformer_b2_clothes

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

labels = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}


def process(
    model, processor, image_path, output_dir, do_resize=False, save_empty=False
):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split(".")[0]
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, do_resize=do_resize, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    Image.fromarray(pred_seg.numpy().astype(np.uint8)).save(
        os.path.join(output_dir, base_name + "_seg_ori.png")
    )
    plt.imsave(
        os.path.join(output_dir, base_name + "_seg.png"), pred_seg, cmap="viridis"
    )

    # Segment the images into multiple parts using pred_seg and save into RGBA format
    data = np.array(image.convert("RGBA"))
    for label in labels.keys():
        a = (pred_seg == label).numpy().astype(np.uint8)

        if not save_empty and a.sum() == 0:
            continue

        r, g, b = data[:, :, 0] * a, data[:, :, 1] * a, data[:, :, 2] * a
        a = Image.fromarray(a * 255, "L")
        _data = np.dstack((r, g, b, a))
        image = Image.fromarray(_data, "RGBA")
        image.save(os.path.join(output_dir, base_name + f"_{labels[label]}.png"))

        # Save the mask as well
        a.save(os.path.join(output_dir, base_name + f"_{labels[label]}_mask.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="Segformer B2 Clothes")
    parser.add_argument("--image-path", type=str, help="Path to the image")
    parser.add_argument("--output-dir", type=str, help="Path to the save directory")
    parser.add_argument("--do-resize", action="store_true", help="Resize the image")
    parser.add_argument("--save-empty", action="store_true", help="Save empty images")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    processor = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )

    process(model, processor, args.image_path, args.output_dir)
