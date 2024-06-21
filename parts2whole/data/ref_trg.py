import json
import random
from os.path import join as osp
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class RefTrgDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        jsonl_path: str,
        image_wh: Union[int, Tuple] = (768, 768),
        num_samples: Optional[int] = None,
        load_rgba: bool = False,
        bg_color: Union[str, float] = "gray",
    ):
        super().__init__()

        with open(jsonl_path, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]

        if num_samples is not None:
            data = data[:num_samples]

        image_wh = (
            (image_wh, image_wh) if isinstance(image_wh, int) else tuple(image_wh)
        )
        bg_color = (
            torch.from_numpy(self.get_bg_color(bg_color))
            if load_rgba and bg_color is not None
            else None
        )

        self.random_crop_resize = transforms.RandomResizedCrop(
            image_wh,
            scale=(0.9, 1.0),
            ratio=(0.9, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        self.data = data
        self.root_dir = root_dir
        self.image_wh = image_wh
        self.load_rgba = load_rgba
        self.bg_color = bg_color

    def __len__(self) -> int:
        return len(self.data)

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = random.uniform(0.3, 0.7)
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(
        self,
        img_path: str,
        bg_color: Optional[torch.Tensor] = None,
        rescale: bool = True,
        permute: bool = True,
    ):
        img = np.array(self.random_crop_resize(Image.open(img_path).convert("RGBA")))
        img = torch.from_numpy((img.astype(np.float32) / 255.0))  # [0, 1]

        mask = img[..., 3:4]
        img = (
            img[..., :3] * mask + bg_color * (1 - mask)
            if bg_color is not None
            else img[..., :3]
        )

        if rescale:
            img = img * 2.0 - 1.0  # to -1 ~ 1

        if permute:
            img = img.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)

        return img, mask

    def load_mask(self, mask_path: str, permute: bool = True):
        mask = np.array(self.random_crop_resize(Image.open(mask_path).convert("L")))
        mask = torch.from_numpy(mask / 255.0).unsqueeze(-1)

        if permute:
            mask = mask.permute(2, 0, 1)

        return mask

    def __getitem__(self, index) -> Any:
        sample = self.data[index]
        target_image, _ = self.load_image(osp(self.root_dir, sample["target"]))

        appearance, structure, mask = {}, {}, {}

        for key, value in sample["appearance"].items():
            ref_img, ref_mask = self.load_image(
                osp(self.root_dir, value), bg_color=self.bg_color
            )
            appearance[key] = ref_img
            mask[key] = ref_mask

        for key, value in sample["structure"].items():
            structure[key], _ = self.load_image(
                osp(self.root_dir, value), rescale=False
            )

        if not self.load_rgba and "mask" in sample:
            for key, value in sample["mask"].items():
                mask[key] = self.load_mask(osp(self.root_dir, value))

        return {
            "pixel_values": target_image,
            "appearance": appearance,
            "structure": structure,
            "mask": mask,
            "caption": sample["caption"],
        }


if __name__ == "__main__":
    import torchvision
    from torch.utils.data import DataLoader

    dataset = RefTrgDataset(
        root_dir="data/DeepFashion-MultiModal-Parts2Whole",
        jsonl_path="data/DeepFashion-MultiModal-Parts2Whole/train.jsonl",
        image_wh=(768, 768),
        load_rgba=False,
        bg_color="gray",
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    print(len(dataset))
    for step, batch in enumerate(dataloader):
        print(step, batch["pixel_values"].shape)
        torchvision.utils.save_image(
            batch["pixel_values"] / 2 + 0.5, "check_target.jpg"
        )
        for key, value in batch["appearance"].items():
            torchvision.utils.save_image(value / 2 + 0.5, f"check_{key}.jpg")
        for key, value in batch["structure"].items():
            torchvision.utils.save_image(value, f"check_{key}.jpg")
        for key, value in batch["mask"].items():
            torchvision.utils.save_image(value, f"check_{key}_mask.jpg")

        break
