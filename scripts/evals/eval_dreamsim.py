# pip install dreamsim
import functools
import os

import clip
import numpy as np
import torch
import torchvision.transforms as TF
from dreamsim import dreamsim
from PIL import Image
from tqdm import tqdm


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files_1, files_2, transforms=None):
        assert len(files_1) == len(files_2)

        self.files_1 = files_1
        self.files_2 = files_2
        self.transforms = transforms

    def __len__(self):
        return len(self.files_1)

    def __getitem__(self, i):
        image_1 = Image.open(self.files_1[i]).convert("RGB")
        image_2 = Image.open(self.files_2[i]).convert("RGB")
        if self.transforms is not None:
            image_1 = self.transforms(image_1)
            image_2 = self.transforms(image_2)

        return {
            "image_1": image_1,
            "image_2": image_2,
        }


def calculate_dreamsim_given_image_lists(images_list, batch_size, num_workers, device):
    # Prepare model
    model, preprocess = dreamsim(pretrained=True, device=device)

    # Prepare Dataset and DataLoader
    dataset = ImagePathDataset(images_list[0], images_list[1], transforms=preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # Calculate CLIP
    score_list = []
    for batch_data in tqdm(dataloader):
        image_1 = batch_data["image_1"].squeeze(1).to(device)
        image_2 = batch_data["image_2"].squeeze(1).to(device)

        with torch.no_grad():
            distance = model(image_1, image_2)

        score_list.append(distance.mean().item())

    return np.mean(score_list)


if __name__ == "__main__":
    # Prepare Hyperparameters
    batch_size = 16
    num_workers = 16
    device = "cuda"

    # load two image lists
    image1_list, image2_list = [], []
    root = "outputs/test-results"
    for dir in sorted(os.listdir(root)):
        image1 = os.path.join(root, dir, "reference.jpg")
        image2 = os.path.join(root, dir, "output_0.jpg")

        if not os.path.exists(image1) or not os.path.exists(image2):
            continue

        image1_list.append(image1)
        image2_list.append(image2)

    print("Number of images in image1_list: ", len(image1_list))
    print("Number of images in image2_list: ", len(image2_list))

    # Calculate DreamSim
    score = calculate_dreamsim_given_image_lists(
        [image1_list, image2_list], batch_size, num_workers, device
    )
    print("DreamSim Score:", score.item())
