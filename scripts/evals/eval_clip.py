# https://github.com/Taited/clip-score
# pip install git+https://github.com/openai/CLIP.git
import os

import clip
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


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


def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == "img":
        features = model.encode_image(data.to(device))
    elif flag == "txt":
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features


def calculate_clip_given_image_lists(images_list, batch_size, num_workers, device):
    # Prepare model
    model, preprocess = clip.load("ViT-B/32", device=device)

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
    logit_scale = model.logit_scale.exp()
    for batch_data in tqdm(dataloader):
        real = batch_data["image_1"]
        real_features = forward_modality(model, real, "img")
        fake = batch_data["image_2"]
        fake_features = forward_modality(model, fake, "img")

        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True)

        # calculate scores
        score = logit_scale * (fake_features * real_features).sum()
        score_list.append(score.detach().cpu().numpy() / real.shape[0])

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

    # Calculate CLIP
    score = calculate_clip_given_image_lists(
        [image1_list, image2_list], batch_size, num_workers, device
    )
    print("CLIP Score: ", score.item())
