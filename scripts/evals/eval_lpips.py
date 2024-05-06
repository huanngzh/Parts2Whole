# https://github.com/richzhang/PerceptualSimilarity
import os

import lpips
import numpy as np
import torch
import torchvision.transforms as TF
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


def calculate_lpips_given_image_lists(images_list, batch_size, num_workers, device):
    # Prepare Dataset and DataLoader
    transform = TF.Compose(
        [
            TF.ToTensor(),
            TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = ImagePathDataset(images_list[0], images_list[1], transforms=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare LPIPS Model
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)
    loss_fn_alex.eval()

    # Calculate LPIPS
    lpips_list = []
    for batch in tqdm(dataloader):
        image_1 = batch["image_1"].to(device)
        image_2 = batch["image_2"].to(device)
        lpips_list.append(loss_fn_alex(image_1, image_2).detach().cpu().numpy())
    lpips_list = np.concatenate(lpips_list)

    mean_lpips = np.mean(lpips_list)

    return lpips_list, mean_lpips


if __name__ == "__main__":
    # Prepare Hyperparameters
    batch_size = 32
    num_workers = 16
    device = "cuda"

    # load two image lists
    image1_list, image2_list = [], []
    root = "outputs/test-results"
    for dir in sorted(os.listdir(root)):
        image1 = os.path.join(root, dir, "target.jpg")
        image2 = os.path.join(root, dir, "output_0.jpg")

        if not os.path.exists(image1) or not os.path.exists(image2):
            continue

        image1_list.append(image1)
        image2_list.append(image2)

    print("Number of images in image1_list: ", len(image1_list))
    print("Number of images in image2_list: ", len(image2_list))

    lpips_list, mean_lpips = calculate_lpips_given_image_lists(
        [image1_list, image2_list],
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    print("Mean LPIPS: ", mean_lpips)
