import os

import numpy as np
import torch
import torchvision
from PIL import Image

#####################################
######  Image IO Utils  #######
#####################################


def save_images_grid(images: torch.Tensor, path: str, rescale=False):
    assert images.shape[2] == 1  # no time dimension
    images = images.squeeze(2)
    if rescale:
        images = (images + 1.0) / 2.0  # -1,1 -> 0,1
    grid = torchvision.utils.make_grid(images)
    grid = (grid * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)


#####################################
#####  Training and Model Util  #####
#####################################


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


def seed_everything(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_parameters_without_gradients(model):
    """
    Returns a list of names of the model parameters that have no gradients.

    Args:
    model (torch.nn.Module): The model to check.

    Returns:
    List[str]: A list of parameter names without gradients.
    """
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad_params.append(name)
    return no_grad_params
