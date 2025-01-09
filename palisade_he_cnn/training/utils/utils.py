# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import ast
from collections import OrderedDict, defaultdict
from typing import Dict, Callable

import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class PadChannel(object):
    def __init__(self, npad: int=1):
        self.n = npad

    def __call__(self, x):
        _, width, height = x.shape
        x = torch.cat([x, torch.zeros(self.n, width, height)])
        return x

def get_gelu_poly_coeffs(degree, filename='gelu_poly_approx_params.txt'):
    with open(filename, 'r') as fp:
        params = []
        for line in fp:
            x = line[:-1]
            params.append(ast.literal_eval(x))

        if degree==2:
            return params[0]
        if degree==4:
            return params[1]
        elif degree==8:
            return params[2]
        elif degree==16:
            return params[3]
        elif degree==32:
            return params[4]
        else:
            print("Defaulting to deg8")
            return params[2]

def patch_whitening(data, patch_size=(3, 3)):
    # Compute weights from data such that
    # torch.std(F.conv2d(data, weights), dim=(2, 3))
    # is close to 1.
    h, w = patch_size
    c = data.size(1)
    patches = data.unfold(2, h, 1).unfold(3, w, 1)
    patches = patches.transpose(1, 3).reshape(-1, c, h, w).to(torch.float32)

    n, c, h, w = patches.shape
    X = patches.reshape(n, c * h * w)
    X = X / (X.size(0) - 1) ** 0.5
    covariance = X.t() @ X

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    eigenvalues = eigenvalues.flip(0)

    eigenvectors = eigenvectors.t().reshape(c * h * w, c, h, w).flip(0)

    return eigenvectors / torch.sqrt(eigenvalues + 1e-2).view(-1, 1, 1, 1)


def get_cifar10_dataloader(batch_size,
                           data_dir: str='../../datasets/cifar10/',
                           num_workers: int=4):
    stats = ((0.4914, 0.4822, 0.4465),
             (0.2023, 0.1994, 0.2010))

    train_tfms = tt.Compose([
        tt.RandomCrop(32,padding=4,padding_mode='reflect'), 
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize(*stats,inplace=True),
        PadChannel(npad=1)
    ])

    val_tfms = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats,inplace=True),
        PadChannel(npad=1)
    ])

    train_ds = ImageFolder(data_dir+'train',transform=train_tfms)
    val_ds = ImageFolder(data_dir+'test',transform=val_tfms)

    train_dl = DataLoader(train_ds,
                          batch_size,
                          pin_memory = True,
                          num_workers = num_workers,
                          shuffle = True)
    val_dl = DataLoader(val_ds,
                        batch_size,
                        pin_memory = True,
                        num_workers = num_workers)
    return train_dl, val_dl


def remove_all_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_hooks(child)
            
# Given a model and an input, get intermediate layer output
def get_intermediate_output(model):
    activation = defaultdict(list)

    def get_activation(name):
        def hook(model, input, output):
            x = output.detach().cpu()
            activation[name].append(x)

        return hook
    
    BatchNorm_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for i, b in enumerate(BatchNorm_layers):
        b.register_forward_hook(
            get_activation(f"bn_{i + 1}")
        )
    return activation


def get_all_bn_activations(model, val_dl, DEVICE):
    activation = get_intermediate_output(model)
    
    model.to(DEVICE)
    model.eval()

    for img, label in (val_dl):
        img, label = img.to(DEVICE), label.to(DEVICE)
        out = model(img)
    
    remove_all_hooks(model)
    
    activation = {k:torch.cat(v) for k,v in activation.items()}

    return activation
    