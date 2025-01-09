# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import torch
import torch.nn as nn
import torchvision


def load_cifar10(device, dtype, data_dir='./datasets/cifar10/'):
    print("Loading CIFAR10")
    train = torchvision.datasets.CIFAR10(root=data_dir, download=True)
    valid = torchvision.datasets.CIFAR10(root=data_dir, train=False)

    train_data = preprocess_cifar10_data(train.data, device, dtype)
    valid_data = preprocess_cifar10_data(valid.data, device, dtype)

    train_targets = torch.tensor(train.targets).to(device)
    valid_targets = torch.tensor(valid.targets).to(device)

    # Pad 32x32 to 40x40
    train_data = nn.ReflectionPad2d(4)(train_data)

    return train_data, train_targets, valid_data, valid_targets

def load_cifar100(device, dtype, data_dir='./datasets/cifar100/'):
    print("Loading CIFAR100")
    train = torchvision.datasets.CIFAR100(root=data_dir, download=True)
    valid = torchvision.datasets.CIFAR100(root=data_dir, train=False)

    train_data = preprocess_cifar100_data(train.data, device, dtype)
    valid_data = preprocess_cifar100_data(valid.data, device, dtype)

    train_targets = torch.tensor(train.targets).to(device)
    valid_targets = torch.tensor(valid.targets).to(device)

    # Pad 32x32 to 40x40
    train_data = nn.ReflectionPad2d(4)(train_data)

    return train_data, train_targets, valid_data, valid_targets

def random_crop(data, crop_size):
    crop_h, crop_w = crop_size
    h = data.size(2)
    w = data.size(3)
    x = torch.randint(w - crop_w, size=(1,))[0]
    y = torch.randint(h - crop_h, size=(1,))[0]
    return data[:, :, y : y + crop_h, x : x + crop_w]

def preprocess_cifar10_data(data, device, dtype):
    # Convert to torch float16 tensor
    data = torch.tensor(data, device=device).to(dtype)

    # Normalize
    mean = torch.tensor([125.31, 122.95, 113.87], device=device).to(dtype)
    std = torch.tensor([62.99, 62.09, 66.70], device=device).to(dtype)
    data = (data - mean) / std

    # Permute data from NHWC to NCHW format
    data = data.permute(0, 3, 1, 2)

    return data

def preprocess_cifar100_data(data, device, dtype):
    # Convert to torch float16 tensor
    data = torch.tensor(data, device=device).to(dtype)

    # Normalize
    mean = torch.tensor([129.30, 124.07, 112.43], device=device).to(dtype)
    std = torch.tensor([68.17, 65.39, 70.42], device=device).to(dtype)
    data = (data - mean) / std

    # Permute data from NHWC to NCHW format
    data = data.permute(0, 3, 1, 2)

    return data

