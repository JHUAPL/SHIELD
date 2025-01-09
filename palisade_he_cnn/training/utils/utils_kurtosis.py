# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

from collections import defaultdict, OrderedDict
from typing import Dict, Callable

import torch
import torch.nn.functional as F

def get_intermediate_output(model):
    activations = defaultdict(list)

    def get_activation(name):
        def hook(model, input, output):
            x = input[0]
            activations[name].append(x)

        return hook
    
    GELU_layers = [m for m in model.modules() if isinstance(m, torch.nn.GELU)]
    for i, b in enumerate(GELU_layers):
        b.register_forward_hook(
            get_activation(f"GELU_{i + 1}")
        )
    return activations

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

def moment(x: torch.Tensor, std: float, mean: float, deg: int=4, eps: float=1e-4) -> torch.Tensor:
    x = x.double()
    temp = (x-mean)**deg / x.shape[0]
    return torch.sum(temp) / (std**deg + eps)

def get_statistics(activations):
    n = len(activations)
    means = torch.zeros(n)
    stds  = torch.zeros(n)
    kurts = torch.zeros(n)
    
    for layer_index,name in enumerate(sorted(activations.keys(), key=lambda x:int(x.split('_')[1]))):
        dist = activations[name]
        dist = dist.flatten()
        
        std, mean = torch.std_mean(dist)
        kurt = moment(dist, std, mean, deg=4)
        
        means[layer_index] = mean
        stds[layer_index] = std
        kurts[layer_index] = kurt

    loss_means = F.mse_loss(means, torch.zeros(n))
    loss_stds  = F.mse_loss(stds, torch.ones(n))
    loss_kurts = F.mse_loss(kurts, 3*torch.ones(n))
    
    return loss_means, loss_stds, loss_kurts
