# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import numpy as np
import torch
import json
import glob
from palisade_he_cnn.training.models.resnetN_multiplexed import *

def get_model(model_type, kwargs):
    if model_type=='resnet20':
        return resnet20(**kwargs)
    elif model_type=='resnet32':
        return resnet32(**kwargs)
    elif model_type=='resnet44':
        return resnet44(**kwargs)
    elif model_type=='resnet56':
        return resnet56(**kwargs)
    elif model_type=='resnet110':
        return resnet110(**kwargs)
    elif model_type=='resnet_test':
        return resnet_test(**kwargs)
    else: 
        raise ValueError("Returning None bc you are wrong!")

def get_best_weights(loc, dataset, model_type):
    loc = '%s%s/' % (loc,dataset)
    log_file = None
    for log in glob.glob(loc+'logs/*.json'):
        if model_type in log:
            log_file = log
            break
                
    if log_file is None:
        raise ValueError("model_type number must be resnet9,20,32,44,56, or 110")
        
    with open(log_file) as f:
        contents = json.load(f)
    
    print("Finding the best model according to logs...")
    print(contents)
    
    runs = {"run%d"%i : contents["run%d"%i] for i in range(5)}
    mean, std = contents["accuracy"]
    accs = [contents["run%d"%i] for i in range(5)]
    idx = accs.index(max(accs))
    
    print("\nAverage (5 runs): %1.3f%% +/- %1.3f%%" % (100*mean, 100*std))
    print("Best (idx %d): %1.3f" % (idx,runs["run%d"%idx]))
    weight_file = loc + "weights/%s_%s_run%d.pt" % (model_type, dataset, idx)
    return weight_file

def num_params(model) -> int:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def update_ema(train_model, valid_model, rho):
    # The trained model is not used for validation directly. Instead, the
    # validation model weights are updated with exponential moving averages.
    train_weights = train_model.state_dict().values()
    valid_weights = valid_model.state_dict().values()
    for train_weight, valid_weight in zip(train_weights, valid_weights):
        if valid_weight.dtype in [torch.float16, torch.float32]:
            valid_weight *= rho
            valid_weight += (1 - rho) * train_weight

def update_nesterov(weights, lr, weight_decay, momentum):
    for weight, velocity in weights:
        if weight.requires_grad:
            gradient = weight.grad.data
            weight = weight.data

            gradient.add_(weight, alpha=weight_decay).mul_(-lr)
            velocity.mul_(momentum).add_(gradient)
            weight.add_(gradient.add_(velocity, alpha=momentum))

def label_smoothing_loss(inputs, targets, alpha):
    log_probs = torch.nn.functional.log_softmax(inputs, dim=1, _stacklevel=5)
    kl = -log_probs.mean(dim=1)
    xent = torch.nn.functional.nll_loss(log_probs, targets, reduction="none")
    loss = (1 - alpha) * xent + alpha * kl
    return loss

def patch_whitening(data, patch_size=(3, 3)):
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
