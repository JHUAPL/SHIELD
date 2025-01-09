# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import argparse
import copy
import json
import time

import torch.nn as nn

from optuna_params import get_optuna_params
from palisade_he_cnn.training.utils.utils_dataloading import random_crop
from palisade_he_cnn.training.utils.utils_kurtosis import *
from palisade_he_cnn.training.utils.utils_resnetN import (
    get_model, update_nesterov, update_ema, label_smoothing_loss
)

# training time augmentation
use_TTA = False


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_accuracy = 0.0

    def early_stop(self, accuracy):
        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
            self.counter = 0
        elif accuracy <= (self.max_accuracy + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def argparsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nlayers',
                        help='ResNet model depth',
                        type=int,
                        choices=[20, 32, 44, 56, 110],
                        required=True)
    parser.add_argument('-bs', '--batch',
                        help='Batch size',
                        type=int,
                        required=False,
                        default=256)
    parser.add_argument('-e', '--epochs',
                        help='Number of epochs',
                        type=int,
                        required=False,
                        default=100)
    parser.add_argument('-d', '--debug',
                        help='Debugging mode',
                        type=bool,
                        required=False,
                        default=False)
    parser.add_argument('-c', '--cuda',
                        help='CUDA device number',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('-dataset', '--dataset',
                        help='CIFAR10 or CIFAR100',
                        type=str,
                        choices=['CIFAR10', 'CIFAR100'],
                        required=False,
                        default='CIFAR10')
    parser.add_argument('-s', '--save',
                        help='Save model and log files',
                        type=bool,
                        required=False,
                        default=True)

    return vars(parser.parse_args())


def train(
        dataset,
        epochs,
        batch_size,
        lr,
        lr_bias,
        momentum,
        weight_decay,
        weight_decay_bias,
        ema_update_freq,
        ema_rho,
        device,
        dtype,
        model_type,
        kwargs,
        seed=0
):
    # Load dataset
    if dataset == "CIFAR10":
        train_data, train_targets, valid_data, valid_targets = load_cifar10(device, dtype)
    else:
        train_data, train_targets, valid_data, valid_targets = load_cifar100(device, dtype)

    train_data = torch.cat(
        [train_data, torch.zeros(train_data.size(0), 1, train_data.size(2), train_data.size(3)).to(device)], dim=1)
    valid_data = torch.cat(
        [valid_data, torch.zeros(valid_data.size(0), 1, valid_data.size(2), valid_data.size(3)).to(device)], dim=1)

    N = int(len(train_data) / batch_size)  # 50k / 256, now below is organized by epoch #
    lr_schedule = torch.cat([
        torch.linspace(0.0, lr, N),
        # torch.linspace(lr,  lr, 2*N),
        torch.linspace(lr, 1e-4, 3 * N),
        torch.linspace(1e-4, 1e-4, 50 * N),
        torch.linspace(1e-5, 1e-5, 25 * N),
        torch.linspace(1e-6, 1e-6, 25 * N),
    ])
    lr_schedule_bias = lr_bias * lr_schedule

    kurt_schedule = torch.cat([
        torch.linspace(0, 0, 10 * N),
        torch.linspace(0.05, 0.05, 2 * N),
        torch.linspace(0.1, 0.1, 200 * N),
    ])
    # Print information about hardware on first run
    if seed == 0:
        if device.type == "cuda":
            print("Device :", torch.cuda.get_device_name(device.index))

        print("Dtype  :", dtype)
        print()

    # Start measuring time
    start_time = time.perf_counter()

    # Set random seed to increase chance of reproducability
    torch.manual_seed(seed)

    # Setting cudnn.benchmark to True hampers reproducability, but is faster
    torch.backends.cudnn.benchmark = True

    # Convert model weights to half precision
    train_model = get_model(model_type, kwargs).to(device)
    train_model.to(dtype)

    # Convert BatchNorm back to single precision for better accuracy
    for module in train_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()

    # Collect weights and biases and create nesterov velocity values
    weights = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) > 1
    ]
    biases = [
        (w, torch.zeros_like(w))
        for w in train_model.parameters()
        if w.requires_grad and len(w.shape) <= 1
    ]

    # Copy the model for validation
    valid_model = copy.deepcopy(train_model)

    # Patience:
    early_stopper = EarlyStopper(patience=120, min_delta=0.001)  # this is %

    # Testing non-SGD optimizer
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.001)

    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")
    print("\nepoch    batch    train time [sec]    validation accuracy")

    train_time = 0.0
    batch_count = 0
    best_acc = 0.0
    best_model = None
    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()

        # Randomly shuffle training data
        indices = torch.randperm(len(train_data), device=device)
        data = train_data[indices]
        targets = train_targets[indices]

        # Crop random 32x32 patches from 40x40 training data
        data = [
            random_crop(data[i: i + batch_size], crop_size=(32, 32))
            for i in range(0, len(data), batch_size)
        ]
        data = torch.cat(data)

        # Randomly flip half the training data
        data[: len(data) // 2] = torch.flip(data[: len(data) // 2], [-1])

        for i in range(0, len(data), batch_size):
            # Discard partial batches
            if i + batch_size > len(data):
                break

            # Slice batch from data
            inputs = data[i: i + batch_size]
            target = targets[i: i + batch_size]
            batch_count += 1

            # Compute new gradients
            train_model.zero_grad()
            train_model.train(True)

            # kurtosis setup
            remove_all_hooks(train_model)
            activations = get_intermediate_output(train_model)

            logits = train_model(inputs)
            loss = label_smoothing_loss(logits, target, alpha=0.2)

            # kurtosis scheduler
            kurt_index = min(batch_count, len(kurt_schedule) - 1)
            kurt_scale = kurt_schedule[kurt_index]

            # kurtosis calculation and cleanup
            remove_all_hooks(train_model)
            activations = {k: torch.cat(v) for k, v in activations.items()}
            loss_means, loss_stds, loss_kurts = get_statistics(activations)
            loss += (loss_means + loss_stds + loss_kurts) * kurt_scale

            loss.sum().backward()

            lr_index = min(batch_count, len(lr_schedule) - 1)
            lr = lr_schedule[lr_index]
            lr_bias = lr_schedule_bias[lr_index]

            # Update weights and biases of training model
            update_nesterov(weights, lr, weight_decay, momentum)
            update_nesterov(biases, lr_bias, weight_decay_bias, momentum)

            # Update validation model with exponential moving averages
            if (i // batch_size % ema_update_freq) == 0:
                update_ema(train_model, valid_model, ema_rho)

        # Add training time
        train_time += time.perf_counter() - start_time

        valid_correct = []
        for i in range(0, len(valid_data), batch_size):
            valid_model.train(False)
            regular_inputs = valid_data[i: i + batch_size]
            logits = valid_model(regular_inputs).detach()

            if use_TTA:
                flipped_inputs = torch.flip(regular_inputs, [-1])
                logits2 = valid_model(flipped_inputs).detach()
                logits = torch.mean(torch.stack([logits, logits2], dim=0), dim=0)

            # Compute correct predictions
            correct = logits.max(dim=1)[1] == valid_targets[i: i + batch_size]

            valid_correct.append(correct.detach().type(torch.float64))

        # Accuracy is average number of correct predictions
        valid_acc = torch.mean(torch.cat(valid_correct)).item()
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = train_model

        if early_stopper.early_stop(valid_acc):
            print("Early stopping")
            break

        print(f"{epoch:5} {batch_count:8d} {train_time:19.2f} {valid_acc:22.4f}")

    return best_acc, best_model


def main():
    args = argparsing()

    model_type = "resnet%s" % args["nlayers"]
    cifar_dataset = args["dataset"]
    save = args["save"]
    weight_name = 'weights/%s_%s' % (model_type, cifar_dataset)

    print("ResNet%s" % args["nlayers"])
    print("Weight file:", weight_name)

    kwargs = {
        "num_classes": 10 if cifar_dataset == 'CIFAR10' else 100,
        "debug": args["debug"]
    }

    device = torch.device("cuda:%s" % args["cuda"] if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Optuna:
    optuna_params = get_optuna_params(model_type, cifar_dataset)
    lr = optuna_params["lr"]
    lr_bias = optuna_params["lr_bias"]
    momentum = optuna_params["momentum"]
    weight_decay = optuna_params["weight_decay"]

    # Configurable parameters
    ema_update_freq = 5
    params = {
        "dataset": cifar_dataset,
        "epochs": args["epochs"],
        "batch_size": args["batch"],
        "lr": lr,
        "lr_bias": lr_bias,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "weight_decay_bias": 0.004,
        "ema_update_freq": ema_update_freq,
        "ema_rho": 0.99 ** ema_update_freq,
        "model_type": model_type,
        "kwargs": kwargs
    }

    nruns = 5
    log = {
        "weights": weight_name,
        "model_type": model_type,
        "kwargs": kwargs,
        "params": params
    }
    accuracies = []
    for run in range(nruns):
        weight_name_seed = weight_name + "_run%d.pt" % run

        best_acc, best_model = train(**params,
                                     device=device,
                                     dtype=dtype,
                                     seed=run)
        accuracies.append(best_acc)
        print("Best Run Accuracy: %1.4f" % best_acc)
        log["run%s" % run] = best_acc

        if save:
            print("Saving %s" % weight_name_seed)
            torch.save(best_model.state_dict(), weight_name_seed)

    mean = sum(accuracies) / len(accuracies)
    variance = sum((acc - mean) ** 2 for acc in accuracies) / len(accuracies)
    std = variance ** 0.5
    print("Accuracy: %1.4f +/- %1.4f" % (mean, std))
    log["accuracy"] = [mean, std]

    if save:
        with open("logs/logs_resnet%s_%s.json" % (args["nlayers"], cifar_dataset), 'w') as fp:
            json.dump(log, fp)


if __name__ == "__main__":
    main()
