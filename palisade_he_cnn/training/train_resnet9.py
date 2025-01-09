# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import argparse
import copy
import json
import time

from palisade_he_cnn.training.models.resnet9 import ResNet9
from palisade_he_cnn.training.utils.utils_dataloading import *
from palisade_he_cnn.training.utils.utils_kurtosis import *
from palisade_he_cnn.training.utils.utils_resnetN import (
    patch_whitening, update_nesterov, update_ema, label_smoothing_loss
)


def argparsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch',
                        help='Batch size',
                        type=int,
                        required=False,
                        default=512)
    parser.add_argument('-e', '--epochs',
                        help='Number of epochs',
                        type=int,
                        required=False,
                        default=100)
    parser.add_argument('-c', '--cuda',
                        help='CUDA device number',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('-r', '--nruns',
                        help='Number of training runs',
                        type=int,
                        required=False,
                        default=5)
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
        momentum,
        weight_decay,
        weight_decay_bias,
        ema_update_freq,
        ema_rho,
        device,
        dtype,
        kwargs,
        use_TTA,
        seed=0
):
    lr_schedule = torch.cat([
        torch.linspace(0e+0, 2e-3, 194),
        torch.linspace(2e-3, 2e-4, 582),
    ])

    lr_schedule_bias = 64.0 * lr_schedule

    kurt_schedule = torch.cat([
        torch.linspace(0, 1e-1, 2000),
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

    # Load dataset
    if dataset == "CIFAR10":
        train_data, train_targets, valid_data, valid_targets = load_cifar10(device, dtype)
    else:
        train_data, train_targets, valid_data, valid_targets = load_cifar100(device, dtype)

    train_data = torch.cat(
        [train_data, torch.zeros(train_data.size(0), 1, train_data.size(2), train_data.size(3)).to(device)], dim=1)
    valid_data = torch.cat(
        [valid_data, torch.zeros(valid_data.size(0), 1, valid_data.size(2), valid_data.size(3)).to(device)], dim=1)

    temp = train_data[:10000, :, 4:-4, 4:-4]
    weights = patch_whitening(temp)

    train_model = ResNet9(c_in=weights.size(1),
                          c_out=weights.size(0),
                          **kwargs).to(device)
    train_model.set_conv1_weights(
        weights=weights.to(device),
        bias=torch.zeros(weights.size(0)).to(device)
    )
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

    print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    # Train and validate
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
            # discard partial batches
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

            # Test time agumentation: Test model on regular and flipped data
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

        print(f"{epoch:5} {batch_count:8d} {train_time:19.2f} {valid_acc:22.4f}")

    return best_acc, best_model


def main():
    args = argparsing()

    model_type = "resnet9"
    cifar_dataset = args["dataset"]
    save = args["save"]
    weight_name = 'weights/%s_%s' % (model_type, cifar_dataset)
    kwargs = {
        "num_classes": 10 if cifar_dataset == 'CIFAR10' else 100,
        "scale_out": 0.125
    }

    device = torch.device("cuda:%s" % args["cuda"] if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Configurable parameters
    ema_update_freq = 5
    params = {
        "dataset": cifar_dataset,
        "epochs": args["epochs"],
        "batch_size": args["batch"],
        "momentum": 0.9,
        "weight_decay": 0.256,
        "weight_decay_bias": 0.004,
        "ema_update_freq": ema_update_freq,
        "ema_rho": 0.99 ** ema_update_freq,
        "kwargs": kwargs,
        "use_TTA": False
    }

    log = {
        "weights": weight_name,
        "model_type": model_type,
        "kwargs": kwargs,
        "params": params
    }

    nruns = args["nruns"]

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
        with open("logs/logs_resnet9_%s.json" % cifar_dataset, 'w') as fp:
            json.dump(log, fp)


if __name__ == "__main__":
    main()
