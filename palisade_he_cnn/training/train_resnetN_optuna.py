# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import argparse
import copy
import time

import optuna
import joblib
from optuna.trial import TrialState

from palisade_he_cnn.training.utils.utils_dataloading import *
from palisade_he_cnn.training.utils.utils_kurtosis import *
from palisade_he_cnn.training.utils.utils_resnetN import (
    get_model, update_nesterov, update_ema, label_smoothing_loss
)

use_TTA = False


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
        trial,
        dataset,
        epochs,
        batch_size,
        weight_decay_bias,
        ema_update_freq,
        ema_rho,
        device,
        dtype,
        model_type,
        kwargs,
        seed=0
):
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

    # Load dataset
    if dataset == "CIFAR10":
        train_data, train_targets, valid_data, valid_targets = load_cifar10(device, dtype)
    else:
        train_data, train_targets, valid_data, valid_targets = load_cifar100(device, dtype)

    train_data = torch.cat(
        [train_data, torch.zeros(train_data.size(0), 1, train_data.size(2), train_data.size(3)).to(device)], dim=1)
    valid_data = torch.cat(
        [valid_data, torch.zeros(valid_data.size(0), 1, valid_data.size(2), valid_data.size(3)).to(device)], dim=1)

    # Convert model weights to half precision
    train_model = get_model(model_type, kwargs).to(device)
    train_model.to(dtype)
    train_model.train()

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 9e-4, 2e-3)
    lr_bias = trial.suggest_float("lr_bias", 54, 74)
    momentum = trial.suggest_float("momentum", 0.7, .99)
    weight_decay = trial.suggest_float("weight_decay", 0.01, .3)

    N = int(len(train_data) / batch_size)
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
    best_acc = []
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
        loss_epoch = 0.0
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

            loss_epoch += loss.sum().item() / batch_size

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

        correct = []
        with torch.no_grad():
            for i in range(0, len(valid_data), batch_size):
                valid_model.train(False)
                regular_inputs = valid_data[i: i + batch_size]
                logits = valid_model(regular_inputs).detach()

                if use_TTA:
                    flipped_inputs = torch.flip(regular_inputs, [-1])
                    logits2 = valid_model(flipped_inputs).detach()
                    logits = torch.mean(torch.stack([logits, logits2], dim=0), dim=0)

                # Compute correct predictions
                temp = logits.max(dim=1)[1] == valid_targets[i: i + batch_size]

                correct.append(temp.detach().type(torch.float64))

        # Accuracy is average number of correct predictions
        accuracy = torch.mean(torch.cat(correct)).item()
        best_acc.append(accuracy)
        print(f"{epoch:5} {batch_count:8d} {train_time:19.2f} {accuracy:22.4f}")

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return max(best_acc)


def objective(trial):
    args = argparsing()

    model_type = "resnet%s" % args["nlayers"]
    cifar_dataset = args["dataset"]
    save = args["save"]
    weight_name = 'weights/optuna/%s_%s' % (model_type, cifar_dataset)

    print("ResNet%s" % args["nlayers"])
    print("Weight file:", weight_name)

    kwargs = {
        "num_classes": 10 if cifar_dataset == 'CIFAR10' else 100,
        "debug": args["debug"]
    }

    device = torch.device("cuda:%s" % args["cuda"] if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ema_update_freq = 5
    params = {
        "trial": trial,
        "dataset": cifar_dataset,
        "epochs": args["epochs"],
        "batch_size": args["batch"],
        "weight_decay_bias": 0.004,
        "ema_update_freq": ema_update_freq,
        "ema_rho": 0.99 ** ema_update_freq,
        "model_type": model_type,
        "kwargs": kwargs
    }
    accuracy = train(**params,
                     device=device,
                     dtype=dtype,
                     seed=0)
    return accuracy


def main():
    args = argparsing()
    model_type = "resnet%s" % args["nlayers"]
    cifar_dataset = args["dataset"]
    save = args["save"]

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///db.sqlite3",
                                pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=5,
                                                                    min_delta=0.0))
    study.optimize(objective, n_trials=10)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    joblib.dump(study, "study_%s_%s.pkl" % (model_type, cifar_dataset))


if __name__ == "__main__":
    main()
