# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import subprocess
import argparse

layer = [20, 32, 44, 56, 110]
batch = [256, 256, 256, 256, 64]
epoch = [50, 50, 50, 50, 50]


def argparsing():
    parser = argparse.ArgumentParser()
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
    return vars(parser.parse_args())


def main():
    args = argparsing()
    dataset = args["dataset"]
    cuda = args["cuda"]

    for i in range(len(layer)):
        cmd = "python3 train_resnetN_optuna.py -n %s -bs %s -e %s -dataset %s -c %d" \
              % (layer[i], batch[i], epoch[i], dataset, cuda)

        print("\n")
        print(cmd)
        subprocess.run(
            [
                "python3",
                "train_resnetN_optuna.py",
                "-n", "%s" % str(layer[i]),
                "-bs", "%s" % str(batch[i]),
                "-e", "%s" % str(epoch[i]),
                "-c", "%s" % str(cuda),
                "-dataset", dataset
            ],
            shell=False)


if __name__ == "__main__":
    main()
