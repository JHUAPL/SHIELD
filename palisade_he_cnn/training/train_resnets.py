# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import subprocess
import argparse

#layer = [20,32,44,56,110]
#batch = [256,256,256,256,64]
#epoch = [200,200,200,200,200]

#layer = [32]
#batch = [256]
#epoch = [200]

def argparsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--cuda',
                        help='CUDA device number',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('-dataset', '--dataset',
                        help='CIFAR10 or CIFAR100',
                        type=str,
                        choices=['CIFAR10','CIFAR100'],
                        required=False,
                        default='CIFAR10')
    parser.add_argument('-n', '--nlayers',
                        help='List of layers in string format',
                        type=str,
                        required=True,
                        default='[20,32,44,56,110]')
    parser.add_argument('-e', '--epochs',
                        help='List of epochs in string format',
                        type=str,
                        required=True,
                        default='[100,100,100,100,100]')
    parser.add_argument('-bs', '--batch_size',
                        help='List of batch sizes in string format',
                        type=str,
                        required=True,
                        default='[256,256,256,256,64]')
    return vars(parser.parse_args())

def str2list(arg):
    return [int(item) for item in arg.split(',') if item!='']

def main():

    args = argparsing()
    layer = str2list(args["nlayers"])
    batch = str2list(args["epochs"])
    epoch = str2list(args["batch_size"])
                     
    dataset = args["dataset"]
    cuda = args["cuda"]
    
    for i in range(len(layer)):
        cmd = "python3 train_resnetN.py -n %s -bs %s -e %s -dataset %s -c %d" \
            % (layer[i], batch[i], epoch[i], dataset, cuda)

        print("\n")
        print(cmd)
        subprocess.run(
            [
                "python3",
                "train_resnetN.py",
                "-n",  "%s"%str(layer[i]),
                "-bs", "%s"%str(batch[i]),
                "-e",  "%s"%str(epoch[i]),
                "-c",  "%s"%str(cuda),
                "-dataset", dataset
            ],
            shell=False)

if __name__ == "__main__":
    main()
