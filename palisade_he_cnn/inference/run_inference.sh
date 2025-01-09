# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#!/bin/bash

# srun -p hybrid -n 128 --mem=800G --pty bash -i

# ./run_vivian.sh 2>&1 >> logs/imagenet/resnet50_256_log_v2.txt

export OMP_DISPLAY_ENV=TRUE
export OMP_NUM_THREADS=64
export OMP_PROC_BIND=TRUE

for i in `seq 0 50` ; do
    python resnet50_cifar_inference.py -i $i
    # python resnet50_imagenet128_inference.py -i $i
    # python resnet50_imagenet256_inference.py -i $i
done