# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

# srun -p hybrid -n 128 --mem=300G --pty bash -i
# srun -p himem -n 128 --mem=300G --pty bash -i

# export OMP_DISPLAY_ENV=TRUE
# export OMP_NUM_THREADS=32

import torch
import numpy as np
from time import time
import copy
import json
import argparse
from pathlib import Path

from palisade_he_cnn.src.cnn_context import create_cnn_context, TIMING_DICT

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from palisade_he_cnn.src.he_cnn.utils import compare_accuracy, get_keys
from palisade_he_cnn.src.utils import pad_conv_input_channels
from palisade_he_cnn.training.utils.utils import PadChannel

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--idx", default="0")
args = vars(parser.parse_args())
img_idx = int(args["idx"])

print("img_idx", img_idx)

IMAGENET_CHANNEL_MEAN = (0.485, 0.456, 0.406)
IMAGENET_CHANNEL_STD = (0.229, 0.224, 0.225)

stats = (IMAGENET_CHANNEL_MEAN, IMAGENET_CHANNEL_STD)

IMAGENET_DIR = Path("/aoscluster/he-cnn/vivian/imagenet/datasets/ILSVRC/Data/CLS-LOC")
resize_size = 136
crop_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats,inplace=True),
    PadChannel(npad=1),
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size)
    ])

validset = ImageFolder(IMAGENET_DIR / "val", transform=transform)

validloader = DataLoader(validset,
                    batch_size = 1,
                    pin_memory = True,
                    num_workers = 1,
                    shuffle=True)

# top level model
resnet_model = torch.load("weights/resnet50_imagenet128_gelu_kurt.pt")
resnet_model.eval()

print(resnet_model)


##############################################################################

conv1 = resnet_model.conv1
bn1 = resnet_model.bn1

padded_conv1 = pad_conv_input_channels(conv1)

embedder = copy.deepcopy(torch.nn.Sequential(resnet_model.conv1, resnet_model.bn1, resnet_model.relu, resnet_model.maxpool))

for i, (padded_test_data, test_label) in enumerate(validloader):
    if i == img_idx:
        break

unpadded_test_data = padded_test_data[:,:3]
ptxt_embedded = embedder(unpadded_test_data).detach().cpu()

unencrypted = ptxt_embedded

##############################################################################

# create HE cc and keys
mult_depth = 34
scale_factor_bits = 59
batch_size = 32 * 32 * 32

# if using bootstrapping, you must increase scale_factor_bits to 59
cc, keys = get_keys(mult_depth, scale_factor_bits, batch_size, bootstrapping=True)


##############################################################################

cnn_context = create_cnn_context(padded_test_data[0], cc, keys.publicKey, verbose=True)

while cnn_context.shards[0].getTowersRemaining() > 18:
    for i in range(cnn_context.num_shards):
        cnn_context.shards[i] *= 1.0

start = time()

# embedding layer
cnn_context = cnn_context.apply_conv(padded_conv1, bn1)
cnn_context = cnn_context.apply_gelu(bound=50.0, degree=200)
cnn_context = cnn_context.apply_pool(conv=True)

compare_accuracy(keys, cnn_context, unencrypted, "embedding")

###############################################################################


for i, layer in enumerate([resnet_model.layer1, resnet_model.layer2, resnet_model.layer3, resnet_model.layer4]):
    for j, bottleneck in enumerate(layer):

        name = f"bottleneck #{i+1}-{j}"
        cnn_context = cnn_context.apply_bottleneck(bottleneck, bootstrap=True, gelu_params={"bound" : 15.0, "degree": 59})
        unencrypted = bottleneck(unencrypted)
        compare_accuracy(keys, cnn_context, unencrypted, name)

###############################################################################

linear = resnet_model.fc
ctxt_logits = cnn_context.apply_fused_pool_linear(linear)

inference_time = time() - start
print(f"\nTotal Time: {inference_time:.0f} s = {inference_time / 60:.01f} min")

flattened = torch.nn.Flatten()(resnet_model.avgpool(unencrypted))
ptxt_logits = linear(flattened)
ptxt_logits = ptxt_logits.detach().cpu().numpy().ravel()

decrypted_logits = cc.decrypt(keys.secretKey, ctxt_logits)[:linear.out_features]

print(f"[+] decrypted logits = {decrypted_logits}")
print(f"[+] plaintext logits = {ptxt_logits}")

###############################################################################

dataset = "imagenet"
model_type = "resnet50_128"

filename = Path("logs") / dataset / model_type / f"log_{img_idx}.json"
filename.parent.mkdir(exist_ok=True, parents=True)
data = dict(TIMING_DICT)
data["decrypted logits"] = decrypted_logits.tolist()
data["unencrypted logits"] = ptxt_logits.tolist()
data["inference time"] = inference_time

# avoid double-counting the strided conv operations
data['Pool'] = data['Pool'][:1]

with open(filename, "w") as f:
    json.dump(data, f, indent=4)

