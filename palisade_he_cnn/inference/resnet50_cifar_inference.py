# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).


import argparse
import copy
import json
from time import time

import torch
import torchvision
import torchvision.transforms as transforms

from palisade_he_cnn.src.cnn_context import create_cnn_context, TIMING_DICT
from palisade_he_cnn.src.he_cnn.utils import *
from palisade_he_cnn.src.utils import pad_conv_input_channels, PadChannel

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--idx", default="0")
args = vars(parser.parse_args())
img_idx = int(args["idx"])

print("img_idx", img_idx)

# create HE cc and keys
mult_depth = 35
scale_factor_bits = 59
batch_size = 32 * 32 * 32


# if using bootstrapping, you must increase scale_factor_bits to 59
cc, keys = get_keys(mult_depth, scale_factor_bits, batch_size, bootstrapping=True)


stats = ((0.4914, 0.4822, 0.4465), # mean
         (0.247, 0.243, 0.261)) # std

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats,inplace=True),
    PadChannel(npad=1),
    transforms.Resize(32)
    ])
validset = torchvision.datasets.CIFAR10(root="./data", download=True, transform=transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True)

# top level model
resnet_model = torch.load("palisade_he_cnn/src/weights/resnet50_cifar_gelu_kurt.pt")
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


##############################################################################

cnn_context = create_cnn_context(padded_test_data[0], cc, keys.publicKey, verbose=True)

start = time()

# embedding layer
cnn_context = cnn_context.apply_conv(padded_conv1, bn1)
cnn_context = cnn_context.apply_gelu(bound=15.0)

unencrypted = ptxt_embedded

compare_accuracy(keys, cnn_context, unencrypted, "embedding", num_digits=7)

###############################################################################


for i, layer in enumerate([resnet_model.layer1, resnet_model.layer2, resnet_model.layer3, resnet_model.layer4]):
    for j, bottleneck in enumerate(layer):

        bootstrap = False if (i == 0 and j == 0) else True
        name = f"bottleneck #{i+1}-{j}"
        cnn_context = cnn_context.apply_bottleneck(bottleneck, bootstrap=bootstrap, bootstrap_params={"meta" : True})
        unencrypted = bottleneck(unencrypted)
        compare_accuracy(keys, cnn_context, unencrypted, name, num_digits=7)

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

dataset = "cifar10"
model_type = "resnet50_metaBTS"

filename = Path("logs") / dataset / model_type / f"log_{img_idx}.json"
filename.parent.mkdir(exist_ok=True, parents=True)
data = dict(TIMING_DICT)
data["decrypted logits"] = decrypted_logits.tolist()
data["unencrypted logits"] = ptxt_logits.tolist()
data["inference time"] = inference_time

# avoid double-counting the strided conv operations
data['Pool'] = data['Pool'][1:2]

with open(filename, "w") as f:
    json.dump(data, f, indent=4)

