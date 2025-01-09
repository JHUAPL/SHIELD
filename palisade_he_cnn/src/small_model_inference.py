# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

# export OMP_DISPLAY_ENV=TRUE
import os
from time import time

import torch
import torchvision
import torchvision.transforms as transforms

from cnn_context import create_cnn_context
from he_cnn.utils import *
from small_model import SmallModel

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# create HE cc and keys
mult_depth = 30
scale_factor_bits = 40
batch_size = 32 * 32 * 32  # increased batch size b/c the ring dimension is higher due to the mult_depth

# used for a small test of big shards
# batch_size = 128

# if using bootstrapping, you must increase scale_factor_bits to 59
cc, keys = create_cc_and_keys(batch_size, mult_depth=mult_depth, scale_factor_bits=scale_factor_bits,
                              bootstrapping=False)

# load the model
weight_file = "palisade_he_cnn/src/weights/small_model.pt"
print(os.getcwd())
model = SmallModel(activation='gelu', pool_method='avg')
model.load_state_dict(torch.load(weight_file))
model.eval()

# load data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Pad(2)])
validset = torchvision.datasets.MNIST(root="./data", download=True, transform=transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True)

total = 0
correct = 0
total_time = 0

for i, test_data in enumerate(validloader):
    print(f"Inference {i + 1}:")

    x_test, y_test = test_data

    input_img = create_cnn_context(x_test[0], cc, keys.publicKey, verbose=True)

    start = time()

    layer = model.model_layers.conv1
    conv1 = input_img.apply_conv(layer[0], layer[1])
    act1 = conv1.apply_gelu()
    pool1 = act1.apply_pool()

    layer = model.model_layers.conv2
    perm = np.random.permutation(128)  # example of how to use an output permutation
    conv2 = pool1.apply_conv(layer[0], layer[1], output_permutation=perm)
    act2 = conv2.apply_gelu()
    pool2 = act2.apply_pool()

    layer = model.model_layers.conv3
    conv3 = pool2.apply_conv(layer[0], layer[1])
    act3 = conv3.apply_gelu()

    layer = model.model_layers.classifier[1]
    logits = act3.apply_linear(layer)

    logits_dec = cc.decrypt(keys.secretKey, logits)[:10]
    logits_pt = model(x_test).detach().numpy().ravel()

    print(f"[+] decrypted logits   = {logits_dec}")
    print(f"[+] unencrypted logits = {logits_pt}")

    inference_time = time() - start
    total_time += inference_time
    total += 1

    y_label = y_test[0]
    correct += np.argmax(logits_dec) == y_label

    out_string = f"""
    Count: {total}
    Accuracy: {correct / total}
    Average latency: {total_time / total:.02f}s
    """

    print(out_string)
