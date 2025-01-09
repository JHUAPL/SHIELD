# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import pytest
import torch
import numpy as np

from src.cnn_context import create_cnn_context
from src.he_cnn.utils import *

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

class Info():
    def __init__(self, mult_depth = 30, scale_factor_bits = 40, batch_size = 32 * 32 * 32, max = 255, min = 0, h = 128, w = 128, channel_size = 3, ker_size = 3):
        self.mult_depth = mult_depth
        self.scale_factor_bits = scale_factor_bits
        self.batch_size = batch_size
        self.max = max
        self.min = min
        self.h = h
        self.w = w
        self.channel_size = channel_size
        self.ker_size = ker_size

        rand_tensor = (max-min)*torch.rand((channel_size, h, w)) + min
        self.rand_tensor = rand_tensor

        self.cc, self.keys = create_cc_and_keys(batch_size, mult_depth=mult_depth, scale_factor_bits=scale_factor_bits, bootstrapping=False)

        self.input_img = create_cnn_context(self.rand_tensor, self.cc, self.keys.publicKey, verbose=True)

@pytest.fixture
def check1():
    return Info(30, 40, 32 * 32 * 32, 1, -1, 64, 64, 4, 3)

@pytest.fixture
def check2():
    return Info(30, 40, 32 * 32 * 32, 1, -1, 64, 64, 1, 3)

@pytest.fixture
def check3():
    return Info(30, 40, 32, 1, -1, 16, 16, 2, 3)

def test_apply_conv2d_c1(check1) -> None:
    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(ConvLayer, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

        def forward(self, x):
            x = self.conv(x)
            return x
        
    model = ConvLayer(check1.channel_size, check1.channel_size, check1.ker_size)
    model.eval()
    layer = model.conv

    pt_conv = model(check1.rand_tensor)
    pt_conv = torch.squeeze(pt_conv, axis=0).detach().numpy()

    conv1 = check1.input_img.apply_conv(layer)
    dec_conv1 = conv1.decrypt_to_tensor(check1.cc, check1.keys).numpy().squeeze()

    assert np.allclose(dec_conv1, pt_conv, atol=1e-03), "Convolution result did not match between HE and PyTorch, failed image < shard"

def test_apply_conv2d_c2(check2) -> None:
    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(ConvLayer, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

        def forward(self, x):
            x = self.conv(x)
            return x
        
    model = ConvLayer(check2.channel_size, check2.channel_size, check2.ker_size)
    model.eval()
    layer = model.conv

    pt_conv = model(check2.rand_tensor)
    pt_conv = torch.squeeze(pt_conv, axis=0).detach().numpy()

    conv1 = check2.input_img.apply_conv(layer)
    dec_conv1 = conv1.decrypt_to_tensor(check2.cc, check2.keys).numpy().squeeze()

    assert np.allclose(dec_conv1, pt_conv, atol=1e-03), "Convolution result did not match between HE and PyTorch, failed channel < shard"

def test_apply_conv2d_c3(check3) -> None:
    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(ConvLayer, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

        def forward(self, x):
            x = self.conv(x)
            return x
        
    model = ConvLayer(check3.channel_size, check3.channel_size, check3.ker_size)
    model.eval()
    layer = model.conv

    pt_conv = model(check3.rand_tensor)
    pt_conv = torch.squeeze(pt_conv, axis=0).detach().numpy()

    conv1 = check3.input_img.apply_conv(layer)
    dec_conv1 = conv1.decrypt_to_tensor(check3.cc, check3.keys).numpy().squeeze()

    assert np.allclose(dec_conv1, pt_conv, atol=1e-03), "Convolution result did not match between HE and PyTorch, failed channel > shard"


def test_apply_pool_c1(check1) -> None:
    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(ConvLayer, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

        def forward(self, x):
            x = self.conv(x)
            return x

    model = ConvLayer(check1.channel_size, check1.channel_size, check1.ker_size)
    model.eval()
    layer = model.conv

    pt_conv = model(check1.rand_tensor)
    pt_max_pool = torch.nn.AvgPool2d(2)
    pt_pool = pt_max_pool(pt_conv)
    pt_pool = pt_pool.detach().numpy()

    conv1 = check1.input_img.apply_conv(layer)
    pool = conv1.apply_pool()
    dec_pool = pool.decrypt_to_tensor(check1.cc, check1.keys).numpy()

    assert np.allclose(dec_pool, pt_pool, atol=1e-03), "Pooling result did not match between HE and PyTorch, failed image < shard"

def test_apply_pool_c2(check2) -> None:
    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(ConvLayer, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

        def forward(self, x):
            x = self.conv(x)
            return x

    model = ConvLayer(check2.channel_size, check2.channel_size, check2.ker_size)
    model.eval()
    layer = model.conv

    pt_conv = model(check2.rand_tensor)
    pt_max_pool = torch.nn.AvgPool2d(2)
    pt_pool = pt_max_pool(pt_conv)
    pt_pool = pt_pool.detach().numpy()

    conv1 = check2.input_img.apply_conv(layer)
    pool = conv1.apply_pool()
    dec_pool = pool.decrypt_to_tensor(check2.cc, check2.keys).numpy()

    assert np.allclose(dec_pool, pt_pool, atol=1e-03), "Pooling result did not match between HE and PyTorch, failed channel < shard"

def test_apply_pool_c3(check3) -> None:
    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super(ConvLayer, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

        def forward(self, x):
            x = self.conv(x)
            return x

    model = ConvLayer(check3.channel_size, check3.channel_size, check3.ker_size)
    model.eval()
    layer = model.conv

    pt_conv = model(check3.rand_tensor)
    pt_max_pool = torch.nn.AvgPool2d(2)
    pt_pool = pt_max_pool(pt_conv)
    pt_pool = pt_pool.detach().numpy()

    conv1 = check3.input_img.apply_conv(layer)
    pool = conv1.apply_pool()
    dec_pool = pool.decrypt_to_tensor(check3.cc, check3.keys).numpy()

    assert np.allclose(dec_pool, pt_pool, atol=1e-03), "Pooling result did not match between HE and PyTorch, failed channel > shard"

def test_apply_linear_c1(check1) -> None:
    class LinearLayer(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(LinearLayer, self).__init__()
            self.linear_one = torch.nn.Linear(input_size, output_size)

        def forward(self, x):
            x = self.linear_one(x)
            return x

    linear = LinearLayer(len(check1.rand_tensor.flatten()), check1.rand_tensor.shape[0])
    linear.eval()
    pt_linear = linear(check1.rand_tensor.flatten()).detach().numpy()
    
    he_linear = check1.input_img.apply_linear(linear.linear_one)
    dec_linear = check1.cc.decrypt(check1.keys.secretKey, he_linear)[0:check1.rand_tensor.shape[0]]

    assert np.allclose(dec_linear, pt_linear, atol=1e-03), "Linear result did not match between HE and PyTorch, failed image < shard"

def test_apply_linear_c2(check2) -> None:
    class LinearLayer(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(LinearLayer, self).__init__()
            self.linear_one = torch.nn.Linear(input_size, output_size)

        def forward(self, x):
            x = self.linear_one(x)
            return x

    linear = LinearLayer(len(check2.rand_tensor.flatten()), check2.rand_tensor.shape[0])
    linear.eval()
    pt_linear = linear(check2.rand_tensor.flatten()).detach().numpy()
    
    he_linear = check2.input_img.apply_linear(linear.linear_one)
    dec_linear = check2.cc.decrypt(check2.keys.secretKey, he_linear)[0:check2.rand_tensor.shape[0]]

    assert np.allclose(dec_linear, pt_linear, atol=1e-03), "Linear result did not match between HE and PyTorch, failed channel < shard"

def test_apply_gelu_c1(check1) -> None:
    gelu = torch.nn.GELU()
    pt_gelu = gelu(check1.rand_tensor)

    he_gelu = check1.input_img.apply_gelu()
    dec_gelu = he_gelu.decrypt_to_tensor(check1.cc, check1.keys).numpy()

    assert np.allclose(dec_gelu, pt_gelu, atol=1e-03), "GELU result did not match between HE and PyTorch, failed image < shard"

def test_apply_gelu_c2(check2) -> None:
    gelu = torch.nn.GELU()
    pt_gelu = gelu(check2.rand_tensor)

    he_gelu = check2.input_img.apply_gelu()
    dec_gelu = he_gelu.decrypt_to_tensor(check2.cc, check2.keys).numpy()

    assert np.allclose(dec_gelu, pt_gelu, atol=1e-03), "GELU result did not match between HE and PyTorch, failed channel < shard"

def test_apply_gelu_c3(check3) -> None:
    gelu = torch.nn.GELU()
    pt_gelu = gelu(check3.rand_tensor)

    he_gelu = check3.input_img.apply_gelu()
    dec_gelu = he_gelu.decrypt_to_tensor(check3.cc, check3.keys).numpy()

    assert np.allclose(dec_gelu, pt_gelu, atol=1e-03), "GELU result did not match between HE and PyTorch, failed channel > shard"