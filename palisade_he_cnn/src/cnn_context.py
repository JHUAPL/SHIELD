# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import numpy as np
import math
import torch
from time import time
from collections import defaultdict
import palisade_he_cnn.src.he_cnn.utils as utils
import palisade_he_cnn.src.he_cnn.conv as conv
import palisade_he_cnn.src.he_cnn.pool as pool
import palisade_he_cnn.src.he_cnn.linear as linear

from pyOpenFHE import CKKS as pal

TIMING_DICT = defaultdict(list)

DUPLICATED, IMAGE_SHARDED, CHANNEL_SHARDED = range(3)


def reset_timing_dict():
    global TIMING_DICT
    TIMING_DICT.clear()


# an image is a PyTorch 3-tensor
def create_cnn_context(image, cc, publicKey, verbose=False):
    # create these to encrypt
    shard_size = cc.getBatchSize()

    if len(image.shape) != 3:
        raise ValueError("Input image must be a PyTorch 3-tensor")

    # do we want to address rectangular images at some point...?
    if image.shape[1] != image.shape[2]:
        raise ValueError("Non-square channels not currently supported")

    if not utils.is_power_of_2(image.shape[0]):
        raise ValueError("Number of channels must be a power-of-two")

    if not utils.is_power_of_2(image.shape[1]):
        raise ValueError("Image dimensions must be a power-of-two")

    mtx_size = image.shape[1]
    num_channels = image.shape[0]
    total_size = mtx_size * mtx_size * num_channels
    num_shards = math.ceil(total_size / shard_size)

    if total_size <= shard_size:
        duplication_factor = shard_size // total_size
    else:
        duplication_factor = 1

    duplicated_image = np.repeat(image.numpy(), duplication_factor, axis=0).flatten()
    shards = []
    for s in range(num_shards):
        shard = cc.encrypt(publicKey, duplicated_image[shard_size * s: shard_size * (s + 1)])
        shards.append(shard)

    # return the cc and keys as well for decryption at the end
    cnn_context = CNNContext(shards, mtx_size, num_channels, permutation=None, verbose=verbose)

    return cnn_context


def timing_decorator_factory(prefix=""):
    def timing_decorator(func):
        def wrapper_function(*args, **kwargs):
            global TIMING_DICT
            start = time()
            res = func(*args, **kwargs)
            layer_time = time() - start
            self = args[0]
            TIMING_DICT[prefix.strip()].append(layer_time)
            if self.verbose:
                print(prefix + f"Layer took {layer_time:.02f} seconds")
            return res

        return wrapper_function

    return timing_decorator


class CNNContext:
    r"""This class contains methods for applying network layers to an image."""

    def __init__(self, shards, mtx_size, num_channels, permutation=None, verbose=False):
        r"""Initializes the CNNContext object. We only needs shards and channel/matrix size to compute all other metadata."""

        if permutation is None:
            permutation = np.array(range(num_channels))

        self.shards = shards
        self.mtx_size = mtx_size
        self.num_channels = num_channels
        self.permutation = permutation
        self.verbose = verbose

        self.compute_metadata()

    def compute_metadata(self):
        # Shard information
        self.num_shards = len(self.shards)
        self.shard_size = self.shards[0].getBatchSize()
        self.total_size = self.num_shards * self.shard_size

        # Channel information
        self.channel_size = self.mtx_size * self.mtx_size

        # Duplication factor
        self.duplication_factor = (self.total_size // self.channel_size) // self.num_channels

        if self.duplication_factor > 1:
            self.shard_type = DUPLICATED
        elif self.channel_size <= self.shard_size:
            self.shard_type = IMAGE_SHARDED
        else:
            self.shard_type = CHANNEL_SHARDED

        # Channel and shard info
        self.num_phys_chan_per_shard = self.shard_size // self.channel_size
        self.num_phys_chan_total = self.num_shards * self.num_phys_chan_per_shard
        self.num_log_chan_per_shard = self.num_phys_chan_per_shard // self.duplication_factor
        self.num_log_chan_total = self.num_shards * self.num_log_chan_per_shard

    def print_metadata(self):
        # shard information
        print(f"num_shards: {self.num_shards}")
        print(f"shard_size: {self.shard_size}")
        print(f"total_size: {self.total_size}")

        # Channel information
        print(f"channel_size: {self.channel_size}")

        # Duplication factor
        print(f"duplication_factor: {self.duplication_factor}")
        print(f"shard_type: {self.shard_type}")

        # Channel and shard info
        print(f"num_phys_chan_per_shard: {self.num_phys_chan_per_shard}")
        print(f"num_phys_chan_total: {self.num_phys_chan_total}")
        print(f"num_log_chan_per_shard: {self.num_log_chan_per_shard}")
        print(f"num_log_chan_total: {self.num_log_chan_total}")

    def decrypt_to_tensor(self, cc, keys):
        # decrypt the shards
        decrypted_shards = [cc.decrypt(keys.secretKey, shard) for shard in self.shards]
        decrypted_output = np.concatenate(decrypted_shards)

        # reshape with possible duplication
        duplicated_output = decrypted_output.reshape(
            self.num_channels * self.duplication_factor,
            self.mtx_size,
            self.mtx_size
        )

        decrypted_deduplicated_output = duplicated_output[0 :: self.duplication_factor]

        return torch.from_numpy(decrypted_deduplicated_output)

    @timing_decorator_factory("Conv ")
    def apply_conv(self, conv_layer, bn_layer=None, output_permutation=None, drop_levels=False):
        pal.CNN.omp_set_nested(0)
        # pal.CNN.omp_set_dynamic(0)

        # Get filters, biases 
        filters, biases = utils.get_filters_and_biases_from_conv2d(conv_layer)

        # Get batch norm info if one is passed in
        if bn_layer:
            scale, shift = utils.get_scale_and_shift_from_bn(bn_layer)
        else:
            scale = None
            shift = None

        num_out_channels = filters.shape[1]
        if output_permutation is None:
            output_permutation = np.array(range(num_out_channels))
        elif len(output_permutation) != num_out_channels:
            raise ValueError("output permutation is incorrect length")

        # TODO this should be a Compress() call
        if drop_levels:
            L = self.shards[0].getTowersRemaining() - 4
            for j in range(self.num_shards):
                for i in range(L):
                    self.shards[j] *= 1.0

        # Apply conv
        new_shards = conv.conv2d(
            ciphertext_shards=self.shards,
            filters=filters,
            mtx_size=self.mtx_size,
            biases=biases,
            permutation=self.permutation,
            bn_scale=scale,
            bn_shift=shift,
            output_permutation=output_permutation
        )

        # Create new CNN Context
        stride = conv_layer.stride
        cnn_context = CNNContext(new_shards, self.mtx_size, num_out_channels, output_permutation, self.verbose)
        if stride == (1, 1):
            return cnn_context
        elif stride == (2, 2):
            return cnn_context.apply_pool(conv=False)
        else:
            raise ValueError("Unsupported stride: {stride}")

    @timing_decorator_factory("Pool ")
    def apply_pool(self, conv=True):
        pal.CNN.omp_set_nested(0)
        # pal.CNN.omp_set_dynamic(0)

        # Apply pool
        new_shards = pool.pool(self.shards, self.mtx_size, conv)

        # Get permutation
        new_permutation = pool.get_pool_permutation(self.shards, self.num_channels, self.mtx_size)
        new_permutation = pool.compose_permutations(self.permutation, new_permutation)
        new_permutation = np.array(new_permutation)

        # Create new CNN Context
        return CNNContext(new_shards, self.mtx_size // 2, self.num_channels, new_permutation, self.verbose)

    @timing_decorator_factory("Fused adaptive pool and linear ")
    def apply_fused_pool_linear(self, linear_layer):
        has_bias = hasattr(linear_layer, "bias")
        return self.apply_linear(linear_layer, has_bias, pool_factor=self.mtx_size)

    @timing_decorator_factory("Bottleneck block ")
    def apply_bottleneck(self, bottleneck_block, debug=False, gelu_params={}, bootstrap_params={}, bootstrap=True):
        # Bottleneck block's forward pass is here: https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

        skip_connection = self
        downsample_block = bottleneck_block.downsample
        if downsample_block:
            conv_downsample_layer = downsample_block[0]
            bn_downsample_layer = downsample_block[1]
            skip_connection = skip_connection.apply_conv(conv_downsample_layer, bn_downsample_layer)

        conv1_layer = bottleneck_block.conv1
        bn1_layer = bottleneck_block.bn1
        cnn_context = self.apply_conv(conv1_layer, bn1_layer)

        if not debug:
            if bootstrap: cnn_context = cnn_context.apply_bootstrapping(**bootstrap_params)
            cnn_context = cnn_context.apply_gelu(**gelu_params)

        conv2_layer = bottleneck_block.conv2
        bn2_layer = bottleneck_block.bn2
        cnn_context = cnn_context.apply_conv(conv2_layer, bn2_layer)

        if not debug:
            if bootstrap: cnn_context = cnn_context.apply_bootstrapping(**bootstrap_params)
            cnn_context = cnn_context.apply_gelu(**gelu_params)

        conv3_layer = bottleneck_block.conv3
        bn3_layer = bottleneck_block.bn3
        cnn_context = cnn_context.apply_conv(conv3_layer, bn3_layer, output_permutation=skip_connection.permutation)

        cnn_context = cnn_context.apply_residual(skip_connection)

        if not debug:
            if bootstrap: cnn_context = cnn_context.apply_bootstrapping(**bootstrap_params)
            cnn_context = cnn_context.apply_gelu(**gelu_params)

        return cnn_context

    # This operation doesn't return a CNNContext, that's returned by linear
    @timing_decorator_factory("Linear ")
    def apply_linear(self, linear_layer, bias=True, scale=1.0, pool_factor=1):
        pal.CNN.omp_set_nested(0)
        pal.CNN.omp_set_dynamic(1)

        linear_weights, linear_biases = utils.get_weights_and_biases_from_linear(linear_layer,
                                                                                 self.mtx_size,
                                                                                 bias,
                                                                                 pool_factor)
        final_shard = linear.linear(self.shards, linear_weights, linear_biases, self.mtx_size, self.permutation, scale,
                                    pool_factor)

        return final_shard

    @timing_decorator_factory("Square ")
    def apply_square(self):
        new_shards = [shard * shard for shard in self.shards]

        return CNNContext(new_shards, self.mtx_size, self.num_channels, self.permutation, self.verbose)

    @timing_decorator_factory("GELU ")
    def apply_gelu(self, bound=10.0, degree=59):
        """
        bound:
            bound = an upper bound on the absolute value of the inputs.
            the polynomial approximation is valid for [-bound, bound]
        degree:
            degree of Chebyshev polynomial
        """
        if self.num_shards < 8:
            pal.CNN.omp_set_nested(1)
            pal.CNN.omp_set_dynamic(1)
        else:
            pal.CNN.omp_set_nested(0)
            pal.CNN.omp_set_dynamic(1)

        # TODO this can be absorbed into the BN
        new_shards = [x * (1 / bound) for x in self.shards]
        new_shards = pal.CNN.fhe_gelu(new_shards, degree, bound)

        return CNNContext(new_shards, self.mtx_size, self.num_channels, self.permutation, self.verbose)

    @timing_decorator_factory("Bootstrapping ")
    def apply_bootstrapping(self, meta=False):
        if self.num_shards < 8:
            pal.CNN.omp_set_nested(1)
            pal.CNN.omp_set_dynamic(1)
        else:
            pal.CNN.omp_set_nested(0)
            pal.CNN.omp_set_dynamic(1)

        cc = self.shards[0].getCryptoContext()
        if meta:
            new_shards = cc.evalMetaBootstrap(self.shards)
        else:
            new_shards = cc.evalBootstrap(self.shards)

        return CNNContext(new_shards, self.mtx_size, self.num_channels, self.permutation, self.verbose)

    @timing_decorator_factory("Residual ")
    def apply_residual(self, C2):
        if len(self.permutation) != len(C2.permutation):
            raise ValueError("Incompatible number of channels")
        if self.mtx_size != C2.mtx_size:
            raise ValueError("Incompatible matrix size")
        if any([i != j for i, j in zip(self.permutation, C2.permutation)]):
            raise ValueError("Incompatible permutations")

        new_shards = [i + j for i, j in zip(self.shards, C2.shards)]
        return CNNContext(new_shards, self.mtx_size, self.num_channels, self.permutation, self.verbose)
