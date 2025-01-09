# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import numpy as np
import math
from pyOpenFHE import CKKS as pal

conv2d_cpp = pal.CNN.conv2d


def conv2d(ciphertext_shards, filters, mtx_size, biases, permutation=None, bn_scale=None, bn_shift=None,
           output_permutation=None):
    # if we're combining with a batch norm, fold the batch norm scale factor into the filters
    # with sharded convolutions, filters are not duplicated or permuted in any way.
    scaled_filters = filters
    if bn_scale is not None and bn_shift is not None:
        scaled_filters = filters * bn_scale.reshape(1, -1, 1, 1)

    # if we're combining with a batch norm, fold the batch norm shift factor into the biases
    shifted_biases = biases
    if bn_scale is not None and bn_shift is not None:
        shifted_biases = biases * bn_scale + bn_shift

    # all of this should happen somewhere in the CNNContext class
    shard_size = ciphertext_shards[0].getBatchSize()
    num_out_channels = filters.shape[1]
    channel_size = mtx_size * mtx_size
    if channel_size < shard_size:
        channels_per_shard = shard_size // (mtx_size * mtx_size)
        output_dup_factor = math.ceil(channels_per_shard / num_out_channels)
    else:
        output_dup_factor = 1

    num_in_channels = filters.shape[0]
    if permutation is None:
        permutation = np.array(range(num_in_channels))

    if output_permutation is None:
        output_permutation = np.array(range(num_out_channels))

    if len(permutation) != num_in_channels:
        raise ValueError("incorrect number of input channels")

    if len(output_permutation) != num_out_channels:
        raise ValueError("incorrect number of output channels")

    scaled_filters = scaled_filters[:, output_permutation, :, :]
    shifted_biases = shifted_biases[output_permutation]

    # compute the convolution
    conv_shards = conv2d_cpp(ciphertext_shards, scaled_filters, mtx_size, permutation)

    repeated_shifted_biases = np.repeat(shifted_biases, mtx_size * mtx_size * output_dup_factor)
    for s in range(len(conv_shards)):
        conv_shards[s] += repeated_shifted_biases[s * shard_size: (s + 1) * shard_size]

    return conv_shards
