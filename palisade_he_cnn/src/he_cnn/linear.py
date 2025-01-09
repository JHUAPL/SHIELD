# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import numpy as np
from pyOpenFHE import CKKS as pal_ckks

linear_cpp = pal_ckks.CNN.linear


def linear(channel_shards, weights, biases, mtx_size, permutation=None, scale=1.0, pool_factor=1):
    shard_size = channel_shards[0].getBatchSize()
    num_shards = len(channel_shards)
    num_inputs = weights.shape[1]
    channel_size = mtx_size * mtx_size
    duplication_factor = max(shard_size // num_inputs, 1)
    num_physical_channels_per_shard = shard_size // channel_size
    num_physical_channels = num_physical_channels_per_shard * num_shards
    num_logical_channels = num_physical_channels // duplication_factor

    if permutation is None:
        permutation = np.array(range(num_logical_channels))

    output = linear_cpp(channel_shards, weights * scale, mtx_size, permutation, pool_factor)

    # FO: if np.all(biases==0), then we do not need to compute biases*scale 
    num_out_activs = biases.shape[0]
    output += np.pad(biases * scale, [(0, shard_size - num_out_activs)])

    return output
