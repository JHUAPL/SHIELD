# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import math
import numpy as np
from pyOpenFHE import CKKS as pal

upsample_cpp = pal.CNN.upsample


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def interleave_lists(lists):
    return [val for tup in zip(*lists) for val in tup]


def invert_permutation(P):
    inverse_permutation = [0] * len(P)
    for i, v in enumerate(P):
        inverse_permutation[v] = i
    return inverse_permutation


def compose_permutations(P1, P2):
    if len(P1) != len(P2):
        raise ValueError("permutations must have equal size")
    permutation = [P1[P2[i]] for i in range(len(P1))]
    return permutation


"""
metadata includes:
    - the new channel permutation
    - the duplication factor
    - the new number of shards
"""


def get_upsample_permutation(shards, num_channels, mtx_size):
    initial_num_shards = len(shards)
    shard_size = shards[0].getBatchSize()
    channel_size = mtx_size * mtx_size
    initial_num_physical_channels_per_shard = math.ceil(shard_size / channel_size)
    final_num_physical_channels_per_shard = math.ceil(shard_size / channel_size / 4)
    num_physical_channels = initial_num_shards * initial_num_physical_channels_per_shard
    initial_dup_factor = math.ceil(num_physical_channels / num_channels)

    # if we start with channel sharding, then no permutation
    if channel_size >= shard_size:
        P = list(range(num_channels))
        return P

    if (initial_dup_factor > 1) and (initial_num_shards > 1):
        raise ValueError("Should not have both duplication and shards at the same time")

    # if we have duplication factor >= 4, then no permutation
    if initial_dup_factor > 2:
        P = list(range(num_channels))
        return P

    # if we have two-fold duplication
    if initial_dup_factor == 2:
        P = list(range(num_channels))
        if num_channels == 1: return P
        P = P[::2] + P[1::2]
        return P

    I = list(range(num_channels))
    I = list(divide_chunks(I, initial_num_physical_channels_per_shard))
    I = [list(divide_chunks(J, 4)) for J in I]
    P = [interleave_lists(J) for J in I]
    P = sum(P, start=[])

    return np.array(P)


"""
This takes a permuted list of ciphertexts stored using channel sharding,
and it reorders them into the identity permutation.

mtx_size and permutation refer to the values after upsampling, not of the input shards
"""


def undo_channel_sharding_permutation(shards, num_channels, mtx_size, permutation):
    num_shards = len(shards)
    shard_size = shards[0].getBatchSize()
    channel_size = mtx_size * mtx_size

    if shard_size > channel_size:
        raise ValueError("This function should only be called on a channel sharded image")

    num_shards_per_channel = channel_size // shard_size

    final_shards = [None for _ in range(num_shards)]
    for i, x in enumerate(shards):
        channel_idx = i // num_shards_per_channel
        subshard_idx = i % num_shards_per_channel
        correct_idx = permutation[channel_idx] * num_shards_per_channel + subshard_idx
        final_shards[correct_idx] = x

    return final_shards
