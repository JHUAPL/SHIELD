# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

from .utils import *
import math
from pyOpenFHE import CKKS as pal

pool = pal.CNN.pool


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


def get_pool_permutation(shards, num_channels, mtx_size):
    initial_num_shards = len(shards)
    shard_size = shards[0].getBatchSize()
    channel_size = mtx_size * mtx_size
    initial_num_physical_channels_per_shard = math.ceil(shard_size / channel_size)
    num_physical_channels = initial_num_shards * initial_num_physical_channels_per_shard
    initial_dup_factor = math.ceil(num_physical_channels / num_channels)

    # if we have channel sharding, then no permutation
    if channel_size >= shard_size:
        C = num_channels
        P = list(range(C))
        return P

    if (initial_dup_factor > 1) and (initial_num_shards > 1):
        raise ValueError("Should not have both duplication and shards at the same time")

    # if we have duplication, then no permutation
    if initial_dup_factor > 1:
        C = initial_num_physical_channels_per_shard // initial_dup_factor
        P = list(range(C))
        return P

    C = initial_num_physical_channels_per_shard * initial_num_shards
    I = list(range(C))
    I = list(divide_chunks(I, initial_num_physical_channels_per_shard))
    I = list(divide_chunks(I, 4))
    P = [interleave_lists(J) for J in I]
    P = sum(P, start=[])

    return np.array(P)
