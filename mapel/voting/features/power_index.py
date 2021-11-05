#!/usr/bin/env python


import itertools
import struct

import numpy as np


def banzhaf(rgWeights, fpThold=0.5, normalize=True):
    """ Compute Banzhaf power index """
    n = len(rgWeights)
    wSum = sum(rgWeights)
    wThold = fpThold * wSum
    rgWeights = np.array(rgWeights)
    cDecisive = np.zeros(n, dtype=np.uint64)
    for bitmask in range(0, 2**n-1):
        w = rgWeights * np.unpackbits(np.uint8(struct.unpack('B' * (8), np.uint64(bitmask))), bitorder='little')[:n]
        wSum = sum(w)
        if wSum >= wThold:
            cDecisive = np.add(cDecisive, (w > (wSum - wThold)))
    phi = cDecisive / (2**n) * 2
    if normalize:
        return phi / sum(phi)
    else:
        return phi


def shapley(rgWeights, fpThold=0.5):
    """ Compute Shapley-Shubik power index """
    n = len(rgWeights)
    wSum = sum(rgWeights)
    wThold = fpThold * wSum
    rgWeights = np.array(rgWeights)
    cDecisive = np.zeros(n, dtype=np.uint64)
    for perm in itertools.permutations(range(n)):
        w = rgWeights[list(perm)]
        dec = 0
        wSum = w[dec]
        while wSum < wThold:
            dec += 1
            wSum += w[dec]
        cDecisive[perm[dec]] += 1
    return cDecisive / sum(cDecisive)