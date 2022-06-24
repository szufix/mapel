
import random as rand
import numpy as np
import math
import os
from collections import Counter
from scipy.stats import gamma
from numpy import number


def generate_didi_votes(num_voters=None, num_candidates=None, params=None):

    alphas = get_vector('linear', num_candidates)

    for i in range(len(alphas)):
        if alphas[i] == 0.:
            alphas[i] = 0.00001

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    for j in range(num_voters):
        points = np.random.dirichlet(alphas)
        cand = [q for q in range(num_candidates)]
        tmp_candidates = [x for _, x in sorted(zip(points, cand))]
        for k in range(num_candidates):
            votes[j][k] = tmp_candidates[num_candidates - k - 1]

    return votes


# AUXILIARY (alphas)

def get_vector(type, num_candidates):
    if type == "uniform":
        return [1.] * num_candidates
    elif type == "linear":
        return [(num_candidates - x) for x in range(num_candidates)]
    elif type == "linear_low":
        return [(float(num_candidates) - float(x)) / float(num_candidates) for x in range(num_candidates)]
    elif type == "square":
        return [(float(num_candidates) - float(x)) ** 2 / float(num_candidates) ** 2 for x in
                range(num_candidates)]
    elif type == "square_low":
        return [(num_candidates - x) ** 2 for x in range(num_candidates)]
    elif type == "cube":
        return [(float(num_candidates) - float(x)) ** 3 / float(num_candidates) ** 3 for x in
                range(num_candidates)]
    elif type == "cube_low":
        return [(num_candidates - x) ** 3 for x in range(num_candidates)]
    elif type == "split_2":
        values = [1.] * num_candidates
        for i in range(num_candidates / 2):
            values[i] = 10.
        return values
    elif type == "split_4":
        size = num_candidates / 4
        values = [1.] * num_candidates
        for i in range(size):
            values[i] = 1000.
        for i in range(size, 2 * size):
            values[i] = 100.
        for i in range(2 * size, 3 * size):
            values[i] = 10.
        return values
    else:
        return type