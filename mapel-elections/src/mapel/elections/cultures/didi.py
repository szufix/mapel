
import random as rand
import numpy as np
import math
import os
from collections import Counter
from scipy.stats import gamma
from numpy import number
from mapel.core.utils import get_vector


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
