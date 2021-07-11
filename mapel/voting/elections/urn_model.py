#!/usr/bin/env python

import random as rand
import numpy as np


def generate_urn_model_election(num_voters=None, num_candidates=None, param_1=None):
    """ Generate Polya-Eggenberger urn model election"""

    alpha = param_1
    votes = np.zeros([num_voters, num_candidates])
    urn_size = 1.
    for j in range(num_voters):
        rho = rand.uniform(0, urn_size)
        if rho <= 1.:
            votes[j] = np.random.permutation(num_candidates)
        else:
            votes[j] = votes[rand.randint(0, j - 1)]
        urn_size += alpha

    return votes
