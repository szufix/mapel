import random as rand
import numpy as np
import math


def generate_impartial_anonymous_culture_election(num_voters=None, num_candidates=None):
    alpha = 1. / math.factorial(num_candidates)

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    urn_size = 1.
    for j in range(num_voters):
        rho = rand.random()
        if rho <= urn_size:
            votes[j] = np.random.permutation(num_candidates)
        else:
            votes[j] = votes[rand.randint(0, j - 1)]
        urn_size = 1. / (1. + alpha * (j + 1.))

    return votes


def generate_impartial_culture_election(num_voters=None, num_candidates=None):
    """ helper function: generate impartial culture elections """

    votes = np.zeros([num_voters, num_candidates], dtype=int)

    for j in range(num_voters):
        votes[j] = np.random.permutation(num_candidates)

    return votes