#!/usr/bin/env python

import random as rand

import numpy as np


def generate_urn_model_election(num_voters=None, num_candidates=None, params=None):
    """ Generate (ordinal) votes from Polya-Eggenberger model """

    votes = np.zeros([num_voters, num_candidates])
    urn_size = 1.
    for j in range(num_voters):
        rho = rand.uniform(0, urn_size)
        if rho <= 1.:
            votes[j] = np.random.permutation(num_candidates)
        else:
            votes[j] = votes[rand.randint(0, j - 1)]
        urn_size +=  params['alpha']

    return votes


def generate_approval_urn_election(num_voters=None, num_candidates=None, params=None):
    """ Generate (approval) votes from Polya-Eggenberger model """

    votes = []
    urn_size = 1.
    for j in range(num_voters):
        rho = rand.uniform(0, urn_size)
        if rho <= 1.:
            vote = set()
            for c in range(num_candidates):
                if rand.random() <= params['p']:
                    vote.add(c)
            votes.append(vote)
        else:
            votes.append(votes[rand.randint(0, j - 1)])
        urn_size += params['alpha']

    return votes

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
