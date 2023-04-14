#!/usr/bin/env python
import logging

import numpy as np


###########
# ORDINAL #
###########
def generate_urn_votes(num_voters: int = None,
                       num_candidates: int = None,
                       params: dict = None) -> np.ndarray:
    """ Return: ordinal votes from Polya-Eggenberger urn culture
        Params: there is a single 'alpha' parameter which is the contagion parameter
                (default alpha=0.1)"""
    if params is None:
        params = {}
    alpha = params.get('alpha', 0.1)
    if alpha < 0:
        logging.warning(f'Incorrect value of alpha: {alpha}. Value should be >=0')

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    urn_size = 1.
    for j in range(num_voters):
        rho = np.random.uniform(0, urn_size)
        if rho <= 1.:
            votes[j] = np.random.permutation(num_candidates)
        else:
            votes[j] = votes[np.random.randint(0, j)]
        urn_size += alpha

    return votes


############
# APPROVAL #
############
def generate_approval_truncated_urn_votes(num_voters: int = None,
                                          num_candidates: int = None,
                                          params: dict = None) -> list:
    """ Return: approval votes from a truncated variant of Polya-Eggenberger urn culture
        Params: there is a single 'p' parameter which defines the probability of
                a voter approving a candidate (default p=0.5) """

    ordinal_votes = generate_urn_votes(num_voters=num_voters,
                                       num_candidates=num_candidates,
                                       params=params)
    if params is None:
        params = {}
    p = params.get('p', 0.5)
    if p > 1 or p < 0:
        logging.warning(f'Incorrect value of p: {p}. Value should be in [0,1]')
    votes = []
    k = int(params['p'] * num_candidates)
    for v in range(num_voters):
        set_ = set(ordinal_votes[v][0:k])
        set_ = {int(x) for x in set_}
        votes.append(set_)

    return votes


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.04.2023 #
# # # # # # # # # # # # # # # #
