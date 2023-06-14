#!/usr/bin/env python
import logging

import numpy as np

from mapel.core.features.mallows import mallows_votes


###########
# ORDINAL #
###########
def generate_urn_votes(num_voters: int = None,
                       num_candidates: int = None,
                       alpha: float = 0.1) -> np.ndarray:
    """ Return: ordinal votes from Polya-Eggenberger urn culture
        Params: there is a single 'alpha' parameter which is the contagion parameter
                (default alpha=0.1)"""

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


def generate_mallows_urn_votes(num_voters: int = None,
                       num_candidates: int = None,
                       alpha: float = 0.1,
                       phi: float = 0.25,
                       **kwargs) -> np.ndarray:

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

    votes = mallows_votes(votes, phi=phi)
    return votes



############
# APPROVAL #
############
def generate_approval_truncated_urn_votes(num_voters: int = None,
                                          num_candidates: int = None,
                                          alpha: float = 0.1,
                                          p: float = 0.5) -> list:
    """ Return: approval votes from a truncated variant of Polya-Eggenberger urn culture
        Params: there is a single 'p' parameter which defines the probability of
                a voter approving a candidate (default p=0.5) """

    ordinal_votes = generate_urn_votes(num_voters=num_voters,
                                       num_candidates=num_candidates,
                                       alpha=alpha)
    if p > 1 or p < 0:
        logging.warning(f'Incorrect value of p: {p}. Value should be in [0,1]')
    votes = []
    k = int(p * num_candidates)
    for v in range(num_voters):
        set_ = set(ordinal_votes[v][0:k])
        set_ = {int(x) for x in set_}
        votes.append(set_)

    return votes


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.04.2023 #
# # # # # # # # # # # # # # # #
