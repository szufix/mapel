#!/usr/bin/env python
import logging

import numpy as np

from mapel.core.features.mallows import mallows_votes


def generate_mallows_urn_votes(num_voters: int = None,
                       num_candidates: int = None,
                       alpha: float = None,
                       phi: float = None,
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
