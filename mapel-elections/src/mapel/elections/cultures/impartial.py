#!/usr/bin/env python
import logging
import math

import numpy as np


###########
# ORDINAL #
###########
def generate_ordinal_ic_votes(num_voters: int = None,
                              num_candidates: int = None) -> np.ndarray:
    """ Return: ordinal votes from impartial culture """
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    for j in range(num_voters):
        votes[j] = np.random.permutation(num_candidates)
    return votes


def generate_impartial_anonymous_culture_election(num_voters: int = None,
                                                  num_candidates: int = None) -> np.ndarray:
    """ Return: ordinal votes from impartial anonymous culture """
    alpha = 1. / math.factorial(num_candidates)

    votes = [list() for _ in range(num_voters)]

    urn_size = 1.
    for j in range(num_voters):
        rho = np.random.random()
        if rho <= urn_size:
            votes[j] = list(np.random.permutation(num_candidates))
        else:
            votes[j] = votes[np.random.randint(0, j - 1)]
        urn_size = 1. / (1. + alpha * (j + 1.))

    return np.array(votes)


############
# APPROVAL #
############
def generate_approval_ic_votes(num_voters: int = None,
                               num_candidates: int = None,
                               p: float = 0.5) -> list:
    """ Return: approval votes from impartial culture
        Params: there is a single 'p' parameter which defines the ratio
                of candidates approves by each candidate (default p=0.5) """
    if p > 1 or p < 0:
        logging.warning(f'Incorrect value of p: {p}. Value should be in [0,1]')

    votes = [set(j for j in range(num_candidates) if np.random.random() <= p)
             for _ in range(num_voters)]

    return votes


def generate_approval_id_votes(num_voters: int = None,
                               num_candidates: int = None,
                               p: float = 0.5) -> list:
    """ Return: approval votes from identity culture
        Params: there is a single 'p' parameter which defines the ratio
                of candidates approves by each candidate (default p=0.5) """
    if p > 1 or p < 0:
        logging.warning(f'Incorrect value of p: {p}. Value should be in [0,1]')
    k = int(p * num_candidates)
    vote = {i for i in range(k)}
    return [vote for _ in range(num_voters)]


def generate_approval_full_votes(num_voters: int = None,
                                 num_candidates: int = None,
                                 **kwargs) -> list:
    """ Return: approval votes where each voter approves all the candidates """
    vote = {i for i in range(num_candidates)}
    return [vote for _ in range(num_voters)]


def generate_approval_empty_votes(num_voters: int = None,
                                  num_candidates: int = None,
                                  **kwargs) -> list:
    """ Return: approval votes where each vote is empty """
    return [set() for _ in range(num_voters)]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.04.2023 #
# # # # # # # # # # # # # # # #
