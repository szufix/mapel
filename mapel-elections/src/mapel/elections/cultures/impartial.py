import math

import numpy as np


def generate_approval_ic_votes(num_voters: int = None, num_candidates: int = None,
                               params: dict = None) -> list:
    """ Return: approval votes from Impartial Culture """
    if params is None:
        params = {}
    p = params.get('p', 0.5)
    votes = [set() for _ in range(num_voters)]
    for i in range(num_voters):
        for j in range(num_candidates):
            if np.random.random() <= p:
                votes[i].add(j)
    return votes


def generate_approval_id_votes(num_voters: int = None, num_candidates: int = None,
                               params: dict = None) -> list:
    """ Return: approval votes from Identity for approval """
    if params is None:
        params = {}
    p = params.get('p', 0.5)
    k = int(p * num_candidates)
    vote = {i for i in range(k)}
    return [vote for _ in range(num_voters)]


def generate_approval_full_votes(num_voters: int = None, num_candidates: int = None) -> list:
    """ Return: approval votes from Identity for approval """
    vote = {i for i in range(num_candidates)}
    return [vote for _ in range(num_voters)]


def generate_approval_empty_votes(num_voters: int = None) -> list:
    """ Return: approval votes from Identity for approval """
    return [set() for _ in range(num_voters)]


def generate_impartial_anonymous_culture_election(num_voters: int = None,
                                                  num_candidates: int = None) -> list:
    """ Return: ordinal votes from Impartial Anonymous Culture """
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

    return votes


def generate_ordinal_ic_votes(num_voters: int = None,
                              num_candidates: int = None) -> np.ndarray:
    """ Return: ordinal votes from Impartial Culture """
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    for j in range(num_voters):
        votes[j] = np.random.permutation(num_candidates)
    return votes


def generate_ic_party(num_voters: int = None, params: dict = None) -> list:
    """ Return: party votes from Impartial Culture"""
    num_parties = params['num_parties']
    party_size = params['num_winners']

    votes = np.zeros([num_voters, num_parties], dtype=int)

    for j in range(num_voters):
        votes[j] = np.random.permutation(num_parties)

    new_votes = [[] for _ in range(num_voters)]
    for i in range(num_voters):
        for j in range(num_parties):
            for w in range(party_size):
                _id = votes[i][j] * party_size + w
                new_votes[i].append(_id)
    return new_votes


def generate_weighted_stratification_votes(num_voters: int = None, num_candidates: int = None,
                                           params=None):
    if params is None:
        params = {}

    w = params.get('w', 0.5)

    return [list(np.random.permutation(int(w*num_candidates))) +
             list(np.random.permutation([j for j in range(int(w*num_candidates), num_candidates)]))
            for _ in range(num_voters)]

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 17.08.2022 #
# # # # # # # # # # # # # # # #
