import math
import random as rand

import numpy as np


def generate_approval_ic_election(num_voters=None, num_candidates=None, params=None):
    """ Generate (approval) votes from Impartial Culture """

    if params is None:
        params = {}
    if 'p' not in params:
        params['p'] = 0.5
    votes = [set() for _ in range(num_voters)]
    for i in range(num_voters):
        for j in range(num_candidates):
            if rand.random() <= params['p']:
                votes[i].add(j)
    return votes


def generate_approval_id_election(num_voters=None, num_candidates=None, params=None):
    """ Generate (approval) votes from Identity for approval """

    if params is None:
        params = {}
    if 'p' not in params:
        params['p'] = 0.5
    k = int(params['p'] * num_candidates)
    vote = {i for i in range(k)}
    votes = [vote for _ in range(num_voters)]
    return votes


def generate_approval_full(num_voters=None, num_candidates=None):
    """ Generate (approval) votes from Identity for approval """
    vote = {i for i in range(num_candidates)}
    return [vote for _ in range(num_voters)]


def generate_approval_empty(num_voters=None):
    """ Generate (approval) votes from Identity for approval """
    return [set() for _ in range(num_voters)]


def generate_impartial_anonymous_culture_election(num_voters=None, num_candidates=None):
    """ Generate (ordinal) votes from Impartial Anonymous Culture """
    alpha = 1. / math.factorial(num_candidates)

    votes = [list() for _ in range(num_voters)]

    urn_size = 1.
    for j in range(num_voters):
        rho = rand.random()
        if rho <= urn_size:
            votes[j] = list(np.random.permutation(num_candidates))
        else:
            votes[j] = votes[rand.randint(0, j - 1)]
        urn_size = 1. / (1. + alpha * (j + 1.))

    return votes


def generate_impartial_culture_election(num_voters=None, num_candidates=None):
    """ Generate (ordinal) votes from Impartial Culture """
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    for j in range(num_voters):
        votes[j] = np.random.permutation(num_candidates)
    return votes


def generate_ic_party(num_voters=None, params=None):
    """ Generate (party) votes from Impartial Culture"""
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

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
