import copy
import os
import pickle
import random

import numpy as np
import logging

import mapel.core.features.mallows as ml


def generate_mallows_votes(*args, **kwargs):
    return ml.generate_mallows_votes(*args, **kwargs)


def generate_norm_mallows_mixture_votes(num_voters, num_candidates, params):
    phi_1 = ml.phi_from_normphi(num_candidates, float(params['normphi_1']))
    params_1 = {'weight': 0, 'phi': phi_1}
    votes_1 = generate_mallows_votes(num_voters, num_candidates, params_1)

    phi_2 = ml.phi_from_normphi(num_candidates, float(params['normphi_2']))
    params_2 = {'weight': 1, 'phi': phi_2}
    votes_2 = generate_mallows_votes(num_voters, num_candidates, params_2)

    votes = []
    size_1 = int((1 - float(params['weight'])) * num_voters)
    for i in range(size_1):
        votes.append(votes_1[i])
    for i in range(size_1, num_voters):
        votes.append(votes_2[i])

    return votes


def calculateZpoly(m):
    res = [1]
    for i in range(1, m + 1):
        mult = [1] * i
        res2 = [0] * (len(res) + len(mult) - 1)
        for o1, i1 in enumerate(res):
            for o2, i2 in enumerate(mult):
                res2[o1 + o2] += i1 * i2
        res = res2
    return res


def evaluatePolynomial(coeff, x):
    res = 0
    for i, c in enumerate(coeff):
        res += c * (x ** i)
    return res


def calculateZ(m, phi):
    coeff = calculateZpoly(m)
    return evaluatePolynomial(coeff, phi)


# mat[i][j] is the probability with which candidate i ends up in position j
def mallowsMatrix(num_candidates, lphi, pos, normalize=True):
    mat = np.zeros([num_candidates, num_candidates])
    if normalize:
        phi = ml.phi_from_normphi(num_candidates, lphi)
    else:
        phi = lphi
    Z = calculateZ(num_candidates, phi)
    for i in range(num_candidates):
        for j in range(num_candidates):
            freqs = [pos[k][i][j] for k in
                     range(1 + int(num_candidates * (num_candidates - 1) / 2))]
            unnormal_prob = evaluatePolynomial(freqs, phi)
            mat[i][j] = unnormal_prob / Z
    return mat


def get_mallows_matrix(num_candidates, params, normalize=True):
    lphi = params['normphi']
    if 'weight' not in params:
        weight = 0
    else:
        weight = params['weight']

    if 'sec_normphi' not in params:
        lphi_2 = lphi
    else:
        lphi_2 = params['sec_normphi']

    try:
        path = os.path.join(os.getcwd(), 'mapel', 'elections', 'cultures',
                            'mallows_positionmatrices',
                            str(num_candidates) + "_matrix.txt")
        with open(path, "r") as file:
            pos = pickle.load(file)
    except FileNotFoundError:
        print("Mallows matrix only supported for up to 30 candidates")
    mat1 = mallowsMatrix(num_candidates, lphi, pos, normalize)
    mat2 = mallowsMatrix(num_candidates, lphi_2, pos, normalize)
    res = np.zeros([num_candidates, num_candidates])
    for i in range(num_candidates):
        for j in range(num_candidates):
            res[i][j] = (1. - weight) * mat1[i][j] + (weight) * mat2[i][num_candidates - 1 - j]
    return res


def get_mallows_vectors(num_candidates, fake_param):
    return get_mallows_matrix(num_candidates, fake_param).transpose()


def generate_mallows_party(num_voters=None,
                           num_candidates=None,
                           election_model=None,
                           params=None):
    num_parties = params['num_parties']
    num_winners = params['num_winners']
    party_size = num_winners

    params['phi'] = ml.phi_from_normphi(num_parties, normphi=params['main-phi'])
    mapping = generate_mallows_votes(num_voters, num_parties, params)[0]

    params['phi'] = ml.phi_from_normphi(num_parties, normphi=params['normphi'])
    votes = generate_mallows_votes(num_voters, num_parties, params)

    for i in range(num_voters):
        for j in range(num_parties):
            votes[i][j] = mapping[votes[i][j]]

    new_votes = [[] for _ in range(num_voters)]

    for i in range(num_voters):
        for j in range(num_parties):
            for w in range(party_size):
                _id = votes[i][j] * party_size + w
                new_votes[i].append(_id)

    return new_votes


def generate_approval_truncated_mallows_votes(num_voters=None,
                                              num_candidates=None,
                                              max_range=1,
                                              normphi=None,
                                              weight=None,
                                              **kwargs):
    phi = ml.phi_from_normphi(num_candidates, normphi=normphi)

    ordinal_votes = generate_mallows_votes(num_voters, num_candidates, phi=phi, weight=weight)

    votes = []
    k = np.random.randint(low=1., high=int(max_range * num_candidates))
    for v in range(num_voters):
        votes.append(set(ordinal_votes[v][0:k]))

    return votes


def runif_in_simplex(n):
    ''' Return uniformly random vector in the n-simplex '''

    k = np.random.exponential(scale=1.0, size=n)
    return k / sum(k)
