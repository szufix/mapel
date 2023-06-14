import os
import pickle

import numpy as np
from mapel.roommates.cultures._utils import convert
from mapel.core.utils import *

import mapel.core.features.mallows as ml


def generate_mallows_votes(*args, **kwargs):
    return ml.generate_mallows_votes(*args, **kwargs)



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
        phi = phi_from_relphi(num_candidates, lphi)
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
    lphi = params['norm-phi']
    weight = params['weight']
    # print(lphi, weight)
    try:
        path = os.path.join(os.getcwd(), 'mapel', 'voting', 'elections', 'mallows_positionmatrices',
                            str(num_candidates) + "_matrix.txt")
        print(path)
        with open(path, "rb") as file:
            pos = pickle.load(file)
    except FileNotFoundError:
        print("Mallows matrix only supported for up to 30 candidates")
    # print(pos)
    mat1 = mallowsMatrix(num_candidates, lphi, pos, normalize)
    res = np.zeros([num_candidates, num_candidates])
    for i in range(num_candidates):
        for j in range(num_candidates):
            res[i][j] = weight * mat1[i][j] + (1 - weight) * mat1[i][num_candidates - 1 - j]
    return res



def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)


def generate_roommates_norm_mallows_votes(num_agents: int = None,
                                          normphi: float = 0.5,
                                          weight: float = 0,
                                          **kwargs):

    phi = ml.phi_from_normphi(num_agents, normphi=normphi)

    votes = generate_mallows_votes(num_agents, num_agents, phi=phi, weight=weight)

    return convert(votes)


def mallows_vote(vote, phi):
    num_candidates = len(vote)
    raw_vote = generate_mallows_votes(1, num_candidates, phi=phi, weight=0)[0]
    new_vote = [0] * len(vote)
    for i in range(num_candidates):
        new_vote[raw_vote[i]] = vote[i]
    return new_vote


def mallows_votes(votes, phi):
    for i in range(len(votes)):
        votes[i] = mallows_vote(votes[i], phi)
    return votes


def generate_roommates_malasym_votes(num_agents: int = None,
                                     normphi=0.5,
                                     **kwargs):
    """ Mallows on top of Asymmetric instance """

    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]

    # if 'norm-phi' not in params:
    #     params['norm-phi'] = np.random.rand()

    phi = phi_from_relphi(num_agents, relphi=normphi)
    votes = mallows_votes(votes, phi)

    return convert(votes)