import copy
import os
import pickle

import numpy as np
from mapel.roommates.cultures._utils import convert
from mapel.core.utils import *
from mapel.marriages.cultures.impartial import generate_asymmetric_votes

# Given the number m of candidates and a phi\in [0,1] function computes the expected number of swaps
# in a vote sampled from Mallows culture_id
def calculateExpectedNumberSwaps(num_candidates, phi):
    res = phi * num_candidates / (1 - phi)
    for j in range(1, num_candidates + 1):
        res = res + (j * (phi ** j)) / ((phi ** j) - 1)
    return res


# Given the number m of candidates and a absolute number of expected swaps exp_abs, this function
# returns a value of phi such that in a vote sampled from Mallows culture_id with this parameter
# the expected number of swaps is exp_abs
def phi_from_relphi(num_candidates, relphi=None):
    if relphi is None:
        relphi = np.random.random()
    if relphi == 1:
        return 1
    exp_abs = relphi * (num_candidates * (num_candidates - 1)) / 4
    low = 0
    high = 1
    while low <= high:
        mid = (high + low) / 2
        cur = calculateExpectedNumberSwaps(num_candidates, mid)
        if abs(cur - exp_abs) < 1e-5:
            return mid
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1


def computeInsertionProbas(i, phi):
    probas = (i + 1) * [0]
    for j in range(i + 1):
        probas[j] = pow(phi, (i + 1) - (j + 1))
    return probas


def weighted_choice(choices):
    total = 0
    for w in choices:
        total = total + w
    r = np.random.uniform(0, total)
    upto = 0.0
    for i, w in enumerate(choices):
        if upto + w >= r:
            return i
        upto = upto + w
    assert False, "Shouldn't get here"


def mallowsVote(m, insertion_probabilites_list):
    vote = [0]
    for i in range(1, m):
        index = weighted_choice(insertion_probabilites_list[i - 1])
        vote.insert(index, i)
    return vote


def generate_mallows_votes(num_voters, num_candidates, params):
    insertion_probabilites_list = []
    for i in range(1, num_candidates):
        insertion_probabilites_list.append(computeInsertionProbas(i, params['phi']))
    V = []
    for i in range(num_voters):
        vote = mallowsVote(num_candidates, insertion_probabilites_list)
        if params['weight'] > 0:
            probability = np.random.random()
            if probability >= params['weight']:
                vote.reverse()
        V += [vote]
    return V


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


def generate_norm_mallows_votes(num_agents=None, params=None):

    if 'norm-phi' not in params:
        params['norm-phi'] = np.random.rand()

    params['phi'] = phi_from_relphi(num_agents, relphi=params['norm-phi'])
    if 'weight' not in params:
        params['weight'] = 0.

    return generate_mallows_votes(num_agents, num_agents, params)


def generate_mallows_asymmetric_votes(num_agents: int = None, params=None):
    """ Mallows on top of Asymmetric instance """

    votes_left, votes_right = generate_asymmetric_votes(num_agents=num_agents)

    votes_left = mallows_votes(votes_left, params['phi'])
    votes_right = mallows_votes(votes_right, params['phi'])

    return [votes_left, votes_right]

# def generate_norm_mallows__id_votes(num_agents=None, params=None):
#
#     if 'norm-phi' not in params:
#         params['norm-phi'] = np.random.rand()
#
#     params['phi'] = phi_from_relphi(num_agents, relphi=params['norm-phi'])
#     if 'weight' not in params:
#         params['weight'] = 0.
#
#     votes_1 = generate_mallows_votes(num_agents, num_agents, params)
#     votes_2 = [list(range(num_agents)) for _ in range(num_agents)]
#
#     return [votes_1, votes_2]


def mallows_vote(vote, phi):
    num_candidates = len(vote)
    params = {'weight': 0, 'phi': phi}
    raw_vote = generate_mallows_votes(1, num_candidates, params)[0]
    new_vote = [0] * len(vote)
    for i in range(num_candidates):
        new_vote[raw_vote[i]] = vote[i]
    return new_vote


def mallows_votes(votes, phi):
    for i in range(len(votes)):
        votes[i] = mallows_vote(votes[i], phi)
    return votes
