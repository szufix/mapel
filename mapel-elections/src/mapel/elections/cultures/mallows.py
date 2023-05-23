import copy
import os
import pickle
import random

import numpy as np
import logging


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
        logging.warning('normphi is not defined')
        return -1
    if relphi == 1:
        return 1
    if relphi > 2 or relphi < 0:
        logging.warning("Incorrect normphi value")
    if relphi > 1:
        return 2-relphi
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


def phi_from_normphi(num_candidates=10, normphi=None):
    return phi_from_relphi(num_candidates, relphi=normphi)

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


def generate_mallows_votes(num_voters, num_candidates, phi=None, weight=0, **kwargs):
    if phi is None:
        logging.warning('phi is not defined')
    insertion_probabilites_list = []
    for i in range(1, num_candidates):
        insertion_probabilites_list.append(computeInsertionProbas(i, phi))
    V = []
    for i in range(num_voters):
        vote = mallowsVote(num_candidates, insertion_probabilites_list)
        if weight > 0:
            probability = np.random.random()
            if probability <= weight:
                vote.reverse()
        V += [vote]
    return V


def generate_norm_mallows_mixture_votes(num_voters, num_candidates, params):
    phi_1 = phi_from_normphi(num_candidates, float(params['normphi_1']))
    params_1 = {'weight': 0, 'phi': phi_1}
    votes_1 = generate_mallows_votes(num_voters, num_candidates, params_1)


    phi_2 = phi_from_normphi(num_candidates, float(params['normphi_2']))
    params_2 = {'weight': 1, 'phi': phi_2}
    votes_2 = generate_mallows_votes(num_voters, num_candidates, params_2)

    votes = []
    size_1 = int((1-float(params['weight']))*num_voters)
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
        path = os.path.join(os.getcwd(), 'mapel', 'elections', 'cultures', 'mallows_positionmatrices',
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
            res[i][j] = (1.-weight) * mat1[i][j] + (weight) * mat2[i][num_candidates - 1 - j]
    return res


def get_mallows_vectors(num_candidates, fake_param):
    return get_mallows_matrix(num_candidates, fake_param).transpose()


def generate_mallows_party(num_voters=None, num_candidates=None,
                           election_model=None, params=None):
    num_parties = params['num_parties']
    num_winners = params['num_winners']
    party_size = num_winners

    params['phi'] = phi_from_relphi(num_parties, relphi=params['main-phi'])
    mapping = generate_mallows_votes(num_voters, num_parties, params)[0]

    params['phi'] = phi_from_relphi(num_parties, relphi=params['normphi'])
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





# def generate_approval_hamming_noise_model_votes(num_voters=None, num_candidates=None, params=None):
#     k = int(params['p'] * num_candidates)
#     central_vote = {i for i in range(k)}
#
#     votes = [set() for _ in range(num_voters)]
#     for v in range(num_voters):
#         vote = set()
#         for c in range(num_candidates):
#             if c in central_vote:
#                 if np.random.random() <= 1 - params['phi']:
#                     vote.add(c)
#             else:
#                 if np.random.random() < params['phi']:
#                     vote.add(c)
#         votes[v] = vote
#
#     return votes


def generate_approval_truncated_mallows_votes(num_voters=None, num_candidates=None, max_range=1,
                                              normphi=None, weight=None):

    phi = phi_from_relphi(num_candidates, relphi=normphi)

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




def mallows_vote(vote, phi):
    num_candidates = len(vote)
    raw_vote = generate_mallows_votes(1, num_candidates, phi)[0]
    new_vote = [0] * len(vote)
    for i in range(num_candidates):
        new_vote[raw_vote[i]] = vote[i]
    return new_vote


def mallows_votes(votes, phi):
    for i in range(len(votes)):
        votes[i] = mallows_vote(votes[i], phi)
    return votes


def generate_norm_mallows_with_walls_votes(num_voters, num_candidates, params):
    votes = []
    upper_half_size = int(num_candidates*params['p'])
    if upper_half_size == 0 or upper_half_size == num_candidates:
        return np.array(generate_mallows_votes(num_voters, num_candidates, params))

    lower_half_size = num_candidates - upper_half_size
    upper_half_votes = np.array(generate_mallows_votes(num_voters, upper_half_size, params))
    lower_half_votes = np.array(generate_mallows_votes(num_voters, lower_half_size, params)) \
                       + upper_half_size
    for i in range(num_voters):
        v = np.concatenate([upper_half_votes[i], lower_half_votes[i]])
        votes.append(v)
    return votes

