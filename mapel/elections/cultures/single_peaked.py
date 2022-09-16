import numpy as np
from random import *

from scipy.special import binom
from mapel.elections.cultures.mallows import phi_from_relphi, mallows_votes


def generate_sp_party(model=None, num_voters=None, num_candidates=None, params=None) -> np.ndarray:
    candidates = [[] for _ in range(num_candidates)]
    _ids = [i for i in range(num_candidates)]

    for j in range(params['num_parties']):
        for w in range(params['num_winners']):
            _id = j * params['num_winners'] + w
            candidates[_id] = [np.random.normal(params['party'][j][0], params['var'])]

    mapping = [x for _, x in sorted(zip(candidates, _ids))]

    if model == 'conitzer_party':
        votes = generate_ordinal_sp_conitzer_votes(num_voters=num_voters, num_candidates=num_candidates)
    elif model == 'walsh_party':
        votes = generate_ordinal_sp_walsh_votes(num_voters=num_voters, num_candidates=num_candidates)
    for i in range(num_voters):
        for j in range(num_candidates):
            votes[i][j] = mapping[votes[i][j]]

    return votes


def generate_ordinal_sp_conitzer_votes(num_voters=None, num_candidates=None) -> np.ndarray:
    """ helper function: generate conitzer single-peaked elections """

    votes = np.zeros([num_voters, num_candidates])
    for j in range(num_voters):
        votes[j][0] = np.random.choice(range(num_candidates))
        left = votes[j][0] - 1
        right = votes[j][0] + 1
        for k in range(1, num_candidates):
            side = np.random.choice([0, 1])
            if side == 0:
                if left >= 0:
                    votes[j][k] = left
                    left -= 1
                else:
                    votes[j][k] = right
                    right += 1
            else:
                if right < num_candidates:
                    votes[j][k] = right
                    right += 1
                else:
                    votes[j][k] = left
                    left -= 1

    return votes


def generate_ordinal_spoc_conitzer_votes(num_voters=None, num_candidates=None) -> np.ndarray:
    """ helper function: generate spoc_conitzer single-peaked elections"""

    votes = np.zeros([num_voters, num_candidates])

    for j in range(num_voters):
        votes[j][0] = np.random.choice(range(num_candidates))
        left = votes[j][0] - 1
        left %= num_candidates
        right = votes[j][0] + 1
        right %= num_candidates
        for k in range(1, num_candidates):
            side = np.random.choice([0, 1])
            if side == 0:
                votes[j][k] = left
                left -= 1
                left %= num_candidates
            else:
                votes[j][k] = right
                right += 1
                right %= num_candidates

    return votes


def generate_ordinal_sp_walsh_votes(num_voters=None, num_candidates=None) -> np.ndarray:
    """ helper function: generate walsh single-peaked elections"""

    votes = np.zeros([num_voters, num_candidates])

    for j in range(num_voters):
        votes[j] = walsh_sp(0, num_candidates - 1)

    return votes.astype(int)


# AUXILIARY
def walsh_sp(a, b):
    if a == b:
        return [a]
    elif np.random.choice([0, 1]) == 0:
        return walsh_sp(a + 1, b) + [a]
    else:
        return walsh_sp(a, b - 1) + [b]


### MATRICES ###

# WALSH

def f(i, j):
    if i < 0: return 0
    return (1.0 / (2 ** (i + j))) * binom(i + j, i)


def probW(m, i, t):
    # probability that c_i is ranked t among m candidates
    return 0.5 * f(i - 1, m - t - (i - 1)) + 0.5 * f(i - t, m - i)


# RANDOM CONITZER

def random_conitzer(C):
    # generate a random vote from the Conitzer culture_id for axis
    # C[0], ..., C[m-1]
    m = len(C)
    center = np.random.randint(0, m)
    left = center
    right = center
    vote = [C[center]]
    for i in range(m - 1):
        L = False
        R = False

        if left > 0 and right < m - 1:
            if random() < 0.5:
                L = True
            else:
                R = True
        elif left > 0:
            L = True
        else:
            R = True

        if L:
            left -= 1
            vote.append(C[left])
        else:
            right += 1
            vote.append(C[right])

    return vote


# CONITZER

def g(m, i, j):
    if i > j: return 0
    if i == j: return 1.0 / m
    if i == 1 and j < m: return g(m, 1, j - 1) + 0.5 * g(m, 2, j)
    if j == m and i > 1: return g(m, i + 1, m) + 0.5 * g(m, i, m - 1)
    if i == 1 and j == m: return 1.0
    return 1.0 / m


#  return 0.5*g(m,i+1,j) + 0.5*g(m,i,j-1)


def probC(m, i, t):
    # probability that c_i is ranked t among m candidates
    p = 0.0
    if t == 1: return 1.0 / m

    if i - (t - 1) > 1:
        p += 0.5 * g(m, i - (t - 1), i - 1)
    elif i - (t - 1) == 1:
        p += g(m, i - (t - 1), i - 1)

    if i + (t - 1) < m:
        p += 0.5 * g(m, i + 1, i + (t - 1))
    elif i + (t - 1) == m:
        p += g(m, i + 1, i + (t - 1))

    return p


PRECISION = 1000
DIGITS = 4


def get_conitzer_matrix(m):
    return get_conitzer_vectors(m).transpose()


def get_walsh_matrix(m):
    return get_walsh_vectors(m).transpose()


def get_conitzer_vectors(m):
    P = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            P[i][j] = probC(m, i + 1, j + 1)
    return P


def simconitzer(m):
    P = [[0] * m for _ in range(m)]
    T = 100000

    C = list(range(m))
    for t in range(T):
        if t % 10000 == 0: print(t)
        v = random_conitzer(C)
        for i in range(m):
            P[v[i]][i] += 1

    for j in range(m):
        for i in range(m):
            P[i][j] = str(int(PRECISION * (P[i][j] / T))).rjust(DIGITS)
    return P


def get_walsh_vectors(m):
    P = np.zeros([m, m])
    for i in range(m):
        for t in range(m):
            P[i][t] = probW(m, i + 1, t + 1)
    return P


########  MALLOWS SP  ########
def generate_conitzer_mallows_votes(num_voters, num_candidates, params):
    params['phi'] = phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    votes = generate_ordinal_sp_conitzer_votes(num_voters=num_voters, num_candidates=num_candidates)

    votes = mallows_votes(votes, params['phi'])

    return votes


def generate_walsh_mallows_votes(num_voters, num_candidates, params):
    params['phi'] = phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    votes = generate_ordinal_sp_walsh_votes(num_voters=num_voters, num_candidates=num_candidates)

    votes = mallows_votes(votes, params['phi'])

    return votes