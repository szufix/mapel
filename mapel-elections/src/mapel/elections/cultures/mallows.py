import copy
import os
import pickle
import random

import numpy as np
import scipy.special
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
        relphi = np.random.random()
    if relphi == 1:
        return 1
    if relphi > 2 or relphi < 0:
        logging.warning("Incorrect norm-phi value")
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


def phi_from_norm_phi(num_candidates=10, norm_phi=None):
    return phi_from_relphi(num_candidates, relphi=norm_phi)

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
    # print(params['norm-phi'], params['phi'])
    insertion_probabilites_list = []
    for i in range(1, num_candidates):
        insertion_probabilites_list.append(computeInsertionProbas(i, params['phi']))
    V = []
    for i in range(num_voters):
        vote = mallowsVote(num_candidates, insertion_probabilites_list)
        if 'weight' in params and params['weight'] > 0:
            probability = np.random.random()
            if probability <= params['weight']:
                vote.reverse()
        V += [vote]
    return V


def generate_norm_mallows_mixture_votes(num_voters, num_candidates, params):
    phi_1 = phi_from_norm_phi(num_candidates, float(params['norm-phi_1']))
    params_1 = {'weight': 0, 'phi': phi_1}
    votes_1 = generate_mallows_votes(num_voters, num_candidates, params_1)


    phi_2 = phi_from_norm_phi(num_candidates, float(params['norm-phi_2']))
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
    lphi = params['norm-phi']
    if 'weight' not in params:
        weight = 0
    else:
        weight = params['weight']

    if 'sec_norm-phi' not in params:
        lphi_2 = lphi
    else:
        lphi_2 = params['sec_norm-phi']

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

    params['phi'] = phi_from_relphi(num_parties, relphi=params['norm-phi'])
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


def generate_approval_resampling_votes(num_voters=None, num_candidates=None, params=None):
    # central_vote = set()
    # for c in range(num_candidates):
    #     if rand.np() <= params['p']:
    #         central_vote.add(c)

    k = int(params['p'] * num_candidates)
    # print(k)
    central_vote = {i for i in range(k)}

    votes = [set() for _ in range(num_voters)]
    for v in range(num_voters):
        vote = set()
        for c in range(num_candidates):
            if np.random.random() <= params['phi']:
                if np.random.random() <= params['p']:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote

    return votes


def generate_approval_moving_resampling_votes(num_voters=None, num_candidates=None, params=None):
    # central_vote = set()
    # for c in range(num_candidates):
    #     if rand.random() <= params['p']:
    #         central_vote.add(c)
    num_legs = params.get('legs', 1)
    breaks = [int(num_voters/num_legs)*i for i in range(num_legs)]
    print(breaks)

    k = int(params['p'] * num_candidates)
    central_vote = {i for i in range(k)}
    ccc = copy.deepcopy(central_vote)

    votes = [set() for _ in range(num_voters)]
    votes[0] = copy.deepcopy(central_vote)

    for v in range(1, num_voters):
        vote = set()
        for c in range(num_candidates):
            # if np.random.random() <= params['phi']**3:
            if np.random.random() <= params['phi']:
                if np.random.random() <= params['p']:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote
        central_vote = copy.deepcopy(vote)

        if v in breaks:
            central_vote = copy.deepcopy(ccc)

    return votes


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


def generate_approval_noise_model_votes(num_voters=None, num_candidates=None,
                                                    params=None):
    if params is None:
        params = {}

    model_type = params.get('type_id', 'hamming')

    k = int(params['p'] * num_candidates)
    # if k == 0:
    #     k = 1
    # elif k == num_candidates:
    #     k = num_candidates-1

    A = {i for i in range(k)}
    B = set(range(num_candidates)) - A

    phi = params['phi']
    # phi = 0.1**(10)

    choices = []
    probabilites = []

    print(phi)

    # PREPARE BUCKETS
    for x in range(len(A) + 1):
        num_options_in = scipy.special.binom(len(A), x)
        for y in range(len(B) + 1):
            num_options_out = scipy.special.binom(len(B), y)

            if model_type == 'hamming':
                factor = phi ** (len(A) - x + y)  # Hamming
            elif model_type == 'jaccard':
                factor = phi ** ((len(A) - x + y) / (len(A) + y))  # Jaccard
            elif model_type == 'zelinka':
                factor = phi ** max(len(A) - x, y)  # Zelinka
            elif model_type == 'bunke-shearer':
                factor = phi ** (max(len(A) - x, y) / max(len(A), x+y))  # Bunke-Shearer
            else:
                "No such model type_id!!!"

            num_options = num_options_in * num_options_out * factor

            choices.append((x, y))
            probabilites.append(num_options)

    # print(probabilites)

    denominator = sum(probabilites)
    probabilites = [p / denominator for p in probabilites]

    # SAMPLE VOTES
    votes = []
    for _ in range(num_voters):
        _id = np.random.choice(range(len(choices)), 1, p=probabilites)[0]
        x, y = choices[_id]
        vote = set(np.random.choice(list(A), x, replace=False))
        vote = vote.union(set(np.random.choice(list(B), y, replace=False)))
        votes.append(vote)

    return votes


def generate_approval_simplex_resampling_votes(num_voters=None, num_candidates=None,
                                                    params=None):
        if 'phi' not in params:
            phi = np.random.random()
        else:
            phi = params['phi']

        if 'g' not in params:
            num_groups = 2
        else:
            num_groups = params['g']

        if 'max_range' not in params:
            params['max_range'] = 1.

        sizes_c = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # sizes_c = runif_in_simplex(num_groups)
        sizes_c = np.concatenate(([0], sizes_c))
        sizes_c = np.cumsum(sizes_c)
        print(sizes_c)

        sum = 71 + 5*8

        sizes_v = [71/sum, 8/sum, 8/sum, 8/sum, 8/sum, 8/sum]
        # sizes_v = runif_in_simplex(num_groups)
        sizes_v = np.concatenate(([0], sizes_v))
        sizes_v = np.cumsum(sizes_v)
        print(sizes_v)

        votes = [set() for _ in range(num_voters)]

        for g in range(num_groups):

            central_vote = {i for i in range(int(sizes_c[g] * num_candidates),
                                             int(sizes_c[g+1] * num_candidates))}

            for v in range(int(sizes_v[g] * num_voters), int(sizes_v[g + 1] * num_voters)):
                vote = set()
                for c in range(num_candidates):
                    if np.random.random() <= phi:
                        if np.random.random() <= params['p']:
                            vote.add(c)
                    else:
                        if c in central_vote:
                            vote.add(c)
                votes[v] = vote

            # sum_p = sum([sum(vote) for vote in votes])
            # avg_p = sum_p / (num_voters * num_candidates)
            # print(avg_p, params['max_range'])
            # if avg_p < params['max_range']:
            #     break

        return votes


def generate_approval_disjoint_resamplin_votes(num_voters=None, num_candidates=None, params=None,
                                               equal_sizes=True):
    if 'phi' not in params:
        phi = np.random.random()
    else:
        phi = params['phi']

    if 'g' not in params:
        num_groups = 2
    else:
        num_groups = params['g']

    k = int(params['p'] * num_candidates)

    if equal_sizes:
        sizes = [1./num_groups for _ in range(num_groups)]
    else:
        sizes = runif_in_simplex(num_groups)
    print(sizes)
    sizes = np.concatenate(([0], sizes))
    sizes = np.cumsum(sizes)


    votes = [set() for _ in range(num_voters)]

    for g in range(num_groups):

        central_vote = {g * k + i for i in range(k)}

        for v in range(int(sizes[g]*num_voters), int(sizes[g+1]*num_voters)):
            vote = set()
            for c in range(num_candidates):
                if np.random.random() <= phi:
                    if np.random.random() <= params['p']:
                        vote.add(c)
                else:
                    if c in central_vote:
                        vote.add(c)
            votes[v] = vote

    return votes


def generate_approval_truncated_mallows_votes(num_voters=None, num_candidates=None, params=None):
    # k = int(params['p'] * num_candidates)

    if 'norm-phi' not in params:
        params['norm-phi'] = np.random.rand()

    params['phi'] = phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    ordinal_votes = generate_mallows_votes(num_voters, num_candidates, params)

    # print(params)
    if 'max_range' not in params:
        params['max_range'] = 1.

    votes = []
    k = np.random.randint(low=1., high=int(params['max_range'] * num_candidates))
    for v in range(num_voters):
        # k = -1
        # while k not in range(0, num_candidates + 1):
        #     k = int(np.random.normal(params['p'],
        #                              params['norm-phi'] / (num_candidates ** 0.5)) * num_candidates)
            # k = int(np.random.normal(params['p'], 0.05) * num_candidates)
        votes.append(set(ordinal_votes[v][0:k]))

    return votes


def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)


def approval_anti_pjr_votes(num_voters=None, num_candidates=None, params=None):

    if 'p' not in params:
        p = np.random.random()
    else:
        p = params['p']

    if 'phi' not in params:
        phi = np.random.random()
    else:
        phi = params['phi']

    if 'g' not in params:
        num_groups = 2
    else:
        num_groups = params['g']

    c_group_size = int(num_candidates/num_groups)
    v_group_size = int(num_voters/num_groups)
    size = int(p * num_candidates)

    votes = []
    for g in range(num_groups):

        core = {g * c_group_size + i for i in range(c_group_size)}
        rest = set(list(range(num_candidates))) - core

        if size <= c_group_size:
            central_vote = set(np.random.choice(list(core), size=size))
        else:
            central_vote = set(np.random.choice(list(rest), size=size-c_group_size))
            central_vote = central_vote.union(core)

        for v in range(v_group_size):
            vote = set()
            for c in range(num_candidates):
                if np.random.random() <= phi:
                    if np.random.random() <= p:
                        vote.add(c)
                else:
                    if c in central_vote:
                        vote.add(c)
            votes.append(vote)

    return votes


def approval_partylist_votes_old(num_voters=None, num_candidates=None, params=None):

    if 'g' not in params:
        num_groups = 2
    else:
        num_groups = params['g']

    if 'shift' not in params:
        shift = False
    else:
        shift = True

    if 'm' not in params:
        m = 0
    else:
        m = params['m']

    c_group_size = int(num_candidates/num_groups)
    v_group_size = int(num_voters/num_groups)

    votes = []
    if not shift:
        for g in range(num_groups):
            for v in range(v_group_size):
                vote = set()
                for c in range(c_group_size):
                    c += g*c_group_size
                    vote.add(c)
                for _ in range(m):
                    el = random.sample(vote, 1)[0]
                    vote.remove(el)
                votes.append(vote)
    else:
        shift = int(c_group_size/2)
        for g in range(num_groups):
            for v in range(int(v_group_size/2)):
                vote = set()
                for c in range(c_group_size):
                    c += g*c_group_size
                    vote.add(c)
                for _ in range(m):
                    el = random.sample(vote, 1)[0]
                    vote.remove(el)
                votes.append(vote)

            for v in range(int(v_group_size/2), v_group_size):
                vote = set()
                for c in range(c_group_size):
                    c += g*c_group_size + shift
                    c %= num_candidates
                    vote.add(c)
                for _ in range(m):
                    el = random.sample(vote, 1)[0]
                    vote.remove(el)
                votes.append(vote)

    return votes


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

