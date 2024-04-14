import numpy as np
import itertools

"""
File contains implemetation of approximate diversity, approximate polarization and approximate agreement indeces.
Because of approximation, this implementation is significantly faster than the original exact computation.
It can deal with experiment containing top truncated elections and elections of different sizes (different number of candidates). 
"""

def remove_diag(mtrx):
    """ Return: Input matrix with diagonal removed (shape[1] - 1) """
    res = np.zeros((mtrx.shape[0], mtrx.shape[1] - 1))
    for i in range(mtrx.shape[0]):
        for j in range(mtrx.shape[0]):
            if j < i:
                res[i, j] = mtrx[i, j]
            elif j > i:
                res[i, j - 1] = mtrx[i, j]
    return res

def vote2pote(vote, m):
    reported = vote[vote != -1]
    part_pote = np.argsort(reported)
    res = []
    i = 0
    non_reported_pos = len(reported) + (m - 1 - len(reported)) / 2
    for c in range(m):
        if c in reported:
            res.append(part_pote[i])
            i = i + 1
        else:
            res.append(non_reported_pos)
    return np.array(res)

def get_potes(election):
    if election.potes is not None:
        return election.potes
    else:
        res = []
        for v in election.votes:
            res.append(vote2pote(v, election.num_candidates))
        res = np.array(res)
        election.potes = res
        return res

def swap_distance_between_potes(pote_1: list, pote_2: list, m : int) -> int:
    """ Return: Swap distance between two potes """
    swap_distance = 0
    for a in range(m):
        for b in range(m):
    # for a, b in itertools.combinations(range(m), 2):
            if (pote_1[a] < pote_1[b] and pote_2[a] >= pote_2[b]):
                swap_distance += 0.5
            if (pote_1[a] <= pote_1[b] and pote_2[a] > pote_2[b]):
                swap_distance += 0.5
    return swap_distance

def get_vote_dists(election):
    try:
        return election.vote_dists
    except:
        potes = get_potes(election)
        distances = np.zeros([election.num_voters, election.num_voters])
        for v1 in range(election.num_voters):
            for v2 in range(v1 + 1, election.num_voters):
                distances[v1][v2] = swap_distance_between_potes(potes[v1], potes[v2], election.num_candidates)
                distances[v2][v1] = distances[v1][v2]
        election.vote_dists = distances
        return distances

def get_candidate_dists(election):
    try:
        return election.candidate_dists
    except:
        potes = get_potes(election)
        distances = np.zeros([election.num_candidates, election.num_candidates])
        for a in range(election.num_candidates):
            for b in range(a + 1, election.num_candidates):
                for v in range(election.num_voters):
                    distances[a][b] += abs(potes[v][a] - potes[v][b])
                distances[b][a] = distances[a][b]
        election.candidate_dists = distances
        return distances

def agreement_index(election) -> dict:
    if election.fake:
        return {'value': None}
    potes = get_potes(election)
    res = 0
    for a, b in itertools.combinations(range(election.num_candidates), 2):
        a_b = 0
        b_a = 0
        for p in potes:
            if p[a] < p[b]:
                a_b += 1
            elif p[b] < p[a]:
                b_a += 1
        res += max(abs(a_b - b_a), election.num_voters - a_b - b_a)
    return {'value': res / election.num_voters / (election.num_candidates - 1) / election.num_candidates * 2}

def cand_pos_dist_std(election) -> dict:
    if election.fake:
        return {'value': None}
    distances = get_candidate_dists(election)
    distances = remove_diag(distances)
    return {'value': - distances.std() / election.num_voters}


def distances_to_rankings(rankings, distances):
    dists = distances[rankings]
    return np.sum(dists.min(axis=0))


def find_improvement(distances, d, starting, rest, n, k, l):
    for cut in itertools.combinations(range(k), l):
        # print(cut)
        for paste in itertools.combinations(rest, l):
            ranks = []
            j = 0
            for i in range(k):
                if i in cut:
                    ranks.append(paste[j])
                    j = j + 1
                else:
                    ranks.append(starting[i])
            # check if unique
            if len(set(ranks)) == len(ranks):
                # check if better
                d_new = distances_to_rankings(ranks, distances)
                if d > d_new:
                    return ranks, d_new, True
    return starting, d, False


def local_search_kKemeny_single_k(election, k, l, starting=None) -> dict:
    if starting is None:
        starting = list(range(k))
    distances = get_vote_dists(election)

    n = election.num_voters

    d = distances_to_rankings(starting, distances)
    iter = 0
    check = True
    while (check):
        # print(iter)
        # print(starting)
        # print(d)
        # print()
        iter = iter + 1
        rest = [i for i in range(n) if i not in starting]
        for j in range(l):
            starting, d, check = find_improvement(distances, d, starting, rest, n, k, j + 1)
            if check:
                break
        # print()
    return {'value': d}

def polarization_index(election) -> dict:
    if election.fake:
        return {'value': None}

    distances = get_vote_dists(election)
    best_1 = np.argmin(distances.sum(axis=1))
    best_vec = distances[best_1]
    first_kemeny = best_vec.sum()
    second_kemeny = local_search_kKemeny_single_k(election, 2, 1)['value']

    max_dist = (election.num_candidates) * (election.num_candidates - 1) / 2
    return {'value': 2 * (first_kemeny - second_kemeny) / election.num_voters / max_dist}

def diversity_index(election) -> dict:
    if election.fake:
        return {'value': None}

    k_1 = local_search_kKemeny_single_k(election, 1, 1)['value']
    k_2 = local_search_kKemeny_single_k(election, 2, 1)['value']
    k_3 = local_search_kKemeny_single_k(election, 3, 1)['value']
    k_4 = local_search_kKemeny_single_k(election, 4, 1)['value']
    k_5 = local_search_kKemeny_single_k(election, 5, 1)['value']

    max_dist = (election.num_candidates) * (election.num_candidates - 1) / 2
    return {'value': (k_1 + k_2 + k_3 + k_4 + k_5) / election.num_voters / max_dist }



