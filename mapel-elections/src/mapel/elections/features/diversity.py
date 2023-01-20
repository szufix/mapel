import math

import numpy as np
import itertools
# AUXILIARY FUNCTIONS
from mapel.core.glossary import LIST_OF_FAKE_MODELS


def kemeny_ranking(election):
    print(election.culture_id)
    m = election.num_candidates
    wmg = election.votes_to_pairwise_matrix()
    best_d = np.infty
    for test_ranking in itertools.permutations(list(range(m))):
        dist = 0
        for i in range(m):
            for j in range(i+1,m):
                dist = dist + wmg[test_ranking[j],test_ranking[i]]
            if dist > best_d:
                break
        if dist < best_d:
            best = test_ranking
            best_d = dist
    return best, best_d

def gini_coef(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    return (mad / x.mean() / 2)

def geom_mean(x):
    x = np.log(x)
    return np.exp(x.mean())

def swap_distance_between_potes(pote_1: list, pote_2: list) -> int:
    """ Return: Swap distance between two potes """
    swap_distance = 0
    for i, j in itertools.combinations(pote_1, 2):
        if (pote_1[i] > pote_1[j] and
            pote_2[i] < pote_2[j]) or \
                (pote_1[i] < pote_1[j] and
                 pote_2[i] > pote_2[j]):
            swap_distance += 1
    return swap_distance

def remove_diag(mtrx):
    """ Return: Input matrix with diagonal removed (shape[1] - 1) """
    res = np.zeros((mtrx.shape[0], mtrx.shape[1] - 1))
    for i in range(mtrx.shape[0]):
        for j in range(mtrx.shape[0]):
            if j < i:
                res[i,j] = mtrx[i,j]
            elif j > i:
                res[i,j-1] = mtrx[i,j]
    return res

def calculate_borda_scores(election):
    m = election.num_candidates
    borda = np.zeros(m, int)
    for v in election.votes:
        for i, c in enumerate(v):
            borda[c] = borda[c] + m - i - 1
    return borda

def calculate_cand_dom_dist(election):
    distances = election.votes_to_pairwise_matrix()
    distances = np.abs(distances - 0.5)
    np.fill_diagonal(distances, 0)
    return distances

def calculate_cand_pos_dist(election):
    election.compute_potes()
    distances = np.zeros([election.num_candidates, election.num_candidates])
    for c1 in range(election.num_candidates):
        for c2 in range(election.num_candidates):
            dist = 0
            for pote in election.potes:
                dist += abs(pote[c1] - pote[c2])
            distances[c1][c2] = dist
    return distances

def calculate_vote_swap_dist(election):
    election.compute_potes()
    distances = np.zeros([election.num_voters, election.num_voters])
    for v1 in range(election.num_voters):
        for v2 in range(election.num_voters):
            distances[v1][v2] = swap_distance_between_potes(
                election.potes[v1], election.potes[v2])
    return distances

# DIVERSITY INDICES

def borda_gini(election):
    if election.fake:
        return 'None'
    all_scores = calculate_borda_scores(election)
    return gini_coef(all_scores)

def borda_meandev(election):
    if election.fake:
        return 'None'
    all_scores = calculate_borda_scores(election)
    all_scores = np.abs(all_scores - all_scores.mean())
    return all_scores.mean()

def borda_std(election):
    if election.fake:
        return 'None'
    all_scores = calculate_borda_scores(election)
    return all_scores.std()

# FROM other.py
# def borda_std(election):
#     all_scores = np.zeros(election.num_candidates)
#     vectors = election.votes_to_positionwise_matrix()
#     for i in range(election.num_candidates):
#         for j in range(election.num_candidates):
#             all_scores[i] += vectors[i][j] * (election.num_candidates - j - 1)
#     return np.std(all_scores)

def borda_range(election):
    if election.fake:
        return 'None'
    all_scores = calculate_borda_scores(election)
    return (np.max(all_scores) - np.min(all_scores))

def cand_dom_dist_mean(election):
    if election.fake:
        return 'None'
    distances = calculate_cand_dom_dist(election)
    return distances.sum() / (election.num_candidates - 1) / election.num_candidates * 2

def cand_dom_dist_std(election):
    if election.fake:
        return 'None'
    distances = calculate_cand_dom_dist(election)
    distances = remove_diag(distances)
    return distances.std()

# def cand_pos_dist_mean(election):
#     if election.fake:
#         return 'None'
#     distances = np.zeros([election.num_candidates, election.num_candidates])
#     for c1 in range(election.num_candidates):
#         for c2 in range(election.num_candidates):
#             dist = 0
#             for pote in election.potes:
#                 dist += abs(pote[c1] - pote[c2])
#             distances[c1][c2] = dist
#     print(election.culture_id)
#     print(distances)
#     return distances.sum() / (election.num_candidates - 1) / election.num_candidates
# NO SENSE – ALWAYS CONSTANT

def cand_pos_dist_std(election):
    if election.fake:
        return 'None'
    distances = calculate_cand_pos_dist(election)
    distances = remove_diag(distances)
    return distances.std()

def cand_pos_dist_meandev(election):
    if election.fake:
        return 'None'
    distances = calculate_cand_pos_dist(election)
    distances = remove_diag(distances)
    distances = np.abs(distances - distances.mean())
    return distances.mean()

def cand_pos_dist_gini(election):
    if election.fake:
        return 'None'
    distances = calculate_cand_pos_dist(election)
    distances = remove_diag(distances)
    return gini_coef(distances)

def med_cands_summed(election):
    if election.fake:
        return 'None'
    m = election.num_candidates
    distances = calculate_cand_pos_dist(election)
    res = [0] * m
    for i in range(1,m):
        best_d = np.inf
        for comb in itertools.combinations(range(m),i):
            d_total = 0
            for c1 in range(m):
                min_d = np.inf
                for c2 in comb:
                    d_cand = distances[c1,c2]
                    if d_cand < min_d:
                        min_d = d_cand
                d_total = d_total + min_d
            if d_total < best_d:
                best_d = d_total
        res[i] = best_d
    return sum(res)

def vote_dist_mean(election):
    if election.fake:
        return 'None'
    distances = calculate_vote_swap_dist(election)
    return distances.sum() / election.num_voters / (election.num_voters - 1)

def vote_dist_max(election):
    if election.fake:
        return 'None'
    distances = calculate_vote_swap_dist(election)
    return distances.max()

def vote_dist_med(election):
    if election.fake:
        return 'None'
    distances = calculate_vote_swap_dist(election)
    distances = remove_diag(distances)
    return np.median(distances)

def vote_dist_gini(election):
    if election.fake:
        return 'None'
    distances = calculate_vote_swap_dist(election)
    distances = remove_diag(distances)
    return gini_coef(distances)

def vote_sqr_dist_mean(election):
    if election.fake:
        return 'None'
    distances = calculate_vote_swap_dist(election)
    distances = remove_diag(distances)
    distances = np.sqrt(distances)
    return distances.mean()

def vote_sqr_dist_med(election):
    if election.fake:
        return 'None'
    distances = calculate_vote_swap_dist(election)
    distances = remove_diag(distances)
    distances = np.sqrt(distances)
    return np.median(distances)

def vote_diversity_Karpov(election):
    if election.fake:
        return 'None'
    distances = calculate_vote_swap_dist(election)
    distances = remove_diag(distances)
    distances = distances + 0.5
    distances[distances == 0.5] = 1
    return geom_mean(distances)

def dist_sqr_to_Kemeny_mean(election):
    if election.fake:
        return 'None'
    return 'None'

def dist_to_Kemeny_mean(election):
    if election.fake:
        return 'None'
    _, dist = kemeny_ranking(election)
    return dist / election.num_voters

def dist_to_Kemeny_med(election):
    if election.fake:
        return 'None'
    return 'None'

def dist_to_Borda_mean(election):
    if election.fake:
        return 'None'
    m = election.num_candidates
    borda = calculate_borda_scores(election)
    ranking = np.argsort(-borda)
    wmg = election.votes_to_pairwise_matrix()
    dist = 0
    for i in range(m):
        for j in range(i + 1, m):
            dist = dist + wmg[ranking[j], ranking[i]]
    return dist / election.num_voters

def lexi_diversity(election):
    if election.fake:
        return 'None'
    return 'None'

def greedy_kKemenys_summed(election):
    if election.fake:
        return 'None'
    res = [0] * election.num_voters
    distances = calculate_vote_swap_dist(election)
    best = np.argmin(distances.sum(axis=1))
    best_vec = distances[best]
    res[0] = best_vec.sum()
    distances = np.vstack((distances[:best],distances[best+1:]))

    for i in range(1,election.num_voters):
        relatives = distances - best_vec
        relatives = relatives*(relatives < 0)
        best = np.argmin(relatives.sum(axis=1))
        best_vec = best_vec + relatives[best]
        res[i] = best_vec.sum()
        distances = np.vstack((distances[:best],distances[best+1:]))

    return sum(res)
    max_dist = (election.num_candidates) * (election.num_candidates - 1) / 2
    return sum(res) / election.num_voters / max_dist

def greedy_kKemenys_divk_summed(election):
    if election.fake:
        return 'None'
    res = [0] * election.num_voters
    distances = calculate_vote_swap_dist(election)
    best = np.argmin(distances.sum(axis=1))
    best_vec = distances[best]
    res[0] = best_vec.sum()
    distances = np.vstack((distances[:best], distances[best+1:]))

    for i in range(1, election.num_voters):
        relatives = distances - best_vec
        relatives = relatives*(relatives < 0)
        best = np.argmin(relatives.sum(axis=1))
        best_vec = best_vec + relatives[best]
        res[i] = best_vec.sum() / (i + 1)
        distances = np.vstack((distances[:best], distances[best+1:]))

    # res[0] = 0 # for disregarding one Kemeny (AN = ID)
    max_dist = (election.num_candidates) * (election.num_candidates - 1) / 2
    return sum(res) / election.num_voters / max_dist

def greedy_2kKemenys_summed(election):
    if election.fake:
        return 'None'
    res = []
    distances = calculate_vote_swap_dist(election)
    best = np.argmin(distances.sum(axis=1))
    best_vec = distances[best]
    res.append(best_vec.sum())
    distances = np.vstack((distances[:best], distances[best+1:]))

    k = 2
    for i in range(1, election.num_voters):
        relatives = distances - best_vec
        relatives = relatives*(relatives < 0)
        best = np.argmin(relatives.sum(axis=1))
        best_vec = best_vec + relatives[best]
        if (i + 1) == k:
            res.append(best_vec.sum())
            k = k * 2
        distances = np.vstack((distances[:best], distances[best+1:]))

    # res[0] = 0 # for disregarding one Kemeny (AN = ID)
    max_dist = (election.num_candidates) * (election.num_candidates - 1) / 2
    return sum(res) / election.num_voters / max_dist / 2

def polarization_1by2Kemenys(election):
    if election.fake:
        return 'None'
    distances = calculate_vote_swap_dist(election)
    best = np.argmin(distances.sum(axis=1))
    best_vec = distances[best]
    first_kemeny = best_vec.sum()
    distances = np.vstack((distances[:best],distances[best+1:]))

    relatives = distances - best_vec
    relatives = relatives*(relatives < 0)
    best = np.argmin(relatives.sum(axis=1))
    best_vec = best_vec + relatives[best]
    second_kemeny = best_vec.sum()

    max_dist = (election.num_candidates) * (election.num_candidates - 1) / 2
    return (first_kemeny - second_kemeny) / election.num_voters / max_dist

def greedy_kmeans_summed(election):
    if election.fake:
        return 'None'
    res = [0] * election.num_voters
    distances = calculate_vote_swap_dist(election)
    distances = distances * distances
    best = np.argmin(distances.sum(axis=1))
    best_vec = distances[best]
    res[0] = best_vec.sum()
    distances = np.vstack((distances[:best],distances[best+1:]))

    for i in range(1,election.num_voters):
        relatives = distances - best_vec
        relatives = relatives*(relatives < 0)
        best = np.argmin(relatives.sum(axis=1))
        best_vec = best_vec + relatives[best]
        res[i] = best_vec.sum()
        distances = np.vstack((distances[:best],distances[best+1:]))

    return sum(res)

def support_diversity(election, tuple_len):
    if election.fake:
        return 'None'
    m = election.num_candidates
    res = 0
    for subset in itertools.combinations(range(m), tuple_len):
        support = []
        for v in election.votes:
            trimmed_v = []
            for c in v:
                if c in subset:
                    trimmed_v.append(c)
            if not(trimmed_v in support):
                support.append(trimmed_v)
        res = res + len(support)
    return res

def support_diversity_normed(election, tuple_len):
    if election.fake:
        return 'None'
    m = election.num_candidates
    res = 0
    count = 0
    for subset in itertools.combinations(range(m), tuple_len):
        count = count + 1
        support = []
        for v in election.votes:
            trimmed_v = []
            for c in v:
                if c in subset:
                    trimmed_v.append(c)
            if not(trimmed_v in support):
                support.append(trimmed_v)
        res = res + len(support)
    return res / count

def support_diversity_normed2(election, tuple_len):
    if election.fake:
        return 'None'
    m = election.num_candidates
    res = 0
    count = 0
    for subset in itertools.combinations(range(m), tuple_len):
        count = count + 1
        support = []
        for v in election.votes:
            trimmed_v = []
            for c in v:
                if c in subset:
                    trimmed_v.append(c)
            if not(trimmed_v in support):
                support.append(trimmed_v)
        res = res + len(support)
    return res / count / math.factorial(tuple_len)

def support_diversity_normed3(election, tuple_len):
    if election.fake:
        return 'None'
    m = election.num_candidates
    res = 0
    count = 0
    for subset in itertools.combinations(range(m), tuple_len):
        count = count + 1
        support = []
        for v in election.votes:
            trimmed_v = []
            for c in v:
                if c in subset:
                    trimmed_v.append(c)
            if not(trimmed_v in support):
                support.append(trimmed_v)
        res = res + len(support)
    max_times = min(math.factorial(tuple_len),election.num_voters)
    return res / count / max_times

def support_pairs(election):
    return support_diversity(election,2)

def support_triplets(election):
    return support_diversity(election,3)

def support_votes(election):
    if election.fake:
        return 'None'
    m = election.num_candidates
    return support_diversity(election,m)

def support_diversity_summed(election):
    if election.fake:
        return 'None'
    m = election.num_candidates
    res = 0
    for i in range(2,m+1):
        res = res + support_diversity(election, i)
    return res

def support_diversity_normed_summed(election):
    if election.fake:
        return 'None'
    m = election.num_candidates
    res = 0
    for i in range(2,m+1):
        res = res + support_diversity_normed(election, i)
    return res

def support_diversity_normed2_summed(election):
    if election.fake:
        return 'None'
    m = election.num_candidates
    res = 0
    for i in range(2,m+1):
        res = res + support_diversity_normed2(election, i)
    return res

def support_diversity_normed3_summed(election):
    if election.fake:
        return 'None'
    m = election.num_candidates
    res = 0
    for i in range(2,m+1):
        res = res + support_diversity_normed3(election, i)
    return res

def votes_std(election):
    if election.fake:
        return 'None'
    return 'None'





# FROM vc_diversity.py
# def num_of_diff_votes(election):
#     if election.fake:
#         return 'None'
#     str_votes = [str(vote) for vote in election.votes]
#     return len(set(str_votes))
#
#
# def voterlikeness_sqrt(election):
#     if election.fake:
#         return 'None'
#     vectors = election.votes_to_voterlikeness_vectors()
#     score = 0.
#     for vector in vectors:
#         for value in vector:
#             score += value**0.5
#     return score
#
#
# def voterlikeness_harmonic(election):
#     if election.fake:
#         return 'None'
#     vectors = election.votes_to_voterlikeness_vectors()
#     score = 0.
#     for vector in vectors:
#         vector = sorted(vector)
#         for pos, value in enumerate(vector):
#             score += 1/(pos+2)*value
#     return score
#
#
