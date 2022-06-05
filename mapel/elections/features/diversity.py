import numpy as np
import itertools
# AUXILIARY FUNCTIONS

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
    all_scores = np.zeros(election.num_candidates)
    vectors = election.votes_to_positionwise_matrix()
    for i in range(election.num_candidates):
        for j in range(election.num_candidates):
            all_scores[i] += vectors[i][j] * (election.num_candidates - j - 1)
    return all_scores

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
    all_scores = calculate_borda_scores(election)
    return gini_coef(all_scores)

# def borda_std(election):
#     if election.fake:
#         return 'None'
#     return 'None'

# FROM other.py
# def borda_std(election):
#     all_scores = np.zeros(election.num_candidates)
#     vectors = election.votes_to_positionwise_matrix()
#     for i in range(election.num_candidates):
#         for j in range(election.num_candidates):
#             all_scores[i] += vectors[i][j] * (election.num_candidates - j - 1)
#     return np.std(all_scores)

def borda_range(election):
    all_scores = calculate_borda_scores(election)
    return (np.max(all_scores) - np.min(all_scores))

def cand_dom_dist_mean(election):
    if election.fake:
        return 'None'
    distances = calculate_cand_dom_dist(election)
    return distances.sum() / (election.num_candidates - 1) / election.num_candidates

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
#     print(election.model_id)
#     print(distances)
#     return distances.sum() / (election.num_candidates - 1) / election.num_candidates
# NO SENSE – ALWAYS CONSTANT

def cand_pos_dist_std(election):
    if election.fake:
        return 'None'
    distances = calculate_cand_pos_dist(election)
    distances = remove_diag(distances)
    return distances.std()

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
    return geom_mean(distances)

def vote_dist_Kemeny_sqr_mean(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_Kemeny_mean(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_Kemeny_med(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_Borda_mean(election):
    if election.fake:
        return 'None'
    return 'None'

def lexi_diversity(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_kKemenys_summed(election):
    if election.fake:
        return 'None'
    return 'None'

def support_pairs(election):
    if election.fake:
        return 'None'
    return 'None'

def support_triplets(election):
    if election.fake:
        return 'None'
    return 'None'

def support_votes(election):
    if election.fake:
        return 'None'
    return 'None'

def support_diversity_mc(election, tuple_len, max_no_tuples):
    if election.fake:
        return 'None'
    return 'None'

def support_diversity_mc_summed(election, max_no_tuples):
    if election.fake:
        return 'None'
    return 'None'

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
# def borda_diversity(election):
#     vector = election.votes_to_bordawise_vector()
#     avg = sum(vector)/len(vector)
#     return sum([abs(value-avg) for value in vector])