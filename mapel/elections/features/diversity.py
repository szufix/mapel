def borda_gini(election):
    if election.fake:
        return 'None'
    return 'None'

def borda_std(election):
    if election.fake:
        return 'None'
    return 'None'

def borda_range(election):
    if election.fake:
        return 'None'
    return 'None'

def cand_dom_dist_mean(election):
    if election.fake:
        return 'None'
    return 'None'

def cand_dom_dist_std(election):
    if election.fake:
        return 'None'
    return 'None'

def cand_pos_dist_mean(election):
    if election.fake:
        return 'None'
    return 'None'

def cand_pos_dist_std(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_mean(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_max(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_med(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_gini(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_sqr_dist_mean(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_sqr_dist_med(election):
    if election.fake:
        return 'None'
    return 'None'

def vote_dist_Karpov (election):
    if election.fake:
        return 'None'
    return 'None'

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


# FROM other.py
# def borda_std(election):
#     all_scores = np.zeros(election.num_candidates)
#     vectors = election.votes_to_positionwise_matrix()
#     for i in range(election.num_candidates):
#         for j in range(election.num_candidates):
#             all_scores[i] += vectors[i][j] * (election.num_candidates - j - 1)
#     return np.std(all_scores)


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