

import numpy as np
import scipy.special
import math


def justified_ratio(election, feature_params) -> float:
    # 1-large, 1-cohesive
    election.compute_reverse_approvals()
    threshold = election.num_voters / feature_params['committee_size']
    covered = set()
    for _set in election.reverse_approvals:
        if len(_set) >= threshold:
            covered = covered.union(_set)
    print(len(covered) / float(election.num_voters))
    return len(covered) / float(election.num_voters)

    # 2-large, 2-cohesive
    #
    # election.compute_reverse_approvals()
    # threshold = 2 * election.num_voters / feature_params['committee_size']
    # covered = set()
    # for set_1, set_2 in combinations(election.reverse_approvals, 2):
    #     _intersection = set_1.intersection(set_2)
    #     if len(_intersection) >= threshold:
    #         covered = covered.union(_intersection)
    # print(len(covered) / float(election.num_voters))
    # return len(covered) / float(election.num_voters)

    # 3-large, 3-cohesive
    # election.compute_reverse_approvals()
    # threshold = 3 * election.num_voters / features_params['committee_size']
    # covered = set()
    # for set_1, set_2, set_3 in combinations(election.reverse_approvals, 3):
    #     _intersection = set_1.intersection(set_2).intersection(set_3)
    #     if len(_intersection) >= threshold:
    #         covered = covered.union(_intersection)
    # print(len(covered) / float(election.num_voters))
    # return len(covered) / float(election.num_voters)

def abstract(election) -> float:
    n = election.num_voters
    election.votes_to_approvalwise_vector()
    vector = election.approvalwise_vector
    total_value = 0
    for i in range(election.num_candidates):
        k = vector[i] * n
        x = scipy.special.binom(n, k)
        x = math.log(x)
        total_value += x
    return total_value


def borda_std(election):
    all_scores = np.zeros(election.num_candidates)
    vectors = election.get_vectors()
    for i in range(election.num_candidates):
        for j in range(election.num_candidates):
            all_scores[i] += vectors[i][j] * (election.num_candidates - j - 1) * election.num_voters
    return np.std(all_scores)


def get_effective_num_candidates(election, mode='Borda') -> float:
    """ Compute effective number of candidates """

    c = election.num_candidates
    vectors = election.votes_to_positionwise_matrix()

    if mode == 'Borda':
        all_scores = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) / (c * (c - 1) / 2)
                      for j in range(c)]
    elif mode == 'Plurality':
        all_scores = [sum([vectors[j][i] for i in range(1)]) for j in range(c)]
    else:
        all_scores = []

    return 1. / sum([x * x for x in all_scores])


def max_approval_score(election):
    score = np.zeros([election.num_candidates])
    for vote in election.votes:
        for c in vote:
            score[c] += 1
    return max(score)


def is_condorcet(election):
    """ Check if election witness Condorcet winner"""
    for i in range(election.num_candidates):

        condocret_winner = True
        for j in range(election.num_candidates):

            diff = 0
            for k in range(election.num_voters):

                if election.potes[k][i] <= election.potes[k][j]:
                    diff += 1

            if diff < math.ceil((election.num_voters + 1) / 2.):
                condocret_winner = False
                break

        if condocret_winner:
            return True

    return False