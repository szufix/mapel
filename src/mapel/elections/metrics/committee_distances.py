#!/usr/bin/env python

from mapel.voting.metrics.main_approval_distances import solve_matching_vectors


def get_matching_cost_committee(election, committee_1, committee_2, distance_id):
    if distance_id == 'discrete':
        return len(committee_1.symmetric_difference(committee_2))
    elif distance_id == 'hamming':
        return hamming_distance_between_committees(election, committee_1, committee_2)
    elif distance_id == 'asymmetric':
        return asymmetric_distance_between_committees(election, committee_1, committee_2)


# MAIN FUNCTIONS
def hamming_distance_between_committees(election, committee_1, committee_2):
    cost_table = get_matching_cost_committee_hamming(election,
                                                     list(committee_1), list(committee_2))
    return solve_matching_vectors(cost_table)[0]


def asymmetric_distance_between_committees(election, committee_1, committee_2):
    cost_table = get_matching_cost_committee_asymmetric(election,
                                                        list(committee_1), list(committee_2))
    return solve_matching_vectors(cost_table)[0]


# HELPER FUNCTIONS
def get_matching_cost_committee_hamming(election, committee_1, committee_2):
    size = len(committee_1)
    return [[election.candidatelikeness_original_vectors[committee_1[i]][committee_2[j]]
             for i in range(size)] for j in range(size)]


def get_matching_cost_committee_asymmetric(election, committee_1, committee_2):
    size = len(committee_1)
    election.compute_reverse_approvals()
    return [[compare_candidates(election, committee_1[i], committee_2[j])
             for i in range(size)] for j in range(size)]


def compare_candidates(election, c1, c2):
    app_c1 = election.reverse_approvals[c1]
    app_c2 = election.reverse_approvals[c2]
    if len(app_c1.union(app_c2)) == 0:
        return 1
    return 1 - len(app_c1.intersection(app_c2)) / len(app_c1.union(app_c2))
