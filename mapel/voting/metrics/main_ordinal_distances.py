
import os
import random as rand

import numpy as np
from mapel.voting.metrics import lp

from scipy.optimize import linear_sum_assignment

from mapel.voting.metrics.inner_distances import map_str_to_func


def _minus_one(vector):
    if vector is None:
        return None
    new_vector = [0 for _ in range(len(vector))]
    for i in range(len(vector)):
        new_vector[vector[i]] = i
    return new_vector


# MAIN DISTANCES
def compute_positionwise_distance(election_1, election_2, inner_distance):
    """ Compute Positionwise distance between elections """

    cost_table = get_matching_cost_positionwise(
        election_1, election_2, map_str_to_func(inner_distance))
    objective_value, matching = solve_matching_vectors(cost_table)

    return objective_value, matching


def compute_agg_voterlikeness_distance(election_1, election_2, inner_distance):
    """ Compute Aggregated-Voterlikeness distance between elections """
    vector_1, num_possible_scores = election_1.votes_to_agg_voterlikeness_vector()
    vector_2, _ = election_2.votes_to_agg_voterlikeness_vector()
    inner_distance = map_str_to_func(inner_distance)
    return inner_distance(vector_1, vector_2, num_possible_scores)


def compute_bordawise_distance(election_1, election_2, inner_distance):
    """ Compute Bordawise distance between elections """
    vector_1, num_possible_scores = election_1.votes_to_bordawise_vector()
    vector_2, _ = election_2.votes_to_bordawise_vector()
    inner_distance = map_str_to_func(inner_distance)
    return inner_distance(vector_1, vector_2, num_possible_scores)


def compute_pairwise_distance(election_1, election_2, inner_distance):
    """ Compute Pairwise distance between elections """
    length = election_1.num_candidates
    matrix_1 = election_1.votes_to_pairwise_matrix()
    matrix_2 = election_2.votes_to_pairwise_matrix()
    matching_cost = solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)
    return matching_cost


def compute_voterlikeness_distance(election_1, election_2, inner_distance):
    """ Compute Voterlikeness distance between elections """
    length = election_1.num_voters
    matrix_1 = election_1.votes_to_voterlikeness_matrix()
    matrix_2 = election_2.votes_to_voterlikeness_matrix()
    matching_cost = \
        solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)
    return matching_cost


def compute_spearman_distance(election_1, election_2):
    """ Compute Spearman distance between elections """

    votes_1 = election_1.votes
    votes_2 = election_2.votes
    params = {'voters': election_1.num_voters,
              'candidates': election_1.num_candidates}

    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_ilp_distance(path, votes_1, votes_2, params, 'spearman')
    objective_value = lp.solve_ilp_distance(path, votes_1,
                                            votes_2, params, 'spearman')
    lp.remove_lp_file(path)
    return objective_value


def compute_discrete_distance(election_1, election_2):
    """ Compute Discrete distance between elections """
    return election_1.num_voters - compute_voter_subelection(election_1,
                                                             election_2)


# SUBELECTIONS #
def compute_voter_subelection(election_1, election_2):
    """ Compute Voter-Subelection """
    objective_value = lp.solve_lp_voter_subelection(election_1, election_2)
    return objective_value, None


def compute_candidate_subelection(election_1, election_2):
    """ Compute Candidate-Subelection """
    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    objective_value = lp.solve_lp_candidate_subelections(path, election_1,
                                                         election_2)
    lp.remove_lp_file(path)
    return objective_value, None


# HELPER FUNCTIONS #
def convert_to_cumulative(vector):
    # print(vector)
    tmp = [0 for _ in range(len(vector))]
    tmp[0] = vector[0]
    for i in range(1, len(vector)):
        tmp[i] = tmp[i-1] + vector[i]
    return sum(tmp)


def get_matching_cost_positionwise(ele_1, ele_2, inner_distance):
    """ Get matching cost for positionwise distances """
    vectors_1 = ele_1.get_vectors()
    vectors_2 = ele_2.get_vectors()

    # import copy
    # def emd(vector_1, vector_2):
    #     """ compute EMD metric """
    #     vector_1 = copy.deepcopy(vector_1)
    #     dirt = 0.
    #     for i in range(len(vector_1) - 1):
    #         surplus = vector_1[i] - vector_2[i]
    #         dirt += abs(surplus)
    #         vector_1[i + 1] += surplus
    #     return dirt
    #
    # c_2 = []
    # for vector in vectors_2:
    #     c_2.append(convert_to_cumulative(vector))
    #
    # order = [i for i in range(len(vectors_1))]
    # tmp = [x for _, x in sorted(zip(c_2, order), reverse=True)]
    #
    # total = 0
    # for i in range(len(vectors_1)):
    #     v1 = np.array(vectors_1[i])
    #     v2 = np.array(vectors_2[tmp[i]])
    #     val = emd(v1, v2)
    #     total += val
    # print(total)

    size = ele_1.num_candidates
    cost_table = [[inner_distance(list(vectors_1[i]), list(vectors_2[j]))
                   for i in range(size)] for j in range(size)]
    return cost_table


def solve_matching_vectors(cost_table):
    cost_table = np.array(cost_table)
    row_ind, col_ind = linear_sum_assignment(cost_table)
    return cost_table[row_ind, col_ind].sum(), list(col_ind)


def solve_matching_matrices(matrix_1, matrix_2, length, inner_distance):
    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_lp_file_matching_matrix(path, matrix_1, matrix_2, length,
                                        inner_distance)
    matching_cost = lp.solve_lp_matrix(path, matrix_1, matrix_2, length)
    lp.remove_lp_file(path)
    return matching_cost
