
import os
import random as rand

import numpy as np
from mapel.voting.metrics import lp

from scipy.optimize import linear_sum_assignment

from mapel.voting.metrics.inner_distances import map_str_to_func


# MAIN APPROVAL DISTANCES

def compute_approval_frequency(ele_1, ele_2, inner_distance):
    vector_1 = ele_1.get_approval_frequency_vector()
    vector_2 = ele_2.get_approval_frequency_vector()
    inner_distance = map_str_to_func(inner_distance)
    return inner_distance(vector_1, vector_2), None


def compute_coapproval_frequency_vectors(ele_1, ele_2, inner_distance):
    cost_table = get_matching_cost_coapproval_frequency_vectors(
        ele_1, ele_2, map_str_to_func(inner_distance))
    objective_value, matching = solve_matching_vectors(cost_table)
    return objective_value, matching


def compute_coapproval_pairwise(ele_1, ele_2, inner_distance):
    length = ele_1.num_candidates
    matrix_1 = ele_1.votes_to_coapproval_pairwise_matrix()
    matrix_2 = ele_2.votes_to_coapproval_pairwise_matrix()
    inner_distance = map_str_to_func('single_' + inner_distance)
    matching_cost = solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)
    return matching_cost, None


# HELPER FUNCTIONS #
def get_matching_cost_coapproval_frequency_vectors(ele_1, ele_2, inner_distance):
    vectors_1 = ele_1.get_coapproval_frequency_vectors()
    vectors_2 = ele_2.get_coapproval_frequency_vectors()
    size = ele_1.num_candidates
    cost_table = [[inner_distance(vectors_1[i], vectors_2[j])
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