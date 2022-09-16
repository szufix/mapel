#!/usr/bin/env python

import numpy as np
from scipy.optimize import linear_sum_assignment

from mapel.elections.metrics import lp


def solve_matching_vectors(cost_table) -> (float, list):
    """ Return: objective value, optimal matching """
    cost_table = np.array(cost_table)
    row_ind, col_ind = linear_sum_assignment(cost_table)
    return cost_table[row_ind, col_ind].sum(), list(col_ind)


def solve_matching_matrices(matrix_1, matrix_2, length, inner_distance) -> float:
    """ Return: objective value"""
    return lp.generate_lp_file_matching_matrix(matrix_1, matrix_2, length, inner_distance)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 17.08.2022 #
# # # # # # # # # # # # # # # #
