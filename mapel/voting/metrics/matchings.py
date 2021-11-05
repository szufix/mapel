#!/usr/bin/env python

import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from mapel.voting.metrics import lp


def solve_matching_vectors(cost_table) -> (float, list):
    """ Return: objective value, optimal matching """
    cost_table = np.array(cost_table)
    row_ind, col_ind = linear_sum_assignment(cost_table)
    return cost_table[row_ind, col_ind].sum(), list(col_ind)


def solve_matching_matrices(matrix_1, matrix_2, length, inner_distance) -> float:
    """ Return: objective value"""
    file_name = str(np.random.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_lp_file_matching_matrix(path, matrix_1, matrix_2, length, inner_distance)
    matching_cost = lp.solve_lp_matrix(path)
    lp.remove_lp_file(path)
    return matching_cost


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
