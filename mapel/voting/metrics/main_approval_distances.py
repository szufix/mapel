
import os
import random as rand
import networkx as nx

import numpy as np
from mapel.voting.metrics import lp

from scipy.optimize import linear_sum_assignment

from mapel.voting.metrics.inner_distances import map_str_to_func, hamming


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


def compute_voterlikeness_vectors(ele_1, ele_2, inner_distance):
    cost_table = get_matching_cost_voterlikeness_vectors(
        ele_1, ele_2, map_str_to_func(inner_distance))
    objective_value, matching = solve_matching_vectors(cost_table)
    return objective_value, matching


def compute_flow(ele_1, ele_2, inner_distance):
    cost_table = get_flow_helper_1(ele_1, ele_2)
    objective_value, matching = solve_matching_vectors(cost_table)
    objective_value /= 1000.
    return objective_value, matching


def compute_coapproval_pairwise(ele_1, ele_2, inner_distance):
    length = ele_1.num_candidates
    matrix_1 = ele_1.votes_to_coapproval_pairwise_matrix()
    matrix_2 = ele_2.votes_to_coapproval_pairwise_matrix()
    inner_distance = map_str_to_func('single_' + inner_distance)
    matching_cost = solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)
    return matching_cost, None

import math
# HELPER FUNCTIONS #
def flow_helper(v_1, v_2, num_voters=50):
    # print(v_1, v_2)

    def func(x):
        if x == 0:
            return x
        x /= Q
        x = int(math.log(1/x)*100)
        # print(x)
        return x

    max_capacity = num_voters
    num_candidates = int(len(v_1)/2)
    Q = num_candidates
    total_demand = num_voters

    n = float(num_candidates)
    int_n = int(n)
    source = 0
    sink = 2*num_candidates+1
    graph = nx.DiGraph()

    graph.add_node(source, demand=-total_demand)
    for i in range(1, 2*int(n)+1):
        graph.add_node(i, demand=0)
    graph.add_node(sink, demand=total_demand)

    for i in range(1, 2*int(n)+1):
        graph.add_edge(source, i, capacity=int(v_1[i-1]*max_capacity), weight=0)
        graph.add_edge(i, sink, capacity=int(v_2[i-1]*max_capacity), weight=0)

    # upper row
    for k in range(1, int_n+1):
        prob_right = (n - k) / n
        prob_left = k / n
        # if k == 1:
        #     prob_left_down = 1/n
        #     prob_left_up = 0
        # else:
        prob_left_down = prob_left / k
        prob_left_up = prob_left - prob_left_down

        # go left down
        graph.add_edge(k, k + int_n, capacity=int(max_capacity), weight=1)#func(int(prob_left_down*Q)))
        # go left up
        if 1 < k:
            graph.add_edge(k, k - 1, capacity=int(max_capacity), weight=1)#func(int(prob_left_up*Q)))
        # go right
        if k < n:
            graph.add_edge(k, k + 1, capacity=int(max_capacity), weight=1)#func(int(prob_right*Q)))
        # print(k, prob_left_up, prob_left_down, prob_right)

    # lower row
    for k in range(0, int_n):
        prob_right = (n - k) / n
        prob_left = k / n
        # if k == 0:
        #     prob_right_up = 0.5
        #     prob_right_down = 0.5
        # else:
        prob_right_up = prob_right / (n-k)
        prob_right_down = prob_right - prob_right_up

        # go right up
        my_pos = k + int_n + 1
        graph.add_edge(my_pos, k + 1, capacity=int(max_capacity), weight=1)#func(int(prob_right_up*Q)))
        # go right down
        if k < n - 1:
            graph.add_edge(my_pos, my_pos + 1, capacity=int(max_capacity), weight=1)#func(int(prob_right_down*Q)))
        # go left
        if 0 < k:
            graph.add_edge(my_pos, my_pos - 1, capacity=int(max_capacity), weight=1)#func(int(prob_left*Q)))
        # print("p", prob_right_up, prob_right_down, prob_left)

    # print("start")
    objective_value = nx.min_cost_flow_cost(graph)
    # print("stop")
    # print('obj', objective_value)

    return objective_value


def get_flow_helper_1(ele_1, ele_2):
    vectors_1 = ele_1.get_coapproval_frequency_vectors()
    vectors_2 = ele_2.get_coapproval_frequency_vectors()
    size = ele_1.num_candidates
    cost_table = [[flow_helper(vectors_1[i], vectors_2[j])
                   for i in range(size)] for j in range(size)]
    # print(cost_table)
    return cost_table

def get_matching_cost_coapproval_frequency_vectors(ele_1, ele_2, inner_distance):
    vectors_1 = ele_1.get_coapproval_frequency_vectors()
    vectors_2 = ele_2.get_coapproval_frequency_vectors()
    size = ele_1.num_candidates
    cost_table = [[inner_distance(vectors_1[i], vectors_2[j])
                   for i in range(size)] for j in range(size)]
    return cost_table


def get_matching_cost_voterlikeness_vectors(ele_1, ele_2, inner_distance):
    vectors_1 = ele_1.get_voterlikeness_vectors()
    vectors_2 = ele_2.get_voterlikeness_vectors()
    size = ele_1.num_voters
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