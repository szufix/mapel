#!/usr/bin/env python

import math
from typing import Callable, List

import os
import networkx as nx

from mapel.core.matchings import *
from mapel.elections.objects.ApprovalElection import ApprovalElection


# MAIN APPROVAL DISTANCES

def compute_approvalwise(election_1: ApprovalElection, election_2: ApprovalElection,
                         inner_distance: Callable) -> float:
    """ Return: approvalwise distance """
    election_1.votes_to_approvalwise_vector()
    election_2.votes_to_approvalwise_vector()
    return inner_distance(election_1.approvalwise_vector, election_2.approvalwise_vector)


def compute_coapproval_frequency_vectors(election_1: ApprovalElection, election_2: ApprovalElection,
                                         inner_distance: Callable) -> (float, list):
    """ Return: coapproval frequency distance, optimal matching """
    cost_table = get_matching_cost_coapproval_frequency_vectors(
        election_1, election_2, inner_distance)
    return solve_matching_vectors(cost_table)


def compute_candidatelikeness(election_1: ApprovalElection, election_2: ApprovalElection,
                              inner_distance: Callable) -> (float, list):
    """ Return: candidatelikeness distance, optimal matching """
    cost_table = get_matching_cost_candidatelikeness(
        election_1, election_2, inner_distance)
    return solve_matching_vectors(cost_table)


def compute_hamming(election_1: ApprovalElection, election_2: ApprovalElection) -> float:
    """ Return: Hamming distance """
    votes_1 = election_1.votes
    votes_2 = election_2.votes
    params = {'voters': election_1.num_voters, 'candidates': election_2.num_candidates}
    file_name = f'{np.random.random()}.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_ilp_distance(path, votes_1, votes_2, params, 'hamming')
    objective_value = lp.solve_ilp_distance(path, votes_1, votes_2, params, 'hamming')
    objective_value /= election_1.num_candidates  # ANALYZE THIS LINE
    lp.remove_lp_file(path)
    return objective_value


def compute_voterlikeness(election_1: ApprovalElection, election_2: ApprovalElection,
                          inner_distance: Callable) -> (float, list):
    """ Return: voterlikeness distance, optimal matching """
    cost_table = get_matching_cost_voterlikeness_vectors(
        election_1, election_2, inner_distance)
    return solve_matching_vectors(cost_table)


def compute_pairwise(election_1: ApprovalElection, election_2: ApprovalElection,
                     inner_distance: Callable) -> (float, list):
    """ Return: approval pairwise distance, optimal matching """
    length = election_1.num_candidates
    matrix_1 = election_1.pairwise_matrix
    matrix_2 = election_2.pairwise_matrix
    return solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)


def compute_flow(ele_1, ele_2):
    cost_table = get_flow_helper_1(ele_1, ele_2)
    objective_value, matching = solve_matching_vectors(cost_table)
    objective_value /= 1000.
    return objective_value, matching


# HELPER FUNCTIONS #
def flow_helper_advanced(v_1, v_2, num_candidates=1, num_voters=1):
    """ Return: Objective value """

    def normalize(x):
        if x == 0:
            return 0
        x = math.log(1 / x)
        x = int(x * 1000)
        return x

    max_capacity = num_voters * num_candidates
    total_demand = num_voters * num_candidates

    m = float(num_candidates)
    int_m = int(m)
    source = 'source'
    sink = 'sink'
    graph = nx.DiGraph()
    epsilon = 0.1

    graph.add_node(source, demand=-total_demand)
    for i in range(1, 2 * int(m) + 1):
        graph.add_node(i, demand=0)
    graph.add_node(sink, demand=total_demand)

    for i in range(1, 2 * int(m) + 1):
        graph.add_edge(source, i, capacity=int(v_1[i - 1] * total_demand + epsilon), weight=0)
        graph.add_edge(i, sink, capacity=int(v_2[i - 1] * total_demand + epsilon), weight=0)

    # upper row
    for k in range(1, int_m + 1):
        prob_right = (m - k) / m
        prob_left = k / m
        prob_left_down = prob_left / k
        prob_left_up = prob_left - prob_left_down

        # go left down
        graph.add_edge(k, k + int_m, capacity=int(max_capacity), weight=normalize(prob_left_down))
        # go left up
        if 1 < k:
            graph.add_edge(k, k - 1, capacity=int(max_capacity), weight=normalize(prob_left_up))
        # go right
        if k < m:
            graph.add_edge(k, k + 1, capacity=int(max_capacity), weight=normalize(prob_right))
        # print(k, prob_left_up, prob_left_down, prob_right)

    # lower row
    for k in range(0, int_m):
        prob_right = (m - k) / m
        prob_left = k / m
        prob_right_up = prob_right / (m - k)
        prob_right_down = prob_right - prob_right_up

        # go right up
        my_pos = k + int_m + 1
        graph.add_edge(my_pos, k + 1, capacity=int(max_capacity), weight=normalize(prob_right_up))
        # go right down
        if k < m - 1:
            graph.add_edge(my_pos, my_pos + 1, capacity=int(max_capacity),
                           weight=normalize(prob_right_down))
        # go left
        if 0 < k:
            graph.add_edge(my_pos, my_pos - 1, capacity=int(max_capacity),
                           weight=normalize(prob_left))

    return nx.min_cost_flow_cost(graph)


def flow_helper_simple(v_1, v_2, num_candidates=1, num_voters=1):
    """ Return: Objective value """

    x = num_candidates

    max_capacity = num_voters * num_candidates
    total_demand = num_voters * num_candidates

    m = float(num_candidates)
    int_m = int(m)
    source = 'source'
    sink = 'sink'
    graph = nx.DiGraph()
    epsilon = 0.1

    graph.add_node(source, demand=-total_demand)
    for i in range(1, 2 * int(m) + 1):
        graph.add_node(i, demand=0)
    graph.add_node(sink, demand=total_demand)

    for i in range(1, 2 * int(m) + 1):
        graph.add_edge(source, i, capacity=int(v_1[i - 1] * total_demand + epsilon), weight=0)
        graph.add_edge(i, sink, capacity=int(v_2[i - 1] * total_demand + epsilon), weight=0)

    # upper row
    for k in range(1, int_m + 1):

        # go down
        graph.add_edge(k, k + int_m, capacity=int(max_capacity), weight=x)
        # go left up
        if 1 < k:
            graph.add_edge(k, k - 1, capacity=int(max_capacity), weight=1)
        # go right up
        if k < m:
            graph.add_edge(k, k + 1, capacity=int(max_capacity), weight=1)

    # lower row
    for k in range(0, int_m):

        # go up
        my_pos = k + int_m + 1
        graph.add_edge(my_pos, k + 1, capacity=int(max_capacity), weight=x)
        # go right down
        if k < m - 1:
            graph.add_edge(my_pos, my_pos + 1, capacity=int(max_capacity),
                           weight=1)
        # go left down
        if 0 < k:
            graph.add_edge(my_pos, my_pos - 1, capacity=int(max_capacity),
                           weight=1)

    return nx.min_cost_flow_cost(graph)


def get_flow_helper_1(election_1: ApprovalElection, election_2: ApprovalElection) -> List[list]:
    """ Return: Cost table """
    vectors_1 = election_1.coapproval_frequency_vectors
    vectors_2 = election_2.coapproval_frequency_vectors
    size = election_1.num_candidates
    return [[flow_helper_simple(vectors_1[i], vectors_2[j],
                         num_candidates=election_1.num_candidates,
                         num_voters=election_1.num_voters)
             for i in range(size)] for j in range(size)]


def get_matching_cost_coapproval_frequency_vectors(election_1: ApprovalElection,
                                                   election_2: ApprovalElection,
                                                   inner_distance: Callable) -> List[list]:
    """ Return: Cost table """
    vectors_1 = election_1.coapproval_frequency_vectors
    vectors_2 = election_2.coapproval_frequency_vectors
    size = election_1.num_candidates
    return [[inner_distance(vectors_1[i], vectors_2[j]) for i in range(size)] for j in range(size)]


def get_matching_cost_candidatelikeness(election_1: ApprovalElection,
                                        election_2: ApprovalElection,
                                        inner_distance: Callable) -> List[list]:
    """ Return: Cost table """
    vectors_1 = election_1.candidatelikeness_sorted_vectors
    vectors_2 = election_2.candidatelikeness_sorted_vectors
    size = election_1.num_candidates
    return [[inner_distance(vectors_1[i], vectors_2[j]) for i in range(size)] for j in range(size)]


def get_matching_cost_voterlikeness_vectors(election_1: ApprovalElection,
                                            election_2: ApprovalElection,
                                            inner_distance: Callable) -> List[list]:
    """ Return: Cost table """
    vectors_1 = election_1.voterlikeness_vectors
    vectors_2 = election_2.voterlikeness_vectors
    size = election_1.num_voters
    return [[inner_distance(vectors_1[i], vectors_2[j]) for i in range(size)] for j in range(size)]

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
