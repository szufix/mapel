import os
import random as rand

import numpy as np
from scipy.optimize import linear_sum_assignment

from mapel.voting.metrics import lp
from mapel.voting.objects.OrdinalElection import OrdinalElection


# MAIN DISTANCES
def compute_positionwise_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                                  inner_distance):
    """ Compute Positionwise distance between ordinal elections """

    cost_table = get_matching_cost_positionwise(
        election_1, election_2, inner_distance)
    objective_value, matching = solve_matching_vectors(cost_table)
    return objective_value, matching


def compute_agg_voterlikeness_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                                       inner_distance):
    """ Compute Aggregated-Voterlikeness distance between ordinal elections """
    vector_1, num_possible_scores = election_1.votes_to_agg_voterlikeness_vector()
    vector_2, _ = election_2.votes_to_agg_voterlikeness_vector()
    return inner_distance(vector_1, vector_2, num_possible_scores)


def compute_bordawise_distance(election_1, election_2, inner_distance):
    """ Compute Bordawise distance between ordinal elections """
    vector_1, num_possible_scores = election_1.votes_to_bordawise_vector()
    vector_2, _ = election_2.votes_to_bordawise_vector()
    return inner_distance(vector_1, vector_2, num_possible_scores)


def compute_pairwise_distance(election_1, election_2, inner_distance):
    """ Compute Pairwise distance between ordinal elections """
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
    matching_cost = solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)
    return matching_cost


def compute_spearman_distance(election_1, election_2):
    """ Compute Spearman distance between elections """

    votes_1 = election_1.votes
    votes_2 = election_2.votes
    params = {'voters': election_1.num_voters, 'candidates': election_1.num_candidates}

    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_ilp_distance(path, votes_1, votes_2, params, 'spearman')
    objective_value = lp.solve_ilp_distance(path, votes_1, votes_2, params, 'spearman')
    lp.remove_lp_file(path)
    return objective_value


def compute_discrete_distance(election_1, election_2):
    """ Compute Discrete distance between elections """
    return election_1.num_voters - compute_voter_subelection(election_1, election_2)


# SUBELECTIONS #
def compute_voter_subelection(election_1, election_2):
    """ Compute Voter-Subelection """
    return lp.solve_lp_voter_subelection(election_1, election_2)


def compute_candidate_subelection(election_1, election_2):
    """ Compute Candidate-Subelection """
    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    objective_value = lp.solve_lp_candidate_subelections(path, election_1,
                                                         election_2)
    lp.remove_lp_file(path)
    return objective_value


# HELPER FUNCTIONS #
def get_matching_cost_positionwise(ele_1, ele_2, inner_distance):
    """ Get matching cost for positionwise distances """
    vectors_1 = ele_1.get_vectors()
    vectors_2 = ele_2.get_vectors()
    size = ele_1.num_candidates
    return [[inner_distance(list(vectors_1[i]), list(vectors_2[j]))
             for i in range(size)] for j in range(size)]


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
