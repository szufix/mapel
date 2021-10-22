from typing import Callable, List

from mapel.voting.metrics.matchings import *
from mapel.voting.objects.OrdinalElection import OrdinalElection


# MAIN DISTANCES
def compute_positionwise_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                                  inner_distance: Callable) -> (float, list):
    """ Compute Positionwise distance between ordinal elections """
    cost_table = get_matching_cost_positionwise(election_1, election_2, inner_distance)
    return solve_matching_vectors(cost_table)


def compute_agg_voterlikeness_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                                       inner_distance: Callable) -> float:
    """ Compute Aggregated-Voterlikeness distance between ordinal elections """
    vector_1, num_possible_scores = election_1.votes_to_agg_voterlikeness_vector()
    vector_2, _ = election_2.votes_to_agg_voterlikeness_vector()
    return inner_distance(vector_1, vector_2, num_possible_scores)


def compute_bordawise_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                               inner_distance: Callable) -> float:
    """ Compute Bordawise distance between ordinal elections """
    vector_1 = election_1.votes_to_bordawise_vector()
    vector_2 = election_2.votes_to_bordawise_vector()
    return inner_distance(vector_1, vector_2)


def compute_pairwise_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                              inner_distance: Callable) -> float:
    """ Compute Pairwise distance between ordinal elections """
    length = election_1.num_candidates
    matrix_1 = election_1.votes_to_pairwise_matrix()
    matrix_2 = election_2.votes_to_pairwise_matrix()
    return solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)


def compute_voterlikeness_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                                   inner_distance: Callable) -> float:
    """ Compute Voterlikeness distance between elections """
    length = election_1.num_voters
    matrix_1 = election_1.votes_to_voterlikeness_matrix()
    matrix_2 = election_2.votes_to_voterlikeness_matrix()
    return solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)


def compute_spearman_distance(election_1: OrdinalElection, election_2: OrdinalElection) -> float:
    """ Compute Spearman distance between elections """

    votes_1 = election_1.votes
    votes_2 = election_2.votes
    params = {'voters': election_1.num_voters, 'candidates': election_1.num_candidates}

    file_name = f'{rand.random()}.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_ilp_distance(path, votes_1, votes_2, params, 'spearman')
    objective_value = lp.solve_ilp_distance(path, votes_1, votes_2, params, 'spearman')
    lp.remove_lp_file(path)
    return objective_value


def compute_discrete_distance(election_1: OrdinalElection, election_2: OrdinalElection) -> int:
    """ Compute Discrete distance between elections """
    return election_1.num_voters - compute_voter_subelection(election_1, election_2)


# SUBELECTIONS #
def compute_voter_subelection(election_1: OrdinalElection, election_2: OrdinalElection) -> int:
    """ Compute Voter-Subelection """
    return lp.solve_lp_voter_subelection(election_1, election_2)


def compute_candidate_subelection(election_1: OrdinalElection, election_2: OrdinalElection) -> int:
    """ Compute Candidate-Subelection """
    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), 'trash', file_name)
    objective_value = lp.solve_lp_candidate_subelections(path, election_1, election_2)
    lp.remove_lp_file(path)
    return objective_value


# HELPER FUNCTIONS #
def get_matching_cost_positionwise(election_1: OrdinalElection, election_2: OrdinalElection,
                                   inner_distance: Callable) -> List[list]:
    """ Return: Cost table """
    vectors_1 = election_1.get_vectors()
    vectors_2 = election_2.get_vectors()
    size = election_1.num_candidates
    return [[inner_distance(vectors_1[i], vectors_2[j]) for i in range(size)] for j in range(size)]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
