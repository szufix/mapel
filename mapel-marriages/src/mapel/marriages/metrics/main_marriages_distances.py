from typing import Callable, List
from itertools import combinations, permutations

from mapel.core.matchings import *
from mapel.marriages.objects.Marriages import Marriages
from mapel.core.inner_distances import swap_distance


# MAIN DISTANCES
def compute_retrospective_distance(instance_1, instance_2, inner_distance):

    results = []
    for crossing in [False, True]:
        cost_table_1, cost_table_2 = get_matching_cost_retrospective(
            instance_1, instance_2, inner_distance, crossing=crossing)

        a, _ = solve_matching_vectors(cost_table_1)
        b, _ = solve_matching_vectors(cost_table_2)
        results.append(a+b)

    return min(results)


def compute_positionwise_distance(instance_1, instance_2, inner_distance):
    cost_table = get_matching_cost_positionwise(instance_1, instance_2, inner_distance)
    return solve_matching_vectors(cost_table)


def compute_pos_swap_distance(instance_1: Marriages, instance_2: Marriages,
                              inner_distance: Callable) -> (float, list):
    """ Compute Positionwise distance between ordinal elections """
    cost_table = get_matching_cost_positionwise(instance_1, instance_2, inner_distance)
    obj_val, matching = solve_matching_vectors(cost_table)
    votes_1 = instance_1.votes
    votes_2 = instance_2.votes
    size = instance_1.num_agents
    return sum([swap_distance(votes_1[i], votes_2[matching[i]], matching=matching) for i in range(size)])


def compute_swap_bf_distance(instance_1: Marriages, instance_2: Marriages,
                              inner_distance: Callable) -> int:

    obj_values = []
    tmp = True
    for matching in permutations(range(instance_1.num_agents)):
        votes_1 = instance_1.votes
        votes_2 = instance_2.votes
        size = instance_1.num_agents
        value = sum([swap_distance(votes_1[matching[i]], votes_2[i], matching=matching) for i in
                    range(size)])
        obj_values.append(value)
        if tmp:
            print(matching, value)
            tmp = False

    return min(obj_values)


def compute_pairwise_distance(instance_1: Marriages, instance_2: Marriages,
                              inner_distance: Callable) -> float:
    """ Compute Pairwise distance between ordinal elections """
    length = instance_1.num_agents
    matrix_1 = instance_1.votes_to_pairwise_matrix()
    matrix_2 = instance_2.votes_to_pairwise_matrix()
    return solve_matching_matrices(matrix_1, matrix_2, length, inner_distance)

# HELPER FUNCTIONS #
def print_matrix_for_tex(matrix):
    for row in matrix:
        line = ''
        for value in row[0:-1]:
            line += f' {value} &'
        line += f' {row[-1]} \\\\'
        print(line)


def get_matching_cost_retrospective(instance_1: Marriages, instance_2: Marriages,
                                    inner_distance: Callable, crossing=False):
    """ Return: Cost table """
    vectors_1 = instance_1.get_retrospective_vectors()
    vectors_2 = instance_2.get_retrospective_vectors()

    size = instance_1.num_agents

    if crossing:
        return [[inner_distance(vectors_1[0][i], vectors_2[1][j]) for i in range(size)] for j in range(size)], \
               [[inner_distance(vectors_1[1][i], vectors_2[0][j]) for i in range(size)] for j in range(size)]

    else:
        return [[inner_distance(vectors_1[0][i], vectors_2[0][j]) for i in range(size)] for j in range(size)], \
               [[inner_distance(vectors_1[1][i], vectors_2[1][j]) for i in range(size)] for j in range(size)]



def get_matching_cost_positionwise(instance_1: Marriages, instance_2: Marriages,
                                    inner_distance: Callable) -> List[list]:
    """ Return: Cost table """
    vectors_1 = instance_1.get_positionwise_vectors()
    vectors_2 = instance_2.get_positionwise_vectors()
    size = instance_1.num_agents
    return [[inner_distance(vectors_1[i], vectors_2[j]) for i in range(size)] for j in range(size)]

# def get_matching_cost_pos_swap(instance_1: Roommates, instance_2: Roommates,
#                                 matching) -> List[list]:
#     """ Return: Cost table """
#     votes_1 = instance_1.votes
#     votes_2 = instance_2.votes
#
#     size = instance_1.num_agents
#     return [[swap_distance(votes_1[i], votes_2[j], matching=matching) for i in range(size)]
#             for j in range(size)]

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
