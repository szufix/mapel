from typing import Callable, List
from itertools import combinations, permutations

from mapel.main.matchings import *
from mapel.roommates.objects.Roommates import Roommates


# from mapel.elections.metrics.inner_distances import swap_distance


# MAIN DISTANCES
def compute_retrospective(instance_1, instance_2, inner_distance):
    cost_table = get_matching_cost_retrospective(instance_1, instance_2, inner_distance)
    return solve_matching_vectors(cost_table)


# HELPER FUNCTIONS #
def get_matching_cost_retrospective(instance_1: Roommates, instance_2: Roommates,
                                    inner_distance: Callable) -> List[list]:
    """ Return: Cost table """
    vectors_1 = instance_1.get_retrospective_vectors()
    vectors_2 = instance_2.get_retrospective_vectors()
    size = instance_1.num_agents
    return [[inner_distance(vectors_1[i], vectors_2[j]) for i in range(size)] for j in range(size)]

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
