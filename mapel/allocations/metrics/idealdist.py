import logging
import itertools
import numpy as np

import mapel.main.matchings as matchings

def ideal_distance(left_task, right_task, *args, **kwargs):
  logging.info("Computing single ideal distance")
  _validate(left_task, right_task)

  optimal_matchings = []
  for agents_mapping in itertools.permutations(range(left_task.agents_count)):
      costs = _get_resource_mapping_costs(left_task, right_task, agents_mapping)
      optimal_matchings.append(matchings.solve_matching_vectors(costs))

  return min(optimal_matchings, key = lambda x: x[0])

def _get_resource_mapping_costs(left_task, right_task, agents_mapping):
    """ Return: Cost table """
    cost_table = np.zeros([left_task.resources_count, left_task.resources_count])

    for left_res in range(left_task.resources_count):
      for right_res in range(right_task.resources_count):
        ell_one_dist = 0
        for left_agent in range(left_task.agents_count):
          right_agent = agents_mapping[left_agent]
          ell_one_diff = abs(left_task[left_agent][left_res] - \
          right_task[right_agent][right_res])
          ell_one_dist += ell_one_diff
        cost_table[left_res][right_res] = ell_one_dist
    return cost_table

def _validate(left_task, right_task):
  if left_task.agents_count != right_task.agents_count:
    raise ValueError("Cannot compute distance between two tasks with different "
    "counts of agents")
  if left_task.resources_count != right_task.resources_count:
    raise ValueError("Cannot compute distance between two tasks with different "
    "counts of resources")
