
import os
import random as rand

import numpy as np
from mapel.voting.metrics import lp

from scipy.optimize import linear_sum_assignment

from mapel.voting.metrics.inner_distances import map_str_to_func


# MAIN APPROVAL DISTANCES

def compute_approval_frequency(election_1, election_2, inner_distance):
    vector_1 = election_1.votes_to_approval_frequency_vector()
    vector_2 = election_2.votes_to_approval_frequency_vector()
    inner_distance = map_str_to_func(inner_distance)
    return inner_distance(vector_1, vector_2), None
