#!/usr/bin/env python

import math
from typing import Callable, List

import os
import networkx as nx

from mapel.core.matchings import *
from mapel.elections.objects.ApprovalElection import ApprovalElection


# MAIN APPROVAL DISTANCES

def compute_approvalwise(election_1: ApprovalElection, election_2: ApprovalElection,
                         inner_distance: Callable) -> (float, list):
    """ Return: approvalwise distance """
    election_1.votes_to_approvalwise_vector()
    election_2.votes_to_approvalwise_vector()
    return inner_distance(election_1.approvalwise_vector, election_2.approvalwise_vector), None


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


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
