#!/usr/bin/env python

from typing import Callable

import os

from mapel.core.matchings import *
from mapel.elections.objects.ApprovalElection import ApprovalElection


# MAIN APPROVAL DISTANCES
def compute_approvalwise(election_1: ApprovalElection, election_2: ApprovalElection,
                         inner_distance: Callable) -> (float, list):
    """ Return: approvalwise distance """
    election_1.votes_to_approvalwise_vector()
    election_2.votes_to_approvalwise_vector()
    return inner_distance(election_1.approvalwise_vector, election_2.approvalwise_vector), None


# WAITING FOR UPDATE
def compute_hamming(election_1: ApprovalElection, election_2: ApprovalElection) -> float:
    """ Return: Hamming distance """
    return -1


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
