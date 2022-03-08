import numpy as np


def generate_ic_votes(num_agents: int = None, params=None):

    return [list(np.random.permutation(num_agents)) for _ in range(num_agents)]


def generate_id_votes(num_agents: int = None, params=None):

    return [list(range(num_agents)) for _ in range(num_agents)]


def generate_asymmetric_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    return [rotate(vote, shift) for shift, vote in enumerate(votes)]


# def generate_ic__id_votes(num_agents: int = None, params=None):
#
#     votes_1 = [list(np.random.permutation(num_agents)) for _ in range(num_agents)]
#     votes_2 = [list(range(num_agents)) for _ in range(num_agents)]
#
#     return [votes_1, votes_2]
#
#
# def generate_asymmetric__id_votes(num_agents: int = None, params=None):
#     votes = [list(range(num_agents)) for _ in range(num_agents)]
#
#     votes_1 = [rotate(vote, shift) for shift, vote in enumerate(votes)]
#     votes_2 = [list(range(num_agents)) for _ in range(num_agents)]
#
#     return [votes_1, votes_2]


# HELPER
def rotate(vector, shift):
    shift = shift % len(vector)
    return vector[shift:] + vector[:shift]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.10.2021 #
# # # # # # # # # # # # # # # #
