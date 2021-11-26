import numpy as np
from mapel.roommates.models._utils import convert


def generate_roommates_ic_votes(num_agents: int = None, params=None):

    votes = [list(np.random.permutation(num_agents)) for _ in range(num_agents)]

    return convert(votes)


def generate_roommates_id_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    return convert(votes)


def generate_roommates_cy_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]

    return convert(votes)

# HELPER
def rotate(vector, shift):
    return vector[shift:] + vector[:shift]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.10.2021 #
# # # # # # # # # # # # # # # #
