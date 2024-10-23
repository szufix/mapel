

import numpy as np
from mapel.roommates.cultures._utils import *
from mapel.core.utils import *
import logging


def generate_roommates_ic_votes(num_agents: int = None,
                                incompleteness: int = 0,
                                ties: int = 0,
                                srti_func = None,
                                **kwargs):
    """ Impartial Culture """

    votes = [list(np.random.permutation(num_agents)) for _ in range(num_agents)]
    
    out = convert_votes_to_srti(votes)
    return out if srti_func is None else srti_func(out, incompleteness, ties)


def generate_roommates_group_ic_votes(num_agents: int = None,
                                      proportion: float = 0.5,
                                      incompleteness: int = 0,
                                      ties: int = 0,
                                      srti_func = None,
                                      **kwargs):
    """ Impartial Culture with two groups """

    size_1 = int(proportion * num_agents)
    size_2 = int(num_agents - size_1)

    votes_1 = [list(np.random.permutation(size_1)) +
               list(np.random.permutation([j for j in range(size_1, num_agents)]))
               for _ in range(size_1)]

    votes_2 = [list(np.random.permutation([j for j in range(size_1, num_agents)])) +
               list(np.random.permutation(size_1))
               for _ in range(size_2)]

    votes = votes_1 + votes_2

    out = convert_votes_to_srti(votes)
    return out if srti_func is None else srti_func(out, incompleteness, ties)


# Retorna a instância para o Identity
def generate_roommates_id_votes(num_agents: int = None, **kwargs):
    """ One of four extreme points for Compass """

    votes = [list(range(num_agents)) for _ in range(num_agents)]
    
    return convert_votes_to_srti(votes)


# Retorna a instância para o Mutual disagreement
def generate_roommates_asymmetric_votes(num_agents: int = None, **kwargs):
    """ One of four extreme points for Compass """
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]
    
    return convert_votes_to_srti(votes)

# Retorna a instância para o Mutual agreement
def generate_roommates_symmetric_votes(num_agents: int = None, **kwargs):
    """ One of four extreme points for Compass """

    num_rounds = num_agents - 1

    def next(agents):
        first = agents[0]
        last = agents[-1]
        middle = agents[1:-1]
        new_agents = [first, last]
        new_agents.extend(middle)
        return new_agents

    agents = [i for i in range(num_agents)]
    rounds = []

    for _ in range(num_rounds):
        pairs = []
        for i in range(num_agents // 2):
            agent_1 = agents[i]
            agent_2 = agents[num_agents - 1 - i]
            pairs.append([agent_1, agent_2])
        rounds.append(pairs)
        agents = next(agents)

    votes = np.zeros([num_agents, num_agents - 1], dtype=int)

    for pos, partition in enumerate(rounds):
        for x, y in partition:
            votes[x][pos] = y
            votes[y][pos] = x
    return [[[votes[i][j]] for j in range(len(votes[i]))] for i in range(len(votes))]


# Retorna a instância para o Chaos
def generate_roommates_chaos_votes(num_agents: int = None, **kwargs):
    """ One of four extreme points for Compass """

    if num_agents-1 % 3 == 0:
        logging.warning("Incorrect realization of Chaos instance")

    num_rooms = num_agents // 2
    matrix = np.zeros([num_agents, num_agents - 1], dtype=int)

    matrix[0] = [i for i in range(num_agents - 1)]

    for i in range(1, num_agents):
        for j in range(num_rooms):
            matrix[i][2 * j] = (i + j - 1) % (num_agents - 1)
            if j < num_rooms - 1:
                matrix[i][2 * j + 1] = (num_rooms + i + j - 1) % (num_agents - 1)

    votes = np.zeros([num_agents, num_agents - 1], dtype=int)

    for k1 in range(num_agents):
        for k2 in range(num_agents - 1):
            for i in range(num_agents):
                if k1 != i and matrix[i][matrix[k1][k2]] == matrix[k1][k2]:
                    votes[k1][k2] = i
    
    return [[[votes[i][j]] for j in range(len(votes[i]))] for i in range(len(votes))]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 16.03.2022 #
# # # # # # # # # # # # # # # #
