import numpy as np


def generate_ic_votes(num_agents: int = None, params=None):

    return [list(np.random.permutation(num_agents)) for _ in range(num_agents)]


def generate_id_votes(num_agents: int = None, params=None):

    return [list(range(num_agents)) for _ in range(num_agents)]


def generate_asymmetric_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]
    votes_left = [rotate(vote, shift+1) for shift, vote in enumerate(votes)]
    votes = [list(range(num_agents)) for _ in range(num_agents)]
    votes_right = [rotate(vote, shift) for shift, vote in enumerate(votes)]
    return [votes_left, votes_right]


def generate_group_ic_votes(num_agents: int = None, params: dict = None):
    """ Impartial Culture with two groups """

    if 'proportion' not in params:
        params['proportion'] = 0.5

    size_1 = int(params['proportion'] * num_agents)
    size_2 = int(num_agents - size_1)

    votes_1 = [list(np.random.permutation(size_1)) +
               list(np.random.permutation([j for j in range(size_1, num_agents)]))
               for _ in range(size_1)]

    votes_2 = [list(np.random.permutation([j for j in range(size_1, num_agents)])) +
               list(np.random.permutation(size_1))
               for _ in range(size_2)]

    votes = votes_1 + votes_2

    return votes

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


def generate_symmetric_votes(num_agents: int = None, params=None):

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

    votes = np.zeros([num_agents, num_agents], dtype=int)

    for pos, partition in enumerate(rounds):
        for x, y in partition:
            votes[x][pos+1] = y
            votes[y][pos+1] = x

    for i in range(num_agents):
        votes[i][0] = i

    return votes

# HELPER
def rotate(vector, shift):
    shift = shift % len(vector)
    return vector[shift:] + vector[:shift]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.10.2021 #
# # # # # # # # # # # # # # # #
