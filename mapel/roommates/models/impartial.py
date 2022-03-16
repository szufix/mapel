import math

import numpy as np
from mapel.roommates.models._utils import convert
from mapel.roommates.models.mallows import mallows_votes


# Compass
def generate_roommates_id_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    return convert(votes)


def generate_roommates_chaos_votes(num_agents: int = None, params=None):

    num_rooms = num_agents//2
    matrix = np.zeros([num_agents, num_agents-1], dtype=int)

    matrix[0] = [i for i in range(num_agents-1)]

    for i in range(1, num_agents):
        for j in range(num_rooms):
            matrix[i][2*j] = (i + j - 1) % (num_agents-1)
            if j < num_rooms - 1:
                matrix[i][2*j+1] = (num_rooms + i + j-1) % (num_agents-1)

    votes = np.zeros([num_agents, num_agents-1], dtype=int)
    for i in range(num_agents):
        for j in range(num_agents-1):
            votes[i][j] = -1

    for k1 in range(num_agents):
        for k2 in range(num_agents-1):
            for i in range(num_agents):
                if k1 != i and matrix[i][matrix[k1][k2]] == matrix[k1][k2]:
                    votes[k1][k2] = i

    return votes


def generate_roommates_asymmetric_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]

    return convert(votes)


def generate_roommates_symmetric_votes(num_agents: int = None, params=None):

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

    votes = np.zeros([num_agents, num_agents-1], dtype=int)

    for pos, partition in enumerate(rounds):
        for x, y in partition:
            votes[x][pos] = y
            votes[y][pos] = x

    return votes


# Impartial Culture
def generate_roommates_ic_votes(num_agents: int = None, params=None):

    votes = [list(np.random.permutation(num_agents)) for _ in range(num_agents)]

    return convert(votes)


def generate_roommates_group_ic_votes(num_agents: int = None, params=None):

    num_groups = params['g']

    votes_1 = [list(np.random.permutation(int(num_agents/2))) +
             list(np.random.permutation([j for j in range(int(num_agents/2), num_agents)]))
            for _ in range(int(num_agents/2))]

    votes_2 = [list(np.random.permutation([j for j in range(int(num_agents/2), num_agents)])) +
               list(np.random.permutation(int(num_agents/2)))
            for _ in range(int(num_agents/2))]

    votes = votes_1 + votes_2

    # print(votes)

    return convert(votes)



def generate_roommates_an_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    for i in range(int(num_agents/2), num_agents):
        votes[i].reverse()

    return convert(votes)


def generate_roommates_an2_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    for i in range(0, num_agents, 2):
        votes[i+1].reverse()

    # for i in range(int(num_agents*0.25), int(num_agents*0.75)):
    #     votes[i].reverse()

    return convert(votes)




def generate_roommates_malasym_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]

    votes = mallows_votes(votes, params['phi'])

    return convert(votes)


def generate_roommates_cy3_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, -shift) for shift, vote in enumerate(votes)]

    return convert(votes)

def f(n):
    if n == 0:
        return 0
    s = sum([x+1 for x in range(n)])
    print(s)
    return s

def generate_roommates_cy2_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, f(shift)) for shift, vote in enumerate(votes)]
    # print(votes)

    return convert(votes)


def generate_roommates_revcy_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]

    for i in range(0, num_agents, 2):
        votes[i+1].reverse()

    return convert(votes)



def generate_roommates_ideal_votes(num_agents: int = None, params=None):
    votes = np.zeros([num_agents, num_agents], dtype=int)

    for i in range(num_agents):
        votes[i][0] = i
    print((votes))

    tri_sets = math.floor(num_agents/3)
    for x in range(tri_sets):
        i = x*3
        votes[i][1] = i+1
        votes[i][2] = i+2
        votes[i+1][1] = i+2
        votes[i+1][2] = i
        votes[i+2][1] = i
        votes[i+2][2] = i+1

        base = list(range(num_agents))
        remove = [i, i+1, i+2]
        tmp = np.setdiff1d(base, remove)
        for j in range(i, i+3):
            ending = np.random.permutation(tmp)
            votes[j][3:] = ending

    for i in range(tri_sets*3, num_agents):
        base = list(range(num_agents))
        remove = [i]
        tmp = np.setdiff1d(base, remove)
        ending = np.random.permutation(tmp)
        votes[i][1:] = ending

    return convert(votes)


    #
    # for i in range(num_agents):
    #     if i%4 == 0:
    #         votes[i][0] = i
    #         for j in range(1, num_agents): # normal
    #             if j%2 == 0:
    #                 votes[i][j] = (i + int(j/2)) % num_agents
    #             else:
    #                 votes[i][j] = (i - int(j/2) - 1) % num_agents
    #     elif i%4 == 1:
    #         votes[i][0] = i
    #         votes[i][1] = (i+1)% num_agents
    #         for j in range(2, num_agents, 2): # change
    #             if j%4 == 2:
    #                 votes[i][j] = (i+1 - int(j/2) - 1) % num_agents
    #                 votes[i][j+1] = (i+1 - int(j/2) - 2) % num_agents
    #             else:
    #                 votes[i][j] = (i+1 + int(j/2) - 1) % num_agents
    #                 votes[i][j+1] = (i+1 + int(j/2)) % num_agents
    #     elif i%4 == 2:
    #         votes[i][0] = i
    #         votes[i][1] = (i-1)% num_agents
    #         for j in range(2, num_agents, 2): # change
    #             if j%4 == 2:
    #                 votes[i][j] = (i + int(j/2)) % num_agents
    #                 votes[i][j+1] = (i + int(j/2) + 1) % num_agents
    #             else:
    #                 votes[i][j] = (i - int(j/2)) % num_agents
    #                 votes[i][j+1] = (i - int(j/2) - 1) % num_agents
    #     else:
    #         votes[i][0] = i
    #         for j in range(1, num_agents): # normal
    #             if j%2 == 0:
    #                 votes[i][j] = (i+1 - int(j/2) - 1) % num_agents
    #             else:
    #                 votes[i][j] = (i+1 + int(j/2)) % num_agents
    # print(votes[0:4])
    # for i in [4,5,6,7,12,13,14,15]:
    #     right = int(num_agents/2)
    #     left = right-1
    #     tmp = votes[i][left]
    #     votes[i][left] = votes[i][right]
    #     votes[i][right] = tmp

    # print(votes)
    # exit()

# HELPER
def rotate(vector, shift):
    shift = shift%len(vector)
    return vector[shift:] + vector[:shift]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.10.2021 #
# # # # # # # # # # # # # # # #
