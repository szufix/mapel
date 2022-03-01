import numpy as np
from mapel.roommates.models._utils import convert
from mapel.roommates.models.mallows import mallows_votes


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


def generate_roommates_id_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    return convert(votes)


def generate_roommates_teo_votes(num_agents: int = None, params=None):
    votes = np.zeros([num_agents, num_agents], dtype=int)

    votes[0] = [i for i in range(num_agents)]

    for i in range(1, num_agents):
        votes[i][0] = i

    for i in range(1, num_agents):
        pos = i
        pos_2 = i
        for j in range(1, num_agents):
            if i != j:
                votes[pos_2][pos] = i
            pos += 3
            if pos >= num_agents:
                pos = pos % num_agents + 1
            pos_2 += 1
            if pos_2 >= num_agents:
                pos_2 = pos_2%num_agents + 1

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


def generate_roommates_cy_votes(num_agents: int = None, params=None):
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]

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
    votes = [list(range(num_agents)) for _ in range(num_agents)]

    for i in range(num_agents):
        if i%4 == 0:
            votes[i][0] = i
            for j in range(1, num_agents): # normal
                if j%2 == 0:
                    votes[i][j] = (i + int(j/2)) % num_agents
                else:
                    votes[i][j] = (i - int(j/2) - 1) % num_agents
        elif i%4 == 1:
            votes[i][0] = i
            votes[i][1] = (i+1)% num_agents
            for j in range(2, num_agents, 2): # change
                if j%4 == 2:
                    votes[i][j] = (i+1 - int(j/2) - 1) % num_agents
                    votes[i][j+1] = (i+1 - int(j/2) - 2) % num_agents
                else:
                    votes[i][j] = (i+1 + int(j/2) - 1) % num_agents
                    votes[i][j+1] = (i+1 + int(j/2)) % num_agents
        elif i%4 == 2:
            votes[i][0] = i
            votes[i][1] = (i-1)% num_agents
            for j in range(2, num_agents, 2): # change
                if j%4 == 2:
                    votes[i][j] = (i + int(j/2)) % num_agents
                    votes[i][j+1] = (i + int(j/2) + 1) % num_agents
                else:
                    votes[i][j] = (i - int(j/2)) % num_agents
                    votes[i][j+1] = (i - int(j/2) - 1) % num_agents
        else:
            votes[i][0] = i
            for j in range(1, num_agents): # normal
                if j%2 == 0:
                    votes[i][j] = (i+1 - int(j/2) - 1) % num_agents
                else:
                    votes[i][j] = (i+1 + int(j/2)) % num_agents
    # print(votes[0:4])
    # for i in [4,5,6,7,12,13,14,15]:
    #     right = int(num_agents/2)
    #     left = right-1
    #     tmp = votes[i][left]
    #     votes[i][left] = votes[i][right]
    #     votes[i][right] = tmp

    print(votes)
    # exit()
    return convert(votes)

# HELPER
def rotate(vector, shift):
    shift = shift%len(vector)
    return vector[shift:] + vector[:shift]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.10.2021 #
# # # # # # # # # # # # # # # #
