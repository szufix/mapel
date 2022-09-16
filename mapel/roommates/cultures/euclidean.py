import numpy as np
import math
from numpy import linalg
from mapel.roommates.cultures._utils import convert
from mapel.roommates.cultures.mallows import mallows_votes

################################################################

def get_range(params):
    if params['p_dist'] == 'beta':
        return np.random.beta(params['a'], params['b'])
    elif params['p_dist'] == 'uniform':
        return np.random.uniform(low=params['a'], high=params['b'])


def weighted_l1(a1, a2, w):
    total = 0
    for i in range(len(a1)):
        total += abs(a1[i]-a2[i])*w[i]
    return total


def generate_attributes_votes(num_agents: int = None, params: dict = None):

    name = f'{params["dim"]}d_{params["space"]}'

    agents_skills = np.array([get_rand(name) for _ in range(num_agents)])
    agents_weights = np.array([get_rand(name) for _ in range(num_agents)])

    votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)
    ones = np.ones([params["dim"]], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):
            votes[v][c] = c
            if params["dim"] == 1:
                distances[v][c] = (1. - agents_skills[c]) * agents_weights[v]
            else:
                distances[v][c] = weighted_l1(ones, agents_skills[c], agents_weights[v])
        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return convert(votes)


def generate_roommates_euclidean_votes(num_agents: int = None, params: dict = None):

    name = f'{params["dim"]}d_{params["space"]}'

    agents = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])

    votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):

            votes[v][c] = c
            distances[v][c] = np.linalg.norm(agents[v] - agents[c])
        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return convert(votes)


def generate_roommates_reverse_euclidean_votes(num_agents: int = None, params: dict = None):

    name = f'{params["dim"]}d_{params["space"]}'

    agents = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])

    votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):

            votes[v][c] = c
            distances[v][c] = np.linalg.norm(agents[v] - agents[c])
        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    if 'proportion' in params:
        p = params['proportion']
    else:
        p = 0.5

    for i in range(int(num_agents * (1.-p))):
        tmp = list(votes[i])
        tmp.reverse()
        votes[i] = tmp

    return convert(votes)


def generate_expectation_votes(num_agents: int = None, params: dict = None):

    name = f'{params["dim"]}d_{params["space"]}'

    agents_reality = np.array([get_rand(name) for _ in range(num_agents)])
    agents_wishes = np.zeros([num_agents, 2])

    for v in range(num_agents):
        # while agents_wishes[v][0] <= 0 or agents_wishes[v][0] >= 1:
        agents_wishes[v][0] = np.random.normal(agents_reality[v][0], params['std'])
        # while agents_wishes[v][1] <= 0 or agents_wishes[v][1] >= 1:
        agents_wishes[v][1] = np.random.normal(agents_reality[v][1], params['std'])

    votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):

            votes[v][c] = c
            distances[v][c] = np.linalg.norm(agents_reality[c] - agents_wishes[v])
        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return convert(votes)


def generate_fame_votes(num_agents: int = None, params: dict = None):
    # Also known as radius model

    name = f'{params["dim"]}d_{params["space"]}'

    agents = np.array([get_rand(name) for _ in range(num_agents)])
    votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)
    rays = np.array([np.random.uniform(0, params['radius']) for _ in range(num_agents)])

    for v in range(num_agents):
        for c in range(num_agents):
            votes[v][c] = c
            distances[v][c] = np.linalg.norm(agents[v] - agents[c])
            distances[v][c] = distances[v][c] - rays[c]
        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return convert(votes)


def generate_roommates_mallows_euclidean_votes(num_agents: int = None, params: dict = None):

    name = f'{params["dim"]}d_{params["space"]}'

    agents = np.array([get_rand(name, i=i, num_agents=num_agents) for i in range(num_agents)])

    votes = np.zeros([num_agents, num_agents], dtype=int)
    distances = np.zeros([num_agents, num_agents], dtype=float)

    for v in range(num_agents):
        for c in range(num_agents):

            votes[v][c] = c
            distances[v][c] = np.linalg.norm(agents[v] - agents[c])
        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    votes = mallows_votes(votes, params['phi'])

    return convert(votes)

# AUXILIARY
def random_ball(dimension, num_points=1, radius=1):
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = np.random.random(num_points) ** (1 / dimension)
    return radius * (random_directions * random_radii).T


GEN_CTR = 0

def get_rand(model: str, i: int = 0, num_agents: int = 0, cat: str = "voters") -> list:
    """ generate random values"""
    # print(model ==  "1d_uniform")

    point = [0]
    if model in {"1d_uniform",  "1d_interval"}:
        return np.random.rand()
    elif model in {'1d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=1)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=1)
    elif model in {"1d_gaussian"}:
        point = np.random.normal(0.5, 0.15)
        while point > 1 or point < 0:
            point = np.random.normal(0.5, 0.15)
    elif model == "1d_one_sided_triangle":
        point = np.random.uniform(0, 1) ** 0.5
    elif model == "1d_full_triangle":
        point = np.random.choice([np.random.uniform(0, 1) ** 0.5, 2 - np.random.uniform(0, 1) ** 0.5])
    elif model == "1d_two_party":
        point = np.random.choice([np.random.uniform(0, 1), np.random.uniform(2, 3)])
    elif model in {"2d_disc", "2d_range_disc"}:
        phi = 2.0 * 180.0 * np.random.random()
        radius = math.sqrt(np.random.random()) * 0.5
        point = [0.5 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
    elif model == "2d_range_overlapping":
        phi = 2.0 * 180.0 * np.random.random()
        radius = math.sqrt(np.random.random()) * 0.5
        if cat == "voters":
            point = [0.25 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
        elif cat == "candidates":
            point = [0.75 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
    elif model in {"2d_square", "2d_uniform"}:
        point = [np.random.random(), np.random.random()]
    elif model in {'2d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=2)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=2)
    elif model == "2d_sphere":
        alpha = 2 * math.pi * np.random.random()
        x = 1. * math.cos(alpha)
        y = 1. * math.sin(alpha)
        point = [x, y]
    elif model in ["2d_gaussian", "2d_range_gaussian"]:
        point = [np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)]
        while np.linalg.norm(point - np.array([0.5, 0.5])) > 0.5:
            point = [np.random.normal(0.5, 0.15), np.random.normal(0.5, 0.15)]
    elif model in ["2d_range_fourgau"]:
        r = np.random.randint(1, 4)
        size = 0.06
        if r == 1:
            point = [np.random.normal(0.25, size), np.random.normal(0.5, size)]
        if r == 2:
            point = [np.random.normal(0.5, size), np.random.normal(0.75, size)]
        if r == 3:
            point = [np.random.normal(0.75, size), np.random.normal(0.5, size)]
        if r == 4:
            point = [np.random.normal(0.5, size), np.random.normal(0.25, size)]
    elif model in ["3d_cube", "3d_uniform"]:
        point = [np.random.random(), np.random.random(), np.random.random()]
    elif model in ["5d_uniform"]:
        dim = 5
        point = [np.random.random() for _ in range(dim)]
    elif model in ["10d_uniform"]:
        dim = 10
        point = [np.random.random() for _ in range(dim)]
    elif model in {'3d_asymmetric'}:
        if np.random.rand() < 0.3:
            return np.random.normal(loc=0.25, scale=0.15, size=3)
        else:
            return np.random.normal(loc=0.75, scale=0.15, size=3)
    elif model in ['3d_gaussian']:
        point = [np.random.normal(0.5, 0.15),
                 np.random.normal(0.5, 0.15),
                 np.random.normal(0.5, 0.15)]
        while np.linalg.norm(point - np.array([0.5, 0.5, 0.5])) > 0.5:
            point = [np.random.normal(0.5, 0.15),
                     np.random.normal(0.5, 0.15),
                     np.random.normal(0.5, 0.15)]
    elif model == "4d_cube":
        dim = 4
        point = [np.random.random() for _ in range(dim)]
    elif model == "5d_cube":
        dim = 5
        point = [np.random.random() for _ in range(dim)]
    elif model == '1d_extreme':
        if i%2 == 1:
            i -= 0.1
        point = i
    elif model == '2d_extreme':
        if i % 2 == 1:
            alpha = 2 * math.pi * ((i-np.random.random()/100) / num_agents)
        else:
            alpha = 2 * math.pi * ((i+np.random.random()/100) / num_agents)
        x = 1. * math.cos(alpha)
        y = 1. * math.sin(alpha)
        point = [x, y]
    else:
        print('unknown culture_id', model)
        point = [0, 0]
    return point
