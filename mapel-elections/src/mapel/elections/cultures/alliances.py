import numpy as np

from mapel.elections.cultures.impartial import generate_ordinal_ic_votes
from mapel.elections.cultures.urn import generate_urn_votes
from mapel.core.features.mallows import generate_mallows_votes, phi_from_normphi


def generate_ordinal_alliance_ic_votes(num_voters: int = None,
                                       num_candidates: int = None,
                                       params: dict = None):
    """ Return: ordinal votes from Impartial Culture with alliances """
    votes = generate_ordinal_ic_votes(num_voters, num_candidates)

    alliances = np.random.choice([i for i in range(params['num_alliances'])],
                                 size=num_candidates, replace=True)
    return np.array(votes), alliances


def generate_ordinal_alliance_urn_votes(num_voters: int = None,
                                       num_candidates: int = None,
                                       params: dict = None):

    votes = generate_urn_votes(num_voters, num_candidates, params)

    alliances = np.random.choice([i for i in range(params['num_alliances'])],
                                 size=num_candidates, replace=True)

    return np.array(votes), alliances


def generate_ordinal_alliance_norm_mallows_votes(num_voters: int = None,
                                        num_candidates: int = None,
                                        params: dict = None):
    params['phi'] = phi_from_normphi(num_candidates, params['normphi'])
    votes = generate_mallows_votes(num_voters, num_candidates, params)

    alliances = np.random.choice([i for i in range(params['num_alliances'])],
                                 size=num_candidates, replace=True)

    return np.array(votes), alliances



def generate_ordinal_alliance_euclidean_votes(num_voters: int = None,
                                       num_candidates: int = None,
                                       params: dict = None):

    dim = params['dim']
    num_alliances = params['num_alliances']

    voters = np.zeros([num_voters, dim])
    candidates = np.zeros([num_candidates, dim])
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    voters = np.random.rand(num_voters, dim)
    centers = np.random.rand(num_alliances, dim)
    alliances = np.random.choice([i for i in range(params['num_alliances'])],
                                 size=num_candidates, replace=True)
    candidates = []
    for c in range(num_candidates):
        candidates.append(np.random.normal(loc=centers[alliances[c]], scale=0.15, size=(1, dim)))

    for v in range(num_voters):
        for c in range(num_candidates):
            votes[v][c] = c
            distances[v][c] = np.linalg.norm(voters[v] - candidates[c], ord=params['dim'])

        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return votes, alliances