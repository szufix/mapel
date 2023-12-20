import numpy as np

from prefsampling.ordinal import impartial as generate_ordinal_ic_votes
from prefsampling.ordinal import urn as generate_urn_votes


def get_alliances(num_candidates, num_alliances):
    while True:
        alliances = np.random.choice([i for i in range(num_alliances)],
                                     size=num_candidates, replace=True)
        if len(set(alliances)) > 1:
            return alliances


def generate_ordinal_alliance_ic_votes(num_voters: int = None,
                                       num_candidates: int = None,
                                       params: dict = None):
    """ Return: ordinal votes from Impartial Culture with alliances """
    votes = generate_ordinal_ic_votes(num_voters, num_candidates)

    alliances = get_alliances(num_candidates, params['num_alliances'])

    return np.array(votes), alliances


def generate_ordinal_alliance_urn_votes(num_voters: int = None,
                                        num_candidates: int = None,
                                        params: dict = None):
    votes = generate_urn_votes(num_voters, num_candidates, params)

    alliances = get_alliances(num_candidates, params['num_alliances'])

    return np.array(votes), alliances


def generate_ordinal_alliance_norm_mallows_votes(num_voters: int = None,
                                                 num_candidates: int = None,
                                                 params: dict = None):

    params['phi'] = phi_from_normphi(num_candidates, params['normphi'])
    votes = generate_mallows_votes(num_voters, num_candidates, **params)

    alliances = get_alliances(num_candidates, params['num_alliances'])

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
    alliances = get_alliances(num_candidates, params['num_alliances'])

    candidates = []
    for c in range(num_candidates):
        candidates.append(np.random.normal(loc=centers[alliances[c]], scale=0.15, size=(1, dim)))

    for v in range(num_voters):
        for c in range(num_candidates):
            votes[v][c] = c
            distances[v][c] = np.linalg.norm(voters[v] - candidates[c], ord=params['dim'])

        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return votes, alliances


def assign_to_closest_center(candidates, centers):
    """
    Assign each candidate to the closest center.

    Parameters:
    - candidates: (num_candidates, dim) array, coordinates for each candidate point.
    - centers: (num_alliances, dim) array, coordinates for each center point.

    Returns:
    - assignment: (num_candidates,) array, index of the closest center for each candidate.
    """
    distances = np.linalg.norm(candidates[:, None] - centers, axis=2)
    assignment = np.argmin(distances, axis=1)
    return assignment


def generate_ordinal_alliance_allied_euclidean_votes(num_voters: int = None,
                                       num_candidates: int = None,
                                       params: dict = None):
    dim = params['dim']
    num_alliances = params['num_alliances']

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    while True:
        voters = np.random.rand(num_voters, dim)
        centers = np.random.rand(num_alliances, dim)
        candidates = np.random.rand(num_candidates, dim)

        alliances = assign_to_closest_center(candidates, centers)
        if len(set(alliances)) == num_alliances:
            break


    for v in range(num_voters):
        for c in range(num_candidates):
            votes[v][c] = c
            distances[v][c] = np.linalg.norm(voters[v] - candidates[c])

        votes[v] = [x for _, x in sorted(zip(distances[v], votes[v]))]

    return votes, alliances
