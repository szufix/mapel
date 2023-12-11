import random
import numpy as np

from prefsampling.ordinal import single_peaked_conitzer as generate_ordinal_sp_conitzer_votes
from prefsampling.ordinal import single_peaked_walsh as generate_ordinal_sp_walsh_votes


def generate_ic_party(num_voters: int = None, params: dict = None) -> list:
    """ Return: party votes from Impartial Culture"""
    num_parties = params['num_parties']
    party_size = params['num_winners']

    votes = np.zeros([num_voters, num_parties], dtype=int)

    for j in range(num_voters):
        votes[j] = np.random.permutation(num_parties)

    new_votes = [[] for _ in range(num_voters)]
    for i in range(num_voters):
        for j in range(num_parties):
            for w in range(party_size):
                _id = votes[i][j] * party_size + w
                new_votes[i].append(_id)
    return new_votes



def generate_sp_party(model=None, num_voters=None, num_candidates=None, params=None) -> np.ndarray:
    candidates = [[] for _ in range(num_candidates)]
    _ids = [i for i in range(num_candidates)]

    for j in range(params['num_parties']):
        for w in range(params['num_winners']):
            _id = j * params['num_winners'] + w
            candidates[_id] = [np.random.normal(params['party'][j][0], params['var'])]

    mapping = [x for _, x in sorted(zip(candidates, _ids))]

    if model == 'conitzer_party':
        votes = generate_ordinal_sp_conitzer_votes(num_voters=num_voters,
                                                   num_candidates=num_candidates)
    elif model == 'walsh_party':
        votes = generate_ordinal_sp_walsh_votes(num_voters=num_voters,
                                                num_candidates=num_candidates)
    for i in range(num_voters):
        for j in range(num_candidates):
            votes[i][j] = mapping[votes[i][j]]

    return votes


def generate_approval_exp_partylist_votes(num_voters=None, num_candidates=None, params=None):
    if params is None:
        params = {}

    num_groups = params.get('g', 5)
    exp = params.get('experiment', 2.)

    sizes = np.array([1. / exp ** (i + 1) for i in range(num_groups)])
    sizes = sizes / np.sum(sizes)
    party_votes = np.random.choice([i for i in range(num_groups)], num_voters, p=sizes)

    party_size = int(num_candidates / num_groups)
    votes = []

    for i in range(num_voters):
        shift = party_votes[i] * party_size
        vote = set([int(c + shift) for c in range(party_size)])
        votes.append(vote)

    return votes
