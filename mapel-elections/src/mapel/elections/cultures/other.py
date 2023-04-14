import numpy as np
from single_peaked import generate_ordinal_sp_conitzer_votes, generate_ordinal_sp_walsh_votes

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


def generate_weighted_stratification_votes(num_voters: int = None, num_candidates: int = None,
                                           params=None):
    if params is None:
        params = {}

    w = params.get('w', 0.5)

    return [list(np.random.permutation(int(w*num_candidates))) +
             list(np.random.permutation([j for j in range(int(w*num_candidates), num_candidates)]))
            for _ in range(num_voters)]

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



def generate_approval_urn_votes(num_voters: int = None,
                                num_candidates: int = None,
                                params: dict = None) -> list:
    """ Return: approval votes from an approval variant of Polya-Eggenberger urn culture """

    votes = []
    urn_size = 1.
    for j in range(num_voters):
        rho = np.random.uniform(0, urn_size)
        if rho <= 1.:
            vote = set()
            for c in range(num_candidates):
                if np.random.random() <= params['p']:
                    vote.add(c)
            votes.append(vote)
        else:
            votes.append(votes[np.random.randint(0, j)])
        urn_size += params['alpha']

    return votes
