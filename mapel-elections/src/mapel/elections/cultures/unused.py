import numpy as np
from mapel.elections.cultures.single_peaked import generate_ordinal_sp_conitzer_votes, generate_ordinal_sp_walsh_votes


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


def generate_approval_simplex_resampling_votes(num_voters=None, num_candidates=None,
                                                    params=None):
        if 'phi' not in params:
            phi = np.random.random()
        else:
            phi = params['phi']

        if 'g' not in params:
            num_groups = 2
        else:
            num_groups = params['g']

        if 'max_range' not in params:
            params['max_range'] = 1.

        sizes_c = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # sizes_c = runif_in_simplex(num_groups)
        sizes_c = np.concatenate(([0], sizes_c))
        sizes_c = np.cumsum(sizes_c)
        print(sizes_c)

        sum = 71 + 5*8

        sizes_v = [71/sum, 8/sum, 8/sum, 8/sum, 8/sum, 8/sum]
        # sizes_v = runif_in_simplex(num_groups)
        sizes_v = np.concatenate(([0], sizes_v))
        sizes_v = np.cumsum(sizes_v)
        print(sizes_v)

        votes = [set() for _ in range(num_voters)]

        for g in range(num_groups):

            central_vote = {i for i in range(int(sizes_c[g] * num_candidates),
                                             int(sizes_c[g+1] * num_candidates))}

            for v in range(int(sizes_v[g] * num_voters), int(sizes_v[g + 1] * num_voters)):
                vote = set()
                for c in range(num_candidates):
                    if np.random.random() <= phi:
                        if np.random.random() <= params['p']:
                            vote.add(c)
                    else:
                        if c in central_vote:
                            vote.add(c)
                votes[v] = vote

            # sum_p = sum([sum(vote) for vote in votes])
            # avg_p = sum_p / (num_voters * num_candidates)
            # print(avg_p, params['max_range'])
            # if avg_p < params['max_range']:
            #     break

        return votes

def approval_anti_pjr_votes(num_voters=None, num_candidates=None, params=None):

    if 'p' not in params:
        p = np.random.random()
    else:
        p = params['p']

    if 'phi' not in params:
        phi = np.random.random()
    else:
        phi = params['phi']

    if 'g' not in params:
        num_groups = 2
    else:
        num_groups = params['g']

    c_group_size = int(num_candidates/num_groups)
    v_group_size = int(num_voters/num_groups)
    size = int(p * num_candidates)

    votes = []
    for g in range(num_groups):

        core = {g * c_group_size + i for i in range(c_group_size)}
        rest = set(list(range(num_candidates))) - core

        if size <= c_group_size:
            central_vote = set(np.random.choice(list(core), size=size))
        else:
            central_vote = set(np.random.choice(list(rest), size=size-c_group_size))
            central_vote = central_vote.union(core)

        for v in range(v_group_size):
            vote = set()
            for c in range(num_candidates):
                if np.random.random() <= phi:
                    if np.random.random() <= p:
                        vote.add(c)
                else:
                    if c in central_vote:
                        vote.add(c)
            votes.append(vote)

    return votes


def approval_partylist_votes_old(num_voters=None, num_candidates=None, params=None):

    if 'g' not in params:
        num_groups = 2
    else:
        num_groups = params['g']

    if 'is_shifted' not in params:
        shift = False
    else:
        shift = True

    if 'm' not in params:
        m = 0
    else:
        m = params['m']

    c_group_size = int(num_candidates/num_groups)
    v_group_size = int(num_voters/num_groups)

    votes = []
    if not shift:
        for g in range(num_groups):
            for v in range(v_group_size):
                vote = set()
                for c in range(c_group_size):
                    c += g*c_group_size
                    vote.add(c)
                for _ in range(m):
                    el = random.sample(vote, 1)[0]
                    vote.remove(el)
                votes.append(vote)
    else:
        shift = int(c_group_size/2)
        for g in range(num_groups):
            for v in range(int(v_group_size/2)):
                vote = set()
                for c in range(c_group_size):
                    c += g*c_group_size
                    vote.add(c)
                for _ in range(m):
                    el = random.sample(vote, 1)[0]
                    vote.remove(el)
                votes.append(vote)

            for v in range(int(v_group_size/2), v_group_size):
                vote = set()
                for c in range(c_group_size):
                    c += g*c_group_size + shift
                    c %= num_candidates
                    vote.add(c)
                for _ in range(m):
                    el = random.sample(vote, 1)[0]
                    vote.remove(el)
                votes.append(vote)

    return votes


def generate_approval_exp_partylist_votes(num_voters=None, num_candidates=None, params=None):

    if params is None:
        params = {}

    num_groups = params.get('g', 5)
    exp = params.get('exp', 2.)

    sizes = np.array([1./exp**(i+1) for i in range(num_groups)])
    sizes = sizes / np.sum(sizes)
    party_votes = np.random.choice([i for i in range(num_groups)], num_voters, p=sizes)

    party_size = int(num_candidates/num_groups)
    votes = []

    for i in range(num_voters):
        shift = party_votes[i]*party_size
        vote = set([int(c+shift) for c in range(party_size)])
        votes.append(vote)

    return votes
