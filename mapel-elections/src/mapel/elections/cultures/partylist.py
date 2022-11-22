
import numpy as np
from mapel.core.utils import get_vector


def generate_approval_partylist_votes(num_voters=None, num_candidates=None, params=None):

    if params is None:
        params = {}

    num_groups = params.get('g', 5)

    alphas = get_vector('linear', num_groups)

    # for i in range(len(alphas)):
    #     if alphas[i] == 0.:
    #         alphas[i] = 0.00001

    sizes = np.random.dirichlet(alphas)
    cumv = np.cumsum(sizes)
    cumv = np.insert(cumv, 0, 0)
    print(cumv)

    votes = []

    for i in range(0,0):
        print(i)

    for g in range(1, num_groups+1):
        vote = set()
        print(int(num_candidates*cumv[g-1]))
        print(int(num_candidates*cumv[g]))
        for i in range(int(num_candidates*cumv[g-1]), int(num_candidates*cumv[g])):
            print(i)
            vote.add(i)
        for i in range(int(num_candidates * cumv[g - 1]), int(num_candidates * cumv[g])):
            votes.append(vote)
    print(votes)
    return votes


def generate_approval_urn_partylist_votes(num_voters=None, num_candidates=None, params=None):

    if params is None:
        params = {}

    num_groups = params.get('g', 5)

    party_votes = np.zeros([num_voters])
    urn_size = 1.
    for j in range(num_voters):
        rho = np.random.uniform(0, urn_size)
        if rho <= 1.:
            party_votes[j] = np.random.randint(0, num_groups)
        else:
            party_votes[j] = party_votes[np.random.randint(0, j)]
        urn_size += params['alpha']

    party_size = int(num_candidates/num_groups)
    votes = []

    for i in range(num_voters):
        shift = party_votes[i]*party_size
        vote = set([int(c+shift) for c in range(party_size)])
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
