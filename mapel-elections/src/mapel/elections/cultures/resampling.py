import numpy as np
import copy


def generate_approval_resampling_votes(num_voters=None, num_candidates=None,
                                       phi=0.5, p=0.5):
    k = int(p * num_candidates)
    central_vote = {i for i in range(k)}

    votes = [{c for c in range(num_candidates) if
              (np.random.random() <= phi and np.random.random() <= p) or c in central_vote}
             for _ in range(num_voters)]

    return votes


def generate_approval_disjoint_resampling_votes(num_voters=None, num_candidates=None,
                                                phi=0.5, p=0.5, g=2):
    num_groups = g
    k = int(p * num_candidates)
    sizes = [1./num_groups for _ in range(num_groups)]
    sizes = np.concatenate(([0], sizes))
    sizes = np.cumsum(sizes)

    votes = [set() for _ in range(num_voters)]

    for g in range(num_groups):

        central_vote = {g * k + i for i in range(k)}

        for v in range(int(sizes[g]*num_voters), int(sizes[g+1]*num_voters)):
            vote = set()
            for c in range(num_candidates):
                if np.random.random() <= phi:
                    if np.random.random() <= p:
                        vote.add(c)
                else:
                    if c in central_vote:
                        vote.add(c)
            votes[v] = vote

    return votes


def generate_approval_moving_resampling_votes(num_voters=None,
                                              num_candidates=None,
                                              p=0.5,
                                              phi=0.5,
                                              legs=1):
    # central_vote = set()
    # for c in range(num_candidates):
    #     if rand.random() <= params['p']:
    #         central_vote.add(c)
    num_legs = legs
    breaks = [int(num_voters/num_legs)*i for i in range(num_legs)]

    k = int(p * num_candidates)
    central_vote = {i for i in range(k)}
    ccc = copy.deepcopy(central_vote)

    votes = [set() for _ in range(num_voters)]
    votes[0] = copy.deepcopy(central_vote)

    for v in range(1, num_voters):
        vote = set()
        for c in range(num_candidates):
            if np.random.random() <= phi:
                if np.random.random() <= p:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote
        central_vote = copy.deepcopy(vote)

        if v in breaks:
            central_vote = copy.deepcopy(ccc)

    return votes

