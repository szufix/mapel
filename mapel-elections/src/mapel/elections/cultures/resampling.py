import numpy as np
import copy
import random
import logging


def generate_approval_resampling_votes(num_voters=None, num_candidates=None,
                                       phi=0.5, p=0.5):

    k = int(p * num_candidates)
    central_vote = {i for i in range(k)}

    votes = [set() for _ in range(num_voters)]
    for v in range(num_voters):
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


def generate_approval_disjoint_resampling_votes(num_voters=None, num_candidates=None,
                                                phi=0.5, p=0.5, g=2):
    if p*g > 1:
        logging.warning(f'Disjoint resampling model is not well defined when p * g > 1')

    num_groups = g
    k = int(p * num_candidates)

    votes = [set() for _ in range(num_voters)]

    central_votes = []
    for g in range(num_groups):
        central_votes.append({g * k + i for i in range(k)})

    for v in range(num_voters):

        central_vote = random.choice(central_votes)

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



