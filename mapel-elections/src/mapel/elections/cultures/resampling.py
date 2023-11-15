import numpy as np
import copy
import random
import logging


def generate_approval_resampling_votes(num_voters: int = None,
                                       num_candidates: int = None,
                                       phi: float = 0.5,
                                       p: float = 0.5,
                                       seed: int = None) -> list:
    """
    Generates approval votes from resampling model.

        Parameters
        ----------
        num_voters : int
            Number of voters.
        num_candidates : int
            Number of candidates.
        phi : float, default: 0.5
            Resampling model parameter, denoting the noise.
        p : float, default: 0.5
            Resampling model parameter, denoting the average vote length.
        seed : int
            Seed for numpy random number generator.

        Returns
        -------
        list[set]
            Approval votes.

        Raises
        ------
        ValueError
            When `phi` not in [0,1] interval.
            When `p` not in [0,1] interval.
    """

    if phi < 0 or 1 < phi:
        logging.warning(f'Resampling model is not well defined for `phi` not in [0,1] interval')

    if p < 0 or 1 < p:
        logging.warning(f'Resampling model is not well defined for `p` not in [0,1] interval')

    rng = np.random.default_rng(seed)

    k = int(p * num_candidates)
    central_vote = {i for i in range(k)}

    votes = [set() for _ in range(num_voters)]
    for v in range(num_voters):
        vote = set()
        for c in range(num_candidates):
            if rng.random() <= phi:
                if rng.random() <= p:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote

    return votes


def generate_approval_disjoint_resampling_votes(num_voters: int = None,
                                                num_candidates: int = None,
                                                phi: float = 0.5,
                                                p: float = 0.5,
                                                g: int = 2,
                                                seed: int = None) -> list:

    if p*g > 1:
        logging.warning(f'Disjoint resampling model is not well defined when p * g > 1')

    rng = np.random.default_rng(seed)

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
            if rng.random() <= phi:
                if rng.random() <= p:
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
                                              legs=1,
                                              seed=None) -> list:
    rng = np.random.default_rng(seed)

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
            if rng.random() <= phi:
                if rng.random() <= p:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote
        central_vote = copy.deepcopy(vote)

        if v in breaks:
            central_vote = copy.deepcopy(ccc)

    return votes



