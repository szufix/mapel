#!/usr/bin/env python
import numpy as np


def generate_ordinal_sp_conitzer_votes(num_voters=None, num_candidates=None) -> np.ndarray:
    """ Return: ordinal votes from Conitzer single-peaked culture """

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    for j in range(num_voters):
        votes[j][0] = np.random.choice(range(num_candidates))
        left = votes[j][0] - 1
        right = votes[j][0] + 1
        for k in range(1, num_candidates):
            side = np.random.choice([0, 1])
            if side == 0:
                if left >= 0:
                    votes[j][k] = left
                    left -= 1
                else:
                    votes[j][k] = right
                    right += 1
            else:
                if right < num_candidates:
                    votes[j][k] = right
                    right += 1
                else:
                    votes[j][k] = left
                    left -= 1

    return votes


def generate_ordinal_spoc_conitzer_votes(num_voters=None, num_candidates=None) -> np.ndarray:
    """ Return: ordinal votes from SPOC culture """

    votes = np.zeros([num_voters, num_candidates], dtype=int)

    for j in range(num_voters):
        votes[j][0] = np.random.choice(range(num_candidates))
        left = votes[j][0] - 1
        left %= num_candidates
        right = votes[j][0] + 1
        right %= num_candidates
        for k in range(1, num_candidates):
            side = np.random.choice([0, 1])
            if side == 0:
                votes[j][k] = left
                left -= 1
                left %= num_candidates
            else:
                votes[j][k] = right
                right += 1
                right %= num_candidates

    return votes


def generate_ordinal_sp_walsh_votes(num_voters=None, num_candidates=None) -> np.ndarray:
    """ Return: ordinal votes from Walsh single-peaked culture """

    votes = np.zeros([num_voters, num_candidates], dtype=int)

    for j in range(num_voters):
        votes[j] = _sp_walsh(0, num_candidates - 1)

    return votes.astype(int)


#############
# AUXILIARY #
#############
def _sp_walsh(a, b):
    if a == b:
        return [a]
    elif np.random.choice([0, 1]) == 0:
        return _sp_walsh(a + 1, b) + [a]
    else:
        return _sp_walsh(a, b - 1) + [b]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.04.2023 #
# # # # # # # # # # # # # # # #
