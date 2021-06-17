import random as rand
import numpy as np


def generate_conitzer_election(num_voters=None, num_candidates=None):
    """ helper function: generate conitzer single-peaked elections """

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    for j in range(num_voters):
        votes[j][0] = rand.randint(0, num_candidates - 1)
        left = votes[j][0] - 1
        right = votes[j][0] + 1
        for k in range(1, num_candidates):
            side = rand.choice([0, 1])
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


def generate_spoc_conitzer_election(num_voters=None, num_candidates=None):
    """ helper function: generate spoc_conitzer single-peaked elections"""

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    for j in range(num_voters):
        votes[j][0] = rand.randint(0, num_candidates - 1)
        left = votes[j][0] - 1
        left %= num_candidates
        right = votes[j][0] + 1
        right %= num_candidates
        for k in range(1, num_candidates):
            side = rand.choice([0, 1])
            if side == 0:
                votes[j][k] = left
                left -= 1
                left %= num_candidates
            else:
                votes[j][k] = right
                right += 1
                right %= num_candidates

    return votes


def generate_walsh_election(num_voters=None, num_candidates=None):
    """ helper function: generate walsh single-peaked elections"""

    votes = np.zeros([num_voters, num_candidates])

    for j in range(num_voters):
        votes[j] = walsh_sp(0, num_candidates - 1)

    return votes


# AUXILIARY
def walsh_sp(a, b):
    if a == b:
        return [a]
    elif rand.choice([0, 1]) == 0:
        return walsh_sp(a + 1, b) + [a]
    else:
        return walsh_sp(a, b - 1) + [b]