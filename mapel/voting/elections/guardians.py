import numpy as np

from mapel.voting.elections.impartial import generate_impartial_culture_election


def generate_real_identity_election(num_voters=None, num_candidates=None):
    """ Generate real election that approximates identity (ID) """
    return [[j for j in range(num_candidates)] for _ in range(num_voters)]


def generate_real_uniformity_election(num_voters=None, num_candidates=None):
    """ Generate real election that approximates uniformity (UN) """
    return generate_impartial_culture_election(num_voters=num_voters, num_candidates=num_candidates)


def generate_real_antagonism_election(num_voters=None, num_candidates=None):
    """ Generate real election that approximates antagonism (AN) """
    return [[j for j in range(num_candidates)] for _ in range(int(num_voters / 2))] + \
           [[num_candidates - j - 1 for j in range(num_candidates)] for _ in
            range(int(num_voters / 2))]


def generate_real_stratification_election(num_voters=None, num_candidates=None):
    """ Generate real election that approximates stratification (ST) """
    votes = np.zeros([num_voters, num_candidates], dtype=int)

    for i in range(num_voters):
        votes[i] = list(np.random.permutation(int(num_candidates / 2))) + \
                   list(np.random.permutation([j for j in range(int(num_candidates / 2),
                                                                num_candidates)]))
    return votes

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
