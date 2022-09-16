import numpy as np

from mapel.elections.cultures.impartial import generate_ordinal_ic_votes


def generate_real_identity_votes(num_voters=None, num_candidates=None):
    """ Generate real election that approximates identity (ID) """
    return [[j for j in range(num_candidates)] for _ in range(num_voters)]


def generate_real_uniformity_votes(num_voters=None, num_candidates=None):
    """ Generate real election that approximates uniformity (UN) """
    return generate_ordinal_ic_votes(num_voters=num_voters, num_candidates=num_candidates)


def generate_real_antagonism_votes(num_voters=None, num_candidates=None):
    """ Generate real election that approximates antagonism (AN) """
    return [[j for j in range(num_candidates)] for _ in range(int(num_voters / 2))] + \
           [[num_candidates - j - 1 for j in range(num_candidates)] for _ in
            range(int(num_voters / 2))]


def generate_real_stratification_votes(num_voters=None, num_candidates=None):
    """ Generate real election that approximates stratification (ST) """
    return [list(np.random.permutation(int(num_candidates/2))) +
             list(np.random.permutation([j for j in range(int(num_candidates/2), num_candidates)]))
            for _ in range(num_voters)]


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
