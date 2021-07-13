#!/usr/bin/env python

import os

import numpy as np
import random as rand
import math

from . import objects as obj
from .metrics import lp
from . import development as dev

from .objects.Experiment import Experiment, Experiment_xD, Experiment_2D, Experiment_3D

### MAPPING ###
def get_feature(name):
    return {'borda_std': borda_std,
            'separation': separation,
            'both': both,
            'highest_borda_score': highest_borda_score,
            'highest_plurality_score': highest_plurality_score,
            'highest_copeland_score': highest_copeland_score,
            'lowest_dodgson_score': lowest_dodgson_score,
            }.get(name)


### MAIN FUNCTION ###

def compute_feature(experiment_id, name=None):
    experiment = Experiment_xD(experiment_id)
    values = []

    for election in experiment.elections:
        statistic = get_feature(name)
        value = statistic(election)
        values.append(value)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", str(name) + '.txt')

    file_scores = open(file_name, 'w')
    for i in range(experiment.num_elections):
        file_scores.write(str(values[i]) + "\n")
    file_scores.close()


def borda_std(election):
    scores = np.zeros(election.num_candidates)

    vectors = election.votes_to_positionwise_vectors()

    for i in range(election.num_candidates):
        for j in range(election.num_candidates):
            scores[i] += vectors[i][j] * (election.num_candidates - j - 1)

    std = np.std(scores)
    return std


# def separation_2(election):
# def separation_2(election):
#
#     if election.fake:
#         return 0
#
#     first_half = np.zeros(election.num_candidates)
#
#     for i in range(election.num_voters):
#         for j in range(int(election.num_candidates/2)):
#             first_half[election.votes[i][j]] += 1
#
#     value = 0
#     shift = election.num_voters/2
#     for i in range(len(first_half)):
#         if first_half[i] > shift:
#             first_half[i] -= 2*shift
#         value += first_half[i]**2
#
#     return value


def separation(election):
    # todo: policzyć to na podstawie positionwise vectors
    if election.fake:
        return 0

    half = int(election.num_candidates / 2)

    ranking = dev.get_borda_ranking(election.votes, election.num_voters, election.num_candidates)
    first_half = ranking[0:half]

    distance = 0

    for i in range(election.num_voters):
        for j in range(half):
            if election.votes[i][j] not in first_half:
                distance += half - j

    for i in range(election.num_voters):
        for j in range(half, election.num_candidates):
            if election.votes[i][j] in first_half:
                distance += j - half

    return distance


def both(election):
    v1 = borda_std(election) / 2.9
    v2 = separation(election) / 1235.
    return v1 + v2


# SCORING FUNCTIONS

def highest_borda_score(election):
    """ Compute highest BORDA score of a given election """
    c = election.num_candidates
    v = election.num_voters
    vectors = election.votes_to_positionwise_vectors()
    # return sum([vectors[0][i] * (c - i - 1) for i in range(c)]) * v
    # todo: rewrite this function
    return -1


def highest_plurality_score(election):
    """ compute highest PLURALITY score of a given election"""

    if election.fake:
        c = election.num_candidates
        v = election.num_voters
        vectors = election.votes_to_positionwise_vectors()
        first = []
        for i in range(c):
            first.append(vectors[i][0])
        return max(first) * v

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        scores[vote[0]] += 1

    return max(scores)


def highest_copeland_score(potes, num_voters, num_candidates):
    """ compute highest COPELAND score of a given election """

    scores = np.zeors([num_candidates])

    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            result = 0
            for k in range(num_voters):
                if potes[k][i] < potes[k][j]:
                    result += 1
            if result > num_voters / 2:
                scores[i] += 1
                scores[j] -= 1
            elif result < num_voters / 2:
                scores[i] -= 1
                scores[j] += 1

    return max(scores)


def potes_to_unique_potes(potes):
    """ Remove repetitions from potes (positional votes) """
    unique_potes = []
    N = []
    for pote in potes:
        flag_new = True
        for i, p in enumerate(unique_potes):
            if list(pote) == list(p):
                N[i] += 1
                flag_new = False
        if flag_new:
            unique_potes.append(pote)
            N.append(1)
    return unique_potes, N


def lowest_dodgson_score(election):
    """ compute lowest DODGSON score of a given election """

    min_score = math.inf

    for target_id in range(election.num_candidates):

        # PREPARE N
        unique_potes, N = potes_to_unique_potes(election.potes)

        e = np.zeros([len(N), election.num_candidates, election.num_candidates])

        # PREPARE e
        for i, p in enumerate(unique_potes):
            for j in range(election.num_candidates):
                for k in range(election.num_candidates):
                    if p[target_id] <= p[k] + j:
                        e[i][j][k] = 1

        # PREPARE D
        D = [0 for _ in range(election.num_candidates)]
        threshold = math.ceil((election.num_voters + 1) / 2.)
        for k in range(election.num_candidates):
            diff = 0
            for i, p in enumerate(unique_potes):
                if p[target_id] < p[k]:
                    diff += N[i]
                if diff >= threshold:
                    D[k] = 0
                else:
                    D[k] = threshold - diff
        D[target_id] = 0  # always winning

        file_name = str(rand.random()) + '.lp'
        path = os.path.join(os.getcwd(), "trash", file_name)
        lp.generate_lp_file_dodgson_score(path, N=N, e=e, D=D)
        score = lp.solve_lp_dodgson_score(path)

        lp.remove_lp_file(path)

        if score < min_score:
            min_score = score

    return min_score


def get_effective_num_candidates(election, mode='Borda'):
    """ Compute effective number of candidates """

    c = election.num_candidates
    vectors = election.votes_to_positionwise_vectors()

    if mode == 'Borda':
        scores = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) / (c * (c - 1) / 2) for j in range(c)]
    elif mode == 'Plurality':
        scores = [sum([vectors[j][i] for i in range(1)]) for j in range(c)]

    return 1. / sum([x * x for x in scores])

