#!/usr/bin/env python

import os

import numpy as np
import random as rand
import math
import itertools
import struct

from mapel.voting.metrics import lp
import mapel.voting.development as dev

from mapel.voting.metrics.inner_distances import l2


# MAPPING #
def get_feature(name):
    return {'borda_std': borda_std,
            'separation': separation,
            'both': both,
            'highest_borda_score': highest_borda_score,
            'highest_plurality_score': highest_plurality_score,
            'highest_copeland_score': highest_copeland_score,
            'lowest_dodgson_score': lowest_dodgson_score,
            'avg_distortion_from_guardians': avg_distortion_from_guardians,
            'worst_distortion_from_guardians': worst_distortion_from_guardians,
            }.get(name)


def borda_std(election):
    scores = np.zeros(election.num_candidates)

    vectors = election.votes_to_positionwise_matrix()

    for i in range(election.num_candidates):
        for j in range(election.num_candidates):
            scores[i] += vectors[i][j] * (election.num_candidates - j - 1)

    std = np.std(scores)
    return std



def separation(election):

    if election.fake:
        return 0

    half = int(election.num_candidates / 2)

    ranking = dev.get_borda_ranking(election.votes, election.num_voters,
                                    election.num_candidates)
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
    vectors = election.get_vectors()
    borda = [sum([vectors[i][pos] * (c - pos - 1) for pos in range(c)])
             for i in range(c)]
    return max(borda) * election.num_voters


def highest_plurality_score(election):
    """ compute highest PLURALITY score of a given election"""
    first_pos = election.get_matrix()[0]
    return max(first_pos) * election.num_voters


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
    n = []
    for pote in potes:
        flag_new = True
        for i, p in enumerate(unique_potes):
            if list(pote) == list(p):
                n[i] += 1
                flag_new = False
        if flag_new:
            unique_potes.append(pote)
            n.append(1)
    return unique_potes, n


def lowest_dodgson_score(election):
    """ compute lowest DODGSON score of a given election """

    min_score = math.inf

    for target_id in range(election.num_candidates):

        # PREPARE N
        unique_potes, N = potes_to_unique_potes(election.potes)

        e = np.zeros([len(N), election.num_candidates,
                      election.num_candidates])

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
    vectors = election.votes_to_positionwise_matrix()

    if mode == 'Borda':
        scores = [sum([vectors[j][i] * (c - i - 1) for i in range(c)])
                  / (c * (c - 1) / 2) for j in range(c)]
    elif mode == 'Plurality':
        scores = [sum([vectors[j][i] for i in range(1)]) for j in range(c)]
    else:
        scores = []

    return 1. / sum([x * x for x in scores])


########################################################################
def map_diameter(c):
    """ Compute the diameter """
    return 1 / 3 * (c + 1) * (c - 1)


def distortion_from_guardians(experiment, election_id):
    values = np.array([])
    election_id_1 = election_id

    for election_id_2 in experiment.elections:
        if election_id_2 in {'identity_10_100_0', 'uniformity_10_100_0',
                             'antagonism_10_100_0', 'stratification_10_100_0'}:
            if election_id_1 != election_id_2:
                m = experiment.elections[election_id_1].num_candidates
                true_distance = \
                    experiment.distances[election_id_1][election_id_2]
                true_distance /= map_diameter(m)
                embedded_distance = l2(experiment.coordinates[election_id_1],
                                       experiment.coordinates[election_id_2],
                                       2)

                embedded_distance /= \
                    l2(experiment.coordinates['identity_10_100_0'],
                       experiment.coordinates['uniformity_10_100_0'], 2)
                ratio = float(true_distance) / float(embedded_distance)
                values = np.append(values, ratio)

    return values


def avg_distortion_from_guardians(experiment, election_id):
    values = distortion_from_guardians(experiment, election_id)
    return np.mean(values)


def worst_distortion_from_guardians(experiment, election_id):
    values = distortion_from_guardians(experiment, election_id)
    return np.max(values)


def banzhaf(rgWeights, fpThold=0.5, normalize=True):
    """ Compute Banzhaf power index """
    n = len(rgWeights)
    wSum = sum(rgWeights)
    wThold = fpThold * wSum
    rgWeights = np.array(rgWeights)
    cDecisive = np.zeros(n, dtype=np.uint64)
    for bitmask in range(0, 2**n-1):
        w = rgWeights * np.unpackbits(np.uint8(struct.unpack('B' * (8), np.uint64(bitmask))), bitorder='little')[:n]
        wSum = sum(w)
        if wSum >= wThold:
            cDecisive = np.add(cDecisive, (w > (wSum - wThold)))
    phi = cDecisive / (2**n) * 2
    if normalize:
        return phi / sum(phi)
    else:
        return phi


def shapley(rgWeights, fpThold=0.5):
    """ Compute Shapley-Shubik power index """
    n = len(rgWeights)
    wSum = sum(rgWeights)
    wThold = fpThold * wSum
    rgWeights = np.array(rgWeights)
    cDecisive = np.zeros(n, dtype=np.uint64)
    for perm in itertools.permutations(range(n)):
        w = rgWeights[list(perm)]
        dec = 0
        wSum = w[dec]
        while wSum < wThold:
            dec += 1
            wSum += w[dec]
        cDecisive[perm[dec]] += 1
    return cDecisive / sum(cDecisive)