#!/usr/bin/env python

import math

import networkx as nx
import numpy as np
import scipy.special

import mapel.voting.other.development as dev
import mapel.voting.features.cohesive as cohesive
import mapel.voting.features.scores as scores
from mapel.voting.metrics.inner_distances import l2


# MAPPING #
def get_feature(name):
    return {'borda_std': borda_std,
            'separation': separation,
            'both': both,
            'highest_borda_score': scores.highest_borda_score,
            'highest_plurality_score': scores.highest_plurality_score,
            'highest_copeland_score': scores.highest_copeland_score,
            'lowest_dodgson_score': scores.lowest_dodgson_score,
            'avg_distortion_from_guardians': avg_distortion_from_guardians,
            'worst_distortion_from_guardians': worst_distortion_from_guardians,
            'graph_diameter': graph_diameter,
            'graph_diameter_log': graph_diameter_log,
            'max_approval_score': max_approval_score,
            'largest_cohesive_group': cohesive.count_largest_cohesiveness_level_l_of_cohesive_group,
            'abstract': abstract,
            }.get(name)


def abstract(election):
    n = election.num_voters
    election.votes_to_approvalwise_vector()
    vector = election.approvalwise_vector
    total_value = 0
    for i in range(election.num_candidates):
        k = vector[i] * n
        x = scipy.special.binom(n, k)
        x = math.log(x)
        total_value += x
    return total_value


def borda_std(election):
    all_scores = np.zeros(election.num_candidates)

    vectors = election.votes_to_positionwise_matrix()

    for i in range(election.num_candidates):
        for j in range(election.num_candidates):
            all_scores[i] += vectors[i][j] * (election.num_candidates - j - 1)

    std = np.std(all_scores)
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


def get_effective_num_candidates(election, mode='Borda'):
    """ Compute effective number of candidates """

    c = election.num_candidates
    vectors = election.votes_to_positionwise_matrix()

    if mode == 'Borda':
        all_scores = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) / (c * (c - 1) / 2)
                      for j in range(c)]
    elif mode == 'Plurality':
        all_scores = [sum([vectors[j][i] for i in range(1)]) for j in range(c)]
    else:
        all_scores = []

    return 1. / sum([x * x for x in all_scores])


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
                                       experiment.coordinates[election_id_2])

                embedded_distance /= \
                    l2(experiment.coordinates['identity_10_100_0'],
                       experiment.coordinates['uniformity_10_100_0'])
                ratio = float(true_distance) / float(embedded_distance)
                values = np.append(values, ratio)

    return values


def avg_distortion_from_guardians(experiment, election_id):
    values = distortion_from_guardians(experiment, election_id)
    return np.mean(values)


def worst_distortion_from_guardians(experiment, election_id):
    values = distortion_from_guardians(experiment, election_id)
    return np.max(values)


def graph_diameter(election):
    try:
        return nx.diameter(election.votes)
    except:
        return 100


def graph_diameter_log(election):
    try:
        return math.log(nx.diameter(election.votes))
    except:
        return math.log(100)
##################################


def max_approval_score(election):
    score = np.zeros([election.num_candidates])
    for vote in election.votes:
        for c in vote:
            score[c] += 1
    return max(score)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
