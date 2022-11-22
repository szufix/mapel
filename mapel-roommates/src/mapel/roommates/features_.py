#!/usr/bin/env python

import mapel.roommates.features.basic_features as basic
from mapel.core.glossary import MAIN_LOCAL_FEATUERS, MAIN_GLOBAL_FEATUERS
from mapel.core.features_main import get_main_local_feature, get_main_global_feature
import numpy as np
from itertools import combinations
from mapel.core.inner_distances import l2


def get_local_feature(feature_id):
    if feature_id in MAIN_LOCAL_FEATUERS:
        return get_main_local_feature(feature_id)

    return {'summed_rank_minimal_matching': basic.summed_rank_minimal_matching,
            'summed_rank_maximal_matching': basic.summed_rank_maximal_matching,
            'minimal_rank_maximizing_matching': basic.minimal_rank_maximizing_matching,
            'min_num_bps_matching': basic.min_num_bps_matching,
            'num_of_bps_min_weight': basic.num_of_bps_maximumWeight,
            'avg_num_of_bps_for_rand_matching': basic.avg_num_of_bps_for_random_matching,
            'mutuality': basic.mutuality,
            'dist_from_id_1': basic.dist_from_id_1,
            'dist_from_id_2': basic.dist_from_id_2,
            }.get(feature_id)


def get_global_feature(feature_id):
    if feature_id in MAIN_GLOBAL_FEATUERS:
        return get_main_global_feature(feature_id)

    return {'monotonicity': monotonicity,
            'distortion_from_all': distortion_from_all,
            }.get(feature_id)


# TMP FUNCS
def monotonicity(experiment, instance) -> float:
    e0 = instance.instance_id
    c0 = np.array(experiment.coordinates[e0])
    distortion = 0
    for e1, e2 in combinations(experiment.instances, 2):
        if e1 != e0 and e2 != e0:
            original_d1 = experiment.distances[e0][e1]
            original_d2 = experiment.distances[e0][e2]
            original_proportion = original_d1 / original_d2
            embedded_d1 = np.linalg.norm(c0 - experiment.coordinates[e1])
            embedded_d2 = np.linalg.norm(c0 - experiment.coordinates[e2])
            embedded_proportion = embedded_d1 / embedded_d2
            _max = max(original_proportion, embedded_proportion)
            _min = min(original_proportion, embedded_proportion)
            distortion += _max / _min
    return distortion

#
def distortion_from_all(experiment, election):
    values = np.array([])
    one_side_values = np.array([])
    election_id_1 = election.instance_id

    for election_id_2 in experiment.instances:
        # if election_id_2 in {'identity_10_100_0', 'uniformity_10_100_0',
        #                      'antagonism_10_100_0', 'stratification_10_100_0'}:
        if election_id_1 != election_id_2:
            # m = experiment.instances[election_id_1].num_candidates
            # print(election_id_1, election_id_2)
            true_distance = experiment.distances[election_id_1][election_id_2]
            true_distance /= experiment.distances['MD']['MA']
            embedded_distance = l2(np.array(experiment.coordinates[election_id_1]),
                                   np.array(experiment.coordinates[election_id_2]))

            embedded_distance /= \
                l2(np.array(experiment.coordinates['MD']),
                   np.array(experiment.coordinates['MA']))
            # try:
            #     ratio = float(embedded_distance) / float(true_distance)
            # except:
            #     ratio = 1.
            one_side_ratio = embedded_distance / true_distance
            one_side_values = np.append(one_side_values, one_side_ratio)

            ratio = max(embedded_distance, true_distance) / min(embedded_distance, true_distance)

            values = np.append(values, ratio)

    # print(min(one_side_values), max(one_side_values))
    # if election_id_1 == 'IC_0':
    #     print(values)

    return np.mean(values)
