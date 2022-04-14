#!/usr/bin/env python

import mapel.roommates.features.basic_features as basic
from mapel.main._glossary import MAIN_LOCAL_FEATUERS, MAIN_GLOBAL_FEATUERS
from mapel.main._features_main import get_main_local_feature, get_main_global_feature


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

    return {}.get(feature_id)
