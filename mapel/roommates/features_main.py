#!/usr/bin/env python

import mapel.roommates.features.basic_features as basic


# MAPPING #
def get_feature(feature_id):
    return {'summed_rank_minimal_matching': basic.summed_rank_minimal_matching,
            'min_num_bps_matching': basic.min_num_bps_matching,
            'num_of_bps_maximumWeight': basic.num_of_bps_maximumWeight,
            'avg_num_of_bps_for_rand_matching': basic.avg_num_of_bps_for_random_matching,
            }.get(feature_id)


