#!/usr/bin/env python

import mapel.roommates.features.basic_features as basic


# MAPPING #
def get_feature(feature_id):
    return {'summed_rank_minimal_matching': basic.summed_rank_minimal_matching,
            'min_num_bps_matching': basic.min_num_bps_matching,
            'num_of_bps_maximumWeight': basic.num_of_bps_maximumWeight,
            'avg_num_of_bps_for_rand_matching': basic.avg_num_of_bps_for_random_matching,
            # 'minimal_rank_maximizing_matching': basic.minimal_rank_maximizing_matching,
            'mutuality': basic.mutuality,
            'dist_from_id_1': basic.dist_from_id_1,
            'dist_from_id_2': basic.dist_from_id_2,

            }.get(feature_id)


