#!/usr/bin/env python

from mapel.main.features.distortion import calculate_distortion
from mapel.main.features.monotonicity import calculate_monotonicity, calculate_monotonicity_naive


def get_main_local_feature(feature_id):
    return {}.get(feature_id)


def get_main_global_feature(feature_id):
    return {
        'distortion': calculate_distortion,
        # 'calculate_monotonicity': calculate_monotonicity, # needs update
        'monotonicity': calculate_monotonicity_naive,
            }.get(feature_id)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 17.08.2022 #
# # # # # # # # # # # # # # # #
