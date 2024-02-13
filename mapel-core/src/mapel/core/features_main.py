#!/usr/bin/env python

from mapel.core.features.distortion import calculate_distortion, calculate_distortion_naive
from mapel.core.features.monotonicity import calculate_monotonicity, calculate_monotonicity_naive


def get_main_local_feature(feature_id):
    return {}.get(feature_id)


def get_main_global_feature(feature_id):
    return {
        # 'calculate_distortion': calculate_distortion, # needs update
        'distortion': calculate_distortion_naive,
        # 'calculate_monotonicity': calculate_monotonicity, # needs update
        'monotonicity': calculate_monotonicity_naive,
            }.get(feature_id)