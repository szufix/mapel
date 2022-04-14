#!/usr/bin/env python

from mapel.main.features.distortion import calculate_distortion
from mapel.main.features.monotonicity import calculate_monotonicity


def get_main_local_feature(feature_id):
    return {}.get(feature_id)


def get_main_global_feature(feature_id):

    return {
        'calculate_distortion': calculate_distortion,
        'calculate_monotonicity': calculate_monotonicity,
            }.get(feature_id)
