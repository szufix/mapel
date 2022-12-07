#!/usr/bin/env python

from typing import Union

import numpy as np

import mapel.roommates.cultures.euclidean as euclidean
import mapel.roommates.cultures.impartial as impartial
import mapel.roommates.cultures.mallows as mallows
import mapel.roommates.cultures.urn as urn
import mapel.roommates.cultures.group_separable as group_separable


def generate_votes(culture_id: str = None, num_agents: int = None,
                   params: dict = None) -> Union[list, np.ndarray]:

    main_models_with_params = {
        'roommates_norm-mallows': mallows.generate_roommates_norm_mallows_votes,
        'roommates_urn': urn.generate_roommates_urn_votes,
        'roommates_euclidean': euclidean.generate_roommates_euclidean_votes,
        'roommates_reverse_euclidean': euclidean.generate_roommates_reverse_euclidean_votes,
        'roommates_gs': group_separable.generate_roommates_gs_votes,
        'roommates_radius': euclidean.generate_fame_votes,  # deprecated
        'fame': euclidean.generate_fame_votes,
        'roommates_double': euclidean.generate_expectation_votes,  # deprecated
        'expectation': euclidean.generate_expectation_votes,
        'roommates_mallows_euclidean': euclidean.generate_roommates_mallows_euclidean_votes,
        'roommates_vectors': euclidean.generate_attributes_votes,  # deprecated
        'attributes': euclidean.generate_attributes_votes,
        'roommates_malasym': mallows.generate_roommates_malasym_votes,
        'roommates_group_ic': impartial.generate_roommates_group_ic_votes,
    }

    main_models_without_params = {
        'roommates_ic': impartial.generate_roommates_ic_votes,
        'roommates_id': impartial.generate_roommates_id_votes,
        'roommates_chaos': impartial.generate_roommates_chaos_votes,
        'roommates_symmetric': impartial.generate_roommates_symmetric_votes,
        'roommates_asymmetric': impartial.generate_roommates_asymmetric_votes,
    }

    if culture_id in main_models_with_params:
        return main_models_with_params.get(culture_id)(num_agents=num_agents, params=params)

    if culture_id in main_models_without_params:
        return main_models_without_params.get(culture_id)(num_agents=num_agents)

    else:
        print("No such election culture_id!", culture_id)
        return []


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 16.03.2022 #
# # # # # # # # # # # # # # # #

