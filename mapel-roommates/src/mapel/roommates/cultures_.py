#!/usr/bin/env python

from typing import Union

import numpy as np

import mapel.roommates.cultures.euclidean as euclidean
import mapel.roommates.cultures.impartial as impartial
import mapel.roommates.cultures.mallows as mallows
import mapel.roommates.cultures.urn as urn


registered_roommates_culture = {
    'ic': impartial.generate_roommates_ic_votes,
    'id': impartial.generate_roommates_id_votes,
    'chaos': impartial.generate_roommates_chaos_votes,
    'symmetric': impartial.generate_roommates_symmetric_votes,
    'asymmetric': impartial.generate_roommates_asymmetric_votes,
    'urn': urn.generate_roommates_urn_votes,
    'fame': euclidean.generate_roommates_fame_votes,
    'expectation': euclidean.generate_roommates_expectation_votes,
    'attributes': euclidean.generate_roommates_attributes_votes,
    'euclidean': euclidean.generate_roommates_euclidean_votes,
    'reverse_euclidean': euclidean.generate_roommates_reverse_euclidean_votes,
    'group_ic': impartial.generate_roommates_group_ic_votes,
    'norm-mallows': mallows.generate_roommates_norm_mallows_votes,
    'mallows_euclidean': euclidean.generate_roommates_mallows_euclidean_votes,
    'malasym': mallows.generate_roommates_malasym_votes,
}


def generate_votes(culture_id: str = None, num_agents: int = None,
                   params: dict = None) -> Union[list, np.ndarray]:

    if culture_id in registered_roommates_culture:
        return registered_roommates_culture.get(culture_id)(num_agents=num_agents, **params)

    else:
        print("No such election culture_id!", culture_id)
        return []


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON:  9.06.2023 #
# # # # # # # # # # # # # # # #

