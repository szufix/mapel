#!/usr/bin/env python
import logging
from typing import Union

import numpy as np

import mapel.marriages.cultures.euclidean as euclidean
import mapel.marriages.cultures.impartial as impartial
import mapel.marriages.cultures.mallows as mallows
import mapel.marriages.cultures.urn as urn


def generate_votes(model_id: str = None, num_agents: int = None,
                   params: dict = None) -> Union[list, np.ndarray]:
    independent_models = {
        'ic': impartial.generate_ic_votes,
        'id': impartial.generate_id_votes,
        'symmetric': impartial.generate_symmetric_votes,
        'norm-mallows': mallows.generate_norm_mallows_votes,
        'urn': urn.generate_urn_votes,
        'group_ic': impartial.generate_group_ic_votes,
    }
    dependent_models = {
        'malasym': mallows.generate_mallows_asymmetric_votes,
        'asymmetric': impartial.generate_asymmetric_votes,
        'euclidean': euclidean.generate_euclidean_votes,
        'reverse_euclidean': euclidean.generate_reverse_euclidean_votes,
        'mallows_euclidean': euclidean.generate_mallows_euclidean_votes,
        'expectation': euclidean.generate_expectation_votes,
        'attributes': euclidean.generate_attributes_votes,
        'fame': euclidean.generate_fame_votes,
    }

    if model_id in independent_models:
        votes_1 = independent_models.get(model_id)(num_agents=num_agents, params=params)
        votes_2 = independent_models.get(model_id)(num_agents=num_agents, params=params)
        return [votes_1, votes_2]

    elif model_id in dependent_models:
        return dependent_models.get(model_id)(num_agents=num_agents, params=params)

    else:
        logging.warning(f'No such model id: {model_id}')
        return []

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
