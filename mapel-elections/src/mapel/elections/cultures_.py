#!/usr/bin/env python

from typing import Union

import logging

import mapel.elections.cultures.euclidean as euclidean
import mapel.elections.cultures.group_separable as group_separable
import mapel.elections.cultures.guardians as guardians
import mapel.elections.cultures.guardians_plus as guardians_plus
import mapel.elections.cultures.impartial as impartial
import mapel.elections.cultures.mallows as mallows
import mapel.elections.cultures.single_crossing as single_crossing
import mapel.elections.cultures.single_peaked as single_peaked
import mapel.elections.cultures.urn as urn
import mapel.elections.cultures.partylist as partylist
from mapel.core.glossary import *
from mapel.elections.cultures.preflib import generate_preflib_votes
import mapel.elections.cultures.field_experiment as fe
import mapel.elections.cultures.didi as didi
import mapel.elections.cultures.unused as unused
import mapel.elections.cultures.sp_matrices as sp_matrices
import mapel.elections.cultures.resampling as resampling
import mapel.elections.cultures.noise as noise

from mapel.elections.cultures.alliances import *

registered_approval_cultures = {
    'ic': impartial.generate_approval_ic_votes,
    'id': impartial.generate_approval_id_votes,
    'resampling': resampling.generate_approval_resampling_votes,
    'disjoint_resampling': resampling.generate_approval_disjoint_resampling_votes,
    'moving_resampling': resampling.generate_approval_moving_resampling_votes,
    'noise': noise.generate_approval_noise_model_votes,
    'euclidean': euclidean.generate_approval_euclidean_votes,
    'truncated_urn': urn.generate_approval_truncated_urn_votes,
    'urn_partylist': partylist.generate_approval_urn_partylist_votes,
    'full': impartial.generate_approval_full_votes,
    'empty': impartial.generate_approval_empty_votes,

    'vcr': euclidean.generate_approval_vcr_votes,  # unsupported culture
    'field': fe.generate_approval_field_votes,  # unsupported culture
    'truncated_mallows': mallows.generate_approval_truncated_mallows_votes,  # unsupported culture

    'impartial_culture': impartial.generate_approval_ic_votes,  # deprecated name
    'approval_full': impartial.generate_approval_full_votes,  # deprecated name
    'approval_empty': impartial.generate_approval_empty_votes,  # deprecated name
}

registered_ordinal_cultures = {
    'ic': impartial.generate_ordinal_ic_votes,
    'iac': impartial.generate_impartial_anonymous_culture_election,
    'euclidean': euclidean.generate_ordinal_euclidean_votes,
    'urn': urn.generate_urn_votes,
    'group-separable': group_separable.generate_ordinal_group_separable_votes,
    'single-crossing': single_crossing.generate_ordinal_single_crossing_votes,
    'weighted_stratification': unused.generate_weighted_stratification_votes,
    'mallows': mallows.generate_mallows_votes,
    'norm-mallows': mallows.generate_mallows_votes,
    'conitzer': single_peaked.generate_ordinal_sp_conitzer_votes,
    'spoc_conitzer': single_peaked.generate_ordinal_spoc_conitzer_votes,
    'walsh': single_peaked.generate_ordinal_sp_walsh_votes,
    'real_identity': guardians.generate_real_identity_votes,

    'mallows_urn': urn.generate_mallows_urn_votes,
    'idan_part': guardians_plus.generate_idan_part_votes,  # unsupported culture
    'idun_part': guardians_plus.generate_idun_part_votes,  # unsupported culture
    'idst_part': guardians_plus.generate_idst_part_votes,  # unsupported culture
    'anun_part': guardians_plus.generate_anun_part_votes,  # unsupported culture
    'anst_part': guardians_plus.generate_anst_part_votes,  # unsupported culture
    'unst_part': guardians_plus.generate_unst_part_votes,  # unsupported culture
    'idan_mallows': guardians_plus.generate_idan_mallows_votes,  # unsupported culture
    'idst_mallows': guardians_plus.generate_idst_mallows_votes,  # unsupported culture
    'anun_mallows': guardians_plus.generate_anun_mallows_votes,  # unsupported culture
    'unst_mallows': guardians_plus.generate_unst_mallows_votes,  # unsupported culture
    'unst_topsize': guardians_plus.generate_unst_topsize_votes,  # unsupported culture
    'idst_blocks': guardians_plus.generate_idst_blocks_votes,
    'didi': didi.generate_didi_votes,  # unsupported culture
    'norm-mallows_mixture': mallows.generate_norm_mallows_mixture_votes,  # unsupported culture
    'walsh_mallows': sp_matrices.generate_walsh_mallows_votes,  # unsupported culture
    'conitzer_mallows': sp_matrices.generate_conitzer_mallows_votes,  # unsupported culture
    'mallows_triangle': mallows.generate_mallows_votes,  # unsupported culture
    '1d_gaussian_party': euclidean.generate_1d_gaussian_party,  # unsupported culture
    '2d_gaussian_party': euclidean.generate_2d_gaussian_party,  # unsupported culture
    'walsh_party': unused.generate_sp_party,  # unsupported culture
    'conitzer_party': unused.generate_sp_party,  # unsupported culture
    'mallows_party': mallows.generate_mallows_party,  # unsupported culture
    'ic_party': unused.generate_ic_party,  # unsupported culture
    'real_uniformity': guardians.generate_real_uniformity_votes,  # unsupported culture
    'real_antagonism': guardians.generate_real_antagonism_votes,  # unsupported culture
    'real_stratification': guardians.generate_real_stratification_votes,  # unsupported culture
    'un_from_matrix': guardians_plus.generate_un_from_matrix_votes,  # unsupported culture
    'un_from_list': guardians_plus.generate_un_from_list,  # unsupported culture
  
    'impartial_culture': impartial.generate_ordinal_ic_votes,  # deprecated name
    'urn_model': urn.generate_urn_votes,  # deprecated name
}


def generate_approval_votes(culture_id: str = None,
                            num_voters: int = None,
                            num_candidates: int = None,
                            params: dict = None) -> Union[list, np.ndarray]:
    if culture_id in registered_approval_cultures:
        return registered_approval_cultures.get(culture_id)(num_voters, num_candidates, **params)

    else:
        logging.warning(f'No such culture id: {culture_id}')
        return []


def generate_ordinal_votes(culture_id: str = None,
                           num_candidates: int = None,
                           num_voters: int = None,
                           params: dict = None,
                           **kwargs) -> Union[list, np.ndarray]:

    if culture_id in LIST_OF_PREFLIB_MODELS:
        return generate_preflib_votes(culture_id=culture_id,
                                      num_candidates=num_candidates,
                                      num_voters=num_voters,
                                      params=params)

    elif culture_id in registered_ordinal_cultures:
        votes = registered_ordinal_cultures.get(culture_id)(num_voters=num_voters,
                                                            num_candidates=num_candidates,
                                                            **params)

    elif culture_id in LIST_OF_FAKE_MODELS:
        votes = [culture_id, num_candidates, num_voters, params]

    else:
        votes = []
        logging.warning(f'No such culture id: {culture_id}')

    if culture_id not in LIST_OF_FAKE_MODELS:
        votes = [[int(x) for x in row] for row in votes]

    return np.array(votes)


def approval_votes_to_vectors(votes, num_candidates=None, num_voters=None):
    vectors = np.zeros([num_candidates, num_candidates])
    for vote in votes:
        denom_in = len(vote)
        denom_out = num_candidates - denom_in
        for i in range(num_candidates):
            if i in vote:
                for j in range(denom_in):
                    vectors[i][j] += 1 / denom_in / num_voters
            else:
                for j in range(denom_out):
                    vectors[i][denom_in + j] += 1 / denom_out / num_voters
    return vectors


def from_approval(num_candidates=None, num_voters=None, params=None):
    votes = generate_approval_votes(culture_id=params['culture_id'],
                                    num_candidates=num_candidates, num_voters=num_voters,
                                    params=params)

    return approval_votes_to_vectors(votes, num_candidates=num_candidates, num_voters=num_voters)


LIST_OF_ORDINAL_ALLIANCE_MODELS = {
    'ic': generate_ordinal_alliance_ic_votes,
    'urn': generate_ordinal_alliance_urn_votes,
    'euc': generate_ordinal_alliance_euclidean_votes,
    'norm-mallows': generate_ordinal_alliance_norm_mallows_votes,
}


def generate_ordinal_alliance_votes(culture_id: str = None,
                                    num_candidates: int = None,
                                    num_voters: int = None,
                                    params: dict = None):
    if culture_id in LIST_OF_ORDINAL_ALLIANCE_MODELS:
        votes, alliances = LIST_OF_ORDINAL_ALLIANCE_MODELS.get(culture_id)(
            num_voters=num_voters,
            num_candidates=num_candidates,
            params=params)
    else:
        votes = []
        alliances = []
        logging.warning(f'No such culture id: {culture_id}')

    return np.array(votes), alliances


def add_approval_culture(name, function):
    registered_approval_cultures[name] = function


def add_ordinal_culture(name, function):
    registered_ordinal_cultures[name] = function

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 16.05.2022 #
# # # # # # # # # # # # # # # #
