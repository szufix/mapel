#!/usr/bin/env python

import logging
from typing import Union


import mapel.elections.cultures.to_be_removed.euclidean as euclidean
import mapel.elections.cultures.to_be_removed.group_separable as group_separable
import mapel.elections.cultures.to_be_removed.impartial as impartial

import mapel.elections.cultures.didi as didi
import mapel.elections.cultures.field_experiment as fe
import mapel.elections.cultures.guardians as guardians
import mapel.elections.cultures.guardians_plus as guardians_plus
import mapel.elections.cultures.mallows as mallows
import mapel.elections.cultures.partylist as partylist
import mapel.elections.cultures.matrices.single_peaked_matrices as sp_matrices
import mapel.elections.cultures.unused as unused
import mapel.elections.cultures.urn as urn
from mapel.core.glossary import *
from mapel.elections.cultures.alliances import *
from mapel.elections.cultures.preflib import generate_preflib_votes

import prefsampling.ordinal as pref_ordinal
import prefsampling.approval as pref_approval

registered_approval_cultures = {
    'impartial': pref_approval.impartial,
    'impartial_culture': pref_approval.impartial,
    'ic': pref_approval.impartial,
    'id': pref_approval.identity,
    'resampling': pref_approval.resampling,
    'disjoint_resampling': pref_approval.disjoint_resampling,
    'moving_resampling': pref_approval.moving_resampling,
    'noise': pref_approval.noise,
    'euclidean': pref_approval.euclidean,

    'full': impartial.generate_approval_full_votes,
    'empty': impartial.generate_approval_empty_votes,

    'truncated_urn': urn.generate_approval_truncated_urn_votes,
    'urn_partylist': partylist.generate_approval_urn_partylist_votes,

    'vcr': euclidean.generate_approval_vcr_votes,  # unsupported culture
    'field': fe.generate_approval_field_votes,  # unsupported culture
    'truncated_mallows': mallows.generate_approval_truncated_mallows_votes,  # unsupported culture

    'approval_full': impartial.generate_approval_full_votes,  # deprecated name
    'approval_empty': impartial.generate_approval_empty_votes,  # deprecated name
}

registered_ordinal_cultures = {

    'impartial': pref_ordinal.impartial,
    'impartial_culture': pref_ordinal.impartial,
    'ic': pref_ordinal.impartial,
    'iac': pref_ordinal.impartial_anonymous,
    'euclidean': pref_ordinal.euclidean,
    'urn': pref_ordinal.urn,
    'single-crossing': pref_ordinal.single_crossing,
    'conitzer': pref_ordinal.single_peaked_conitzer,
    'walsh': pref_ordinal.single_peaked_walsh,
    'spoc': pref_ordinal.single_peaked_circle,
    'weighted_stratification': pref_ordinal.stratification,
    'stratification': pref_ordinal.stratification,
    'mallows': pref_ordinal.mallows,

    'norm-mallows': mallows.generate_mallows_votes,
    'real_identity': guardians.generate_real_identity_votes,

    'group-separable': group_separable.generate_ordinal_group_separable_votes,

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

    'urn_model': pref_ordinal.urn,  # deprecated name
}


def generate_approval_votes(culture_id: str = None,
                            num_voters: int = None,
                            num_candidates: int = None,
                            params: dict = None) -> Union[list, np.ndarray]:
    """
    Generates approval votes according to the given culture id.

    :param culture_id: name of the culture.
    :param num_voters: number of the voters.
    :param num_candidates: number of the candidates.
    :param params: culture parameters.
    :return: a list of sets of approved candidates.
    """
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
    """
    Generates approval votes according to the given culture id.

    :param culture_id: name of the culture.
    :param num_voters: number of the voters.
    :param num_candidates: number of the candidates.
    :param params: culture parameters.
    :param kwargs: additional arguments.
    :return: array of ordinal votes.
    """

    if culture_id in LIST_OF_PREFLIB_MODELS:
        try:
            votes = generate_preflib_votes(culture_id=culture_id,
                                           num_candidates=num_candidates,
                                           num_voters=num_voters,
                                           params=params)
        except:
            votes = []
            logging.warning(
                f'You are trying to create an election based on Preflib '
                f'without having the original source election. '
                f'Please use different culture_id than: {culture_id}')

    elif culture_id in registered_ordinal_cultures:
        votes = registered_ordinal_cultures.get(culture_id)(num_voters=num_voters,
                                                            num_candidates=num_candidates,
                                                            **params)

    elif culture_id in LIST_OF_FAKE_MODELS:
        votes = [culture_id, num_candidates, num_voters, params]

    else:
        votes = []
        logging.warning(
            f'No such culture id: {culture_id} \n'
            f'If you are using your own instances then ignore this warning.')

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


def from_approval(num_candidates: int = None,
                  num_voters: int = None,
                  params: dict = None):
    votes = generate_approval_votes(culture_id=params['culture_id'],
                                    num_candidates=num_candidates, num_voters=num_voters,
                                    params=params)

    return approval_votes_to_vectors(votes, num_candidates=num_candidates, num_voters=num_voters)


LIST_OF_ORDINAL_ALLIANCE_MODELS = {
    'ic': generate_ordinal_alliance_ic_votes,
    'urn': generate_ordinal_alliance_urn_votes,
    'euc': generate_ordinal_alliance_euclidean_votes,
    'allied_euc': generate_ordinal_alliance_allied_euclidean_votes,
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
    """
    Adds a new approval culture to the list of available approval cultures.

    :param name: name of the culture.
    :param function: function that generates the votes.
    :return: None.
    """
    registered_approval_cultures[name] = function


def add_ordinal_culture(name, function):
    """
    Adds a new ordinal culture to the list of available ordinal cultures.

    :param name: name of the culture.
    :param function: function that generates the votes.
    :return: None.
    """
    registered_ordinal_cultures[name] = function
