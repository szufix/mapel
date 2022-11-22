#!/usr/bin/env python

from collections import Counter
from typing import Union

import logging
import numpy as np

import mapel.elections.cultures.euclidean as euclidean
import mapel.elections.cultures.group_separable as group_separable
import mapel.elections.cultures.guardians as guardians
import mapel.elections.cultures.guardians_plus as guardians_plus
import mapel.elections.cultures.impartial as impartial
import mapel.elections.cultures.mallows as mallows
import mapel.elections.cultures.single_crossing as single_crossing
import mapel.elections.cultures.single_peaked as single_peaked
import mapel.elections.cultures.urn_model as urn_model
import mapel.elections.cultures.partylist as partylist
from mapel.core.glossary import *
from mapel.elections.cultures.preflib import generate_preflib_votes
import mapel.elections.cultures.field_experiment as fe
import mapel.elections.cultures.didi as didi


def generate_approval_votes(culture_id: str = None, num_candidates: int = None,
                            num_voters: int = None, params: dict = None) -> Union[list, np.ndarray]:
    main_models = {'impartial_culture': impartial.generate_approval_ic_votes,
                   'ic': impartial.generate_approval_ic_votes,
                   'id': impartial.generate_approval_id_votes,
                   'resampling': mallows.generate_approval_resampling_votes,
                   'noise': mallows.generate_approval_noise_model_votes,
                   'urn': urn_model.generate_approval_urn_votes,
                   'urn_model': urn_model.generate_approval_urn_votes,
                   'euclidean': euclidean.generate_approval_euclidean_votes,
                   'disjoint_resampling': mallows.generate_approval_disjoint_resamplin_votes,
                   'vcr': euclidean.generate_approval_vcr_votes,
                   'truncated_mallows': mallows.generate_approval_truncated_mallows_votes,
                   'truncated_urn': urn_model.generate_approval_truncated_urn_votes,
                   'moving_resampling': mallows.generate_approval_moving_resampling_votes,
                   'simplex_resampling': mallows.generate_approval_simplex_resampling_votes,
                   'anti_pjr': mallows.approval_anti_pjr_votes,
                   'partylist': partylist.generate_approval_partylist_votes,
                   'urn_partylist': partylist.generate_approval_urn_partylist_votes,
                   'exp_partylist': partylist.generate_approval_exp_partylist_votes,
                   'field': fe.generate_approval_field_votes,
                   }

    if culture_id in main_models:
        return main_models.get(culture_id)(num_voters=num_voters,
                                           num_candidates=num_candidates,
                                           params=params)
    elif culture_id in ['approval_full']:
        return impartial.generate_approval_full_votes(num_voters=num_voters,
                                                      num_candidates=num_candidates)
    elif culture_id in ['approval_empty']:
        return impartial.generate_approval_empty_votes(num_voters=num_voters)

    elif culture_id in APPROVAL_FAKE_MODELS:
        return [culture_id, num_candidates, num_voters, params]
    else:
        logging.warning(f'No such model id: {culture_id}')
        return []


LIST_OF_ORDINAL_MODELS_WITH_PARAMS = {
    '1d_interval': euclidean.generate_ordinal_euclidean_votes,
    '1d_gaussian': euclidean.generate_ordinal_euclidean_votes,
    '1d_one_sided_triangle': euclidean.generate_ordinal_euclidean_votes,
    '1d_full_triangle': euclidean.generate_ordinal_euclidean_votes,
    '1d_two_party': euclidean.generate_ordinal_euclidean_votes,
    '2d_disc': euclidean.generate_ordinal_euclidean_votes,
    '2d_square': euclidean.generate_ordinal_euclidean_votes,
    '2d_gaussian': euclidean.generate_ordinal_euclidean_votes,
    '3d_gaussian': euclidean.generate_ordinal_euclidean_votes,
    '3d_cube': euclidean.generate_ordinal_euclidean_votes,
    '4d_cube': euclidean.generate_ordinal_euclidean_votes,
    '5d_cube': euclidean.generate_ordinal_euclidean_votes,
    '10d_cube': euclidean.generate_ordinal_euclidean_votes,
    '2d_sphere': euclidean.generate_ordinal_euclidean_votes,
    '3d_sphere': euclidean.generate_ordinal_euclidean_votes,
    '4d_sphere': euclidean.generate_ordinal_euclidean_votes,
    '5d_sphere': euclidean.generate_ordinal_euclidean_votes,
    '4d_ball': euclidean.generate_ordinal_euclidean_votes,
    '5d_ball': euclidean.generate_ordinal_euclidean_votes,
    '2d_grid': euclidean.generate_elections_2d_grid,
    'euclidean': euclidean.generate_ordinal_euclidean_votes,
    '1d_gaussian_party': euclidean.generate_1d_gaussian_party,
    '2d_gaussian_party': euclidean.generate_2d_gaussian_party,
    'walsh_party': single_peaked.generate_sp_party,
    'conitzer_party': single_peaked.generate_sp_party,
    'mallows_party': mallows.generate_mallows_party,
    'ic_party': impartial.generate_ic_party,
    'urn_model': urn_model.generate_urn_votes,
    'urn': urn_model.generate_urn_votes,
    'group-separable': group_separable.generate_ordinal_group_separable_votes,
    'single-crossing': single_crossing.generate_ordinal_single_crossing_votes,
    'weighted_stratification': impartial.generate_weighted_stratification_votes,
    'idan_part': guardians_plus.generate_idan_part_votes,
    'idun_part': guardians_plus.generate_idun_part_votes,
    'idst_part': guardians_plus.generate_idst_part_votes,
    'anun_part': guardians_plus.generate_anun_part_votes,
    'anst_part': guardians_plus.generate_anst_part_votes,
    'unst_part': guardians_plus.generate_unst_part_votes,
    'idan_mallows': guardians_plus.generate_idan_mallows_votes,
    'idst_mallows': guardians_plus.generate_idst_mallows_votes,
    'anun_mallows': guardians_plus.generate_anun_mallows_votes,
    'unst_mallows': guardians_plus.generate_unst_mallows_votes,
    'unst_topsize': guardians_plus.generate_unst_topsize_votes,
    'idst_blocks': guardians_plus.generate_idst_blocks_votes,
    'didi': didi.generate_didi_votes,
    'mallows': mallows.generate_mallows_votes,
    'norm-mallows': mallows.generate_mallows_votes,
    'norm-mallows_with_walls': mallows.generate_norm_mallows_with_walls_votes,
    'norm-mallows_mixture': mallows.generate_norm_mallows_mixture_votes,
    'walsh_mallows': single_peaked.generate_walsh_mallows_votes,
    'conitzer_mallows': single_peaked.generate_conitzer_mallows_votes,
    'mallows_triangle': mallows.generate_mallows_votes,
}

LIST_OF_ORDINAL_MODELS_WITHOUT_PARAMS = {
    'impartial_culture': impartial.generate_ordinal_ic_votes,
    'iac': impartial.generate_impartial_anonymous_culture_election,
    'conitzer': single_peaked.generate_ordinal_sp_conitzer_votes,
    'spoc_conitzer': single_peaked.generate_ordinal_spoc_conitzer_votes,
    'walsh': single_peaked.generate_ordinal_sp_walsh_votes,
    'real_identity': guardians.generate_real_identity_votes,
    'real_uniformity': guardians.generate_real_uniformity_votes,
    'real_antagonism': guardians.generate_real_antagonism_votes,
    'real_stratification': guardians.generate_real_stratification_votes,
    'un_from_matrix': guardians_plus.generate_un_from_matrix_votes,
}


def generate_ordinal_votes(culture_id: str = None,
                           num_candidates: int = None,
                           num_voters: int = None,
                           params: dict = None) -> Union[list, np.ndarray]:

    if culture_id in LIST_OF_PREFLIB_MODELS:
        return generate_preflib_votes(culture_id=culture_id,
                                      num_candidates=num_candidates,
                                      num_voters=num_voters,
                                      params=params)

    elif culture_id in LIST_OF_ORDINAL_MODELS_WITH_PARAMS:
        votes = LIST_OF_ORDINAL_MODELS_WITH_PARAMS.get(culture_id)(num_voters=num_voters,
                                                                   num_candidates=num_candidates,
                                                                   params=params)
    elif culture_id in LIST_OF_ORDINAL_MODELS_WITHOUT_PARAMS:
        votes = LIST_OF_ORDINAL_MODELS_WITHOUT_PARAMS.get(culture_id)(num_voters=num_voters,
                                                                      num_candidates=num_candidates)

    elif culture_id in LIST_OF_FAKE_MODELS:
        votes = [culture_id, num_candidates, num_voters, params]

    else:
        votes = []
        logging.warning(f'No such culture id: {culture_id}')

    if culture_id not in LIST_OF_FAKE_MODELS:
        votes = [[int(x) for x in row] for row in votes]

    return np.array(votes)


def store_votes_in_a_file(election, culture_id, num_candidates, num_voters,
                          params, path, ballot, votes=None, aggregated=True):
    """ Store votes in a file """
    if votes is None:
        votes = election.votes

    if params is None:
        params = {}

    with open(path, 'w') as file_:
        if culture_id in NICE_NAME:
            file_.write("# " + NICE_NAME[culture_id] + " " + str(params) + "\n")
        else:
            file_.write("# " + culture_id + " " + str(params) + "\n")

        file_.write(str(num_candidates) + "\n")

        for i in range(num_candidates):
            file_.write(str(i) + ', c' + str(i) + "\n")

        if aggregated:

            c = Counter(map(tuple, votes))
            counted_votes = [[count, list(row)] for row, count in c.items()]
            counted_votes = sorted(counted_votes, reverse=True)

            file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' +
                        str(len(counted_votes)) + "\n")

            if ballot == 'approval':
                for i in range(len(counted_votes)):
                    file_.write(str(counted_votes[i][0]) + ', {')
                    for j in range(len(counted_votes[i][1])):
                        file_.write(str(int(counted_votes[i][1][j])))
                        if j < len(counted_votes[i][1]) - 1:
                            file_.write(", ")
                    file_.write("}\n")

            elif ballot == 'ordinal':
                for i in range(len(counted_votes)):
                    file_.write(str(counted_votes[i][0]) + ', ')
                    for j in range(len(counted_votes[i][1])):
                        file_.write(str(int(counted_votes[i][1][j])))
                        if j < len(counted_votes[i][1]) - 1:
                            file_.write(", ")
                    file_.write("\n")
        else:

            file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' +
                        str(num_voters) + "\n")

            if ballot == 'approval':
                for i in range(len(votes)):
                    file_.write('1, {')
                    for j in range(len(votes[i])):
                        file_.write(str(int(list(votes[i])[j])))
                        if j < len(votes[i]) - 1:
                            file_.write(", ")
                    file_.write("}\n")

            elif ballot == 'ordinal':
                for i in range(len(votes)):
                    file_.write('1, ')
                    for j in range(len(votes[i])):
                        file_.write(str(int(votes[i][j])))
                        if j < len(votes[i]) - 1:
                            file_.write(", ")
                    file_.write("\n")



def approval_votes_to_vectors(votes, num_candidates=None, num_voters=None):
    vectors = np.zeros([num_candidates, num_candidates])
    for vote in votes:
        denom_in = len(vote)
        # print(denom_in)
        denom_out = num_candidates - denom_in
        for i in range(num_candidates):
            if i in vote:
                for j in range(denom_in):
                    vectors[i][j] += 1/denom_in/num_voters
            else:
                for j in range(denom_out):
                    vectors[i][denom_in+j] += 1/denom_out/num_voters
    print(vectors)
    return vectors



def from_approval(culture_id=None, num_candidates=None, num_voters=None, params=None):
    # params['phi'] = np.random.rand()
    # params['p'] = np.random.rand()
    # votes = urn_model.generate_approval_urn_votes(num_candidates=num_candidates, num_voters=num_voters,
    #                                     params=params)

    votes = generate_approval_votes(culture_id=params['culture_id'],
                                    num_candidates=num_candidates, num_voters=num_voters, params=params)

    return approval_votes_to_vectors(votes, num_candidates=num_candidates, num_voters=num_voters)




# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 16.05.2022 #
# # # # # # # # # # # # # # # #
