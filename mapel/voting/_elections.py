#!/usr/bin/env python
""" this module is used to generate and import elections"""

from mapel.voting.elections.group_separable import generate_group_separable_election, get_gs_caterpillar_matrix
from mapel.voting.elections.mallows import generate_mallows_election, \
    phi_mallows_helper
from mapel.voting.elections.euclidean import generate_elections_1d_simple, \
    generate_elections_2d_simple, generate_elections_nd_simple
from mapel.voting.elections.single_peaked import generate_conitzer_election, \
    generate_walsh_election, generate_spoc_conitzer_election, get_walsh_matrix, get_conitzer_matrix
from mapel.voting.elections.single_crossing import generate_single_crossing_election, get_single_crossing_matrix
from mapel.voting.elections.impartial import generate_impartial_culture_election, \
    generate_impartial_anonymous_culture_election
from mapel.voting.elections.guardians import generate_real_antagonism_election, \
    generate_real_identity_election, generate_real_stratification_election, generate_real_uniformity_election
from mapel.voting.elections.urn_model import generate_urn_model_election

import os
import random as rand
from collections import Counter

import numpy as np
from scipy.stats import gamma

from .objects.Election import Election
from .objects.Experiment import Experiment, Experiment_xD, Experiment_2D, Experiment_3D

import mapel.voting.elections.preflib as preflib

from mapel.voting.glossary import NICE_NAME, LIST_OF_FAKE_MODELS


def generate_votes(election_model=None, num_candidates=None, num_voters=None, param_1=0, param_2=0):
    if election_model == 'mallows' and int(param_1) == 0:
        param_1 = rand.random()
    elif election_model == 'norm-mallows' and int(param_1) == 0:
        param_1 = phi_mallows_helper(num_candidates)
    elif election_model == 'norm-mallows' and param_1 >= 0:
        # if (num_candidates, param_1) in LOOKUP_TABLE:
        #     param_1 = LOOKUP_TABLE[(num_candidates, param_1)]
        # else:
        param_1 = phi_mallows_helper(num_candidates, rdis=param_1)
    elif election_model == 'urn_model' and int(param_1) == 0:
        param_1 = gamma.rvs(0.8)

    naked_models = {'impartial_culture': generate_impartial_culture_election,
                    'iac': generate_impartial_anonymous_culture_election,
                    'conitzer': generate_conitzer_election,
                    'spoc_conitzer': generate_spoc_conitzer_election,
                    'walsh': generate_walsh_election,
                    'single-crossing': generate_single_crossing_election,
                    'real_identity': generate_real_identity_election,
                    'real_uniformity': generate_real_uniformity_election,
                    'real_antagonism': generate_real_antagonism_election,
                    'real_stratification': generate_real_stratification_election}

    euclidean_models = {'1d_interval': generate_elections_1d_simple,
                        '1d_gaussian': generate_elections_1d_simple,
                        '1d_one_sided_triangle': generate_elections_1d_simple,
                        '1d_full_triangle': generate_elections_1d_simple,
                        '1d_two_party': generate_elections_1d_simple,
                        '2d_disc': generate_elections_2d_simple,
                        '2d_square': generate_elections_2d_simple,
                        '2d_gaussian': generate_elections_2d_simple,
                        '3d_cube': generate_elections_nd_simple,
                        '4d_cube': generate_elections_nd_simple,
                        '5d_cube': generate_elections_nd_simple,
                        '10d_cube': generate_elections_nd_simple,
                        '15d_cube': generate_elections_nd_simple,
                        '20d_cube': generate_elections_nd_simple,
                        '40d_cube': generate_elections_nd_simple,
                        '2d_sphere': generate_elections_2d_simple,
                        '3d_sphere': generate_elections_nd_simple,
                        '4d_sphere': generate_elections_nd_simple,
                        '5d_sphere': generate_elections_nd_simple,
                        '4d_ball': generate_elections_nd_simple,
                        '5d_ball': generate_elections_nd_simple,
                        '40d_ball': generate_elections_nd_simple, }

    single_param_models = {'urn_model': generate_urn_model_election,
                           'group-separable': generate_group_separable_election, }

    double_param_models = {'mallows': generate_mallows_election,
                           'norm-mallows': generate_mallows_election, }

    if election_model in naked_models:
        votes = naked_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates)

    elif election_model in euclidean_models:
        votes = euclidean_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                                     election_model=election_model)

    elif election_model in single_param_models:
        votes = single_param_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                                        param_1=param_1)

    elif election_model in double_param_models:
        votes = double_param_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                                        param_1=param_1, param_2=param_2)
    else:
        votes = []
        print("No such election model!", election_model)

    return votes


def prepare_elections(experiment_id):
    """ Prepare elections for a given experiment """
    experiment = Experiment(experiment_id, raw=True)

    for family_id in experiment.families:
        param_1 = experiment.families[family_id].param_1
        param_2 = experiment.families[family_id].param_2

        election_model = experiment.families[family_id].election_model
        if election_model in preflib.LIST_OF_PREFLIB_MODELS:
            prepare_preflib_family(experiment_id, experiment=experiment, election_model=election_model,
                                   param_1=param_1)
        else:
            prepare_statistical_culture_family(experiment_id, experiment=experiment, election_model=election_model,
                                               family_id=family_id,
                                               param_1=param_1, param_2=param_2)


def extend_elections(experiment_id, folder=None, starting_from=0, ending_at=1000000):
    """ Prepare elections for a given experiment """
    experiment = Experiment(experiment_id, raw=True)

    id_ = 0

    for family_id in experiment.families:
        election_model = experiment.families[family_id].election_model
        param_1 = experiment.families[family_id].param_1
        param_2 = experiment.families[family_id].param_2

        if starting_from <= id_ < ending_at:

            if election_model in preflib.LIST_OF_PREFLIB_MODELS:
                prepare_preflib_family(experiment_id, experiment=experiment, election_model=election_model,
                                       param_1=param_1, id_=id_, folder=folder)
            else:
                prepare_statistical_culture_family(experiment_id, experiment=experiment,
                                                   election_model=election_model,
                                                param_1=param_1, param_2=param_2)

        id_ += experiment.families[family_id].size


# GENERATE
def generate_elections(experiment_id, election_model=None, election_id=None,
                       num_candidates=None, num_voters=None, special=None, second_param=None):
    """ main function: generate elections """

    if election_model in LIST_OF_FAKE_MODELS:
        path = os.path.join("experiments", str(experiment_id), "elections",
                            (str(election_id) + ".soc"))
        file_ = open(path, 'w')
        file_.write('$ fake' + '\n')
        file_.write(str(num_voters) + '\n')
        file_.write(str(num_candidates) + '\n')
        file_.write(str(election_model) + '\n')
        file_.write(str(round(special, 5)) + '\n')
        file_.close()

    else:

        votes = generate_votes(election_model=election_model,
                               num_candidates=num_candidates, num_voters=num_voters,
                               param_1=special, param_2=second_param)

        path = os.path.join("experiments", str(experiment_id), "elections",
                            (str(election_id) + ".soc"))
        file_ = open(path, 'w')

        if election_model in ["norm-mallows", "mallows", "urn_model", 'group-separable']:
            file_.write("# " + NICE_NAME[election_model] + " " + str(round(special, 5)) + "\n")
        elif election_model in NICE_NAME:
            file_.write("# " + NICE_NAME[election_model] + "\n")
        else:
            file_.write("# " + election_model + "\n")

        file_.write(str(num_candidates) + "\n")

        for i in range(num_candidates):
            file_.write(str(i) + ', c' + str(i) + "\n")

        c = Counter(map(tuple, votes))
        counted_votes = [[count, list(row)] for row, count in c.items()]
        counted_votes = sorted(counted_votes, reverse=True)

        file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' + str(len(counted_votes)) + "\n")

        for i in range(len(counted_votes)):
            file_.write(str(counted_votes[i][0]) + ', ')
            for j in range(num_candidates):
                file_.write(str(int(counted_votes[i][1][j])))
                if j < num_candidates - 1:
                    file_.write(", ")
                else:
                    file_.write("\n")

        file_.close()


########################################################################################################################


# IMPORT
def import_election(experiment_id, election_id):
    """ main function: import single election """
    return Election(experiment_id, election_id)


# HELPER FUNCTIONS #
def prepare_preflib_family(experiment_id, experiment=None, election_model=None,
                           param_1=None, id_=None, folder=None):
    # NEEDS UPDATE #

    selection_method = 'random'

    # list of IDs larger than 10
    if election_model == 'irish':
        folder = 'irish_s1'
        # folder = 'irish_f'
        ids = [1, 3]
    elif election_model == 'glasgow':
        folder = 'glasgow_s1'
        # folder = 'glasgow_f'
        ids = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 21]
    elif election_model == 'formula':
        folder = 'formula_s1'
        # 17 races or more
        ids = [17, 35, 37, 40, 41, 42, 44, 45, 46, 47, 48]
    elif election_model == 'skate':
        folder = 'skate_ic'
        # 9 judges
        ids = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48]
    elif election_model == 'sushi':
        folder = 'sushi_ff'
        ids = [1]
    elif election_model == 'grenoble':
        folder = 'grenoble_ff'
        ids = [1]
    elif election_model == 'tshirt':
        folder = 'tshirt_ff'
        ids = [1]
    elif election_model == 'cities_survey':
        folder = 'cities_survey_s1'
        ids = [1, 2]
    elif election_model == 'aspen':
        folder = 'aspen_s1'
        ids = [1]
    elif election_model == 'marble':
        folder = 'marble_ff'
        ids = [1, 2, 3, 4, 5]
    elif election_model == 'cycling_tdf':
        folder = 'cycling_tdf_s1'
        # ids = [e for e in range(1, 69+1)]
        selection_method = 'random'
        ids = [14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]
    elif election_model == 'cycling_gdi':
        folder = 'cycling_gdi_s1'
        ids = [i for i in range(2, 23 + 1)]
    elif election_model == 'ers':
        folder = 'ers_s1'
        # folder = 'ers_f'
        # 500 voters or more
        ids = [3, 9, 23, 31, 32, 33, 36, 38, 40, 68, 77, 79, 80]
    elif election_model == 'ice_races':
        folder = 'ice_races_s1'
        # 80 voters or more
        ids = [4, 5, 8, 9, 15, 20, 23, 24, 31, 34, 35, 37, 43, 44, 49]
    else:
        ids = []

    rand_ids = rand.choices(ids, k=experiment.families[election_model].size)
    for ri in rand_ids:
        election_id = "core_" + str(id_)
        tmp_election_type = election_model + '_' + str(ri)

        preflib.generate_elections_preflib(experiment_id, election_model=tmp_election_type, elections_id=election_id,
                                           num_voters=experiment.families[election_model].num_voters,
                                           num_candidates=experiment.families[election_model].num_candidates,
                                           special=param_1,
                                           folder=folder, selection_method=selection_method)
        id_ += 1


def prepare_statistical_culture_family(experiment_id, experiment=None, election_model=None, family_id=None,
                                       param_1=None, param_2=None):
    copy_param_1 = param_1

    for j in range(experiment.families[family_id].size):

        if election_model in {'unid', 'stan', 'anid', 'stid', 'anun', 'stun'}:
            if copy_param_1 == 0:  # with both
                param_1 = j / (experiment.families[family_id].size - 1)
            elif copy_param_1 == 1:  # without first (which is last)
                param_1 = j / experiment.families[family_id].size
            elif copy_param_1 == 2:  # without second (which is first)
                param_1 = (j + 1) / experiment.families[family_id].size
            elif copy_param_1 == 4:  # without both
                param_1 = (j + 1) / (experiment.families[family_id].size + 1)
        elif election_model == 'urn_model' and copy_param_1 == -2.:
            param_1 = round(j / 10000., 2)

        election_id = family_id + '_' + str(j)
        generate_elections(experiment_id, election_model=election_model, election_id=election_id,
                           num_voters=experiment.families[family_id].num_voters,
                           num_candidates=experiment.families[family_id].num_candidates,
                           special=param_1, second_param=param_2)
