#!/usr/bin/env python
""" this module is used to generate and import elections"""

from mapel.voting.elections.group_separable import generate_group_separable_election

from mapel.voting.elections.mallows import generate_mallows_election, \
    phi_from_relphi, generate_mallows_party, generate_approval_mallows_election

from mapel.voting.elections.euclidean import generate_elections_1d_simple, \
    generate_elections_2d_simple, generate_elections_nd_simple, \
    generate_elections_2d_grid, generate_2d_gaussian_party, get_rand, \
    generate_1d_gaussian_party, \
    generate_approval_2d_disc_elections, generate_approval_1d_interval_elections

from mapel.voting.elections.single_peaked import generate_conitzer_election, \
    generate_walsh_election, generate_spoc_conitzer_election, \
    generate_sp_party
from mapel.voting.elections.single_crossing import \
    generate_single_crossing_election
from mapel.voting.elections.impartial import generate_impartial_culture_election, \
    generate_impartial_anonymous_culture_election, generate_ic_party, \
    generate_approval_ic_election, generate_approval_id_election

from mapel.voting.elections.guardians import \
    generate_real_antagonism_election, \
    generate_real_identity_election, generate_real_stratification_election, \
    generate_real_uniformity_election
from mapel.voting.elections.urn_model import generate_urn_model_election

import os
import random as rand
from collections import Counter
import copy

from scipy.stats import gamma

from mapel.voting.objects.Election import Election

import mapel.voting.elections.preflib as preflib

from mapel.voting.glossary import NICE_NAME, LIST_OF_FAKE_MODELS, PATHS, PARTY_MODELS, \
    APPROVAL_MODELS, GRAPH_MODELS

import networkx as nx


def generate_graph(election_model=None, num_voters=None, num_candidates=None, params=None):

    if election_model == 'erdos_renyi_graph':
        return nx.erdos_renyi_graph(params['n'], params['p'])
    elif election_model == 'watts_strogatz_graph':
        return nx.watts_strogatz_graph(params['n'], params['k'], params['p'])
    elif election_model == 'barabasi_albert_graph':
        return nx.barabasi_albert_graph(params['n'], params['m'])
    elif election_model == 'random_geometric_graph':
        return nx.random_geometric_graph(params['n'], params['radius'])
    elif election_model == 'random_tree':
        return nx.random_tree(params['n'])



def generate_approval_votes(election_model=None, num_candidates=None, num_voters=None, params=None):

    euclidean_models = {'approval_2d_disc': generate_approval_2d_disc_elections,
                        'approval_1d_interval': generate_approval_1d_interval_elections}

    models = {'approval_ic': generate_approval_ic_election,
              'approval_mallows': generate_approval_mallows_election,
              'approval_id': generate_approval_id_election,}

    if election_model in models:
        votes = models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                           params=params)
    elif election_model in euclidean_models:
        votes = euclidean_models.get(election_model)(election_model=election_model,
                                           num_voters=num_voters, num_candidates=num_candidates)

    else:
        votes = []
        print("No such election model!", election_model)

    return votes


def generate_ordinal_votes(election_model=None, num_candidates=None, num_voters=None, params=None):

    if params is None:
        params = {}

    naked_models = {'impartial_culture': generate_impartial_culture_election,
                    'iac': generate_impartial_anonymous_culture_election,
                    'conitzer': generate_conitzer_election,
                    'spoc_conitzer': generate_spoc_conitzer_election,
                    'walsh': generate_walsh_election,
                    'single-crossing': generate_single_crossing_election,
                    'real_identity': generate_real_identity_election,
                    'real_uniformity': generate_real_uniformity_election,
                    'real_antagonism': generate_real_antagonism_election,
                    'real_stratification':
                        generate_real_stratification_election}

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
                        '40d_ball': generate_elections_nd_simple,
                        '2d_grid': generate_elections_2d_grid}

    party_models = {'1d_gaussian_party': generate_1d_gaussian_party,
                    '2d_gaussian_party': generate_2d_gaussian_party,
                    'walsh_party': generate_sp_party,
                    'conitzer_party': generate_sp_party,
                    'mallows_party': generate_mallows_party,
                    'ic_party': generate_ic_party
                    }

    single_param_models = {'urn_model': generate_urn_model_election,
                           'group-separable':
                               generate_group_separable_election}

    double_param_models = {'mallows': generate_mallows_election,
                           'norm-mallows': generate_mallows_election, }

    if election_model in naked_models:
        votes = naked_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates)

    elif election_model in euclidean_models:
        votes = euclidean_models.get(election_model)(
            num_voters=num_voters, num_candidates=num_candidates,
            election_model=election_model)

    elif election_model in party_models:
        votes = party_models.get(election_model)(
            num_voters=num_voters, num_candidates=num_candidates,
            election_model=election_model, params=params)

    elif election_model in single_param_models:
        votes = single_param_models.get(election_model)(
            num_voters=num_voters, num_candidates=num_candidates,
            params=params)

    elif election_model in double_param_models:
        votes = double_param_models.get(election_model)(num_voters, num_candidates, params)

    elif election_model in LIST_OF_FAKE_MODELS:
        votes = [election_model, num_candidates, num_voters, params]
    else:
        votes = []
        print("No such election model!", election_model)

    if election_model not in LIST_OF_FAKE_MODELS:
        # print(votes)
        votes = [[int(x) for x in row] for row in votes]

    return votes

# def generate_family(**kwargs):
#
#     votes = generate_votes(**kwargs)
#     election = Family("virtual", "virtual", votes=votes)
#     return election


# deprecated
# needs update
# def extend_elections(experiment_id, folder=None, starting_from=0,
# ending_at=1000000):
#     """ Prepare elections for a given experiment """
#     experiment = Experiment(experiment_id, raw=True)
#
#     id_ = 0
#
#     for family_id in experiment.families:
#         election_model = experiment.families[family_id].election_model
#         param_1 = experiment.families[family_id].param_1
#         param_2 = experiment.families[family_id].param_2
#
#         if starting_from <= id_ < ending_at:
#
#             if election_model in preflib.LIST_OF_PREFLIB_MODELS:
#                 prepare_preflib_family(experiment_id, experiment=experiment,
#                 election_model=election_model,
#                                        param_1=param_1, id_=id_,
#                                        folder=folder)
#             else:
#                 prepare_statistical_culture_family(experiment_id,
#                 experiment=experiment,
#                                 election_model=election_model,
#                       param_1=param_1, param_2=param_2)
#
#         id_ += experiment.families[family_id].size


# GENERATE
def generate_elections(experiment=None, election_model=None, election_id=None, num_candidates=None,
                       num_voters=None, params=None, ballot='ordinal'):

    """ main function: generate elections """

    if params is None:
        params = {}

    if election_model == 'mallows' and params['phi'] is None:
        params['phi'] = rand.random()
    elif election_model == 'norm-mallows' and params['norm-phi'] is None:
        params['norm-phi'] = rand.random()
    elif election_model == 'urn_model' and params['alpha'] is None:
        params['alpha'] = gamma.rvs(0.8)

    if election_model == 'norm-mallows':
        params['phi'] = phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    if election_model == 'mallows_matrix_path':
        params['norm-phi'] = params['alpha']
        params['phi'] = phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    if 'weight' not in params:
        params['weight'] = 0.

    if ballot == 'ordinal':
        votes = generate_ordinal_votes(election_model=election_model,
                               num_candidates=num_candidates,
                               num_voters=num_voters, params=params)
    elif ballot == 'approval':
        votes = generate_approval_votes(election_model=election_model,
                               num_candidates=num_candidates,
                               num_voters=num_voters, params=params)

    elif ballot == 'graph':
        votes = generate_graph(election_model=election_model,
                                        num_candidates=num_candidates,
                                        num_voters=num_voters, params=params)

    else:
        print("Such ballot does not exist!")
        votes = []

    election = Election("virtual", "virtual", votes=votes,
                        election_model=election_model,
                        num_candidates=num_candidates,
                        num_voters=num_voters,
                        ballot=ballot)

    experiment.elections[election_id] = election

    # For now, storing works only for ordinal ballot
    if experiment.store:

        if election_model in LIST_OF_FAKE_MODELS:
            path = os.path.join("experiments", str(experiment.experiment_id),
                                "elections", (str(election_id) + ".soc"))
            file_ = open(path, 'w')
            file_.write('$ fake' + '\n')
            file_.write(str(num_voters) + '\n')
            file_.write(str(num_candidates) + '\n')
            file_.write(str(election_model) + '\n')
            if election_model == 'norm-mallows_matrix':
                file_.write(str(round(params['norm-phi'], 5)) + '\n')
            elif election_model in PATHS:
                file_.write(str(round(params['alpha'], 5)) + '\n')
                if election_model == 'mallows_matrix_path':
                    file_.write(str(round(params['weight'], 5)) + '\n')
            file_.close()

        else:

            votes = experiment.elections[election_id].votes
            path = os.path.join("experiments",
                                str(experiment.experiment_id), "elections",
                                (str(election_id) + ".soc"))
            with open(path, 'w') as file_:

                if election_model in {'urn_model'}:
                    file_.write("# " + NICE_NAME[election_model] + " " +
                                str(round(params['alpha'], 5)) + "\n")
                elif election_model in ["mallows"]:
                    file_.write("# " + NICE_NAME[election_model] + " " +
                                str(round(params['phi'], 5)) + "\n")
                elif election_model in ["norm-mallows"]:
                    file_.write("# " + NICE_NAME[election_model] + " " +
                                str(round(params['norm-phi'], 5)) + "\n")
                elif election_model in ['group-separable']:
                    try:
                        file_.write("# " + NICE_NAME[election_model] + " " +
                                    str(round(params['param_1'], 5)) + "\n")
                    except Exception:
                        pass

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

                file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' +
                            str(len(counted_votes)) + "\n")

                for i in range(len(counted_votes)):
                    file_.write(str(counted_votes[i][0]) + ', ')
                    for j in range(num_candidates):
                        file_.write(str(int(counted_votes[i][1][j])))
                        if j < num_candidates - 1:
                            file_.write(", ")
                        else:
                            file_.write("\n")

    return experiment.elections[election_id]


###############################################################################

# IMPORT
def import_election(experiment_id, election_id):
    """ main function: import single election """
    return Election(experiment_id, election_id)


# HELPER FUNCTIONS #
def prepare_preflib_family(experiment=None, election_model=None, params=None,
                           id_=None, folder=None):
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
        ids = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
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

        preflib.generate_elections_preflib(
            experiment=experiment, election_model=tmp_election_type,
            elections_id=election_id,
            num_voters=experiment.families[election_model].num_voters,
            num_candidates=experiment.families[election_model].num_candidates,
            special=params,
            folder=folder, selection_method=selection_method)
        id_ += 1


def _get_params_for_crate(j):
    base = []
    my_size = 10
    # with_edge
    for p in range(my_size):
        for q in range(my_size):
            for r in range(my_size):
                a = p / (my_size - 1)
                b = q / (my_size - 1)
                c = r / (my_size - 1)
                d = 1 - a - b - c
                tmp = [a, b, c, d]
                if d >= 0 and sum(tmp) == 1:
                    base.append(tmp)
    params = {'alpha': base[j]}
    return params


def _get_params_for_paths(experiment, family_id, j, copy_param_1=4):

    params = {}
    if copy_param_1 == 0:  # with both
        params['alpha'] = j / (experiment.families[family_id].size - 1)
    elif copy_param_1 == 1:  # without first (which is last)
        params['alpha'] = j / experiment.families[family_id].size
    elif copy_param_1 == 2:  # without second (which is first)
        params['alpha'] = (j + 1) / experiment.families[family_id].size
    elif copy_param_1 == 4:  # without both
        params['alpha'] = (j + 1) / (experiment.families[family_id].size + 1)

    return params


def prepare_parties(experiment=None, election_model=None,
                    family_id=None, params=None):
    parties = []

    if election_model == '2d_gaussian_party':
        for i in range(params['num_parties']):
            point = [rand.random(), rand.random()]
            parties.append(point)

    elif election_model in ['1d_gaussian_party',
                            'conitzer_party', 'walsh_party']:
        for i in range(params['num_parties']):
            point = [rand.random()]
            parties.append(point)

    return parties


def prepare_statistical_culture_family(experiment=None, election_model=None, family_id=None,
                                       params=None):
    keys = []
    ballot = 'ordinal'

    if election_model in APPROVAL_MODELS:
        ballot = 'approval'

    if election_model in GRAPH_MODELS:
        ballot = 'graph'

    if election_model in PARTY_MODELS:
        params['party'] = prepare_parties(experiment=experiment, params=params,
                                          election_model=election_model, family_id=family_id)

    for j in range(experiment.families[family_id].size):

        if election_model in PATHS:
            new_params = _get_params_for_paths(experiment, family_id, j)
            params = {**params, **new_params}

        if election_model in {'crate'}:
            new_params = _get_params_for_crate(j)
            params = {**params, **new_params}

        if experiment.families[family_id].single_election:
            election_id = family_id
        else:
            election_id = family_id + '_' + str(j)

        generate_elections(
            experiment=experiment, election_model=election_model,
            election_id=election_id, num_voters=experiment.families[family_id].num_voters,
            num_candidates=experiment.families[family_id].num_candidates,
            params=copy.deepcopy(params), ballot=ballot)

        keys.append(election_id)

    return keys
