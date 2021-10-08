#!/usr/bin/env python
""" this module is used to generate and import elections"""

from mapel.voting.elections.group_separable import generate_group_separable_election

from mapel.voting.elections.mallows import generate_mallows_election, \
    phi_from_relphi, generate_mallows_party, generate_approval_mallows_election, \
    generate_approval_raw_mallows_election, generate_approval_disjoint_mallows_election

from mapel.voting.elections.euclidean import generate_elections_1d_simple, \
    generate_elections_2d_simple, generate_elections_nd_simple, \
    generate_elections_2d_grid, generate_2d_gaussian_party, \
    generate_1d_gaussian_party, \
    generate_approval_euclidean_election

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

from mapel.voting.elections.urn_model import generate_urn_model_election, \
    generate_approval_urn_election

import os
import random as rand
import numpy as np
from collections import Counter
import copy

from scipy.stats import gamma

from mapel.voting.objects.Election import Election
from mapel.voting.objects.OrdinalElection import OrdinalElection
from mapel.voting.objects.ApprovalElection import ApprovalElection
from mapel.voting.objects.Graph import Graph

import mapel.voting.elections.preflib as preflib

from mapel.voting.glossary import NICE_NAME, LIST_OF_FAKE_MODELS, PATHS, PARTY_MODELS, \
    APPROVAL_MODELS, GRAPH_MODELS, APPROVAL_FAKE_MODELS

import networkx as nx


def generate_graph(model=None, num_nodes=None, params=None):
    non_params_graphs = {'cycle_graph': nx.cycle_graph,
                         'wheel_graph': nx.wheel_graph,
                         'star_graph': nx.star_graph,
                         'ladder_graph': nx.ladder_graph,
                         'circular_ladder_graph': nx.circular_ladder_graph,
                         'random_tree': nx.random_tree,
                         }

    if model in non_params_graphs:
        return non_params_graphs[model](num_nodes)

    elif model in ['erdos_renyi_graph', 'erdos_renyi_graph_path']:
        return nx.erdos_renyi_graph(num_nodes, params['p'])
    elif model == 'watts_strogatz_graph':
        return nx.watts_strogatz_graph(num_nodes, params['k'], params['p'])
    elif model == 'barabasi_albert_graph':
        return nx.barabasi_albert_graph(num_nodes, params['m'])
    elif model == 'random_geometric_graph':
        return nx.random_geometric_graph(num_nodes, params['radius'])


def generate_approval_votes(model=None, num_candidates=None, num_voters=None, params=None):

    models = {'approval_ic': generate_approval_ic_election,
              'approval_id': generate_approval_id_election,
              'approval_mallows': generate_approval_mallows_election,
              'approval_raw_mallows': generate_approval_raw_mallows_election,
              'approval_urn': generate_approval_urn_election,
              'approval_euclidean': generate_approval_euclidean_election,
              'approval_disjoint_mallows': generate_approval_disjoint_mallows_election,
              'approval_id_0.5': generate_approval_id_election,
              'approval_ic_0.5': generate_approval_ic_election,
              }

    if model in models:
        votes = models.get(model)(num_voters=num_voters, num_candidates=num_candidates,
                                  params=params)
    elif model in APPROVAL_FAKE_MODELS:
        votes = []
    else:
        votes = []
        print("No such election model!", model)

    return votes


def generate_ordinal_votes(model=None, num_candidates=None, num_voters=None, params=None):
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

    if model in naked_models:
        votes = naked_models.get(model)(num_voters=num_voters,
                                                 num_candidates=num_candidates)

    elif model in euclidean_models:
        votes = euclidean_models.get(model)(num_voters=num_voters,
                                                     num_candidates=num_candidates,
                                                     model=model)

    elif model in party_models:
        votes = party_models.get(model)(num_voters=num_voters,
                                                 num_candidates=num_candidates,
                                                 model=model, params=params)

    elif model in single_param_models:
        votes = single_param_models.get(model)(num_voters=num_voters,
                                                        num_candidates=num_candidates,
                                                        params=params)

    elif model in double_param_models:
        votes = double_param_models.get(model)(num_voters, num_candidates, params)

    elif model in LIST_OF_FAKE_MODELS:
        votes = [model, num_candidates, num_voters, params]
    else:
        votes = []
        print("No such election model!", model)

    if model not in LIST_OF_FAKE_MODELS:
        # print(votes)
        votes = [[int(x) for x in row] for row in votes]

    return votes


# STORE
def store_ordinal_instances(experiment, model, name, num_candidates, num_voters, params):

        if model in LIST_OF_FAKE_MODELS:
            path = os.path.join("experiments", str(experiment.experiment_id),
                                "elections", (str(name) + ".soc"))
            file_ = open(path, 'w')
            file_.write('$ fake' + '\n')
            file_.write(str(num_voters) + '\n')
            file_.write(str(num_candidates) + '\n')
            file_.write(str(model) + '\n')
            if model == 'norm-mallows_matrix':
                file_.write(str(round(params['norm-phi'], 5)) + '\n')
            elif model in PATHS:
                file_.write(str(round(params['alpha'], 5)) + '\n')
                if model == 'mallows_matrix_path':
                    file_.write(str(round(params['weight'], 5)) + '\n')
            file_.close()

        else:

            votes = experiment.elections[name].votes
            path = os.path.join("experiments",
                                str(experiment.experiment_id), "elections",
                                (str(name) + ".soc"))
            with open(path, 'w') as file_:

                if model in {'urn_model'}:
                    file_.write("# " + NICE_NAME[model] + " " +
                                str(round(params['alpha'], 5)) + "\n")
                elif model in ["mallows"]:
                    file_.write("# " + NICE_NAME[model] + " " +
                                str(round(params['phi'], 5)) + "\n")
                elif model in ["norm-mallows"]:
                    file_.write("# " + NICE_NAME[model] + " " +
                                str(round(params['norm-phi'], 5)) + "\n")
                elif model in ['group-separable']:
                    try:
                        file_.write("# " + NICE_NAME[model] + " " +
                                    str(round(params['param_1'], 5)) + "\n")
                    except Exception:
                        pass

                elif model in NICE_NAME:
                    file_.write("# " + NICE_NAME[model] + "\n")
                else:
                    file_.write("# " + model + "\n")

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


def store_approval_instances(experiment, model, name, num_candidates, num_voters, params):

    if model in APPROVAL_FAKE_MODELS:
        path = os.path.join("experiments", str(experiment.experiment_id),
                            "instances", (str(name) + ".app"))
        file_ = open(path, 'w')
        file_.write('$ fake' + '\n')
        file_.write(str(num_voters) + '\n')
        file_.write(str(num_candidates) + '\n')
        file_.write(str(model) + '\n')
        # if model == 'norm-mallows_matrix':
        #     file_.write(str(round(params['norm-phi'], 5)) + '\n')
        # elif model in PATHS:
        #     file_.write(str(round(params['alpha'], 5)) + '\n')
        #     if model == 'mallows_matrix_path':
        #         file_.write(str(round(params['weight'], 5)) + '\n')
        file_.close()

    else:
        votes = experiment.instances[name].votes
        path = os.path.join("experiments",
                            str(experiment.experiment_id), "instances",
                            (str(name) + ".app"))
        with open(path, 'w') as file_:

            if model in NICE_NAME:
                file_.write("# " + NICE_NAME[model] + " " +
                            str(params) + "\n")
            else:
                file_.write("# " + model + "\n")

            file_.write(str(num_candidates) + "\n")

            for i in range(num_candidates):
                file_.write(str(i) + ', c' + str(i) + "\n")

            c = Counter(map(tuple, votes))
            counted_votes = [[count, list(row)] for row, count in c.items()]
            counted_votes = sorted(counted_votes, reverse=True)

            file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' +
                        str(len(counted_votes)) + "\n")

            for i in range(len(counted_votes)):
                file_.write(str(counted_votes[i][0]) + ', {')
                for j in range(len(counted_votes[i][1])):
                    file_.write(str(int(counted_votes[i][1][j])))
                    if j < len(counted_votes[i][1]) - 1:
                        file_.write(", ")
                file_.write("}\n")


# GENERATE
def generate_instances(experiment=None, model=None, name=None, num_candidates=None,
                       num_voters=None, num_nodes=None, params=None, ballot='ordinal',
                       param_name=None):
    """ main function: generate instances """

    if params is None:
        params = {}

    alpha = 1

    if model == 'mallows' and params['phi'] is None:
        params['phi'] = rand.random()
    elif model == 'norm-mallows' and params['norm-phi'] is None:
        params['norm-phi'] = rand.random()
    elif model == 'urn_model' and params['alpha'] is None:
        params['alpha'] = gamma.rvs(0.8)

    if model == 'norm-mallows':
        params['phi'] = phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    if model == 'mallows_matrix_path':
        params['norm-phi'] = params['alpha']
        params['phi'] = phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    if model == 'erdos_renyi_graph' and params['p'] is None:
        params['p'] = rand.random()

    if param_name is not None:
        alpha = params[param_name]
        params['path_param'] = param_name

    if 'weight' not in params:
        params['weight'] = 0.


    if ballot == 'ordinal':
        votes = generate_ordinal_votes(model=model, num_candidates=num_candidates,
                                       num_voters=num_voters, params=params)
        instance = OrdinalElection("virtual", "virtual", votes=votes, model=model,
                            num_candidates=num_candidates,
                            num_voters=num_voters, ballot=ballot, alpha=alpha)
    elif ballot == 'approval':
        votes = generate_approval_votes(model=model, num_candidates=num_candidates,
                                        num_voters=num_voters, params=params)
        instance = ApprovalElection("virtual", "virtual", votes=votes, model=model,
                            num_candidates=num_candidates,
                            num_voters=num_voters, ballot=ballot, alpha=alpha)
    elif ballot == 'graph':
        graph = generate_graph(model=model, num_nodes=num_nodes, params=params)
        instance = Graph("virtual", "virtual", graph=graph,
                         model=model, num_nodes=num_nodes, alpha=alpha)

    else:
        print("Such ballot does not exist!")
        instance = None

    if experiment is not None:
        experiment.instances[name] = instance

        # For now, storing works only for ordinal ballot
        if experiment.store:
            if ballot == 'ordinal':
                store_ordinal_instances(experiment, model, name, num_candidates, num_voters, params)
            if ballot == 'approval':
                store_approval_instances(experiment, model, name, num_candidates, num_voters, params)

    return instance


###############################################################################

# IMPORT
def import_election(experiment_id, election_id):
    """ main function: import single election """
    return Election(experiment_id, election_id)


# HELPER FUNCTIONS #
def prepare_preflib_family(experiment=None, model=None, params=None,
                           id_=None, folder=None):
    # NEEDS UPDATE #

    selection_method = 'random'

    # list of IDs larger than 10
    if model == 'irish':
        folder = 'irish_s1'
        # folder = 'irish_f'
        ids = [1, 3]
    elif model == 'glasgow':
        folder = 'glasgow_s1'
        # folder = 'glasgow_f'
        ids = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 21]
    elif model == 'formula':
        folder = 'formula_s1'
        # 17 races or more
        ids = [17, 35, 37, 40, 41, 42, 44, 45, 46, 47, 48]
    elif model == 'skate':
        folder = 'skate_ic'
        # 9 judges
        ids = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48]
    elif model == 'sushi':
        folder = 'sushi_ff'
        ids = [1]
    elif model == 'grenoble':
        folder = 'grenoble_ff'
        ids = [1]
    elif model == 'tshirt':
        folder = 'tshirt_ff'
        ids = [1]
    elif model == 'cities_survey':
        folder = 'cities_survey_s1'
        ids = [1, 2]
    elif model == 'aspen':
        folder = 'aspen_s1'
        ids = [1]
    elif model == 'marble':
        folder = 'marble_ff'
        ids = [1, 2, 3, 4, 5]
    elif model == 'cycling_tdf':
        folder = 'cycling_tdf_s1'
        # ids = [e for e in range(1, 69+1)]
        selection_method = 'random'
        ids = [14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]
    elif model == 'cycling_gdi':
        folder = 'cycling_gdi_s1'
        ids = [i for i in range(2, 23 + 1)]
    elif model == 'ers':
        folder = 'ers_s1'
        # folder = 'ers_f'
        # 500 voters or more
        ids = [3, 9, 23, 31, 32, 33, 36, 38, 40, 68, 77, 79, 80]
    elif model == 'ice_races':
        folder = 'ice_races_s1'
        # 80 voters or more
        ids = [4, 5, 8, 9, 15, 20, 23, 24, 31, 34, 35, 37, 43, 44, 49]
    else:
        ids = []

    rand_ids = rand.choices(ids, k=experiment.families[model].size)
    for ri in rand_ids:
        election_id = "core_" + str(id_)
        tmp_election_type = model + '_' + str(ri)

        preflib.generate_elections_preflib(
            experiment=experiment, model=tmp_election_type,
            elections_id=election_id,
            num_voters=experiment.families[model].num_voters,
            num_candidates=experiment.families[model].num_candidates,
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


def _get_params_for_paths(experiment, family_id, j, with_extremes=False):

    path = experiment.families[family_id].path

    param_name = path['param_name']

    if 'with_extremes' in path:
        with_extremes = path['with_extremes']

    params = {}
    if with_extremes:
        params[param_name] = j / (experiment.families[family_id].size - 1)
    elif not with_extremes:
        params[param_name] = (j + 1) / (experiment.families[family_id].size + 1)

    return params, param_name


def prepare_parties(experiment=None, model=None,
                    family_id=None, params=None):
    parties = []

    if model == '2d_gaussian_party':
        for i in range(params['num_parties']):
            point = np.random.rand(1, 2)
            parties.append(point)

    elif model in ['1d_gaussian_party','conitzer_party', 'walsh_party']:
        for i in range(params['num_parties']):
            point = np.random.rand(1, 1)
            parties.append(point)

    return parties


def prepare_statistical_culture_family(experiment=None, model=None, family_id=None,
                                       params=None):
    keys = []
    ballot = 'ordinal'

    if model in APPROVAL_MODELS:
        ballot = 'approval'
    elif model in GRAPH_MODELS:
        ballot = 'graph'

    if model in PARTY_MODELS:
        params['party'] = prepare_parties(experiment=experiment, params=params,
                                          model=model, family_id=family_id)

    for j in range(experiment.families[family_id].size):

        param_name = None
        path = experiment.families[family_id].path
        if path is not None and 'param_name' in path:
            new_params, param_name = _get_params_for_paths(experiment, family_id, j)
            params = {**params, **new_params}

        if model in {'crate'}:
            new_params = _get_params_for_crate(j)
            params = {**params, **new_params}

        if experiment.families[family_id].single_election:
            name = family_id
        else:
            name = family_id + '_' + str(j)

        generate_instances(experiment=experiment, model=model,
                           name=name,
                           num_voters=experiment.families[family_id].num_voters,
                           num_candidates=experiment.families[family_id].num_candidates,
                           num_nodes=experiment.families[family_id].num_nodes,
                           params=copy.deepcopy(params), ballot=ballot, param_name=param_name)

        keys.append(name)
    return keys
