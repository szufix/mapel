#!/usr/bin/env python
""" this module is used to generate and import elections"""

import copy
import os
import random as rand
from collections import Counter

import networkx as nx
import numpy as np
from scipy.stats import gamma
from typing import Union

import mapel.voting.elections.euclidean as euclidean

from mapel.voting.elections.group_separable import generate_group_separable_election
from mapel.voting.elections.guardians import \
    generate_real_antagonism_election, \
    generate_real_identity_election, generate_real_stratification_election, \
    generate_real_uniformity_election
from mapel.voting.elections.impartial import generate_impartial_culture_election, \
    generate_impartial_anonymous_culture_election, generate_ic_party, \
    generate_approval_ic_election, generate_approval_id_election, \
    generate_approval_empty, generate_approval_full
from mapel.voting.elections.mallows import generate_mallows_election, \
    phi_from_relphi, generate_mallows_party, generate_approval_mallows_election, \
    generate_approval_raw_mallows_election, generate_approval_disjoint_mallows_election
from mapel.voting.elections.single_crossing import \
    generate_single_crossing_election
from mapel.voting.elections.single_peaked import generate_conitzer_election, \
    generate_walsh_election, generate_spoc_conitzer_election, \
    generate_sp_party
from mapel.voting.elections.urn_model import generate_urn_model_election, \
    generate_approval_urn_election
from mapel.voting._glossary import NICE_NAME, LIST_OF_FAKE_MODELS, PATHS, PARTY_MODELS, \
    APPROVAL_MODELS, GRAPH_MODELS, APPROVAL_FAKE_MODELS
from mapel.voting.objects.ApprovalElection import ApprovalElection
from mapel.voting.objects.Election import Election
from mapel.voting.objects.Graph import Graph
from mapel.voting.objects.OrdinalElection import OrdinalElection


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


def generate_approval_votes(model: str = None, num_candidates: int = None, num_voters: int = None,
                            params: dict = None) -> Union[list, np.ndarray]:
    models_with_params = {'approval_ic': generate_approval_ic_election,
                          'approval_id': generate_approval_id_election,
                          'approval_mallows': generate_approval_mallows_election,
                          'approval_raw_mallows': generate_approval_raw_mallows_election,
                          'approval_urn': generate_approval_urn_election,
                          'approval_euclidean': euclidean.generate_approval_euclidean_election,
                          'approval_disjoint_mallows': generate_approval_disjoint_mallows_election,
                          'approval_id_0.5': generate_approval_id_election,
                          'approval_ic_0.5': generate_approval_ic_election,
                          }

    if model in models_with_params:
        return models_with_params.get(model)(num_voters=num_voters, num_candidates=num_candidates,
                                             params=params)
    elif model in ['approval_full']:
        return generate_approval_full(num_voters=num_voters, num_candidates=num_candidates)
    elif model in ['approval_empty']:
        return generate_approval_empty(num_voters=num_voters)

    elif model in APPROVAL_FAKE_MODELS:
        return []
    else:
        print("No such election model!", model)
        return []


def generate_ordinal_votes(model: str = None, num_candidates: int = None, num_voters: int = None,
                           params: dict = None) -> Union[list, np.ndarray]:
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

    euclidean_models = {'1d_interval': euclidean.generate_elections_1d_simple,
                        '1d_gaussian': euclidean.generate_elections_1d_simple,
                        '1d_one_sided_triangle': euclidean.generate_elections_1d_simple,
                        '1d_full_triangle': euclidean.generate_elections_1d_simple,
                        '1d_two_party': euclidean.generate_elections_1d_simple,
                        '2d_disc': euclidean.generate_elections_2d_simple,
                        '2d_square': euclidean.generate_elections_2d_simple,
                        '2d_gaussian': euclidean.generate_elections_2d_simple,
                        '3d_cube': euclidean.generate_elections_nd_simple,
                        '4d_cube': euclidean.generate_elections_nd_simple,
                        '5d_cube': euclidean.generate_elections_nd_simple,
                        '10d_cube': euclidean.generate_elections_nd_simple,
                        '15d_cube': euclidean.generate_elections_nd_simple,
                        '20d_cube': euclidean.generate_elections_nd_simple,
                        '40d_cube': euclidean.generate_elections_nd_simple,
                        '2d_sphere': euclidean.generate_elections_2d_simple,
                        '3d_sphere': euclidean.generate_elections_nd_simple,
                        '4d_sphere': euclidean.generate_elections_nd_simple,
                        '5d_sphere': euclidean.generate_elections_nd_simple,
                        '4d_ball': euclidean.generate_elections_nd_simple,
                        '5d_ball': euclidean.generate_elections_nd_simple,
                        '40d_ball': euclidean.generate_elections_nd_simple,
                        '2d_grid': euclidean.generate_elections_2d_grid}

    party_models = {'1d_gaussian_party': euclidean.generate_1d_gaussian_party,
                    '2d_gaussian_party': euclidean.generate_2d_gaussian_party,
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
        votes = [[int(x) for x in row] for row in votes]

    return votes


# GENERATE
def generate_election(experiment=None, model=None, name=None,
                      num_candidates: int = None, num_voters: int = None, num_nodes: int = None,
                      params: dict = None, ballot: str = 'ordinal',
                      variable=None) -> Election:
    """ main function: generate elections """

    if params is None:
        params = {}
    params, alpha = update_params(params, variable, model, num_candidates)

    if ballot == 'ordinal':
        votes = generate_ordinal_votes(model=model, num_candidates=num_candidates,
                                       num_voters=num_voters, params=params)
        election = OrdinalElection("virtual", "virtual", votes=votes, model=model,
                                   num_candidates=num_candidates,
                                   num_voters=num_voters, ballot=ballot, alpha=alpha)
    elif ballot == 'approval':
        votes = generate_approval_votes(model=model, num_candidates=num_candidates,
                                        num_voters=num_voters, params=params)
        election = ApprovalElection(experiment.experiment_id, name, votes=votes, model=model,
                                    num_candidates=num_candidates,
                                    num_voters=num_voters, ballot=ballot, alpha=alpha)
    elif ballot == 'graph':
        graph = generate_graph(model=model, num_nodes=num_nodes, params=params)
        election = Graph("virtual", "virtual", graph=graph,
                         model=model, num_nodes=num_nodes, alpha=alpha)
    else:
        print("Such ballot does not exist!")
        election = None

    if experiment is not None:
        experiment.elections[name] = election

        if experiment.store:
            if ballot == 'ordinal':
                store_ordinal_election(experiment, model, name, num_candidates, num_voters, params)
            if ballot == 'approval':
                store_approval_election(experiment, model, name, num_candidates, num_voters,
                                        params)

    return election


def prepare_preflib_family(experiment=None, model=None, params=None):
    pass


def prepare_statistical_culture_family(experiment=None, model=None, family_id=None,
                                       params=None) -> list:
    keys = []
    ballot = get_ballot_from_model(model)

    if model in PARTY_MODELS:
        params['party'] = prepare_parties(params=params, model=model)

    for j in range(experiment.families[family_id].size):

        variable = None
        path = experiment.families[family_id].path
        if path is not None and 'variable' in path:
            new_params, variable = _get_params_for_paths(experiment, family_id, j)
            params = {**params, **new_params}

        if model in {'crate'}:
            new_params = _get_params_for_crate(j)
            params = {**params, **new_params}

        if experiment.families[family_id].single_election:
            name = family_id
        else:
            name = family_id + '_' + str(j)

        generate_election(experiment=experiment, model=model, name=name,
                          num_voters=experiment.families[family_id].num_voters,
                          num_candidates=experiment.families[family_id].num_candidates,
                          num_nodes=experiment.families[family_id].num_nodes,
                          params=copy.deepcopy(params), ballot=ballot, variable=variable)

        keys.append(name)
    return keys


# HELPER FUNCTIONS
def get_ballot_from_model(model: str) -> str:
    if model in APPROVAL_MODELS:
        return 'approval'
    elif model in GRAPH_MODELS:
        return 'graph'
    else:
        return 'ordinal'


def update_params(params, variable, model, num_candidates):
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

    alpha = 1
    if variable is not None:
        alpha = params[variable]
        params['variable'] = variable

    if 'weight' not in params:
        params['weight'] = 0.

    return params, alpha


# HELPER FUNCTIONS #
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


def _get_params_for_paths(experiment, family_id, j, extremes=False):
    path = experiment.families[family_id].path

    variable = path['variable']

    if 'extremes' in path:
        extremes = path['extremes']

    params = {}
    if extremes:
        params[variable] = j / (experiment.families[family_id].size - 1)
    elif not extremes:
        params[variable] = (j + 1) / (experiment.families[family_id].size + 1)

    if 'scale' in path:
        params[variable] *= path['scale']
    if 'start' in path:
        params[variable] += path['start']

    return params, variable


def prepare_parties(model=None, params=None):
    parties = []

    if model == '2d_gaussian_party':
        for i in range(params['num_parties']):
            point = np.random.rand(1, 2)
            parties.append(point)

    elif model in ['1d_gaussian_party', 'conitzer_party', 'walsh_party']:
        for i in range(params['num_parties']):
            point = np.random.rand(1, 1)
            parties.append(point)

    return parties


# STORE
def store_ordinal_election(experiment, model, name, num_candidates, num_voters, params):
    """ Store ordinal election in a .soc file """

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

        path = os.path.join("experiments",
                            str(experiment.experiment_id), "elections",
                            (str(name) + ".soc"))

        store_votes_in_a_file(experiment, model, name, num_candidates, num_voters, params, path)


def store_approval_election(experiment, model, name, num_candidates, num_voters, params):
    """ Store approval election in an .app file """

    if model in APPROVAL_FAKE_MODELS:
        path = os.path.join("experiments", str(experiment.experiment_id),
                            "elections", (str(name) + ".app"))
        file_ = open(path, 'w')
        file_.write('$ fake' + '\n')
        file_.write(str(num_voters) + '\n')
        file_.write(str(num_candidates) + '\n')
        file_.write(str(model) + '\n')
        file_.close()

    else:
        path = os.path.join("experiments",
                            str(experiment.experiment_id), "elections",
                            (str(name) + ".app"))

        store_votes_in_a_file(experiment, model, name, num_candidates, num_voters, params, path)


def store_votes_in_a_file(experiment, model, name, num_candidates, num_voters, params, path):
    """ Store votes in a file """

    votes = experiment.elections[name].votes

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

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
