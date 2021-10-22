#!/usr/bin/env python

import copy
import os
import random as rand
from collections import Counter
from typing import Union

import networkx as nx
import numpy as np
from scipy.stats import gamma

import mapel.voting.elections.euclidean as euclidean
import mapel.voting.elections.group_separable as group_separable
import mapel.voting.elections.guardians as guardians
import mapel.voting.elections.impartial as impartial
import mapel.voting.elections.mallows as mallows
import mapel.voting.elections.single_crossing as single_crossing
import mapel.voting.elections.single_peaked as single_peaked
import mapel.voting.elections.urn_model as urn_model
from mapel.voting._glossary import *
from mapel.voting.objects.ApprovalElection import ApprovalElection
from mapel.voting.objects.Election import Election
from mapel.voting.objects.Graph import Graph
from mapel.voting.objects.OrdinalElection import OrdinalElection


def generate_approval_votes(model_id: str = None, num_candidates: int = None,
                            num_voters: int = None, params: dict = None) -> Union[list, np.ndarray]:

    main_models = {'approval_ic': impartial.generate_approval_ic_votes,
                   'approval_id': impartial.generate_approval_id_votes,
                   'approval_mallows': mallows.generate_approval_shumallows_votes,
                   'approval_raw_mallows': mallows.generate_approval_hamming_noise_model_votes,
                   'approval_urn': urn_model.generate_approval_urn_votes,
                   'approval_euclidean': euclidean.generate_approval_euclidean_votes,
                   'approval_disjoint_mallows': mallows.generate_approval_disjoint_shumallows_votes,
                   'approval_vcr': euclidean.generate_approval_vcr_votes,
                   'approval_truncated_mallows': mallows.generate_approval_truncated_mallows_votes,
                   'approval_moving_mallows': mallows.generate_approval_moving_shumallows_votes,
                   }

    if model_id in main_models:
        return main_models.get(model_id)(num_voters=num_voters, num_candidates=num_candidates,
                                         params=params)
    elif model_id in ['approval_full']:
        return impartial.generate_approval_full_votes(num_voters=num_voters,
                                                      num_candidates=num_candidates)
    elif model_id in ['approval_empty']:
        return impartial.generate_approval_empty_votes(num_voters=num_voters)

    elif model_id in APPROVAL_FAKE_MODELS:
        return []
    else:
        print("No such election model_id!", model_id)
        return []


def generate_ordinal_votes(model_id: str = None, num_candidates: int = None, num_voters: int = None,
                           params: dict = None) -> Union[list, np.ndarray]:

    naked_models = {'impartial_culture': impartial.generate_ordinal_ic_votes,
                    'iac': impartial.generate_impartial_anonymous_culture_election,
                    'conitzer': single_peaked.generate_ordinal_sp_conitzer_votes,
                    'spoc_conitzer': single_peaked.generate_ordinal_spoc_conitzer_votes,
                    'walsh': single_peaked.generate_ordinal_sp_walsh_votes,
                    'single-crossing': single_crossing.generate_ordinal_single_crossing_votes,
                    'real_identity': guardians.generate_real_identity_votes,
                    'real_uniformity': guardians.generate_real_uniformity_votes,
                    'real_antagonism': guardians.generate_real_antagonism_votes,
                    'real_stratification': guardians.generate_real_stratification_votes}

    euclidean_models = {'1d_interval': euclidean.generate_ordinal_euclidean_votes,
                        '1d_gaussian': euclidean.generate_ordinal_euclidean_votes,
                        '1d_one_sided_triangle': euclidean.generate_ordinal_euclidean_votes,
                        '1d_full_triangle': euclidean.generate_ordinal_euclidean_votes,
                        '1d_two_party': euclidean.generate_ordinal_euclidean_votes,
                        '2d_disc': euclidean.generate_ordinal_euclidean_votes,
                        '2d_square': euclidean.generate_ordinal_euclidean_votes,
                        '2d_gaussian': euclidean.generate_ordinal_euclidean_votes,
                        '3d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '4d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '5d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '10d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '15d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '20d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '40d_cube': euclidean.generate_ordinal_euclidean_votes,
                        '2d_sphere': euclidean.generate_ordinal_euclidean_votes,
                        '3d_sphere': euclidean.generate_ordinal_euclidean_votes,
                        '4d_sphere': euclidean.generate_ordinal_euclidean_votes,
                        '5d_sphere': euclidean.generate_ordinal_euclidean_votes,
                        '4d_ball': euclidean.generate_ordinal_euclidean_votes,
                        '5d_ball': euclidean.generate_ordinal_euclidean_votes,
                        '2d_grid': euclidean.generate_elections_2d_grid}

    party_models = {'1d_gaussian_party': euclidean.generate_1d_gaussian_party,
                    '2d_gaussian_party': euclidean.generate_2d_gaussian_party,
                    'walsh_party': single_peaked.generate_sp_party,
                    'conitzer_party': single_peaked.generate_sp_party,
                    'mallows_party': mallows.generate_mallows_party,
                    'ic_party': impartial.generate_ic_party
                    }

    single_param_models = {'urn_model': urn_model.generate_urn_votes,
                           'group-separable':
                               group_separable.generate_ordinal_group_separable_votes}

    double_param_models = {'mallows': mallows.generate_mallows_votes,
                           'norm-mallows': mallows.generate_mallows_votes, }

    if model_id in naked_models:
        votes = naked_models.get(model_id)(num_voters=num_voters,
                                           num_candidates=num_candidates)

    elif model_id in euclidean_models:
        votes = euclidean_models.get(model_id)(num_voters=num_voters,
                                               num_candidates=num_candidates,
                                               model=model_id, params=params)

    elif model_id in party_models:
        votes = party_models.get(model_id)(num_voters=num_voters,
                                           num_candidates=num_candidates,
                                           model=model_id, params=params)

    elif model_id in single_param_models:
        votes = single_param_models.get(model_id)(num_voters=num_voters,
                                                  num_candidates=num_candidates,
                                                  params=params)

    elif model_id in double_param_models:
        votes = double_param_models.get(model_id)(num_voters, num_candidates, params)

    elif model_id in LIST_OF_FAKE_MODELS:
        votes = [model_id, num_candidates, num_voters, params]
    else:
        votes = []
        print("No such election model_id!", model_id)

    if model_id not in LIST_OF_FAKE_MODELS:
        votes = [[int(x) for x in row] for row in votes]

    return votes


def generate_graph(model_id=None, num_nodes=None, params=None):
    non_params_graphs = {'cycle_graph': nx.cycle_graph,
                         'wheel_graph': nx.wheel_graph,
                         'star_graph': nx.star_graph,
                         'ladder_graph': nx.ladder_graph,
                         'circular_ladder_graph': nx.circular_ladder_graph,
                         'random_tree': nx.random_tree,
                         }

    if model_id in non_params_graphs:
        return non_params_graphs[model_id](num_nodes)

    elif model_id in ['erdos_renyi_graph', 'erdos_renyi_graph_path']:
        return nx.erdos_renyi_graph(num_nodes, params['p'])
    elif model_id == 'watts_strogatz_graph':
        return nx.watts_strogatz_graph(num_nodes, params['k'], params['p'])
    elif model_id == 'barabasi_albert_graph':
        return nx.barabasi_albert_graph(num_nodes, params['m'])
    elif model_id == 'random_geometric_graph':
        return nx.random_geometric_graph(num_nodes, params['radius'])


# GENERATE
def generate_election(experiment=None, model_id=None, election_id=None,
                      num_candidates: int = None, num_voters: int = None, num_nodes: int = None,
                      params: dict = None, ballot: str = 'ordinal',
                      variable=None) -> Election:
    """ main function: generate elections """

    if params is None:
        params = {}
    params, alpha = update_params(params, variable, model_id, num_candidates)

    if ballot == 'ordinal':
        votes = generate_ordinal_votes(model_id=model_id, num_candidates=num_candidates,
                                       num_voters=num_voters, params=params)
        election = OrdinalElection("virtual", "virtual", votes=votes, model_id=model_id,
                                   num_candidates=num_candidates,
                                   num_voters=num_voters, ballot=ballot, alpha=alpha)
    elif ballot == 'approval':
        votes = generate_approval_votes(model_id=model_id, num_candidates=num_candidates,
                                        num_voters=num_voters, params=params)
        election = ApprovalElection(experiment.experiment_id, election_id, votes=votes,
                                    model_id=model_id,
                                    num_candidates=num_candidates,
                                    num_voters=num_voters, ballot=ballot, alpha=alpha)
    elif ballot == 'graph':
        graph = generate_graph(model_id=model_id, num_nodes=num_nodes, params=params)
        election = Graph("virtual", "virtual", graph=graph,
                         model_id=model_id, num_nodes=num_nodes, alpha=alpha)
    else:
        print("Such ballot does not exist!")
        election = None

    if experiment is not None:
        experiment.elections[election_id] = election

        if experiment.store:
            if ballot == 'ordinal':
                store_ordinal_election(experiment, model_id, election_id, num_candidates,
                                       num_voters, params, ballot)
            if ballot == 'approval':
                store_approval_election(experiment, model_id, election_id, num_candidates,
                                        num_voters, params, ballot)

    return election


def prepare_statistical_culture_family(experiment=None, model_id=None, family_id=None,
                                       params=None) -> list:
    keys = []
    ballot = get_ballot_from_model(model_id)

    if model_id in PARTY_MODELS:
        params['party'] = prepare_parties(params=params, model_id=model_id)

    for j in range(experiment.families[family_id].size):

        variable = None
        path = experiment.families[family_id].path
        if path is not None and 'variable' in path:
            new_params, variable = _get_params_for_paths(experiment, family_id, j)
            params = {**params, **new_params}

        if model_id in {'crate'}:
            new_params = _get_params_for_crate(j)
            params = {**params, **new_params}

        if experiment.families[family_id].single_election:
            election_id = family_id
        else:
            election_id = family_id + '_' + str(j)

        generate_election(experiment=experiment, model_id=model_id, election_id=election_id,
                          num_voters=experiment.families[family_id].num_voters,
                          num_candidates=experiment.families[family_id].num_candidates,
                          num_nodes=experiment.families[family_id].num_nodes,
                          params=copy.deepcopy(params), ballot=ballot, variable=variable)

        keys.append(election_id)
    return keys


# HELPER FUNCTIONS
def get_ballot_from_model(model_id: str) -> str:
    if model_id in APPROVAL_MODELS:
        return 'approval'
    elif model_id in GRAPH_MODELS:
        return 'graph'
    else:
        return 'ordinal'


def update_params(params, variable, model_id, num_candidates):
    if 'p' not in params:
        params['p'] = np.random.rand()
    elif type(params['p']) is list:
        params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])

    if model_id == 'mallows' and params['phi'] is None:
        params['phi'] = rand.random()
    elif model_id == 'norm-mallows' and params['norm-phi'] is None:
        params['norm-phi'] = rand.random()
    elif model_id in ['urn_model', 'approval_urn'] and 'alpha' not in params:
        params['alpha'] = gamma.rvs(0.8)

    if model_id == 'norm-mallows':
        params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    if model_id == 'mallows_matrix_path':
        params['norm-phi'] = params['alpha']
        params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])

    if model_id == 'erdos_renyi_graph' and params['p'] is None:
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


def prepare_parties(model_id=None, params=None):
    parties = []

    if model_id == '2d_gaussian_party':
        for i in range(params['num_parties']):
            point = np.random.rand(1, 2)
            parties.append(point)

    elif model_id in ['1d_gaussian_party', 'conitzer_party', 'walsh_party']:
        for i in range(params['num_parties']):
            point = np.random.rand(1, 1)
            parties.append(point)

    return parties


# STORE
def store_ordinal_election(experiment, model_id, election_id, num_candidates, num_voters,
                           params, ballot):
    """ Store ordinal election in a .soc file """

    if model_id in LIST_OF_FAKE_MODELS:
        path = os.path.join("experiments", str(experiment.experiment_id),
                            "elections", (str(election_id) + ".soc"))
        file_ = open(path, 'w')
        file_.write('$ fake' + '\n')
        file_.write(str(num_voters) + '\n')
        file_.write(str(num_candidates) + '\n')
        file_.write(str(model_id) + '\n')
        if model_id == 'norm-mallows_matrix':
            file_.write(str(round(params['norm-phi'], 5)) + '\n')
        elif model_id in PATHS:
            file_.write(str(round(params['alpha'], 5)) + '\n')
            if model_id == 'mallows_matrix_path':
                file_.write(str(round(params['weight'], 5)) + '\n')
        file_.close()

    else:

        path = os.path.join("experiments", str(experiment.experiment_id), "elections",
                            (str(election_id) + ".soc"))

        store_votes_in_a_file(experiment, model_id, election_id, num_candidates, num_voters,
                              params, path, ballot)


def store_approval_election(experiment, model_id, election_id, num_candidates, num_voters,
                            params, ballot):
    """ Store approval election in an .app file """

    if model_id in APPROVAL_FAKE_MODELS:
        path = os.path.join("experiments", str(experiment.experiment_id),
                            "elections", (str(election_id) + ".app"))
        file_ = open(path, 'w')
        file_.write('$ fake' + '\n')
        file_.write(str(num_voters) + '\n')
        file_.write(str(num_candidates) + '\n')
        file_.write(str(model_id) + '\n')
        file_.close()

    else:
        path = os.path.join("experiments", str(experiment.experiment_id), "elections",
                            (str(election_id) + ".app"))

        store_votes_in_a_file(experiment, model_id, election_id, num_candidates, num_voters,
                              params, path, ballot)


def store_votes_in_a_file(experiment, model_id, election_id, num_candidates, num_voters,
                          params, path, ballot, votes=None):
    """ Store votes in a file """
    if votes is None:
        votes = experiment.elections[election_id].votes

    with open(path, 'w') as file_:

        if model_id in NICE_NAME:
            file_.write("# " + NICE_NAME[model_id] + " " + str(params) + "\n")
        else:
            file_.write("# " + model_id + "\n")

        file_.write(str(num_candidates) + "\n")

        for i in range(num_candidates):
            file_.write(str(i) + ', c' + str(i) + "\n")

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


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
