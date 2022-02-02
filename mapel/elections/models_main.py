#!/usr/bin/env python

import copy
import os
from collections import Counter
from typing import Union

import networkx as nx
import numpy as np
from scipy.stats import gamma

import mapel.elections.models.euclidean as euclidean
import mapel.elections.models.group_separable as group_separable
import mapel.elections.models.guardians as guardians
import mapel.elections.models.impartial as impartial
import mapel.elections.models.mallows as mallows
import mapel.elections.models.single_crossing as single_crossing
import mapel.elections.models.single_peaked as single_peaked
import mapel.elections.models.urn_model as urn_model
import mapel.elections.models.preflib as preflib
from mapel.elections._glossary import *
from mapel.elections.objects.ApprovalElection import ApprovalElection
from mapel.elections.objects.Election import Election
from mapel.elections.objects.OrdinalElection import OrdinalElection


def generate_approval_votes(model_id: str = None, num_candidates: int = None,
                            num_voters: int = None, params: dict = None) -> Union[list, np.ndarray]:
    main_models = {'approval_ic': impartial.generate_approval_ic_votes,
                   'approval_id': impartial.generate_approval_id_votes,
                   'approval_resampling': mallows.generate_approval_resampling_votes,
                   'approval_noise_model': mallows.generate_approval_noise_model_votes,
                   'approval_urn': urn_model.generate_approval_urn_votes,
                   'approval_euclidean': euclidean.generate_approval_euclidean_votes,
                   'approval_disjoint_resampling': mallows.generate_approval_disjoint_shumallows_votes,
                   'approval_vcr': euclidean.generate_approval_vcr_votes,
                   'approval_truncated_mallows': mallows.generate_approval_truncated_mallows_votes,
                   'approval_truncated_urn': urn_model.generate_approval_truncated_urn_votes,
                   'approval_moving_resampling': mallows.generate_approval_moving_resampling_votes,
                   'approval_simplex_shumallows': mallows.generate_approval_simplex_shumallows_votes,
                   'approval_anti_pjr': mallows.approval_anti_pjr_votes,
                   'approval_partylist': mallows.approval_partylist_votes,
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
        return [model_id, num_candidates, num_voters, params]
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
                               group_separable.generate_ordinal_group_separable_votes,

                           'single-crossing': single_crossing.generate_ordinal_single_crossing_votes,}

    double_param_models = {'mallows': mallows.generate_mallows_votes,
                           'norm-mallows': mallows.generate_mallows_votes,
                           'norm-mallows_mixture': mallows.generate_norm_mallows_mixture_votes}

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
def generate_election(experiment=None, model_id: str = None, election_id: str = None,
                      num_candidates: int = None, num_voters: int = None,
                      params: dict = None, ballot: str = 'ordinal',
                      variable=None) -> Election:
    """ main function: generate election """

    if params is None:
        params = {}

    if model_id == 'all_votes':
        alpha = 1
    else:
        params, alpha = update_params(params, variable, model_id, num_candidates)

    if ballot == 'ordinal':
        votes = generate_ordinal_votes(model_id=model_id, num_candidates=num_candidates,
                                       num_voters=num_voters, params=params)
        election = OrdinalElection("virtual", "virtual", votes=votes, model_id=model_id,
                                   num_candidates=num_candidates, params=params,
                                   num_voters=num_voters, ballot=ballot, alpha=alpha)
    elif ballot == 'approval':
        votes = generate_approval_votes(model_id=model_id, num_candidates=num_candidates,
                                        num_voters=num_voters, params=params)
        election = ApprovalElection(experiment.experiment_id, election_id, votes=votes,
                                    model_id=model_id,
                                    num_candidates=num_candidates,
                                    num_voters=num_voters, ballot=ballot, alpha=alpha)
    # elif ballot == 'graph':
    #     graph = generate_graph(model_id=model_id, num_nodes=num_nodes, params=params)
    #     election = Graph("virtual", "virtual", graph=graph,
    #                      model_id=model_id, num_nodes=num_nodes, alpha=alpha)
    # elif ballot == 'roommates':
    #     votes = generate_roommates_votes(model_id=model_id, num_agents=num_agents, params=params)
    #     election = RoommatesProblem("virtual", "virtual", votes=votes, model_id=model_id,
    #                                 alpha=1)
    else:
        print("Such ballot does not exist!")
        election = None

    if experiment is not None:
        experiment.instances[election_id] = election

        if experiment.store:
            if ballot == 'ordinal':
                store_ordinal_election(experiment, model_id, election_id, num_candidates,
                                       num_voters, params, ballot)
            if ballot == 'approval':
                store_approval_election(experiment, model_id, election_id, num_candidates,
                                        num_voters, params, ballot)

    return election


def prepare_statistical_culture_family(experiment=None, model_id: str = None,
                                       family_id: str = None, params: dict = None):
    ballot = get_ballot_from_model(model_id)

    if model_id in PARTY_MODELS:
        params['party'] = prepare_parties(params=params, model_id=model_id)

    elections = {}
    for j in range(experiment.families[family_id].size):

        variable = None
        path = experiment.families[family_id].path
        if path is not None and 'variable' in path:
            new_params, variable = _get_params_for_paths(experiment, family_id, j)
            params = {**params, **new_params}

        if params is not None and 'norm-phi' in params:

            params['phi'] = mallows.phi_from_relphi(experiment.families[family_id].num_candidates,
                                                    relphi=params['norm-phi'])

        if model_id in {'all_votes'}:
            params['iter_id'] = j

        if model_id in {'crate'}:
            new_params = _get_params_for_crate(j)
            params = {**params, **new_params}

        if experiment.families[family_id].single_election:
            election_id = family_id
        else:
            election_id = family_id + '_' + str(j)

        election = generate_election(experiment=experiment, model_id=model_id,
                                     election_id=election_id,
                                     num_voters=experiment.families[family_id].num_voters,
                                     num_candidates=experiment.families[family_id].num_candidates,
                                     params=copy.deepcopy(params), ballot=ballot, variable=variable)
        elections[election_id] = election
    return elections


# HELPER FUNCTIONS
def get_ballot_from_model(model_id: str) -> str:
    if model_id in APPROVAL_MODELS:
        return 'approval'
    elif model_id in GRAPH_MODELS:
        return 'graph'
    elif model_id in ROOMMATES_PROBLEM_MODELS:
        return 'roommates'
    else:
        return 'ordinal'


def update_params(params, variable, model_id, num_candidates):

    if variable is not None:
        params['alpha'] = params[variable]
        params['variable'] = variable

        if model_id in APPROVAL_MODELS:
            if 'p' not in params:
                params['p'] = np.random.rand()
            elif type(params['p']) is list:
                params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])

    else:
        if model_id in ['approval_partylist']:
            return params, 1

        if model_id in APPROVAL_MODELS:
            if 'p' not in params:
                params['p'] = np.random.rand()
            elif type(params['p']) is list:
                params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])

        # if 'phi' not in params:
        #     params['phi'] = np.random.rand()
        if 'phi' in params and type(params['phi']) is list:
            params['phi'] = np.random.uniform(low=params['phi'][0], high=params['phi'][1])

        if model_id == 'mallows' and params['phi'] is None:
            params['phi'] = np.random.random()
        elif model_id == 'norm-mallows' and 'norm-phi' not in params:
            params['norm-phi'] = np.random.random()
        elif model_id in ['urn_model', 'approval_urn'] and 'alpha' not in params:
            params['alpha'] = gamma.rvs(0.8)

        if model_id == 'norm-mallows':
            params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])
            if 'weight' not in params:
                params['weight'] = 0.

        if model_id == 'mallows_matrix_path':
            params['norm-phi'] = params['alpha']
            params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])

        if model_id == 'erdos_renyi_graph' and params['p'] is None:
            params['p'] = np.random.random()

        if 'alpha' not in params:
            if 'norm-phi' in params:
                params['alpha'] = params['norm-phi']
            elif 'phi' in params:
                params['alpha'] = params['phi']
            else:
                params['alpha'] = np.random.rand()
        elif type(params['alpha']) is list:
            params['alpha'] = np.random.uniform(low=params['alpha'][0], high=params['alpha'][1])

    return params, params['alpha']


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
    else:
        path['start'] = 0.

    if 'step' in path:
        params[variable] = path['start'] + j * path['step']

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
                           params, ballot, votes=None):
    """ Store ordinal election in a .soc file """

    if model_id in LIST_OF_FAKE_MODELS:
        path = os.path.join("experiments", str(experiment.experiment_id),
                            "elections", (str(election_id) + ".soc"))
        file_ = open(path, 'w')
        file_.write(f'$ {model_id} {params} \n')
        file_.write(str(num_candidates) + '\n')
        file_.write(str(num_voters) + '\n')
        file_.close()

    else:

        path = os.path.join("experiments", str(experiment.experiment_id), "elections",
                            (str(election_id) + ".soc"))

        store_votes_in_a_file(experiment, model_id, election_id, num_candidates, num_voters,
                              params, path, ballot, votes=votes)


def store_approval_election(experiment, model_id, election_id, num_candidates, num_voters,
                            params, ballot):
    """ Store approval election in an .app file """

    if model_id in APPROVAL_FAKE_MODELS:
        path = os.path.join("experiments", str(experiment.experiment_id),
                            "elections", (str(election_id) + ".app"))
        file_ = open(path, 'w')
        file_.write(f'$ {model_id} {params} \n')
        file_.write(str(num_candidates) + '\n')
        file_.write(str(num_voters) + '\n')
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
        votes = experiment.instances[election_id].votes

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
