#!/usr/bin/env python

import copy
import os
from collections import Counter
from typing import Union

import numpy as np

import mapel.roommates.models.euclidean as euclidean
import mapel.roommates.models.impartial as impartial
import mapel.roommates.models.mallows as mallows
import mapel.roommates.models.urn as urn
import mapel.roommates.models.group_separable as group_separable
from mapel.elections._glossary import *
from mapel.roommates.objects.Roommates import Roommates


# GENERATE VOTES
def generate_roommates_votes(model_id: str = None, num_agents: int = None,
                             params: dict = None) -> Union[list, np.ndarray]:
    main_models = {
                    'roommates_ic': impartial.generate_roommates_ic_votes,
                    'roommates_group_ic': impartial.generate_roommates_group_ic_votes,
                    'roommates_id': impartial.generate_roommates_id_votes,
                    'roommates_test': impartial.generate_roommates_id_votes,
                    'roommates_teo': impartial.generate_roommates_teo_votes,
                   'roommates_asymmetric': impartial.generate_roommates_cy_votes,
                   'roommates_cy2': impartial.generate_roommates_cy2_votes,
                   'roommates_cy3': impartial.generate_roommates_cy3_votes,
                   'roommates_revcy': impartial.generate_roommates_revcy_votes,
                   'roommates_an':  impartial.generate_roommates_an_votes,
                   'roommates_an2':  impartial.generate_roommates_an2_votes,
                   'roommates_ideal': impartial.generate_roommates_ideal_votes,
                   'roommates_norm-mallows': mallows.generate_roommates_norm_mallows_votes,
                   'roommates_urn': urn.generate_roommates_urn_votes,
                   'roommates_euclidean': euclidean.generate_roommates_euclidean_votes,
                   'roommates_gs': group_separable.generate_roommates_gs_votes,
                   'roommates_symmetric': group_separable.generate_roommates_gs_ideal_votes,
                   'roommates_revgs_ideal': group_separable.generate_roommates_revgs_ideal_votes,
                    'roommates_radius': euclidean.generate_roommates_radius_votes,
                    'roommates_double': euclidean.generate_roommates_double_votes,
                    'roommates_mallows_euclidean': euclidean.generate_roommates_mallows_euclidean_votes,
                    'roommates_malasym': impartial.generate_roommates_malasym_votes,

                    }

    if model_id in main_models:
        return main_models.get(model_id)(num_agents=num_agents, params=params)
    else:
        print("No such election model_id!", model_id)
        return []


# GENERATE INSTANCE
def generate_roommates_instance(experiment=None, model_id: str = None, instance_id: str = None,
                                num_agents: int = None, params: dict = None) -> Roommates:
    if params is None:
        params = {}

    if model_id == 'roommates_norm-mallows' and 'norm-phi' not in params:
        params['norm-phi'] = np.random.rand()
        params['alpha'] = params['norm-phi']

    elif model_id == 'roommates_urn' and 'alpha' not in params:
        params['alpha'] = np.random.rand()

    else:
        params['alpha'] = 1
    alpha = params['alpha']

    votes = generate_roommates_votes(model_id=model_id, num_agents=num_agents, params=params)

    if experiment.store:
        store_roommates_instance(experiment, model_id, instance_id, params, num_agents, votes)

    return Roommates("virtual", instance_id, votes=votes, model_id=model_id, alpha=alpha,
                     num_agents=num_agents)


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

# PREPARE INSTANCES
def prepare_roommates_instances(experiment=None, model_id: str = None,
                                family_id: str = None, params: dict = None) -> None:
    _keys = []
    for j in range(experiment.families[family_id].size):

        variable = None
        path = experiment.families[family_id].path
        if path is not None and 'variable' in path:
            new_params, variable = _get_params_for_paths(experiment, family_id, j)
            params = {**params, **new_params}

        if params is not None and 'norm-phi' in params:
            params['phi'] = mallows.phi_from_relphi(experiment.families[family_id].num_agents,
                                                    relphi=params['norm-phi'])

        if experiment.families[family_id].single_instance:
            instance_id = family_id
        else:
            instance_id = family_id + '_' + str(j)

        instance = generate_roommates_instance(experiment=experiment, model_id=model_id,
                                               instance_id=instance_id,
                                               num_agents=experiment.families[family_id].num_agents,
                                               params=copy.deepcopy(params))

        experiment.instances[instance_id] = instance
        _keys.append(instance_id)

    experiment.families[family_id].instance_ids = _keys


# STORE
def store_roommates_instance(experiment, model_id, instance_id, params, num_agents, votes):

    path = os.path.join("experiments", experiment.experiment_id, "instances", f'{instance_id}.ri')
    store_votes_in_a_file(model_id, params, path, num_agents, votes)


def store_votes_in_a_file(model_id, params, path, num_agents, votes):
    """ Store votes in a file """

    with open(path, 'w') as file_:

        if model_id in NICE_NAME:
            file_.write("# " + NICE_NAME[model_id] + " " + str(params) + "\n")
        else:
            file_.write("# " + model_id + " " + str(params) + "\n")

        file_.write(str(num_agents) + "\n")

        for i in range(num_agents):
            file_.write(str(i) + ', a' + str(i) + "\n")

        c = Counter(map(tuple, votes))
        counted_votes = [[count, list(row)] for row, count in c.items()]
        # counted_votes = sorted(counted_votes, reverse=True)

        file_.write(str(num_agents) + ', ' + str(num_agents) + ', ' +
                    str(len(counted_votes)) + "\n")


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
