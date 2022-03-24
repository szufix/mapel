#!/usr/bin/env python
import logging
from time import time
from typing import Callable

import numpy as np

from mapel.elections.metrics import main_approval_distances as mad
from mapel.elections.metrics import main_ordinal_distances as mod
from mapel.main._inner_distances import map_str_to_func
from mapel.elections.objects.ApprovalElection import ApprovalElection
from mapel.elections.objects.Election import Election
from mapel.elections.objects.OrdinalElection import OrdinalElection
from mapel.main.objects.Experiment import Experiment


def get_distance(election_1: Election, election_2: Election,
                 distance_id: str = None) -> float or (float, list):
    """ Return: distance between instances, (if applicable) optimal matching """

    if type(election_1) is ApprovalElection and type(election_2) is ApprovalElection:
        return get_approval_distance(election_1, election_2, distance_id=distance_id)
    elif type(election_1) is OrdinalElection and type(election_2) is OrdinalElection:
        return get_ordinal_distance(election_1, election_2, distance_id=distance_id)
    else:
        logging.warning('No such instance!')


def get_approval_distance(election_1: ApprovalElection, election_2: ApprovalElection,
                          distance_id: str = None) -> float or (float, list):
    """ Return: distance between approval elections, (if applicable) optimal matching """

    inner_distance, main_distance = extract_distance_id(distance_id)

    metrics_without_params = {
        'flow': mad.compute_flow,
        'hamming': mad.compute_hamming,
    }

    metrics_with_inner_distance = {
        'approvalwise': mad.compute_approvalwise,
        'coapproval_frequency': mad.compute_coapproval_frequency_vectors,
        'pairwise': mad.compute_pairwise,
        'voterlikeness': mad.compute_voterlikeness,
        'candidatelikeness': mad.compute_candidatelikeness,
    }

    if main_distance in metrics_without_params:
        return metrics_without_params.get(main_distance)(election_1, election_2)

    elif main_distance in metrics_with_inner_distance:
        return metrics_with_inner_distance.get(main_distance)(election_1, election_2,
                                                              inner_distance)


def get_ordinal_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                         distance_id: str = None) -> float or (float, list):
    """ Return: distance between ordinal elections, (if applicable) optimal matching """

    inner_distance, main_distance = extract_distance_id(distance_id)

    metrics_without_params = {
        'discrete': mod.compute_discrete_distance,
        'voter_subelection': mod.compute_voter_subelection,
        'candidate_subelection': mod.compute_candidate_subelection,
        'swap_bf': mod.compute_swap_bf_distance,
    }

    metrics_with_inner_distance = {
        'positionwise': mod.compute_positionwise_distance,
        'bordawise': mod.compute_bordawise_distance,
        'pairwise': mod.compute_pairwise_distance,
        'voterlikeness': mod.compute_voterlikeness_distance,
        'agg_voterlikeness': mod.compute_agg_voterlikeness_distance,
        'pos_swap': mod.compute_pos_swap_distance,
    }

    if main_distance in metrics_without_params:
        return metrics_without_params.get(main_distance)(election_1, election_2)

    elif main_distance in metrics_with_inner_distance:
        return metrics_with_inner_distance.get(main_distance)(election_1, election_2,
                                                              inner_distance)


def extract_distance_id(distance_id: str) -> (Callable, str):
    if '-' in distance_id:
        inner_distance, main_distance = distance_id.split('-')
        inner_distance = map_str_to_func(inner_distance)
    else:
        main_distance = distance_id
        inner_distance = None
    return inner_distance, main_distance


def run_single_thread(experiment: Experiment, thread_ids: list,
                      distances: dict, times: dict, matchings: dict,
                      printing: bool, t) -> None:
    """ Single thread for computing distances """
    for instance_id_1, instance_id_2 in thread_ids:
        if t == 0 and printing:
            print(instance_id_1, instance_id_2)
        start_time = time()
        distance = get_distance(experiment.instances[instance_id_1],
                                experiment.instances[instance_id_2],
                                distance_id=experiment.distance_id)
        if type(distance) is tuple:
            distance, matching = distance
            matching = np.array(matching)
            matchings[instance_id_1][instance_id_2] = matching
            matchings[instance_id_2][instance_id_1] = np.argsort(matching)
        distances[instance_id_1][instance_id_2] = distance
        distances[instance_id_2][instance_id_1] = distances[instance_id_1][instance_id_2]
        times[instance_id_1][instance_id_2] = time() - start_time
        times[instance_id_2][instance_id_1] = times[instance_id_1][instance_id_2]

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 17.03.2022 #
# # # # # # # # # # # # # # # #
