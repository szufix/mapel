#!/usr/bin/env python

from time import time
from typing import Union, Callable

import networkx as nx
import numpy as np

from mapel.roommates.metrics import main_roommates_distances as mrd
from mapel.main.inner_distances import map_str_to_func
from mapel.roommates.objects.Roommates import Roommates
# from mapel.roommates.objects.OrdinalElection import OrdinalElection
from mapel.main.objects.Experiment import Experiment



def get_distance(election_1: Roommates, election_2: Roommates,
                 distance_id: str = None) -> float or (float, list):
    if type(election_1) is Roommates and type(election_2) is Roommates:
        return get_roommates_distance(election_1, election_2, distance_id=distance_id)
    else:
        print('No such instance!')


def get_roommates_distance(election_1: Roommates, election_2: Roommates,
                         distance_id: str = None) -> float or (float, list):
    """ Return: distance between ordinal elections, (if applicable) optimal matching """
    inner_distance, main_distance = extract_distance_id(distance_id)

    metrics_without_params = {

    }

    metrics_with_inner_distance = {
        'retrospective': mrd.compute_retrospective_distance,
        'positionwise': mrd.compute_positionwise_distance,
        'pos_swap': mrd.compute_pos_swap_distance,
        'swap_bf': mrd.compute_swap_bf_distance,
        'pairwise': mrd.compute_pairwise_distance,
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
                      printing: bool) -> None:
    """ Single thread for computing distances """
    for election_id_1, election_id_2 in thread_ids:
        if printing:
            print(election_id_1, election_id_2)
        start_time = time()
        distance = get_distance(experiment.instances[election_id_1],
                                experiment.instances[election_id_2],
                                distance_id=experiment.distance_id)
        if type(distance) is tuple:
            distance, matching = distance
            matching = np.array(matching)
            matchings[election_id_1][election_id_2] = matching
            matchings[election_id_2][election_id_1] = np.argsort(matching)
        distances[election_id_1][election_id_2] = distance
        distances[election_id_2][election_id_1] = distances[election_id_1][election_id_2]
        times[election_id_1][election_id_2] = time() - start_time
        times[election_id_2][election_id_1] = times[election_id_1][election_id_2]

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
