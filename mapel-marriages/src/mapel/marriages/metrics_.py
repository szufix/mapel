#!/usr/bin/env python


from time import time
from typing import Callable

import copy
import os
import csv
import numpy as np

from mapel.core.inner_distances import map_str_to_func
from mapel.core.objects.Experiment import Experiment
from mapel.marriages.metrics import main_marriages_distances as mrd
from mapel.marriages.objects.Marriages import Marriages


def get_distance(election_1: Marriages, election_2: Marriages,
                 distance_id: str = None) -> float or (float, list):
    """ Return: distance between ordinal elections, (if applicable) optimal matching """
    inner_distance, main_distance = extract_distance_id(distance_id)

    metrics_without_params = {

    }

    metrics_with_inner_distance = {
        'mutual_attraction': mrd.compute_retrospective_distance,
        # 'positionwise': mrd.compute_positionwise_distance,
        # 'pos_swap': mrd.compute_pos_swap_distance,
        # 'swap_bf': mrd.compute_swap_bf_distance,
        # 'pairwise': mrd.compute_pairwise_distance,
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


def run_single_process(exp: Experiment, instances_ids: list,
                      distances: dict, times: dict, matchings: dict,
                      printing: bool) -> None:
    """ Single thread for computing distances """

    for instance_id_1, instance_id_2 in instances_ids:
        if printing:
            print(instance_id_1, instance_id_2)
        start_time = time()
        distance = get_distance(copy.deepcopy(exp.instances[instance_id_1]),
                                copy.deepcopy(exp.instances[instance_id_2]),
                                distance_id=copy.deepcopy(exp.distance_id))
        print(distance)
        if type(distance) is tuple:
            distance, matching = distance
            matching = np.array(matching)
            matchings[instance_id_1][instance_id_2] = matching
            matchings[instance_id_2][instance_id_1] = np.argsort(matching)
        distances[instance_id_1][instance_id_2] = distance
        distances[instance_id_2][instance_id_1] = distances[instance_id_1][instance_id_2]
        times[instance_id_1][instance_id_2] = time() - start_time
        times[instance_id_2][instance_id_1] = times[instance_id_1][instance_id_2]


def run_multiple_processes(exp: Experiment, instances_ids: list,
                      distances: dict, times: dict, matchings: dict,
                      printing: bool, t) -> None:
    """ Single thread for computing distances """

    for instance_id_1, instance_id_2 in instances_ids:
        if t == 0 and printing:
            print(instance_id_1, instance_id_2)
        start_time = time()
        distance = get_distance(copy.deepcopy(exp.instances[instance_id_1]),
                                copy.deepcopy(exp.instances[instance_id_2]),
                                distance_id=copy.deepcopy(exp.distance_id))
        if type(distance) is tuple:
            distance, matching = distance
            matching = np.array(matching)
            matchings[instance_id_1][instance_id_2] = matching
            matchings[instance_id_2][instance_id_1] = np.argsort(matching)
        distances[instance_id_1][instance_id_2] = distance
        distances[instance_id_2][instance_id_1] = distances[instance_id_1][instance_id_2]
        times[instance_id_1][instance_id_2] = time() - start_time
        times[instance_id_2][instance_id_1] = times[instance_id_1][instance_id_2]

    if exp.store:
        _store_distances(exp, instances_ids, distances, times, t)


def _store_distances(exp, instances_ids, distances, times, t):
    """ Store distances to file """
    file_name = f'{exp.distance_id}_p{t}.csv'
    path = os.path.join(os.getcwd(), "experiments", exp.experiment_id, "distances", file_name)
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["instance_id_1", "instance_id_2", "distance", "time"])
        for election_id_1, election_id_2 in instances_ids:
            distance = float(distances[election_id_1][election_id_2])
            time_ = float(times[election_id_1][election_id_2])
            writer.writerow([election_id_1, election_id_2, distance, time_])

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
