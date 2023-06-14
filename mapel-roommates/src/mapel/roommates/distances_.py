#!/usr/bin/env python
import copy
import csv
from time import time
import logging
import os
from typing import Callable

import numpy as np

from mapel.core.inner_distances import map_str_to_func
from mapel.core.objects.Experiment import Experiment
from mapel.roommates.distances import main_distances as mrd
from mapel.roommates.objects.Roommates import Roommates

registered_roommates_distances = {
    'mutual_attraction': mrd.compute_retrospective_distance,

    'positionwise': mrd.compute_positionwise_distance,  # unsupported distance
    'pos_swap': mrd.compute_pos_swap_distance,  # unsupported distance
    'swap_bf': mrd.compute_swap_bf_distance,  # unsupported distance
    'pairwise': mrd.compute_pairwise_distance,  # unsupported distance
}


def get_distance(election_1: Roommates, election_2: Roommates,
                 distance_id: str = None) -> float or (float, list):
    """ Return: distance between ordinal elections, (if applicable) optimal matching """

    inner_distance, main_distance = extract_distance_id(distance_id)

    if main_distance in registered_roommates_distances:
        return registered_roommates_distances.get(main_distance)(election_1,
                                                                 election_2,
                                                                 inner_distance)
    else:
        logging.warning('No such metric!')


def extract_distance_id(distance_id: str) -> (Callable, str):
    if '-' in distance_id:
        inner_distance, main_distance = distance_id.split('-')
        inner_distance = map_str_to_func(inner_distance)
    else:
        main_distance = distance_id
        inner_distance = None
    return inner_distance, main_distance


def run_single_thread(experiment: Experiment,
                      thread_ids: list,
                      distances: dict,
                      times: dict,
                      matchings: dict,
                      t) -> None:
    """ Single thread for computing distances """
    for election_id_1, election_id_2 in thread_ids:
        start_time = time()

        distance = get_distance(copy.deepcopy(experiment.instances[election_id_1]),
                                copy.deepcopy(experiment.instances[election_id_2]),
                                distance_id=copy.deepcopy(experiment.distance_id),
                                )
        if type(distance) is tuple:
            distance, matching = distance
            matching = np.array(matching)
            matchings[election_id_1][election_id_2] = matching
            matchings[election_id_2][election_id_1] = np.argsort(matching)
        distances[election_id_1][election_id_2] = distance
        distances[election_id_2][election_id_1] = distances[election_id_1][election_id_2]
        times[election_id_1][election_id_2] = time() - start_time
        times[election_id_2][election_id_1] = times[election_id_1][election_id_2]

    if experiment.is_exported:

        file_name = f'{experiment.distance_id}_p{t}.csv'
        path = os.path.join(os.getcwd(), "election", experiment.experiment_id, "distances",
                            file_name)

        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(
                ["instance_id_1", "instance_id_2", "distance", "time"])

            for election_id_1, election_id_2 in thread_ids:
                distance = float(distances[election_id_1][election_id_2])
                time_ = float(times[election_id_1][election_id_2])
                writer.writerow([election_id_1, election_id_2, distance, time_])

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
