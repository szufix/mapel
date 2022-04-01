from pathlib import Path
from typing import List

import numpy as np

from mapel.main.features.common import extract_selected_coordinates, extract_selected_distances, \
    extract_calculated_distances
from mapel.main.objects.Experiment import Experiment


def _close_zero(number, e=1e-6):
    return e if number <= e else number


def calculate_distortion(experiment: Experiment, election_ids: List[str] = None):
    """

    :param experiment
    :param election_ids: list of elections to take into consideration. If none, takes all.
    :return: dict {election_id: mean distortion}
    """

    if election_ids is None:
        election_ids = list(experiment.distances.keys())

    n = len(election_ids)

    coordinates = extract_selected_coordinates(experiment, election_ids)
    distances = extract_selected_distances(experiment, election_ids)

    calculated_distances = extract_calculated_distances(coordinates)

    max_distance_matrix = np.max([distances, calculated_distances], axis=0)
    min_distance_matrix = np.min([distances, calculated_distances], axis=0)
    np.fill_diagonal(min_distance_matrix, 1)
    distortion_matrix = max_distance_matrix / min_distance_matrix
    np.fill_diagonal(distortion_matrix, 0)
    mean_distortion = np.sum(distortion_matrix, axis=1) / (n - 1)

    return {
        election: mean_distortion[i] for i, election in enumerate(election_ids)
    }
