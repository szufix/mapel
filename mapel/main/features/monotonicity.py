from typing import List

import numpy as np

from mapel.main.features.common import extract_calculated_distances, extract_selected_distances, \
    extract_selected_coordinates
from mapel.main.objects.Experiment import Experiment


def _remove_diagonal(A):
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)


def calculate_monotonicity(experiment: Experiment, election_ids: List[str] = None):
    if election_ids is None:
        election_ids = list(experiment.distances.keys())

    n = len(election_ids)

    coordinates = extract_selected_coordinates(experiment, election_ids)
    distances = extract_selected_distances(experiment, election_ids)

    calculated_distances = extract_calculated_distances(coordinates)
    coordinates_delta = np.linalg.norm(coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2)
    coordinates_delta = np.transpose(coordinates_delta[np.newaxis, :, :] - coordinates_delta[:, np.newaxis, :],
                                     [2, 1, 0])

    print(coordinates_delta[0, 1, 2])
    print(coordinates_delta[1, 0, 2])
    print(coordinates_delta[2, 1, 0])
    print(coordinates_delta[3, 1, 2])

    print(np.linalg.norm(coordinates[0] - coordinates[1]) - np.linalg.norm(coordinates[0] - coordinates[2]))
    print(np.linalg.norm(coordinates[1] - coordinates[0]) - np.linalg.norm(coordinates[1] - coordinates[2]))
    print(np.linalg.norm(coordinates[2] - coordinates[1]) - np.linalg.norm(coordinates[2] - coordinates[0]))
    print(np.linalg.norm(coordinates[3] - coordinates[1]) - np.linalg.norm(coordinates[3] - coordinates[2]))
