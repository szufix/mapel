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

    coordinates = extract_selected_coordinates(experiment, election_ids)

    calculated_triangles_dist = _get_triangles_distances(
        np.linalg.norm(coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2))
    desired_triangles_dist = _get_triangles_distances(extract_selected_distances(experiment, election_ids))


def _get_triangles_distances(distances):
    n = distances.shape[0]
    coordinates_delta = _remove_diagonal(distances)

    coordinates_delta = coordinates_delta[:, :, np.newaxis] - coordinates_delta[:, np.newaxis, :]

    iu1 = np.triu_indices(n - 1, k=1)
    fill = np.zeros(shape=(n - 1, n - 1), dtype=np.bool)
    fill[iu1] = True

    return coordinates_delta[:, fill]
