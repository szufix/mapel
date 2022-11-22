import random
import unittest
import uuid
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np

from mapel.core.features.common import extract_selected_coordinates_from_experiment, extract_selected_distances, \
    extract_calculated_distances, MockExperiment
from mapel.core.objects.Experiment import Experiment


def _close_zero(number, e=1e-6):
    return e if number <= e else number


def calculate_distortion(experiment: Experiment, election_ids: List[str] = None, max_distance_percentage=1.0,
                         normalize=True):
    """

    :param normalize: whether to
    :param max_distance_percentage:
    :param experiment
    :param election_ids: list of elections to take into consideration. If none, takes all.
    :return: dict {election_id: mean distortion}
    """

    if election_ids is None:
        election_ids = list(experiment.distances.keys())

    n = len(election_ids)

    coordinates = extract_selected_coordinates_from_experiment(experiment, election_ids)
    distances = extract_selected_distances(experiment, election_ids)

    calculated_distances = np.linalg.norm(coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2)
    if normalize:
        calculated_distances /= np.max(calculated_distances)
        distances /= np.max(distances)

    max_distance_matrix = np.max([distances, calculated_distances], axis=0)
    min_distance_matrix = np.min([distances, calculated_distances], axis=0)

    max_distance = np.max(distances)
    bad_distances_mask = distances > max_distance * max_distance_percentage

    np.fill_diagonal(min_distance_matrix, 1)
    distortion_matrix = max_distance_matrix / min_distance_matrix
    np.fill_diagonal(distortion_matrix, 0)

    distortion_matrix[bad_distances_mask] = 0

    mean_distortion = np.sum(distortion_matrix, axis=1) / (n - 1 - bad_distances_mask.sum(axis=1))

    return {
        election: mean_distortion[i] for i, election in enumerate(election_ids)
    }


def calculate_distortion_naive(experiment: Experiment, election_ids: List[str] = None, max_distance_percentage=1.0,
                               normalize=True):
    coordinates = extract_selected_coordinates_from_experiment(experiment, election_ids)

    desired_distances = extract_selected_distances(experiment, election_ids)
    calculated_distances = extract_calculated_distances(coordinates)
    if normalize:
        calculated_distances /= np.max(calculated_distances)
        desired_distances /= np.max(desired_distances)

    max_distance = np.max(desired_distances)
    allowed_distance = max_distance * max_distance_percentage

    distortions = defaultdict(list)

    n = len(election_ids)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = desired_distances[i, j]
            if d1 <= allowed_distance:
                d2 = calculated_distances[i, j]
                if d1 > d2:
                    my_distortion = d1 / d2
                else:
                    my_distortion = d2 / d1

                distortions[i].append(my_distortion)
                distortions[j].append(my_distortion)

    return {
        election: np.mean(distortions[i]) for i, election in enumerate(election_ids)
    }


class TestDistortion(unittest.TestCase):
    def test_calculate_monotonicity(self):
        n = 500
        election_ids = [str(uuid.uuid4()) for _ in range(n)]

        experiment = MockExperiment(election_ids)

        elections_subset = random.sample(election_ids, 300)

        m1 = calculate_distortion(experiment, elections_subset, 0.9)
        print("m1 done")
        m2 = calculate_distortion_naive(experiment, elections_subset, 0.9)
        print("m2 done")

        for el_id in elections_subset:
            self.assertAlmostEqual(m1[el_id], m2[el_id])
