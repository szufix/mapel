from typing import List, Dict

import numpy as np

from mapel.core.objects.Experiment import Experiment


def extract_selected_distances(experiment: Experiment, election_ids: List[str]):
    n = len(election_ids)

    distances = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i + 1, n):
            election_id_1 = election_ids[i]
            election_id_2 = election_ids[j]
            distances[i, j] = distances[j, i] = experiment.distances[election_id_1][election_id_2]

    return distances


def extract_selected_coordinates(coordinates: Dict, election_ids: List[str]):
    return np.array([coordinates[election_id] for election_id in election_ids])


def extract_selected_coordinates_from_experiment(experiment: Experiment, election_ids: List[str]):
    return extract_selected_coordinates(experiment.coordinates, election_ids)


def extract_calculated_distances(coordinates: np.array):
    n = coordinates.shape[0]
    calculated_distances = np.zeros(shape=(n, n))

    for i in range(n):
        pos_i = coordinates[i]
        for j in range(i + 1, n):
            pos_j = coordinates[j]
            calculated_distances[i, j] = calculated_distances[j, i] = np.linalg.norm(pos_i - pos_j)

    return calculated_distances


class MockExperiment:
    def __init__(self, election_ids):
        self.coordinates = {
            e: np.random.uniform(-10, 10, size=(2,)) for e in election_ids
        }

        self.distances = {
            e1: {e2: np.random.uniform(1, 20) for e2 in election_ids}
            for e1 in election_ids
        }
