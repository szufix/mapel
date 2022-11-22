import unittest
from typing import List

import numpy as np

from mapel.core.features.common import extract_selected_coordinates
from mapel.core.objects.Experiment import Experiment


def calculate_stability(experiment: Experiment, election_ids: List[str] = None, rotate_to_match=True):
    if election_ids is None:
        election_ids = list(experiment.distances.keys())

    coordinates = []

    for coordinate_dict in experiment.coordinates_lists.values():
        coordinates.append(extract_selected_coordinates(coordinate_dict, election_ids))

    if rotate_to_match:
        for i in range(1, len(coordinates)):
            coordinates[i] = rotate_coordinates_to_match(coordinates[i], coordinates[0])

    coordinates_differences = []

    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            coordinates_differences.append(np.linalg.norm(coordinates[i] - coordinates[j], axis=1))

    coordinates_differences = np.array(coordinates_differences)
    differences_mean = np.mean(coordinates_differences, axis=0)

    return {
        election: differences_mean[i] for i, election in enumerate(election_ids)
    }


def rotate_via_numpy(coordinates, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(coordinates, j)

    return m


def rotate_coordinates_to_match(coordinates_to_rotate, coordinates_to_match):
    assert coordinates_to_rotate.shape == coordinates_to_match.shape

    num_rows, num_cols = coordinates_to_rotate.shape
    if num_cols != 2:
        raise Exception(f"matrix is not Nx2, it is {num_rows}x{num_cols}")

    coordinates_to_rotate = np.transpose(coordinates_to_rotate, [1, 0])
    coordinates_to_match = np.transpose(coordinates_to_match, [1, 0])

    centroid_a = np.mean(coordinates_to_rotate, axis=1)
    centroid_b = np.mean(coordinates_to_match, axis=1)

    # ensure centroids are 1x2
    centroid_a = centroid_a.reshape(-1, 1)
    centroid_b = centroid_b.reshape(-1, 1)

    # subtract mean
    Am = coordinates_to_rotate - centroid_a
    Bm = coordinates_to_match - centroid_b

    H = Am @ np.transpose(Bm)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_a + centroid_b

    return ((R @ coordinates_to_rotate) + t).T


class TestStability(unittest.TestCase):
    def test_rotation(self):
        n = 100
        coordinates = np.random.uniform(-100, 100, size=(n, 2))
        coordinates_rotated = rotate_via_numpy(coordinates, radians=2)
        self.assertFalse(np.allclose(coordinates, coordinates_rotated))

        coordinates_rotated = rotate_coordinates_to_match(coordinates_rotated, coordinates)
        self.assertTrue(np.allclose(coordinates, coordinates_rotated))
