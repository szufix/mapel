import math
import time
from random import random

import numpy as np

from mapel.core.embedding.initial_positions import initial_place_points
from mapel.core.embedding.simulated_annealing.simulated_annealing_energy import get_total_energy


class SimulatedAnnealing:
    def __init__(self, initial_temperature=100000,
                 cooling_temp_factor=0.75,
                 num_stages=10,
                 number_of_trials_for_temp=30,
                 cooling_radius_factor=None,
                 initial_radius=None,
                 initial_positions_algorithm='circumference'):
        self.initial_temperature = initial_temperature
        self.cooling_temp_factor = cooling_temp_factor
        self.num_stages = num_stages
        self.number_of_trials_for_temp = number_of_trials_for_temp
        self.cooling_radius_factor = cooling_radius_factor
        self.initial_radius = initial_radius
        self.initial_positions_algorithm = initial_positions_algorithm

    def embed(self, distances: np.array, initial_positions: dict = None, fix_initial_positions: bool = True):
        """

        :param distances: matrix nxn
        :param initial_positions: optional dictionary {node_index: (x, y)}
        :param fix_initial_positions: if true, initial positions won't change
        :return: list of positions for each vertex [(x1, y1), ...]
        """

        if initial_positions is not None and fix_initial_positions:
            fixed_positions_indexes = list(initial_positions.keys())
        else:
            fixed_positions_indexes = []

        positions = initial_place_points(distances, initial_positions, self.initial_positions_algorithm)

        start_time = time.time()

        ann = SimRunner(
            positions, distances,
            self.initial_temperature,
            num_stages=self.num_stages,
            number_of_trials_for_temp=self.number_of_trials_for_temp,
            frozen_node_indexes=fixed_positions_indexes,
            cooling_temp_factor=self.cooling_temp_factor,
            cooling_radius_factor=self.cooling_radius_factor
        )

        return ann.run()


class SimRunner:
    def __init__(self, initial_positions, distances, temperature, frozen_node_indexes=None, cooling_temp_factor=0.75,
                 num_stages=10, number_of_trials_for_temp=30, cooling_radius_factor=0.75, radius=None):
        if cooling_radius_factor is None:
            cooling_radius_factor = cooling_temp_factor

        self.cooling_radius_factor = cooling_radius_factor
        self.positions = initial_positions
        self.distances = distances
        self.num_elections = initial_positions.shape[0]
        if frozen_node_indexes is None:
            frozen_node_indexes = []

        self.frozen_node_indexes = frozen_node_indexes
        if radius is None:
            self.radius = np.amax(self.distances)
        else:
            self.radius = radius

        self.temperature = temperature
        self.cooling_temp_factor = cooling_temp_factor
        self.num_stages = num_stages

        self.number_of_trials_for_temp = number_of_trials_for_temp * self.num_elections

    def _get_rand_point_index(self):
        index = np.random.randint(0, self.num_elections)
        while index in self.frozen_node_indexes:
            index = np.random.randint(0, self.num_elections)
        return index

    def move(self, positions):
        index = self._get_rand_point_index()
        center = positions[index]
        new_position = _rotate(center, self.radius, np.random.uniform(0, 2 * math.pi))
        positions[index] = new_position
        return positions

    def run(self):
        energy = get_total_energy(self.positions, self.distances)
        new_positions = self.move(np.copy(self.positions))
        new_energy = get_total_energy(new_positions, self.distances)
        print(f"Initial Energy: {energy}")

        for i in range(self.num_stages):
            for j in range(self.number_of_trials_for_temp):
                accept = (new_energy < energy or random() < np.exp((energy - new_energy) / self.temperature))
                if accept:
                    print(
                        f"Accepting new energy: {new_energy} temp: {self.temperature}. Temerature Iteration: {j}/{self.number_of_trials_for_temp}. Global Iteration: {i}/{self.num_stages}")
                    self.positions = new_positions
                    energy = new_energy

                new_positions = self.move(np.copy(self.positions))
                new_energy = get_total_energy(new_positions, self.distances)
            self.temperature *= self.cooling_temp_factor
            self.radius *= self.cooling_radius_factor

            print(f"energy: {energy}")
        print(f"Final energy: {energy}.")
        return self.positions


def _rotate(center, radius, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    point = center + radius
    cx, cy = center
    px, py = point

    qx = cx + math.cos(angle) * (px - cx) - math.sin(angle) * (py - cy)
    qy = cy + math.sin(angle) * (px - cx) + math.cos(angle) * (py - cy)
    return qx, qy