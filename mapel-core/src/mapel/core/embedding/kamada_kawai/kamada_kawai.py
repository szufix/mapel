import time

import numpy as np

from mapel.core.embedding.initial_positions import initial_place_on_circumference, initial_place_inside_square, \
    initial_place_points
from mapel.core.embedding.kamada_kawai.energy_functions import _close_zero, get_total_energy, get_total_energy_dxy
from mapel.core.embedding.kamada_kawai.optimization_algorithms import optimize_bb, _get_delta_energy, _optimize_newton, \
    adam, _get_pos_k_l_x_y_for_i


class KamadaKawai:
    def __init__(self,
                 special_k=10000,
                 max_neighbour_distance_percentage=None,
                 optim_method='bb',
                 initial_positions_algorithm='circumference',
                 epsilon=0.00001):
        self.special_k = special_k
        self.epsilon = epsilon
        self.max_neighbour_distance_percentage = max_neighbour_distance_percentage
        self.optim_method = optim_method
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

        k = _calc_k_with_special_value(distances, self.special_k, fixed_positions_indexes)
        optim_method_to_fun = {
            'kk': _get_positions_kk,
            'bb': _get_positions_bb,
            'adam': _get_positions_adam
        }

        positions = initial_place_points(distances, initial_positions, self.initial_positions_algorithm)

        start_time = time.time()
        positions = optim_method_to_fun[self.optim_method](distances, k, positions, fixed_positions_indexes)

        k = _calc_k_with_special_value(distances, 1, fixed_positions_indexes)
        print("MIDDLE ENERGY:", get_total_energy(positions, k, distances), "TIME:", time.time() - start_time)

        positions = optim_method_to_fun[self.optim_method](distances, k, positions, fixed_positions_indexes)

        print("FINAL ENERGY:", get_total_energy(positions, k, distances), "TIME:", time.time() - start_time)

        if self.max_neighbour_distance_percentage is not None:
            k = _respect_only_close_neighbours_k(k, distances, self.max_neighbour_distance_percentage)

            positions = optim_method_to_fun[self.optim_method](distances, k, positions, fixed_positions_indexes)
            print("Last adjustments:", get_total_energy(positions, k, distances), "TIME:", time.time() - start_time)

        return positions


def _get_max_derivative(k, distances, positions, fixed_positions_indexes=None):
    num_vertices = distances.shape[0]
    max_derivative = 0, 0
    for i in range(0, num_vertices):
        if fixed_positions_indexes is not None and i in fixed_positions_indexes:
            continue
        pos, k, l, x, y = _get_pos_k_l_x_y_for_i(positions, k, distances, i)

        my_energy = _get_delta_energy(pos, k, l, x, y), i
        if my_energy > max_derivative:
            max_derivative = my_energy

    return max_derivative


def _get_positions_kk(distances, k, l, positions, fixed_positions_indexes, epsilon=0.00001):
    max_derivative = _get_max_derivative(k, distances, positions, fixed_positions_indexes)
    print(max_derivative)
    while max_derivative[0] > epsilon:
        max_der, i = max_derivative
        total_energy = get_total_energy(positions, k, distances)
        print(f'Energy: {total_energy}, max der: {max_der}')
        positions[i], succ = _optimize_newton(positions, k, l, i, epsilon)
        if not succ:
            positions[i] += np.random.uniform(-10, 10, size=(2,))
        max_derivative = _get_max_derivative(k, distances, positions, fixed_positions_indexes)

    return positions


def _get_positions_bb(distances, k, positions, fixed_positions_indexes):
    pos_copy = np.copy(positions)
    new_positions = optimize_bb(
        get_total_energy,
        get_total_energy_dxy,
        args=(k, distances, fixed_positions_indexes),
        x0=pos_copy,
        max_iter=int(1e5),
        init_step_size=1e-3,
        max_iter_without_improvement=1000,
        min_improvement_percentage=0.001,
        percentage_lookup_history=1000,
    )

    return new_positions


def _get_positions_adam(distances, k, positions, fixed_positions_indexes):
    pos_copy = np.copy(positions)
    new_positions = adam(
        get_total_energy,
        get_total_energy_dxy,
        x0=pos_copy,
        args=(k, distances, fixed_positions_indexes),
        learning_rate=1.0,
        maxiter=4000
    )
    return new_positions


def _calc_k_with_special_value(distances, special_value, indexes=None):
    square_dist = distances ** 2
    np.fill_diagonal(square_dist, 1)
    _close_zero(square_dist)
    k = np.ones_like(square_dist)

    if indexes is not None:
        for i in indexes:
            k[:, i] = special_value
            k[i, :] = special_value

    k = k / square_dist
    np.fill_diagonal(k, 0)

    return k


def _respect_only_close_neighbours_k(k, distances, max_distance_percentage):
    k = np.copy(k)
    num_vertices = distances.shape[0]
    max_distance = np.max(distances)
    distance_threshold = max_distance_percentage * max_distance
    for i in range(num_vertices):
        for j in range(num_vertices):
            if abs(distances[i, j]) > distance_threshold:
                k[i, j] = 0.0

    return k
