import numpy as np


def get_total_energy(positions, desired_distances, node_distribution_param=1, desired_distance_param=1,
                     borderlines_param=0.1,
                     edge_lengths_param=1, node_edge_distances_param=0):
    positions_delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    pos_delta_squared = positions_delta[:, :, 0] ** 2 + positions_delta[:, :, 1] ** 2
    current_distances = np.sqrt(pos_delta_squared)

    np.fill_diagonal(pos_delta_squared, 1)
    node_dist_factor = node_distribution_param / pos_delta_squared
    np.fill_diagonal(pos_delta_squared, 0)

    desired_dist_factor = desired_distance_param * np.power(desired_distances - current_distances, 2)
    return (node_dist_factor + desired_dist_factor).sum()
