import numpy as np


def get_total_energy(positions, k, l, special_pos=None):
    positions_delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    pos_squared = positions_delta[:, :, 0] ** 2 + positions_delta[:, :, 1] ** 2

    my_matrix = k * (pos_squared + l ** 2 - 2 * l * np.sqrt(pos_squared)) / 2

    return _upper_tri_sum(my_matrix)


def get_total_energy_dy(new_ys, positions, k, l):
    positions[:, 1] = new_ys
    positions_delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    pos_squared = positions_delta[:, :, 0] ** 2 + positions_delta[:, :, 1] ** 2
    np.fill_diagonal(pos_squared, 1)
    pos_squared = _close_zero(pos_squared)
    matrix = k * (positions_delta[:, :, 1] - l * positions_delta[:, :, 1] / np.sqrt(pos_squared))
    np.fill_diagonal(matrix, 0)
    matrix_sum = matrix.sum(axis=1)

    return matrix_sum


def get_total_energy_dx(new_xs, positions, k, l):
    positions[:, 0] = new_xs
    positions_delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    pos_squared = positions_delta[:, :, 0] ** 2 + positions_delta[:, :, 1] ** 2
    np.fill_diagonal(pos_squared, 1)
    pos_squared = _close_zero(pos_squared)
    matrix = k * (positions_delta[:, :, 0] - l * positions_delta[:, :, 0] / np.sqrt(pos_squared))
    np.fill_diagonal(matrix, 0)
    matrix_sum = matrix.sum(axis=1)

    return matrix_sum


def get_total_energy_dxy(positions, k, l, special_pos=None):
    """

    :param special_pos:
    :param positions:
    :param k:
    :param l:
    :return: [E/dx, E/dy]
    """
    positions_delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    pos_squared = positions_delta[:, :, 0] ** 2 + positions_delta[:, :, 1] ** 2
    np.fill_diagonal(pos_squared, 1)
    pos_squared = _close_zero(np.sqrt(pos_squared))
    matrix_x = k * (positions_delta[:, :, 0] - l * positions_delta[:, :, 0] / pos_squared)
    matrix_y = k * (positions_delta[:, :, 1] - l * positions_delta[:, :, 1] / pos_squared)

    np.fill_diagonal(matrix_x, 0)
    np.fill_diagonal(matrix_y, 0)
    if special_pos is not None:
        matrix_x[special_pos] = 0
        matrix_y[special_pos] = 0
    return np.array([matrix_x.sum(axis=1), matrix_y.sum(axis=1)]).T


def get_energy_dy(x, y, k, l, positions):
    """
    we calc derivative energy for i

    :param x:  x pos
    :param y: y pos
    :param k: np.delete(k[i, :], i)
    :param l: np.delete(l[i, :], i)
    :param positions: np.delete(positions, i, axis=0)
    :return:
    """
    xs_delta = x - positions[:, 0]
    ys_delta = y - positions[:, 1]

    my_vector = k * (ys_delta - l * ys_delta / np.sqrt(_close_zero(xs_delta ** 2 + ys_delta ** 2)))

    return my_vector.sum()


def get_energy_dy_dy(x, y, k, l, positions):
    """
    we calc derivative derivative energy for i

    :param x:  x pos
    :param y: y pos
    :param k: np.delete(k[i, :], i)
    :param l: np.delete(l[i, :], i)
    :param positions: np.delete(positions, i, axis=0)
    :return:
    """
    xs_delta_sq = (x - positions[:, 0]) ** 2
    ys_delta_sq = (y - positions[:, 1]) ** 2

    my_vector = k * (1 - l * xs_delta_sq / np.power(_close_zero(xs_delta_sq + ys_delta_sq), 3 / 2))
    return my_vector.sum()


def get_energy_dx(x, y, k, l, positions):
    """
    we calc derivative energy for i

    :param x:  x pos
    :param y: y pos
    :param k: np.delete(k[i, :], i)
    :param l: np.delete(l[i, :], i)
    :param positions: np.delete(positions, i, axis=0)
    :return:
    """
    xs_delta = x - positions[:, 0]
    ys_delta = y - positions[:, 1]

    my_vector = k * (xs_delta - l * xs_delta / np.sqrt(_close_zero(xs_delta ** 2 + ys_delta ** 2)))

    return my_vector.sum()


def get_energy_dx_dx(x, y, k, l, positions):
    """
    we calc derivative derivative energy for i

    :param x:  x pos
    :param y: y pos
    :param k: np.delete(k[i, :], i)
    :param l: np.delete(l[i, :], i)
    :param positions: np.delete(positions, i, axis=0)
    :return:
    """
    xs_delta_sq = (x - positions[:, 0]) ** 2
    ys_delta_sq = (y - positions[:, 1]) ** 2

    my_vector = k * (1 - l * ys_delta_sq / np.power(_close_zero(xs_delta_sq + ys_delta_sq), 3 / 2))
    return my_vector.sum()


def get_energy_dx_dy(x, y, k, l, positions):
    """
    we calc derivative derivative energy for i

    :param x:  x pos
    :param y: y pos
    :param k: np.delete(k[i, :], i)
    :param l: np.delete(l[i, :], i)
    :param positions: np.delete(positions, i, axis=0)
    :return:
    """
    xs_delta = x - positions[:, 0]
    ys_delta = y - positions[:, 1]

    xs_delta_sq = xs_delta ** 2
    ys_delta_sq = ys_delta ** 2

    my_vector = k * (l * xs_delta * ys_delta / np.power(_close_zero(xs_delta_sq + ys_delta_sq), 3 / 2))
    return my_vector.sum()


def get_energy_dy_dx(x, y, k, l, positions):
    """
    we calc derivative derivative energy for i

    :param x:  x pos
    :param y: y pos
    :param k: np.delete(k[i, :], i)
    :param l: np.delete(l[i, :], i)
    :param positions: np.delete(positions, i, axis=0)
    :return:
    """
    return get_energy_dx_dy(x, y, k, l, positions)


def _upper_tri_sum(matrix):
    return np.triu(matrix, 1).sum()


def _close_zero(matrix, eps=1e-5):
    cond = matrix < eps
    matrix[cond] = eps

    return matrix
