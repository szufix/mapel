import numpy as np


def initial_place_points(distances, initial_positions, initial_positions_algorithm):
    type_to_algorithm = {
        'circumference': initial_place_on_circumference,
        'inside-square': initial_place_inside_square
    }

    positions = type_to_algorithm[initial_positions_algorithm](distances)
    _apply_initial_positions(positions, initial_positions)
    return positions


def _apply_initial_positions(positions, initial_positions):
    if initial_positions is not None:
        for pos_index, position in initial_positions.items():
            positions[pos_index] = position


def initial_place_on_circumference(distances):
    num_vertices = distances.shape[0]
    longest_distance = np.max(distances)

    should_split_in_two = num_vertices > 10

    if should_split_in_two:
        positions = _place_on_circumference(
            [longest_distance / 2, longest_distance * 2],
            [num_vertices // 2, num_vertices - num_vertices // 2]
        )
    else:
        positions = _place_on_circumference(
            [longest_distance],
            [num_vertices]
        )

    return positions


def initial_place_inside_square(distances):
    num_vertices = distances.shape[0]
    longest_distance = np.max(distances)

    return _place_inside_square(longest_distance, num_vertices)


def _place_on_circumference(r, n):
    """
    places points on a circumference
    :param r: list of circuit radius, eg [1, 2]
    :param n: list of number of points, eg [10, 100]
    :return: list of positions [(x, y), ...]
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.extend(np.c_[x, y])
    return np.array(circles)


def _place_inside_square(a, n):
    """
    places points inside a square with a square side equal to a
    :param a: square side
    :param n: number of points to place
    :return: list of positions [(x, y), ...]
    """
    positions = []
    for i in range(n):
        x = np.random.uniform(-a / 2, a / 2)
        y = np.random.uniform(-a / 2, a / 2)
        positions.append((x, y))

    return np.array(positions)
