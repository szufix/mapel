from copy import deepcopy
import math
import numpy as np
import itertools


def map_str_to_func(name):
    return {'l1': l1,
            'l2': l2,
            'chebyshev': chebyshev,
            'hellinger': hellinger,
            'emd': emd,
            'emdinf': emd_infty,
            'discrete': discrete,
            'wl1': wl1,
            }.get(name)


def discrete(vector_1, vector_2) -> int:
    """
    Computes the discrete distance.

    Parameters
    ----------
        vector_1
            First vector.
        vector_2
            Second vector.
    Returns
    -------
        int
            Discrete distance.
    """
    for i in range(len(vector_1)):
        if vector_1[i] != vector_2[i]:
            return 1
    return 0


def single_l1(value_1, value_2) -> float:
    """
    Computes the L1 distance between two values.

    Parameters
    ----------
        value_1
            First value.
        value_2
            Second values.
    Returns
    -------
        float
            L1 distance.
    """
    return abs(value_1 - value_2)


def wl1(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """ Return: L1 distance """
    return sum([len(vector_1)-i*abs(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])


def l1(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Computes the L1 distance.

    Parameters
    ----------
        vector_1
            First vector.
        vector_2
            Second vector.
    Returns
    -------
        float
            L1 distance.
    """
    return np.linalg.norm(vector_1 - vector_2, ord=1)


def l2(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Computes the L2 distance.

    Parameters
    ----------
        vector_1
            First vector.
        vector_2
            Second vector.
    Returns
    -------
        float
            L2 distance.
    """
    return np.linalg.norm(vector_1 - vector_2, ord=2)


def chebyshev(vector_1, vector_2) -> float:
    """
    Computes the Chebyshev distance.

    Parameters
    ----------
        vector_1
            First vector.
        vector_2
            Second vector.
    Returns
    -------
        float
            Chebyshev distance.
    """
    return max([abs(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])


def hellinger(vector_1, vector_2) -> float:
    """
    Computes the Hellinger distance.

    Parameters
    ----------
        vector_1
            First vector.
        vector_2
            Second vector.
    Returns
    -------
        float
            Hellinger distance.
    """
    h1 = np.average(vector_1)
    h2 = np.average(vector_2)
    product = sum([math.sqrt(vector_1[i] * vector_2[i])
                   for i in range(len(vector_1))])
    return math.sqrt(1 - (1 / math.sqrt(h1 * h2 * len(vector_1) * len(vector_1)))
                     * product)


def stretch(xx, mult):
    return [x for _ in range(mult) for x in xx]


def emd_infty(xx, yy):
    if len(xx) != len(yy):
        xx = stretch(xx, math.lcm(len(xx), len(yy)))
        yy = stretch(yy, math.lcm(len(xx), len(yy)))

    m = len(xx)
    cum_x = 0
    cum_y = 0
    res = 0
    for x, y in zip(xx, yy):
        cum_x_ = cum_x
        cum_y_ = cum_y
        cum_x += x
        cum_y += y

        if np.sign(cum_x_ - cum_y_) == np.sign(cum_x - cum_y):
            # Trapezoid case
            res += (abs(cum_x_ - cum_y_) + abs(cum_x - cum_y)) / m / 2
        else:
            # Two triangles case (works also for one triangle)
            d_1 = abs(cum_x_ - cum_y_)
            d_2 = abs(cum_x - cum_y)
            res += (d_1 * d_1 + d_2 * d_2) / (d_1 + d_2) / m / 2

    return res


def emd(vector_1, vector_2) -> float:
    """
    Computes the EMD distance.

    Parameters
    ----------
        vector_1
            First vector.
        vector_2
            Second vector.
    Returns
    -------
        float
            EMD distance.
    """
    vector_1 = deepcopy(vector_1)
    dirt = 0.
    for i in range(len(vector_1) - 1):
        surplus = vector_1[i] - vector_2[i]
        dirt += abs(surplus)
        vector_1[i + 1] += surplus
    return dirt


def hamming(set_1: set, set_2: set) -> int:
    """
    Computes the Hamming distance.

    Parameters
    ----------
        vector_1
            First vector.
        vector_2
            Second vector.
    Returns
    -------
        int
            Hamming distance.
    """
    return len(set_1.symmetric_difference(set_2))


# TMP
def vote_to_pote(vote: list) -> list:
    """ Return: Positional vote """
    return [vote.index(i) for i in range(len(vote)+1) if i in vote]


def swap_distance(vote_1: list, vote_2: list, matching=None) -> int:
    """ Return: Swap distance between two votes """

    new_vote_2 = deepcopy(vote_2)
    if matching is not None:
        for i in range(len(vote_2)):
            new_vote_2[i] = matching[vote_2[i]]

    pote_1 = vote_to_pote(vote_1)
    pote_2 = vote_to_pote(new_vote_2)

    swap_distance = 0
    for i, j in itertools.combinations(pote_1, 2):
        if (pote_1[i] > pote_1[j] and pote_2[i] < pote_2[j]) or \
                (pote_1[i] < pote_1[j] and pote_2[i] > pote_2[j]):
            swap_distance += 1
    return swap_distance


def swap_distance_between_potes(pote_1: list, pote_2: list) -> int:
    """ Return: Swap distance between two potes """
    swap_distance = 0
    for i, j in itertools.combinations(pote_1, 2):
        if (pote_1[i] > pote_1[j] and
            pote_2[i] < pote_2[j]) or \
                (pote_1[i] < pote_1[j] and
                 pote_2[i] > pote_2[j]):
            swap_distance += 1
    return swap_distance


def spearman_distance_between_potes(pote_1: list, pote_2: list) -> int:
    return sum([abs(pote_1[c] - pote_2[c]) for c in range(len(pote_1))])
