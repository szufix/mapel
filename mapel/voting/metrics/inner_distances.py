
import math
import numpy as np

### MAPPING ###
def map_str_to_func(name):
    return {'l1': l1,
            'l2': l2,
            'chebyshev': chebyshev,
            'hellinger': hellinger,
            'emd': emd,
            'discrete': discrete,
            }.get(name)


def discrete(vector_1, vector_2, length):
    """ compute DISCRETE metric """
    for i in range(length(vector_1)):
        if vector_1[i] != vector_2[i]:
            return 1
    return 0


def l1(vector_1, vector_2):
    """ compute L1 metric """
    return sum([abs(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])


def l2(vector_1, vector_2):
    """ compute L2 metric """
    return math.pow(sum([math.pow((vector_1[i] - vector_2[i]), 2)
                         for i in range(len(vector_1))]), 0.5)


def chebyshev(vector_1, vector_2):
    """ compute CHEBYSHEV metric """
    return max([abs(vector_1[i] - vector_2[i]) for i in range(len(vector_1))])


def hellinger(vector_1, vector_2, length):
    """ compute HELLINGER metric """
    h1 = np.average(vector_1)
    h2 = np.average(vector_2)
    product = sum([math.sqrt(vector_1[i] * vector_2[i])
                   for i in range(len(vector_1))])
    return math.sqrt(1 - (1 / math.sqrt(h1 * h2 * length * length(vector_1)))
                     * product)


def emd(vector_1, vector_2, length):
    """ compute EMD metric """
    dirt = 0.
    for i in range(len(vector_1) - 1):
        surplus = vector_1[i] - vector_2[i]
        dirt += abs(surplus)
        vector_1[i + 1] += surplus
    return dirt


def hamming(set_1, set_2):
    """ Compute HAMMING metric """
    return len(set_1) + len(set_2) - 2 * len(set_1.intersection(set_2))
