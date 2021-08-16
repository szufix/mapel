import random as rand
import numpy as np


def generate_single_crossing_election(num_voters=None, num_candidates=None):
    """ helper function: generate simple single-crossing elections"""

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    # GENERATE DOMAIN

    domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)

    domain = [[i for i in range(num_candidates)] for _ in range(domain_size)]

    for line in range(1, domain_size):

        poss = []
        for i in range(num_candidates - 1):
            if domain[line - 1][i] < domain[line - 1][i + 1]:
                poss.append([domain[line - 1][i], domain[line - 1][i + 1]])

        r = rand.randint(0, len(poss) - 1)  # random swap

        for i in range(num_candidates):

            domain[line][i] = domain[line - 1][i]

            if domain[line][i] == poss[r][0]:
                domain[line][i] = poss[r][1]

            elif domain[line][i] == poss[r][1]:
                domain[line][i] = poss[r][0]

    # GENERATE VOTES

    for j in range(num_voters):
        r = rand.randint(0, domain_size - 1)
        votes[j] = list(domain[r])

    return votes


# def get_single_crossing_matrix_slow(num_candidates):
#
#     # GENERATE DOMAIN
#
#     domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)
#
#     domain = [[i for i in range(num_candidates)] for _ in range(domain_size)]
#
#     for line in range(1, domain_size):
#
#         poss = []
#         for i in range(num_candidates - 1):
#             if domain[line - 1][i] < domain[line - 1][i + 1]:
#                 poss.append([domain[line - 1][i], domain[line - 1][i + 1]])
#
#         r = 0  # first swap
#
#         for i in range(num_candidates):
#
#             domain[line][i] = domain[line - 1][i]
#
#             if domain[line][i] == poss[r][0]:
#                 domain[line][i] = poss[r][1]
#
#             elif domain[line][i] == poss[r][1]:
#                 domain[line][i] = poss[r][0]
#
#     # GENERATE MATRIX
#
#     matrix = np.zeros([num_candidates, num_candidates])
#
#     for i in range(domain_size):
#         for j in range(num_candidates):
#             matrix[domain[i][j]][j] += 1
#
#     return matrix

def get_single_crossing_matrix(num_candidates):
    return get_single_crossing_vectors(num_candidates).transpose()


def get_single_crossing_vectors(num_candidates):

    matrix = np.zeros([num_candidates, num_candidates])

    for i in range(num_candidates):
        for j in range(num_candidates-i):
            matrix[i][j] = i+j

    for i in range(num_candidates):
        for j in range(i):
            matrix[i][j] += 1

    sums = [1]
    for i in range(num_candidates):
        sums.append(sums[i]+i)

    for i in range(num_candidates):
        matrix[i][i] += sums[i]
        matrix[i][num_candidates-i-1] -= i

    for i in range(num_candidates):
        denominator = sum(matrix[i])
        for j in range(num_candidates):
            matrix[i][j] /= denominator

    return matrix
