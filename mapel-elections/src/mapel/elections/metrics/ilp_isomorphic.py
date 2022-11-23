#!/usr/bin/env python

import os
from contextlib import suppress
import numpy as np
import scipy.special

try:
    import cplex
except ImportError:
    cplex = None


# SPEARMAN
def generate_ilp_spearman_distance(lp_file_name, votes_1, votes_2, params):
    lp_file = open(lp_file_name, 'w')
    lp_file.write("Minimize\n")

    first = True
    for k in range(params['voters']):
        for l in range(params['voters']):

            vote_1 = votes_1[k]
            vote_2 = votes_2[l]

            pote_1 = [0] * params['candidates']
            pote_2 = [0] * params['candidates']

            for i in range(params['candidates']):
                pote_1[vote_1[i]] = i
                pote_2[vote_2[i]] = i

            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    if not first:
                        lp_file.write(" + ")
                    first = False

                    weight = abs(pote_1[i] - pote_2[j])

                    lp_file.write(
                        str(weight) + " P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(
                            j))
    lp_file.write("\n")

    lp_file.write("Subject To\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
                    lp_file.write(" - " + "M" + "i" + str(i) + "j" + str(j) + " <= 0" + "\n")

                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
                    lp_file.write(" - " + "N" + "k" + str(k) + "l" + str(l) + " <= 0" + "\n")

    for k in range(params['voters']):
        first = True
        for l in range(params['voters']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("N" + "k" + str(k) + "l" + str(l))
        lp_file.write(" = 1" + "\n")

    for l in range(params['voters']):
        first = True
        for k in range(params['voters']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("N" + "k" + str(k) + "l" + str(l))
        lp_file.write(" = 1" + "\n")

    for i in range(params['candidates']):
        first = True
        for j in range(params['candidates']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("M" + "i" + str(i) + "j" + str(j))
        lp_file.write(" = 1" + "\n")

    for j in range(params['candidates']):
        first = True
        for i in range(params['candidates']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("M" + "i" + str(i) + "j" + str(j))
        lp_file.write(" = 1" + "\n")

    for k in range(params['voters']):
        for i in range(params['candidates']):
            first = True
            for l in range(params['voters']):
                for j in range(params['candidates']):
                    if not first:
                        lp_file.write(" + ")
                    first = False
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
            lp_file.write(" = 1" + "\n")

    for l in range(params['voters']):
        for j in range(params['candidates']):
            first = True
            for k in range(params['voters']):
                for i in range(params['candidates']):
                    if not first:
                        lp_file.write(" + ")
                    first = False
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
            lp_file.write(" = 1" + "\n")

    lp_file.write("Binary\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    lp_file.write(
                        "P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j) + "\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            lp_file.write("N" + "k" + str(k) + "l" + str(l) + "\n")

    for i in range(params['candidates']):
        for j in range(params['candidates']):
            lp_file.write("M" + "i" + str(i) + "j" + str(j) + "\n")

    lp_file.write("End\n")


def solve_ilp_distance(path):
    cp_lp = cplex.Cplex(path)
    cp_lp.set_results_stream(None)

    cp_lp.parameters.threads.set(1)

    try:
        cp_lp.solve()

    except cplex.CplexSolverError:
        print("Exception raised during solve")
        return

    return cp_lp.solution.get_objective_value()


def spearman_cost(single_votes_1, single_votes_2, params, perm):
    pote_1 = [0] * params['candidates']
    pote_2 = [0] * params['candidates']

    for i in range(params['candidates']):
        id_1 = int(perm[0][single_votes_1[i]])
        pote_1[id_1] = i
        id_2 = int(perm[1][single_votes_2[i]])
        pote_2[id_2] = i

    total_diff = 0.
    for i in range(params['candidates']):
        local_diff = float(abs(pote_1[i] - pote_2[i]))
        total_diff += local_diff

    return total_diff


def spearman_cost_per_cand(single_votes_1, single_votes_2, params, perm):
    pote_1 = [0] * params['candidates']
    pote_2 = [0] * params['candidates']

    for i in range(params['candidates']):
        id_1 = int(perm[0][single_votes_1[i]])
        pote_1[id_1] = i
        id_2 = int(perm[1][single_votes_2[i]])
        pote_2[id_2] = i

    cand_diff = [0] * params['candidates']
    for i in range(params['candidates']):
        cand_diff[i] = float(abs(pote_1[i] - pote_2[i]))

    return cand_diff


def remove_lp_file(path):
    """ Safely remove lp file """
    with suppress(OSError):
        os.remove(path)


# SWAP
def generate_ilp_swap_distance(lp_file_name, votes_1, votes_2, params):

    all = int(params['voters'] * scipy.special.binom(params['candidates'], 2))

    lp_file = open(lp_file_name, 'w')
    lp_file.write("Minimize\n")

    first = True
    for k in range(params['voters']):
        for l in range(params['voters']):

            # pote_1 = [0] * params['candidates']
            # pote_2 = [0] * params['candidates']
            #
            # for i in range(params['candidates']):
            #     pote_1[votes_1[k][i]] = i
            #     pote_2[votes_2[l][i]] = i

            vote_1 = votes_1[k]
            vote_2 = votes_2[l]

            pote_1 = [list(vote_1).index(i) for i, _ in enumerate(vote_1)]
            pote_2 = [list(vote_2).index(i) for i, _ in enumerate(vote_2)]

            # pote_1 = [0] * params['candidates']
            # pote_2 = [0] * params['candidates']

            # for i in range(params['candidates']):
            #     pote_1[vote_1[i]] = i
            #     pote_2[vote_2[i]] = i

            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(i1+1, params['candidates']):
                    # for i2 in range(params['candidates']):
                        for j2 in range(params['candidates']):

                            if i1 == i2 or j1 == j2:
                                continue

                            # if k==1 and l==0 and i1==0 and j1==2 and i2==1 and j2==1:
                            #     print(pote_1[i1], pote_2[j1], pote_1[i2], pote_2[j2])

                            # if (pote_1[i1] > pote_2[j1] and pote_1[i2] < pote_2[j2]) or \
                            #         (pote_1[i1] < pote_2[j1] and pote_1[i2] > pote_2[j2]):
                            if (pote_1[i1] > pote_1[i2] and pote_2[j1] < pote_2[j2]) or \
                                    (pote_1[i1] < pote_1[i2] and pote_2[j1] > pote_2[j2]) :

                                if not first:
                                    lp_file.write(" + ")
                                first = False
                                lp_file.write(" R" + "k" + str(k) + "l" + str(l) + "i" + str(i1) + "j" + str(
                                    j1) + "i" + str(i2) + "j" + str(j2))
    lp_file.write("\n")

    lp_file.write("Subject To\n")

    # N=1
    for k in range(params['voters']):
        first = True
        for l in range(params['voters']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("N" + "k" + str(k) + "l" + str(l))
        lp_file.write(" = 1" + "\n")

    # N=1
    for l in range(params['voters']):
        first = True
        for k in range(params['voters']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("N" + "k" + str(k) + "l" + str(l))
        lp_file.write(" = 1" + "\n")

    # M=1
    for i in range(params['candidates']):
        first = True
        for j in range(params['candidates']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("M" + "i" + str(i) + "j" + str(j))
        lp_file.write(" = 1" + "\n")

    # M=1
    for j in range(params['candidates']):
        first = True
        for i in range(params['candidates']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("M" + "i" + str(i) + "j" + str(j))
        lp_file.write(" = 1" + "\n")

    # P=1
    for k in range(params['voters']):
        for i in range(params['candidates']):
            first = True
            for l in range(params['voters']):
                for j in range(params['candidates']):
                    if not first:
                        lp_file.write(" + ")
                    first = False
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
            lp_file.write(" = 1" + "\n")

    # P=1
    for l in range(params['voters']):
        for j in range(params['candidates']):
            first = True
            for k in range(params['voters']):
                for i in range(params['candidates']):
                    if not first:
                        lp_file.write(" + ")
                    first = False
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
            lp_file.write(" = 1" + "\n")

    # P<N, P<M
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
                    lp_file.write(" - " + "N" + "k" + str(k) + "l" + str(l) + " <= 0" + "\n")

                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
                    lp_file.write(" - " + "M" + "i" + str(i) + "j" + str(j) + " <= 0" + "\n")

    # All
    first = True
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(i1+1, params['candidates']):
                    # for i2 in range(params['candidates']):
                        for j2 in range(params['candidates']):

                            if i1 == i2 or j1 == j2:
                                continue

                            if not first:
                                lp_file.write(" + ")
                            first = False
                            lp_file.write("R" + "k" + str(k) + "l" + str(l) + "i" + str(i1) + "j" + str(j1) + "i" + str(
                                i2) + "j" + str(j2))
    lp_file.write(" = " + str(all) + "\n")

    # R<P, R<P
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(i1+1, params['candidates']):
                    # for i2 in range(params['candidates']):
                        for j2 in range(params['candidates']):

                            if i1 == i2 or j1 == j2:
                                continue

                            lp_file.write("R" + "k" + str(k) + "l" + str(l) + "i" + str(i1) + "j" + str(j1) + "i" + str(
                                i2) + "j" + str(j2))
                            lp_file.write(" - " + "P" + "k" + str(k) + "l" + str(l) + "i" + str(i1) + "j" + str(
                                j1) + " <= 0" + "\n")

                            lp_file.write("R" + "k" + str(k) + "l" + str(l) + "i" + str(i1) + "j" + str(j1) + "i" + str(
                                i2) + "j" + str(j2))
                            lp_file.write(" - " + "P" + "k" + str(k) + "l" + str(l) + "i" + str(i2) + "j" + str(
                                j2) + " <= 0" + "\n")

    lp_file.write("Binary\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(i1+1, params['candidates']):
                    # for i2 in range(params['candidates']):
                        for j2 in range(params['candidates']):

                            if i1 == i2 or j1 == j2:
                                continue

                            lp_file.write("R" + "k" + str(k) + "l" + str(l) + "i" + str(i1) + "j" + str(j1) + "i" + str(
                                i2) + "j" + str(j2) + "\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j) + "\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            lp_file.write("N" + "k" + str(k) + "l" + str(l) + "\n")

    for i in range(params['candidates']):
        for j in range(params['candidates']):
            lp_file.write("M" + "i" + str(i) + "j" + str(j) + "\n")

    lp_file.write("End\n")


def generate_ilp_distance_swap_short(lp_file_name, votes_1, votes_2, params):
    lp_file = open(lp_file_name, 'w')
    lp_file.write("Minimize\n")  # obj: ")

    first = True
    for k in range(params['voters']):
        for l in range(params['voters']):

            pote_1 = [0] * params['candidates']
            pote_2 = [0] * params['candidates']

            for i in range(params['candidates']):
                pote_1[votes_1[k][i]] = i
                pote_2[votes_2[l][i]] = i

            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(params['candidates']):
                        for j2 in range(params['candidates']):



                            if not first:
                                lp_file.write(" + ")
                            first = False

                            if pote_1[i1] > pote_2[i2] and pote_1[j1] < pote_2[j2]:
                                weight = 1
                            elif pote_1[i1] < pote_2[i2] and pote_1[j1] > pote_2[j2]:
                                weight = 1
                            else:
                                weight = 0

                            lp_file.write(str(weight) + " R" + str(k) + str(l) + str(i1) + str(j1) + str(i2) + str(j2))
    lp_file.write("\n")

    lp_file.write("Subject To\n")

    # N=1
    for k in range(params['voters']):
        first = True
        for l in range(params['voters']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("N" + "k" + str(k) + "l" + str(l))
        lp_file.write(" = 1" + "\n")

    # N=1
    for l in range(params['voters']):
        first = True
        for k in range(params['voters']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("N" + "k" + str(k) + "l" + str(l))
        lp_file.write(" = 1" + "\n")

    # M=1
    for i in range(params['candidates']):
        first = True
        for j in range(params['candidates']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("M" + "i" + str(i) + "j" + str(j))
        lp_file.write(" = 1" + "\n")

    # M=1
    for j in range(params['candidates']):
        first = True
        for i in range(params['candidates']):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write("M" + "i" + str(i) + "j" + str(j))
        lp_file.write(" = 1" + "\n")

    # P=1
    for k in range(params['voters']):
        for i in range(params['candidates']):
            first = True
            for l in range(params['voters']):
                for j in range(params['candidates']):
                    if not first:
                        lp_file.write(" + ")
                    first = False
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
            lp_file.write(" = 1" + "\n")

    # P=1
    for l in range(params['voters']):
        for j in range(params['candidates']):
            first = True
            for k in range(params['voters']):
                for i in range(params['candidates']):
                    if not first:
                        lp_file.write(" + ")
                    first = False
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
            lp_file.write(" = 1" + "\n")

    # P<N, P<M
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
                    lp_file.write(" - " + "N" + "k" + str(k) + "l" + str(l) + " <= 0" + "\n")

                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
                    lp_file.write(" - " + "M" + "i" + str(i) + "j" + str(j) + " <= 0" + "\n")

    # R=M
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    first = True
                    for i2 in range(params['candidates']):
                        for j2 in range(params['candidates']):
                            if not first:
                                lp_file.write(" + ")
                            first = False
                            lp_file.write("R" + "k" + str(k) + "l" + str(l) + str(i1) + str(j1) + str(i2) + str(j2))
                    lp_file.write(" = " + str(params['candidates']) + "\n")

    # R=M
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i2 in range(params['candidates']):
                for j2 in range(params['candidates']):
                    first = True
                    for i1 in range(params['candidates']):
                        for j1 in range(params['candidates']):
                            if not first:
                                lp_file.write(" + ")
                            first = False
                            lp_file.write("R" + str(k) + str(l) + str(i1) + str(j1) + str(i2) + str(j2))
                    lp_file.write(" = " + str(params['candidates']) + "\n")

    # R<P, R<P
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(params['candidates']):
                        for j2 in range(params['candidates']):

                            lp_file.write("R" + str(k) + str(l) + str(i1) + str(j1) + str(i2) + str(j2))
                            lp_file.write(" - " + "P" + "k" + str(k) + "l" + str(l) + "i" + str(i1) + "j" + str(
                                j1) + " <= 0" + "\n")


                            lp_file.write("R" + str(k) + str(l) + str(i1) + str(j1) + str(i2) + str(j2))
                            lp_file.write(" - " + "P" + "k" + str(k) + "l" + str(l) + "i" + str(i2) + "j" + str(
                                j2) + " <= 0" + "\n")

    lp_file.write("Binary\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(params['candidates']):
                        for j2 in range(params['candidates']):
                            lp_file.write("R" + str(k) + str(l) + str(i1) + str(j1) + str(i2) + str(j2) + "\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j) + "\n")

    for k in range(params['voters']):
        for l in range(params['voters']):
            lp_file.write("N" + "k" + str(k) + "l" + str(l) + "\n")

    for i in range(params['candidates']):
        for j in range(params['candidates']):
            lp_file.write("M" + "i" + str(i) + "j" + str(j) + "\n")

    lp_file.write("End\n")


def solve_ilp_distance_swap(path, votes_1, votes_2, params):

    cp_lp = cplex.Cplex(path)
    cp_lp.set_results_stream(None)

    cp_lp.parameters.threads.set(1)

    try:
        cp_lp.solve()

    except cplex.CplexSolverError:
        print("Exception raised during solve")
        return

    # total = 0
    # for k in range(params['voters']):
    #     for l in range(params['voters']):
    #
    #         pote_1 = [0] * params['candidates']
    #         pote_2 = [0] * params['candidates']
    #
    #         for i in range(params['candidates']):
    #             pote_1[votes_1[k][i]] = i
    #             pote_2[votes_2[l][i]] = i
    #
    #         for i1 in range(params['candidates']):
    #             for j1 in range(params['candidates']):
    #                 for i2 in range(i1+1, params['candidates']):
    #                     for j2 in range(params['candidates']):
    #
    #                         if i1 == i2 or j1 == j2:
    #                             continue
    #
    #                         if pote_1[i1] > pote_2[j1] and pote_1[i2] < pote_2[j2]:
    #                             weight = 1
    #                         elif pote_1[i1] < pote_2[j1] and pote_1[i2] > pote_2[j2]:
    #                             weight = 1
    #                         else:
    #                             weight = 0
    #
    #                         A = "R" + "k" + str(k) + "l" + str(l) + "i" + str(i1) + "j" + str(j1) + "i" + str(i2) + "j" + str(j2)
    #                         # print(weight, int(cp_lp.solution.get_values(A)))
    #                         if int(cp_lp.solution.get_values(A)) == 1:
    #                             print(A, weight)
    #                         total += weight * int(cp_lp.solution.get_values(A))
    # print(f'total {total}')

    return cp_lp.solution.get_objective_value()