#!/usr/bin/env python

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import cplex
except ImportError:
    cplex = None


def solve_matching_vectors(cost_table) -> (float, list):
    """ Return: objective value, optimal matching """
    cost_table = np.array(cost_table)
    row_ind, col_ind = linear_sum_assignment(cost_table)
    return cost_table[row_ind, col_ind].sum(), list(col_ind)


def solve_matching_matrices(matrix_1, matrix_2, length, inner_distance) -> float:
    """ Return: objective value"""
    return _generate_lp_file_matching_matrix(matrix_1, matrix_2, length, inner_distance)


def _generate_lp_file_matching_matrix(matrix_1, matrix_2, length, inner_distance):

    cp = cplex.Cplex()
    cp.parameters.threads.set(1)
    cp.objective.set_sense(cp.objective.sense.minimize)

    # OBJECTIVE FUNCTION
    names = []
    obj = []
    for k in range(length):
        for l in range(length):
            for i in range(length):
                if i == k:
                    continue
                for j in range(length):
                    if j == l:
                        continue

                    weight = inner_distance(np.array([matrix_1[k][i]]), np.array([matrix_2[l][j]]))

                    names.append(f'Pk{k}l{l}i{i}j{j}')
                    obj.append(weight)

    types = [cp.variables.type.binary] * len(names)
    cp.variables.add(obj=obj, names=names, types=types)


    # ADD MISSING VARIABLES
    names = []
    for i in range(length):
        for j in range(length):
            names.append(f'Mi{i}j{j}')
    cp.variables.add(names=list(names),
                     types=[cp.variables.type.binary] * len(names))


    # FIRST GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for k in range(length):
        for l in range(length):
            for i in range(length):
                if i == k:
                    continue
                for j in range(length):
                    if j == l:
                        continue

                    ind = [f'Pk{k}l{l}i{i}j{j}', f'Mi{i}j{j}']
                    val = [1., -1.]
                    lin_expr.append(cplex.SparsePair(ind=ind, val=val))
                    rhs.append(0)

                    ind = [f'Pk{k}l{l}i{i}j{j}', f'Mi{k}j{l}']
                    val = [1., -1.]
                    lin_expr.append(cplex.SparsePair(ind=ind, val=val))
                    rhs.append(0)

    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['L'] * len(rhs),
                              rhs=rhs,
                              names=['C1_' + str(i) for i in range(len(rhs))])

    # SECOND GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for i in range(length):
        ind = []
        val = []
        for j in range(length):
            ind.append(f'Mi{i}j{j}')
            val.append(1.)
        lin_expr.append(cplex.SparsePair(ind=ind, val=val))
        rhs.append(1.)
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'] * len(rhs),
                              rhs=rhs,
                              names=['C2_' + str(i) for i in range(len(rhs))])

    # THIRD GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for j in range(length):
        ind = []
        val = []
        for i in range(length):
            ind.append(f'Mi{i}j{j}')
            val.append(1.)
        lin_expr.append(cplex.SparsePair(ind=ind, val=val))
        rhs.append(1.)
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'] * len(rhs),
                              rhs=rhs,
                              names=['C3_' + str(i) for i in range(len(rhs))])

    # FORTH GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for k in range(length):
        for i in range(length):
            if k == i:
                continue
            ind = []
            val = []
            for l in range(length):
                for j in range(length):
                    if l == j:
                        continue
                    ind.append(f'Pk{k}l{l}i{i}j{j}')
                    val.append(1.)
            lin_expr.append(cplex.SparsePair(ind=ind, val=val))
            rhs.append(1.)
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'] * len(rhs),
                              rhs=rhs,
                              names=['C4_' + str(i) for i in range(len(rhs))])

    # FIFTH GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for l in range(length):
        for j in range(length):
            if l == j:
                continue
            ind = []
            val = []
            for k in range(length):
                for i in range(length):
                    if k == i:
                        continue
                    ind.append(f'Pk{k}l{l}i{i}j{j}')
                    val.append(1.)
            lin_expr.append(cplex.SparsePair(ind=ind, val=val))
            rhs.append(1.)
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'] * len(rhs),
                              rhs=rhs,
                              names=['C5_' + str(i) for i in range(len(rhs))])

    # SOLVE THE ILP
    cp.set_results_stream(None)
    try:
        cp.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised while solving")
        return

    objective_value = cp.solution.get_objective_value()
    return objective_value

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 17.08.2022 #
# # # # # # # # # # # # # # # #
