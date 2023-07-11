#!/usr/bin/env python

import numpy as np
from scipy.optimize import linear_sum_assignment

import gurobipy as gp
from gurobipy import GRB


def solve_matching_vectors(cost_table) -> (float, list):
    """ Return: objective value, optimal matching """
    cost_table = np.array(cost_table)
    row_ind, col_ind = linear_sum_assignment(cost_table)
    return cost_table[row_ind, col_ind].sum(), list(col_ind)


def solve_matching_matrices(matrix_1, matrix_2, length, inner_distance) -> float:
    """ Return: objective value"""
    m = gp.Model()
    m.setParam('Threads', 1)
    m.ModelSense = GRB.MINIMIZE

    # OBJECTIVE FUNCTION
    variables = {}
    for k in range(length):
        for l in range(length):
            for i in range(length):
                if i == k:
                    continue
                for j in range(length):
                    if j == l:
                        continue
                    weight = inner_distance(np.array([matrix_1[k][i]]), np.array([matrix_2[l][j]]))
                    name = f'Pk{k}l{l}i{i}j{j}'
                    variables[name] = m.addVar(vtype=GRB.BINARY, name=name, obj=weight)

    # ADD MISSING VARIABLES
    for i in range(length):
        for j in range(length):
            name = f'Mi{i}j{j}'
            variables[name] = m.addVar(vtype=GRB.BINARY, name=name)

    m.update()

    # CONSTRAINTS
    for k in range(length):
        for l in range(length):
            for i in range(length):
                if i == k:
                    continue
                for j in range(length):
                    if j == l:
                        continue

                    m.addConstr(variables[f'Pk{k}l{l}i{i}j{j}'] - variables[f'Mi{i}j{j}'] <= 0)
                    m.addConstr(variables[f'Pk{k}l{l}i{i}j{j}'] - variables[f'Mi{k}j{l}'] <= 0)

    for i in range(length):
        m.addConstr(gp.quicksum(variables[f'Mi{i}j{j}'] for j in range(length)) == 1)

    for j in range(length):
        m.addConstr(gp.quicksum(variables[f'Mi{i}j{j}'] for i in range(length)) == 1)

    for k in range(length):
        for i in range(length):
            if k == i:
                continue
            m.addConstr(gp.quicksum(variables[f'Pk{k}l{l}i{i}j{j}']
                                    for l in range(length)
                                    for j in range(length) if l != j) == 1)

    for l in range(length):
        for j in range(length):
            if l == j:
                continue
            m.addConstr(gp.quicksum(variables[f'Pk{k}l{l}i{i}j{j}']
                                    for k in range(length)
                                    for i in range(length) if k != i) == 1)

    # SOLVE THE ILP
    m.setParam('OutputFlag', 0)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        objective_value = m.objVal
        return objective_value
    else:
        print("Exception raised while solving")


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 17.08.2022 #
# # # # # # # # # # # # # # # #
