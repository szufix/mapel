#!/usr/bin/env python
import logging

import scipy.special
import scipy.special
from gurobipy import Model, GRB


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_ilp_spearman_distance(votes_1, votes_2, params):
    model = Model("spearman_distance")
    model.setParam('Threads', 1)
    model.ModelSense = GRB.MINIMIZE

    P = {}
    M = {}
    N = {}

    # Define the P variables
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    weight = abs(votes_1[k].index(i) - votes_2[l].index(j))
                    P[k, l, i, j] = model.addVar(vtype=GRB.BINARY, name=f"P_{k}_{l}_{i}_{j}", obj=weight)

    # Define the M variables
    for i in range(params['candidates']):
        for j in range(params['candidates']):
            M[i, j] = model.addVar(vtype=GRB.BINARY, name=f"M_{i}_{j}")

    # Define the N variables
    for k in range(params['voters']):
        for l in range(params['voters']):
            N[k, l] = model.addVar(vtype=GRB.BINARY, name=f"N_{k}_{l}")

    model.update()

    # Add the constraints
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    model.addConstr(P[k, l, i, j] <= M[i, j], name=f"c1_{k}_{l}_{i}_{j}_1")
                    model.addConstr(P[k, l, i, j] <= N[k, l], name=f"c1_{k}_{l}_{i}_{j}_2")

    # N constraints for voters
    for k in range(params['voters']):
        model.addConstr(sum(N[k, l] for l in range(params['voters'])) == 1, name=f"c2_{k}")

    for l in range(params['voters']):
        model.addConstr(sum(N[k, l] for k in range(params['voters'])) == 1, name=f"c3_{l}")

    # M constraints for candidates
    for i in range(params['candidates']):
        model.addConstr(sum(M[i, j] for j in range(params['candidates'])) == 1, name=f"c4_{i}")

    for j in range(params['candidates']):
        model.addConstr(sum(M[i, j] for i in range(params['candidates'])) == 1, name=f"c5_{j}")

    # P constraints for voters and candidates
    for k in range(params['voters']):
        for i in range(params['candidates']):
            model.addConstr(sum(P[k, l, i, j] for l in range(params['voters']) for j in range(params['candidates'])) == 1, name=f"c6_{k}_{i}")

    for l in range(params['voters']):
        for j in range(params['candidates']):
            model.addConstr(sum(P[k, l, i, j] for k in range(params['voters']) for i in range(params['candidates'])) == 1, name=f"c7_{l}_{j}")

    # Solve the model
    model.optimize()

    if model.status != GRB.OPTIMAL:
        logging.warning("No optimal solution found")

    return model.objVal


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_ilp_swap_distance(votes_1, votes_2, params):
    model = Model("swap_distance")
    model.setParam('Threads', 1)
    model.ModelSense = GRB.MINIMIZE

    all_swaps = int(params['voters'] * scipy.special.binom(params['candidates'], 2))

    # Define variables
    R = {}
    P = {}
    M = {}
    N = {}

    # Create R variables with corresponding conditions and objective function
    for k in range(params['voters']):
        for l in range(params['voters']):
            vote_1 = votes_1[k]
            vote_2 = votes_2[l]

            pote_1 = [list(vote_1).index(i) for i, _ in enumerate(vote_1)]
            pote_2 = [list(vote_2).index(i) for i, _ in enumerate(vote_2)]

            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(i1 + 1, params['candidates']):
                        for j2 in range(params['candidates']):
                            if i1 == i2 or j1 == j2:
                                continue

                            if (pote_1[i1] > pote_1[i2] and pote_2[j1] < pote_2[j2]) or \
                                    (pote_1[i1] < pote_1[i2] and pote_2[j1] > pote_2[j2]):

                                R[k, l, i1, j1, i2, j2] = model.addVar(vtype=GRB.BINARY, name=f"R_{k}_{l}_{i1}_{j1}_{i2}_{j2}", obj=1)

    # Create P, M, and N variables
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    P[k, l, i, j] = model.addVar(vtype=GRB.BINARY, name=f"P_{k}_{l}_{i}_{j}")

    for k in range(params['voters']):
        for l in range(params['voters']):
            N[k, l] = model.addVar(vtype=GRB.BINARY, name=f"N_{k}_{l}")

    for i in range(params['candidates']):
        for j in range(params['candidates']):
            M[i, j] = model.addVar(vtype=GRB.BINARY, name=f"M_{i}_{j}")

    model.update()

    # Add constraints

    # N=1 constraints
    for k in range(params['voters']):
        model.addConstr(sum(N[k, l] for l in range(params['voters'])) == 1, name=f"N_1_{k}")

    for l in range(params['voters']):
        model.addConstr(sum(N[k, l] for k in range(params['voters'])) == 1, name=f"N_2_{l}")

    # M=1 constraints
    for i in range(params['candidates']):
        model.addConstr(sum(M[i, j] for j in range(params['candidates'])) == 1, name=f"M_1_{i}")

    for j in range(params['candidates']):
        model.addConstr(sum(M[i, j] for i in range(params['candidates'])) == 1, name=f"M_2_{j}")

    # P=1 constraints
    for k in range(params['voters']):
        for i in range(params['candidates']):
            model.addConstr(sum(P[k, l, i, j] for l in range(params['voters']) for j in range(params['candidates'])) == 1, name=f"P_1_{k}_{i}")

    for l in range(params['voters']):
        for j in range(params['candidates']):
            model.addConstr(sum(P[k, l, i, j] for k in range(params['voters']) for i in range(params['candidates'])) == 1, name=f"P_2_{l}_{j}")

    # P<N and P<M constraints
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i in range(params['candidates']):
                for j in range(params['candidates']):
                    model.addConstr(P[k, l, i, j] <= N[k, l], name=f"P_N_{k}_{l}_{i}_{j}")
                    model.addConstr(P[k, l, i, j] <= M[i, j], name=f"P_M_{k}_{l}_{i}_{j}")

    # All swaps constraint
    model.addConstr(sum(R[k, l, i1, j1, i2, j2] for k in range(params['voters']) for l in range(params['voters']) for i1 in range(params['candidates']) for j1 in range(params['candidates']) for i2 in range(i1 + 1, params['candidates']) for j2 in range(params['candidates']) if (i1 != i2 and j1 != j2)) == all_swaps, name="All_swaps")

    # R<P constraints
    for k in range(params['voters']):
        for l in range(params['voters']):
            for i1 in range(params['candidates']):
                for j1 in range(params['candidates']):
                    for i2 in range(i1 + 1, params['candidates']):
                        for j2 in range(params['candidates']):
                            if i1 == i2 or j1 == j2:
                                continue

                            model.addConstr(R[k, l, i1, j1, i2, j2] <= P[k, l, i1, j1], name=f"R_P_1_{k}_{l}_{i1}_{j1}_{i2}_{j2}")
                            model.addConstr(R[k, l, i1, j1, i2, j2] <= P[k, l, i2, j2], name=f"R_P_2_{k}_{l}_{i1}_{j1}_{i2}_{j2}")

    model.optimize()

    if model.status != GRB.OPTIMAL:
        logging.warning("No optimal solution found")

    return model.objVal
