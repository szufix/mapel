#!/usr/bin/env python
import logging

import gurobipy as gp
import numpy as np
from gurobipy import Model, GRB


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_ilp_voter_subelection(election_1, election_2, metric_name='0') -> int:

    # Initialize model
    model = gp.Model()

    # Limit the number of threads
    model.setParam('Threads', 1)

    # OBJECTIVE FUNCTION
    names = []
    for v1 in range(election_1.num_voters):
        for v2 in range(election_2.num_voters):
            names.append('N' + str(v1) + '_' + str(v2))

    # Add variables
    N_vars = model.addVars(names, vtype=GRB.BINARY, obj=1.0, name="N")

    # Set objective to maximize
    model.ModelSense = GRB.MAXIMIZE

    # FIRST CONSTRAINT FOR VOTERS
    for v1 in range(election_1.num_voters):
        model.addConstr(gp.quicksum(
            N_vars['N' + str(v1) + '_' + str(v2)] for v2 in range(election_2.num_voters)) <= 1.0,
                        name='C1_' + str(v1))

    # SECOND CONSTRAINT FOR VOTERS
    for v2 in range(election_2.num_voters):
        model.addConstr(gp.quicksum(
            N_vars['N' + str(v1) + '_' + str(v2)] for v1 in range(election_1.num_voters)) <= 1.0,
                        name='C2_' + str(v2))

    # ADD VARIABLES FOR CANDIDATES
    M_names = []
    for c1 in range(election_1.num_candidates):
        for c2 in range(election_2.num_candidates):
            M_names.append('M' + str(c1) + '_' + str(c2))

    M_vars = model.addVars(M_names, vtype=GRB.BINARY, name="M")

    # FIRST CONSTRAINT FOR CANDIDATES
    for c1 in range(election_1.num_candidates):
        model.addConstr(gp.quicksum(M_vars['M' + str(c1) + '_' + str(c2)] for c2 in
                                    range(election_2.num_candidates)) == 1.0,
                        name='C3_' + str(c1))

    # SECOND CONSTRAINT FOR CANDIDATES
    for c2 in range(election_2.num_candidates):
        model.addConstr(gp.quicksum(M_vars['M' + str(c1) + '_' + str(c2)] for c1 in
                                    range(election_1.num_candidates)) == 1.0,
                        name='C4_' + str(c2))

    # MAIN CONSTRAINT FOR VOTES
    potes_1 = election_1.get_potes()
    potes_2 = election_2.get_potes()

    for v1 in range(election_1.num_voters):
        for v2 in range(election_2.num_voters):
            M_constr = gp.LinExpr()
            for c1 in range(election_1.num_candidates):
                for c2 in range(election_2.num_candidates):
                    if abs(potes_1[v1][c1] - potes_2[v2][c2]) <= int(metric_name):
                        M_constr += M_vars['M' + str(c1) + '_' + str(c2)]
            M_constr -= N_vars['N' + str(v1) + '_' + str(v2)] * election_1.num_candidates
            model.addConstr(M_constr >= 0.0, name='C5_' + str(v1) + '_' + str(v2))

    # Optimize the model
    model.optimize()

    # Return the objective value
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        logging.warning("No optimal solution found")


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_ilp_candidate_subelection(election_1, election_2) -> int:
    """ LP solver for candidate subelection problem using Gurobi """

    # PRECOMPUTING
    P = np.zeros([election_1.num_voters, election_2.num_voters, election_1.num_candidates,
                  election_2.num_candidates,
                  election_1.num_candidates, election_2.num_candidates])

    for v in range(election_1.num_voters):
        for u in range(election_2.num_voters):
            for c1 in range(election_1.num_candidates):
                for d1 in range(election_2.num_candidates):
                    for c2 in range(election_1.num_candidates):
                        for d2 in range(election_2.num_candidates):
                            if (election_1.potes[v][c1] > election_1.potes[v][c2] and
                                election_2.potes[u][d1] > election_2.potes[u][d2]) or \
                                    (election_1.potes[v][c1] < election_1.potes[v][c2] and
                                     election_2.potes[u][d1] < election_2.potes[u][d2]):
                                P[v][u][c1][d1][c2][d2] = 1

    # Initialize Gurobi model
    model = Model("Candidate_Subelection")

    # Define variables
    M = {}
    for c in range(election_1.num_candidates):
        for d in range(election_2.num_candidates):
            M[c, d] = model.addVar(vtype=GRB.BINARY, name=f"M_{c}_{d}")

    N = {}
    for v in range(election_1.num_voters):
        for u in range(election_2.num_voters):
            N[v, u] = model.addVar(vtype=GRB.BINARY, name=f"N_{v}_{u}")

    P_vars = {}
    for v in range(election_1.num_voters):
        for u in range(election_2.num_voters):
            for c1 in range(election_1.num_candidates):
                for d1 in range(election_2.num_candidates):
                    for c2 in range(election_1.num_candidates):
                        for d2 in range(election_2.num_candidates):
                            if c1 != c2 and d1 != d2:
                                P_vars[v, u, c1, d1, c2, d2] = model.addVar(vtype=GRB.BINARY,
                                                                            name=f"P_{v}_{u}_{c1}_{d1}_{c2}_{d2}")

    # Set objective
    model.setObjective(sum(M[c, d] for c in range(election_1.num_candidates)
                           for d in range(election_2.num_candidates)), GRB.MAXIMIZE)

    # Constraints for voters
    for v in range(election_1.num_voters):
        model.addConstr(sum(N[v, u] for u in range(election_2.num_voters)) == 1, name=f"c_v_{v}")

    for u in range(election_2.num_voters):
        model.addConstr(sum(N[v, u] for v in range(election_1.num_voters)) == 1, name=f"c_u_{u}")

    # Constraints for candidates
    for c in range(election_1.num_candidates):
        model.addConstr(sum(M[c, d] for d in range(election_2.num_candidates)) <= 1, name=f"c_c_{c}")

    for d in range(election_2.num_candidates):
        model.addConstr(sum(M[c, d] for c in range(election_1.num_candidates)) <= 1, name=f"c_d_{d}")

    # Constraints for P variables
    for v in range(election_1.num_voters):
        for u in range(election_2.num_voters):
            for c1 in range(election_1.num_candidates):
                for d1 in range(election_2.num_candidates):
                    for c2 in range(election_1.num_candidates):
                        if c1 == c2:
                            continue
                        for d2 in range(election_2.num_candidates):
                            if d1 == d2:
                                continue
                            model.addConstr(
                                P_vars[v, u, c1, d1, c2, d2] - 0.34 * N[v, u] - 0.34 * M[c1, d1] - 0.34 * M[c2, d2] <= 0,
                                name=f"p_constr_1_{v}_{u}_{c1}_{d1}_{c2}_{d2}")
                            model.addConstr(
                                P_vars[v, u, c1, d1, c2, d2] - 0.34 * N[v, u] - 0.34 * M[c1, d1] - 0.34 * M[c2, d2] > -1,
                                name=f"p_constr_2_{v}_{u}_{c1}_{d1}_{c2}_{d2}")
                            model.addConstr(
                                P_vars[v, u, c1, d1, c2, d2] <= P[v, u][c1][d1][c2][d2],
                                name=f"p_constr_3_{v}_{u}_{c1}_{d1}_{c2}_{d2}")

    # Optimize model
    model.optimize()

    # Return the objective value
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        logging.warning("No optimal solution found")
