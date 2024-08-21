#!/usr/bin/env python
import logging
import math
import os
from contextlib import suppress

from gurobipy import LinExpr
from gurobipy import Model, GRB

from mapel.elections.features.dependent_rounding import approx_rand_tree


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_lp_matching_vector_with_lp(cost_table, length):

    # Create a new model
    model = Model("vectors_matching")
    model.setParam('Threads', 1)

    # Length of the cost table
    length = len(cost_table)

    # Variables
    x = {}
    for i in range(length):
        for j in range(length):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Objective function
    model.setObjective(
        sum(cost_table[i][j] * x[i, j] for i in range(length) for j in range(length)), GRB.MINIMIZE)

    # First group of constraints
    for i in range(length):
        model.addConstr(sum(x[i, j] for j in range(length)) == 1, name=f"row_{i}")

    # Second group of constraints
    for j in range(length):
        model.addConstr(sum(x[i, j] for i in range(length)) == 1, name=f"col_{j}")

    # Solve the model
    model.optimize()

    # If a solution was found, return the objective value
    if model.status == GRB.OPTIMAL:
        objective_value = model.objVal
        print(f"Objective Value: {objective_value}")
    else:
        print("No optimal solution found")


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_lp_matching_interval(cost_table, length_1, length_2):
    precision = length_1 * length_2

    # Create a new model
    model = Model("lp_matching_interval")
    model.setParam('Threads', 1)

    # Variables
    x = {}
    for i in range(length_1):
        for j in range(length_2):
            x[i, j] = model.addVar(vtype=GRB.INTEGER, name=f"x_{i}_{j}")

    # Objective function
    model.setObjective(sum(cost_table[i][j] * x[i, j] for i in range(length_1) for j in range(length_2)), GRB.MINIMIZE)

    # First group of constraints
    for i in range(length_1):
        model.addConstr(sum(x[i, j] for j in range(length_2)) == length_2, name=f"c1_{i}")

    # Second group of constraints
    for j in range(length_2):
        model.addConstr(sum(x[i, j] for i in range(length_1)) == length_1, name=f"c2_{j}")

    # Save the model to files
    model.write('interval.lp')
    model.write('interval.mps')

    # Solve the model
    model.optimize()

    # If a solution was found, return the objective value divided by precision
    if model.status == GRB.OPTIMAL:
        result = model.objVal / precision
        return result
    else:
        print("No optimal solution found")
        return None


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_lp_file_dodgson_score(N=None, e=None, D=None) -> float:
    # Create a new model
    model = Model("dodgson_score")
    model.setParam('Threads', 1)

    # Objective function variables and coefficients
    names = []
    obj = []
    for i in range(len(N)):
        for j in range(1, len(D)):
            var_name = f"y_{i}_{j}"
            names.append(var_name)
            obj.append(j)
            model.addVar(vtype=GRB.INTEGER, name=var_name, obj=j)

    # Add missing variables for j=0
    for i in range(len(N)):
        var_name = f"y_{i}_0"
        model.addVar(vtype=GRB.INTEGER, name=var_name)

    model.update()

    # Objective function
    model.setObjective(sum(model.getVarByName(f"y_{i}_{j}") * j
                           for i in range(len(N))
                           for j in range(1, len(D))), GRB.MINIMIZE)

    # First group of constraints
    for i in range(len(N)):
        model.addConstr(model.getVarByName(f"y_{i}_0") == N[i], name=f"C1_{i}")

    # Second group of constraints
    for k in range(len(D)):
        val = []
        ind = []
        for i in range(len(N)):
            for j in range(1, len(D)):
                ind.append(model.getVarByName(f"y_{i}_{j}"))
                value = e[i][j][k] - e[i][j - 1][k]
                val.append(value)
        model.addConstr(sum(ind[l] * val[l] for l in range(len(ind))) >= D[k], name=f"C2_{k}")

    # Third group of constraints
    for i in range(len(N)):
        for j in range(1, len(D)):
            model.addConstr(model.getVarByName(f"y_{i}_{j - 1}") - model.getVarByName(f"y_{i}_{j}") >= 0,
                            name=f"C3_{i}_{j}")

    # Solve the model
    model.optimize()

    # If a solution was found, return the objective value
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        logging.warning("No optimal solution found")


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_lp_borda_owa(params, votes, owa):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_orders = params['orders']

    # Create a new model
    model = Model("borda_owa")
    model.setParam('Threads', 1)

    # Objective function variables and coefficients
    x = {}
    for i in range(num_voters):
        for j in range(num_orders):
            for k in range(num_candidates):
                var_name = f"x_{i}_{j}_{k}"
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=var_name, obj=owa[j])

    # Add variables for candidates
    y = {}
    for i in range(num_candidates):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    model.update()

    # Objective function
    model.ModelSense = GRB.MAXIMIZE

    # First group of constraints
    model.addConstr(sum(y[i] for i in range(num_candidates)) == num_orders, name="c1")

    # Second group of constraints
    for i in range(num_voters):
        for j in range(num_candidates):
            model.addConstr(sum(x[i, k, j] for k in range(num_orders)) <= sum(y[int(votes[i][k])] for k in range(0, j + 1)),
                            name=f"c2_{i}_{j}")

    # Solve the model
    model.setParam('OutputFlag', 0)
    start = model.getAttr(GRB.Attr.Runtime)
    model.optimize()
    stop = model.getAttr(GRB.Attr.Runtime)

    if model.status != GRB.OPTIMAL:
        print("Exception raised during solve")
        return None, stop-start

    # Extract winners
    result = [y[i].X for i in range(num_candidates)]
    winners = [i for i in range(num_candidates) if math.isclose(result[i], 1.0)]
    winners = sorted(winners)

    return winners, stop-start


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_lp_bloc_owa(params, votes, owa, t_bloc):
    """This function generates a Gurobi model and solves it."""

    # Create a new model
    model = Model("bloc_owa")
    model.setParam('Threads', 1)

    # Objective function variables and coefficients
    x = {}
    pos = 0
    for i in range(params['voters']):
        for j in range(params['orders']):
            for k in range(params['candidates']):
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"x_{pos}")
                if k == t_bloc - 1:
                    model.setObjective(model.getObjective() + owa[j] * x[i, j, k], GRB.MAXIMIZE)
                pos += 1

    # Add variables for candidates
    y = {}
    for i in range(params['candidates']):
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    model.update()

    # First group of constraints
    model.addConstr(sum(y[i] for i in range(params['candidates'])) == params['orders'], name="c0")

    # Second group of constraints
    for i in range(params['voters']):
        for j in range(params['candidates']):
            expr = sum(x[i, k, j] for k in range(params['orders'])) - sum(y[int(votes[i][k])] for k in range(0, j + 1))
            model.addConstr(expr <= 0, name=f"c_{i}_{j}")

    # Solve the model
    model.setParam('OutputFlag', 0)
    start = model.Runtime
    model.optimize()
    stop = model.Runtime

    if model.status != GRB.OPTIMAL:
        print("Exception raised during solve")
        return None, stop-start

    # Extract winners
    result = [y[i].X for i in range(params['candidates'])]
    winners = [i for i in range(params['candidates']) if math.isclose(result[i], 1.0)]
    winners = sorted(winners[:params['orders']])

    return winners, stop-start


# THIS FUNCTION HAS NOT BEEN TESTED SINCE CONVERSION TO GUROBI
def solve_rand_approx_pav(election, committee_size, W, C, ctr=0, fixed=None):
    if ctr == 1:
        return 0

    if fixed is None:
        fixed = []

    m = election.num_candidates
    n = election.num_voters
    k = committee_size

    # Create a new model
    model = Model()

    # Limit the number of threads
    model.setParam('Threads', 1)

    # Add variables
    x = {}
    for l in range(k):
        for i in range(m):
            for j in range(n):
                x[l, i, j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                          name=f"x_{l}_{i}_{j}")

    y = {}
    for i in range(m):
        y[i] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                            name=f"y_{i}")

    model.update()

    # Set objective
    objective = LinExpr()
    for j in range(n):
        for l in range(k):
            for i in range(m):
                objective.addTerms(W[l] * C[j][i], x[l, i, j])
    model.setObjective(objective, GRB.MINIMIZE)

    # Add constraints
    # C1: sum(y[i]) == committee_size
    model.addConstr(sum(y[i] for i in range(m)) == committee_size, "C1")

    # C2: sum(x[l, i, j] for l in range(k)) <= y[i] for all i, j
    for i in range(m):
        for j in range(n):
            model.addConstr(sum(x[l, i, j] for l in range(k)) <= y[i], f"C2_{i}_{j}")

    # C3: sum(x[l, i, j] for i in range(m)) >= 1 for all j, l
    for j in range(n):
        for l in range(k):
            model.addConstr(sum(x[l, i, j] for i in range(m)) >= 1, f"C3_{j}_{l}")

    # C4: fix the values of some y[i] variables
    for i in fixed:
        model.addConstr(y[i] == 1, f"C4_fixed_{i}")

    # Optimize model
    model.setParam('OutputFlag', 0)  # Turn off the output
    model.optimize()

    if model.status == GRB.Status.OPTIMAL:

        values = [0.] * election.num_candidates
        for i in range(election.num_candidates):
            values[i] = model.getVarByName(f'y_{i}').x

        # Assuming `approx_rand_tree` is a function you have defined elsewhere
        final_values = approx_rand_tree(values)

        winner_id = 0
        winners = [-1] * committee_size
        for i in range(election.num_candidates):
            if final_values[i] == 1.:
                winners[winner_id] = i
                winner_id += 1

        winners = sorted(winners)

        def get_pav_score(election, winners) -> float:

            num_voters = election.num_voters
            num_candidates = election.num_candidates
            votes = election.votes

            score = 0

            vector = [0.] * num_candidates
            for i in range(len(winners)):
                vector[i] = 1.

            for i in range(num_voters):
                ctr = 1.
                for j in range(num_candidates):
                    if votes[i][j] in winners:
                        score += (1. / ctr) * vector[j]
                        ctr += 1

            return score

        try:
            score = get_pav_score(election, winners)
        except:
            score = -1

        return score

    else:
        print("No optimal solution found")
        return None


# OTHER
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


# OTHER
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


# OTHER
def remove_lp_file(path):
    """ Safely remove lp file """
    with suppress(OSError):
        os.remove(path)
