#!/usr/bin/env python

import os
import math
from contextlib import suppress

from mapel.elections.features.dependent_rounding import approx_rand_tree

try:
    import cplex
except ImportError:
    cplex = None

import numpy as np


# NEW ILP
def solve_rand_approx_pav(election, committee_size, W, C, ctr=0, fixed=None):
    if ctr == 1:
        return 0

    if fixed is None:
        fixed = []

    cp = cplex.Cplex()
    cp.parameters.threads.set(1)

    m = election.num_candidates
    n = election.num_voters
    k = committee_size

    # OBJECTIVE FUNCTION
    cp.objective.set_sense(cp.objective.sense.minimize)
    names = []
    obj = []
    for j in range(n):
        for l in range(k):
            for i in range(m):
                names.append(f'x{l}_{i}_{j}')
                obj.append(W[l] * C[j][i])
    types = [cp.variables.type.continuous] * len(names)
    cp.variables.add(obj=obj, names=names, types=types,
                     lb=[0.] * len(names), ub=[1.] * len(names))

    # ADD MISSING VARIABLES
    names = []
    for i in range(m):
        names.append(f'y{i}')
    cp.variables.add(names=list(names),
                     types=[cp.variables.type.continuous] * len(names),
                     lb=[0.] * len(names), ub=[1.] * len(names))

    # FIRST SINGLE CONSTRAINT
    ind = []
    val = []
    for i in range(m):
        ind.append(f'y{i}')
        val.append(1.)
    cp.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                              senses=['E'],
                              rhs=[committee_size],
                              names=['C1'])

    # SECOND GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for i in range(m):
        for j in range(n):
            val = []
            ind = []
            for l in range(k):
                ind.append(f'x{l}_{i}_{j}')
                val.append(1.)
            ind.append(f'y{i}')
            val.append(-1.)
            lin_expr.append(cplex.SparsePair(ind=ind, val=val))
            rhs.append(0.)
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['L'] * len(rhs),
                              rhs=rhs,
                              names=['C2_' + str(i) for i in range(len(rhs))])

    # THIRD GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for j in range(n):
        for l in range(k):
            val = []
            ind = []
            for i in range(m):
                ind.append(f'x{l}_{i}_{j}')
                val.append(1.)
            lin_expr.append(cplex.SparsePair(ind=ind, val=val))
            rhs.append(1.)
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['G'] * len(rhs),
                              rhs=rhs,
                              names=['C3_' + str(i) for i in range(len(rhs))])

    # FIXED
    lin_expr = []
    rhs = []
    for i in fixed:
        ind = [f'y{i}']
        val = [1.]
    lin_expr.append(cplex.SparsePair(ind=ind, val=val))
    rhs.append(1.)
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'] * len(rhs),
                              rhs=rhs,
                              names=['C4_' + str(i) for i in range(len(rhs))])

    cp.set_results_stream(None)
    try:
        cp.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised while solving")
        return

    ### TO DO LATER ###

    # score, fixed = print_results(election, committee_size, cp)

    values = [0.] * election.num_candidates
    for i in range(election.num_candidates):
        values[i] = cp.solution.get_values('y' + str(i))
    # print("values:", values)

    final_values = approx_rand_tree(values)
    # print(final_values)

    winner_id = 0
    winners = [-1] * committee_size
    for i in range(election.num_candidates):
        if final_values[f'x{i}'] == 1.:
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

    # print(score)
    return score

    # if len(fixed) < committee_size:
    #
    #     print(score, fixed)
    #
    #     solve_rand_approx_pav(election, committee_size, W, C, ctr+1, fixed)
    # else:
    #     return score


# FOR SUBELECTIONS
def solve_lp_voter_subelection(election_1, election_2, metric_name='0'):
    """ LP solver for voter subelection problem """

    cp = cplex.Cplex()
    cp.parameters.threads.set(1)

    # OBJECTIVE FUNCTION
    cp.objective.set_sense(cp.objective.sense.maximize)
    names = []
    for v1 in range(election_1.num_voters):
        for v2 in range(election_2.num_voters):
            names.append('N' + str(v1) + '_' + str(v2))
    obj = [1.] * len(names)
    types = [cp.variables.type.binary] * len(names)
    cp.variables.add(obj=obj, names=names, types=types)

    # FIRST CONSTRAINT FOR VOTERS
    lin_expr = []
    for v1 in range(election_1.num_voters):
        val = [1.0] * election_2.num_voters
        ind = []
        for v2 in range(election_2.num_voters):
            ind.append('N' + str(v1) + '_' + str(v2))
        lin_expr.append(cplex.SparsePair(ind=ind, val=val))
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['L'] * election_1.num_voters,
                              rhs=[1.0] * election_1.num_voters,
                              names=['C1_' + str(i) for i in range(election_1.num_voters)])

    # SECOND CONSTRAINT FOR VOTERS
    lin_expr = []
    for v2 in range(election_2.num_voters):
        ind = []
        for v1 in range(election_1.num_voters):
            ind.append('N' + str(v1) + '_' + str(v2))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * election_1.num_voters))
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['L'] * election_2.num_voters,
                              rhs=[1.0] * election_2.num_voters,
                              names=['C2_' + str(i) for i in range(election_2.num_voters)])

    # ADD VARIABLES FOR CANDIDATES
    names = []
    for c1 in range(election_1.num_candidates):
        for c2 in range(election_2.num_candidates):
            names.append('M' + str(c1) + '_' + str(c2))
    cp.variables.add(names=list(names),
                     types=[cp.variables.type.binary] * election_1.num_candidates * election_2.num_candidates)

    # FIRST CONSTRAINT FOR CANDIDATES
    lin_expr = []
    for c1 in range(election_1.num_candidates):
        ind = []
        for c2 in range(election_2.num_candidates):
            ind.append('M' + str(c1) + '_' + str(c2))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * election_2.num_candidates))
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'] * election_1.num_candidates,
                              rhs=[1.0] * election_1.num_candidates,
                              names=['C3_' + str(i) for i in range(election_1.num_candidates)])

    # SECOND CONSTRAINT FOR CANDIDATES
    lin_expr = []
    for c2 in range(election_2.num_candidates):
        ind = []
        for c1 in range(election_1.num_candidates):
            ind.append('M' + str(c1) + '_' + str(c2))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * election_1.num_candidates))
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'] * election_2.num_candidates,
                              rhs=[1.0] * election_2.num_candidates,
                              names=['C4_' + str(i) for i in range(election_2.num_candidates)])


    # MAIN CONSTRAINT FOR VOTES
    potes_1 = election_1.get_potes()
    potes_2 = election_2.get_potes()
    lin_expr = []
    for v1 in range(election_1.num_voters):
        for v2 in range(election_2.num_voters):
            ind = []
            val = []
            for c1 in range(election_1.num_candidates):
                for c2 in range(election_2.num_candidates):
                    ind.append('M' + str(c1) + '_' + str(c2))
                    if abs(potes_1[v1][c1] - potes_2[v2][c2]) <= int(metric_name):
                        val.append(1.)
                    else:
                        val.append(0.)
            ind.append('N' + str(v1) + '_' + str(v2))
            val.append(-election_1.num_candidates)
            lin_expr.append(cplex.SparsePair(ind=ind, val=val))
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['G'] * election_1.num_voters * election_2.num_voters,
                              rhs=[0.0] * election_1.num_voters * election_2.num_voters,
                              names=['C5_' + str(i) for i in
                                     range(election_1.num_voters * election_2.num_voters)])

    # cp.write('new.lp')

    # SOLVE THE ILP
    cp.set_results_stream(None)
    try:
        cp.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised while solving")
        return

    return cp.solution.get_objective_value()


def solve_lp_candidate_subelections(lp_file_name, election_1, election_2):
    """ LP solver for candidate subelection problem """

    # PRECOMPUTING
    # """

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
                                election_2.potes[u][d1] >
                                election_2.potes[u][d2]) or \
                                    (election_1.potes[v][c1] < election_1.potes[v][c2] and
                                     election_2.potes[u][d1] <
                                     election_2.potes[u][d2]):
                                P[v][u][c1][d1][c2][d2] = 1

    # print(P)
    # """

    # CREATE LP FILE
    lp_file = open(lp_file_name, 'w')
    lp_file.write("Maximize\nobj: ")

    first = True
    for c in range(election_1.num_candidates):
        for d in range(election_2.num_candidates):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write(" M_" + str(c) + "_" + str(d))

    lp_file.write("\n")

    """
    first = True
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

                            if P[v][u][c1][d1][c2][d2] == 1:
                                if not first:
                                    lp_file.write(" + ")
                                first = False
                                lp_file.write(" P_" + str(v) + "_" + str(u) + "_" +
                                              str(c1) + "_" + str(d1) + "_" + str(c2) + "_" + str(d2))
    lp_file.write("\n")
    """

    lp_file.write("Subject To\n")
    ctr_c = 0

    # FIRST CONSTRAINT FOR VOTERS
    for v in range(election_1.num_voters):
        lp_file.write("c" + str(ctr_c) + ":")
        first = True
        for u in range(election_2.num_voters):
            if not first:
                lp_file.write(" +")
            first = False
            lp_file.write(" N_" + str(v) + "_" + str(u))
        lp_file.write(" = 1" + "\n")
        ctr_c += 1

    # SECOND CONSTRAINT FOR VOTERS
    for u in range(election_2.num_voters):
        lp_file.write("c" + str(ctr_c) + ":")
        first = True
        for v in range(election_1.num_voters):
            if not first:
                lp_file.write(" +")
            first = False
            lp_file.write(" N_" + str(v) + "_" + str(u))
        lp_file.write(" = 1" + "\n")
        ctr_c += 1

    # FIRST CONSTRAINT FOR CANDIDATES
    for c in range(election_1.num_candidates):
        lp_file.write("c" + str(ctr_c) + ":")
        first = True
        for d in range(election_2.num_candidates):
            if not first:
                lp_file.write(" +")
            first = False
            lp_file.write(" M_" + str(c) + "_" + str(d))
        lp_file.write(" <= 1" + "\n")
        ctr_c += 1

    # SECOND CONSTRAINT FOR CANDIDATES
    for d in range(election_2.num_candidates):
        lp_file.write("c" + str(ctr_c) + ":")
        first = True
        for c in range(election_1.num_candidates):
            if not first:
                lp_file.write(" +")
            first = False
            lp_file.write(" M_" + str(c) + "_" + str(d))
        lp_file.write(" <= 1" + "\n")
        ctr_c += 1

    # FIRST CONSTRAINT FOR P
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

                            # if P[v][u][c1][d1][c2][d2] == 1:
                            lp_file.write("c" + str(ctr_c) + ":")
                            lp_file.write(" P_" + str(v) + "_" + str(u) + "_" +
                                          str(c1) + "_" + str(d1) + "_" + str(c2) + "_" + str(d2))

                            lp_file.write(" - 0.34 N_" + str(v) + "_" + str(u))
                            lp_file.write(" - 0.34 M_" + str(c1) + "_" + str(d1))
                            lp_file.write(" - 0.34 M_" + str(c2) + "_" + str(d2))

                            lp_file.write(" <= 0" + "\n")
                            ctr_c += 1

    # SECOND CONSTRAINT FOR P
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

                            # if P[v][u][c1][d1][c2][d2] == 1:
                            lp_file.write("c" + str(ctr_c) + ":")
                            lp_file.write(" P_" + str(v) + "_" + str(u) + "_" +
                                          str(c1) + "_" + str(d1) + "_" + str(c2) + "_" + str(d2))
                            # lp_file.write(" + 1")
                            lp_file.write(" - 0.34 N_" + str(v) + "_" + str(u))
                            lp_file.write(" - 0.34 M_" + str(c1) + "_" + str(d1))
                            lp_file.write(" - 0.34 M_" + str(c2) + "_" + str(d2))

                            lp_file.write(" > -1" + "\n")
                            ctr_c += 1

    # THIRD CONSTRAINT FOR P
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

                            # if P[v][u][c1][d1][c2][d2] == 1:
                            lp_file.write("c" + str(ctr_c) + ":")
                            lp_file.write(" P_" + str(v) + "_" + str(u) + "_" +
                                          str(c1) + "_" + str(d1) + "_" + str(c2) + "_" + str(d2))

                            lp_file.write(" <= " + str(P[v][u][c1][d1][c2][d2]) + "\n")
                            ctr_c += 1

    """
    # NEW 1
    for c1 in range(election_1.num_candidates):
        for d1 in range(election_2.num_candidates):
            lp_file.write("c" + str(ctr_c) + ":")
            first = True
            for v in range(election_1.num_voters):
                for u in range(election_2.num_voters):
                    for c2 in range(election_1.num_candidates):
                        if c1 == c2:
                            continue
                        for d2 in range(election_2.num_candidates):
                            if d1 == d2:
                                continue

                            if P[v][u][c1][d1][c2][d2] == 1:
                                if not first:
                                    lp_file.write(" +")
                                first = False
                                lp_file.write(" P_" + str(v) + "_" + str(u) + "_" +
                                              str(c1) + "_" + str(d1) + "_" + str(c2) + "_" + str(d2))
            lp_file.write(' - ' + str((magic_param-1)*election_1.num_voters) + ' M_' + str(c1) + '_' + str(d1) + ' = 0' + "\n")
            ctr_c += 1

    # NEW 2
    for v in range(election_1.num_voters):
        for u in range(election_2.num_voters):
            lp_file.write("c" + str(ctr_c) + ":")
            first = True
            for c1 in range(election_1.num_candidates):
                for d1 in range(election_2.num_candidates):
                    for c2 in range(election_1.num_candidates):
                        if c1 == c2:
                            continue
                        for d2 in range(election_2.num_candidates):
                            if d1 == d2:
                                continue

                            if P[v][u][c1][d1][c2][d2] == 1:
                                if not first:
                                    lp_file.write(" +")
                                first = False
                                lp_file.write(" P_" + str(v) + "_" + str(u) + "_" +
                                              str(c1) + "_" + str(d1) + "_" + str(c2) + "_" + str(d2))
            lp_file.write(' - ' + str((magic_param-1)*2) + ' N_' + str(v) + '_' + str(u) + ' = 0' + "\n")
            ctr_c += 1
    """

    lp_file.write("Binary\n")

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

                            # if P[v][u][c1][d1][c2][d2] == 1:
                            lp_file.write("P_" + str(v) + "_" + str(u) + "_" +
                                          str(c1) + "_" + str(d1) + "_" + str(c2) + "_" + str(
                                d2) + "\n")

    for v in range(election_1.num_voters):
        for u in range(election_2.num_voters):
            lp_file.write("N_" + str(v) + "_" + str(u) + "\n")

    for c in range(election_1.num_candidates):
        for d in range(election_2.num_candidates):
            lp_file.write("M_" + str(c) + "_" + str(d) + "\n")

    lp_file.write("End\n")

    lp_file.close()

    ### SECOND PART
    cp_lp = cplex.Cplex(lp_file_name)
    cp_lp.parameters.threads.set(1)
    cp_lp.set_results_stream(None)

    try:
        cp_lp.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised during solve")
        return

    ##########################
    ##########################

    result = np.zeros([election_1.num_candidates, election_1.num_candidates])
    for i in range(election_1.num_candidates):
        for j in range(election_1.num_candidates):
            name = 'M_' + str(i) + '_' + str(j)
            result[i][j] = cp_lp.solution.get_values(name)

    # print('M', result)
    """
    result_2 = np.zeros([election_1.num_voters, election_1.num_voters])
    for i in range(election_1.num_voters):
        for j in range(election_1.num_voters):
            election_id = 'N_' + str(i) + '_' + str(j)
            result_2[i][j] = cp_lp.solution.get_values(election_id)

    print('N', result_2)
    total = 0
    for v in range(election_1.num_voters):
        for u in range(election_1.num_voters):
            for c1 in range(election_1.num_candidates):
                for d1 in range(election_1.num_candidates):
                    for c2 in range(election_1.num_candidates):
                        if c1 == c2:
                            continue
                        for d2 in range(election_1.num_candidates):
                            if d1 == d2:
                                continue

                            #if P[v][u][c1][d1][c2][d2] == 1:
                            election_id = "P_" + str(v) + "_" + str(u) + "_" + str(c1) + "_" + str(d1) + "_" + str(c2) + "_" + str(d2)
                            value = cp_lp.solution.get_values(election_id)
                            #print(value)
                            if value == 1:
                                print(election_id)
                            total += value
    print(total)
    """
    ##########################
    ##########################

    # objective_value = cp_lp.solution.get_objective_value()
    # print('O-V: ', objective_value)
    # print(sum(sum(result)))
    return sum(sum(result))


# FOR METRICS
def solve_lp_matching_vector_with_lp(cost_table, length):
    """ LP solver for vectors' matching """

    # print(cost_table)
    cp = cplex.Cplex()
    cp.parameters.threads.set(1)

    # OBJECTIVE FUNCTION
    cp.objective.set_sense(cp.objective.sense.minimize)
    objective = []
    names = []
    pos = 0
    for i in range(length):
        for j in range(length):
            names.append('x' + str(pos))
            objective.append(cost_table[i][j])
            pos += 1
    cp.variables.add(obj=objective, names=names, types=[cp.variables.type.binary] * length ** 2)

    # FIRST GROUP OF CONSTRAINTS
    lin_expr = []
    for i in range(length):
        ind = []
        for j in range(length):
            pos = i * length + j
            ind.append('x' + str(pos))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * length))
    cp.linear_constraints.add(lin_expr=lin_expr, senses=['E'] * length, rhs=[1.0] * length)

    # SECOND GROUP OF CONSTRAINTS
    lin_expr = []
    for j in range(length):
        ind = []
        for i in range(length):
            pos = i * length + j
            ind.append('x' + str(pos))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * length))
    cp.linear_constraints.add(lin_expr=lin_expr, senses=['E'] * length, rhs=[1.0] * length)

    # c.write('new.lp')

    # SOLVE THE ILP
    cp.set_results_stream(None)
    try:
        cp.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised while solving")
        return

    objective_value = cp.solution.get_objective_value()
    return objective_value


def solve_lp_matching_interval(cost_table, length_1, length_2):
    precision = length_1 * length_2
    # print(cost_table)

    c = cplex.Cplex()
    c.parameters.threads.set(1)

    # OBJECTIVE FUNCTION
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_name("Obj")
    objective = []
    names = []
    pos = 0
    for i in range(length_1):
        for j in range(length_2):
            names.append('x' + str(pos))
            objective.append(cost_table[i][j])
            pos += 1
    c.variables.add(obj=objective,
                    names=names,
                    types=[c.variables.type.integer] * precision)

    # FIRST GROUP OF CONSTRAINTS
    lin_expr = []
    c_names = []
    for i in range(length_1):
        ind = []
        for j in range(length_2):
            pos = i * length_2 + j
            ind.append('x' + str(pos))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * length_2))
        c_names.append('c1_' + str(i))
    c.linear_constraints.add(lin_expr=lin_expr,
                             senses=['E'] * length_1,
                             rhs=[length_2] * length_1,
                             names=c_names)

    # SECOND GROUP OF CONSTRAINTS
    lin_expr = []
    c_names = []
    for j in range(length_2):
        ind = []
        for i in range(length_1):
            pos = i * length_2 + j
            ind.append('x' + str(pos))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * length_1))
        c_names.append('c2_' + str(j))
    c.linear_constraints.add(lin_expr=lin_expr,
                             senses=['E'] * length_2,
                             rhs=[length_1] * length_2,
                             names=c_names)

    c.write('interval.lp')
    c.write('interval.mps')

    # SOLVE THE ILP
    c.set_results_stream(None)
    try:
        c.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised while solving")
        return

    result = c.solution.get_objective_value() / precision

    return result


# DODGSON SCORE
def solve_lp_file_dodgson_score(election, N=None, e=None, D=None):

    cp = cplex.Cplex()

    # OBJECTIVE FUNCTION
    cp.objective.set_sense(cp.objective.sense.minimize)

    names = []
    obj = []
    for i in range(len(N)):
        for j in range(1, len(D)):
            names.append("y" + str(i) + "_" + str(j))
            obj.append(j)
    types = [cp.variables.type.integer] * len(names)
    cp.variables.add(obj=obj, names=names, types=types)

    # ADD MISSING VARIABLES
    names = []
    for i in range(len(N)):
        names.append("y" + str(i) + "_" + str(0))
    cp.variables.add(names=list(names),
                     types=[cp.variables.type.integer] * len(names))

    # FIRST GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for i in range(len(N)):
        ind = ["y" + str(i) + "_" + str(0)]
        val = [1.]
        lin_expr.append(cplex.SparsePair(ind=ind, val=val))
        rhs.append(N[i])
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'] * len(rhs),
                              rhs=rhs,
                              names=['C1_' + str(i) for i in range(len(rhs))])

    # SECOND GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for k in range(len(D)):

        # if election.election_id == 'Impartial Culture_2' and k==0:
        #     continue

        val = []
        ind = []
        for i in range(len(N)):
            for j in range(1, len(D)):
                ind.append("y" + str(i) + "_" + str(j))
                value = e[i][j][k] - e[i][j - 1][k]
                val.append(value)
        lin_expr.append(cplex.SparsePair(ind=ind, val=val))
        rhs.append(D[k])

    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['G'] * len(rhs),
                              rhs=rhs,
                              names=['C2_' + str(i) for i in range(len(rhs))])

    # THIRD GROUP OF CONSTRAINTS
    lin_expr = []
    rhs = []
    for i in range(len(N)):
        for j in range(1, len(D)):
            ind = []
            val = []
            ind.append("y" + str(i) + "_" + str(j - 1))
            ind.append("y" + str(i) + "_" + str(j))
            val.append(1.)
            val.append(-1.)
            lin_expr.append(cplex.SparsePair(ind=ind, val=val))
            rhs.append(0.)
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['G'] * len(rhs),
                              rhs=rhs,
                              names=['C3_' + str(i) for i in range(len(rhs))])
    # cp.write("tmp.lp")
    #
    # cp.read("tmp.lp")

    # SOLVE THE ILP
    cp.parameters.threads.set(1)
    cp.set_results_stream(None)
    try:
        cp.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised while solving")
        return

    # for x in cp.solution.get_values():
    #     if x <0 or x > 1:
    #         print(x)

    return cp.solution.get_objective_value()



def solve_lp_dodgson_score(lp_file_name):
    """ this function ..."""

    cp_lp = cplex.Cplex(lp_file_name)
    cp_lp.parameters.threads.set(1)
    cp_lp.set_results_stream(None)

    start = cp_lp.get_time()
    try:
        cp_lp.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised during solve")
        return
    stop = cp_lp.get_time()

    """
    import numpy as np
    result = np.zeros([len(N), len(D)])
    for i in range(len(N)):
        for j in range(len(D)):
            result[i] = cp_lp.solution.get_values('y' + str(i) + '_' + str(j))
    """

    return cp_lp.solution.get_objective_value(), stop-start


# FOR WINNERS - updated
def solve_lp_borda_owa(params, votes, owa):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_orders = params['orders']

    cp = cplex.Cplex()
    cp.parameters.threads.set(1)

    # OBJECTIVE FUNCTION
    cp.objective.set_sense(cp.objective.sense.maximize)
    objective = []
    names = []
    pos = 0
    for i in range(num_voters):
        for j in range(num_orders):
            for k in range(num_candidates):
                names.append(f'x{pos}')
                objective.append(owa[j])
                pos += 1
    cp.variables.add(obj=objective, names=names,
                     types=[cp.variables.type.binary] * num_voters * num_orders * num_candidates)

    # ADD VARIABLES FOR CANDIDATES
    names = [f'y{i}' for i in range(num_candidates)]
    cp.variables.add(names=list(names),
                     types=[cp.variables.type.binary] * num_candidates)

    # FIRST GROUP OF CONSTRAINTS
    lin_expr = []
    ind = [f'y{i}' for i in range(num_candidates)]
    val = [1.] * num_candidates
    lin_expr.append(cplex.SparsePair(ind=ind, val=val))
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['E'],
                              rhs=[num_orders])

    # SECOND GROUP OF CONSTRAINTS
    lin_expr = []
    for i in range(num_voters):
        for j in range(num_candidates):
            ind = []
            val = []
            pos = i * num_orders * num_candidates + j
            for k in range(num_orders):
                ind.append(f'x{pos + num_candidates * k}')
                val.append(1.)
            for k in range(0, j + 1):
                ind.append(f'y{int(votes[i][k])}')
                val.append(-1.)
            lin_expr.append(cplex.SparsePair(ind=ind, val=val))
    cp.linear_constraints.add(lin_expr=lin_expr,
                              senses=['L'] * num_voters * num_candidates,
                              rhs=[0.0] * num_voters * num_candidates)

    # SOLVE THE ILP
    cp.set_results_stream(None)
    start = cp.get_time()
    try:
        cp.solve()
    except cplex.CplexSolverError:
        print("Exception raised during solve")
        return
    stop = cp.get_time()

    # EXTRACT WINNERS
    result = [0.] * num_candidates
    for i in range(num_candidates):
        result[i] = cp.solution.get_values(f'y{i}')
    winner_id = 0
    winners = [0.] * num_orders
    for i in range(num_candidates):
        if math.isclose(result[i], 1.):
            winners[winner_id] = i
            winner_id += 1
    winners = sorted(winners)

    return winners, stop-start


# FOR WINNERS - needs update
def solve_lp_bloc_owa(params, votes, owa, t_bloc):
    """ this function generates lp file"""


    rand_name = str(np.random.random())
    lp_file_name = str(rand_name + ".lp")

    lp_file = open(lp_file_name, 'w')
    lp_file.write("Maximize\nobj: ")
    pos = 0
    first = True
    for i in range(params['voters']):
        for j in range(params['orders']):
            for k in range(params['candidates']):

                if not first:
                    if k == t_bloc - 1:
                        lp_file.write(" + ")
                first = False
                if k == t_bloc - 1:
                    lp_file.write(str(owa[j]) + " x" + str(pos))
                pos += 1
    lp_file.write("\n")

    lp_file.write("Subject To\n")
    lp_file.write("c0:")
    first = True
    for i in range(params['candidates']):
        if not first:
            lp_file.write(" +")
        first = False
        lp_file.write(" y" + str(i))
    lp_file.write(' = ' + str(params['orders']) + '\n')

    for i in range(params['voters']):
        for j in range(params['candidates']):
            lp_file.write("c" + str(i * params['candidates'] + j + 1) + ": ")
            pos = i * params['orders'] * params['candidates'] + j
            first = True
            for k in range(params['orders']):
                if not first:
                    lp_file.write(" +")
                first = False
                lp_file.write(" x" + str(pos + params['candidates'] * k))

            for k in range(0, j + 1):
                lp_file.write(" - y" + str(int(votes[i][k])))

            lp_file.write(" <= 0 \n")

    lp_file.write("Binary\n")
    for i in range(params['voters'] * params['orders'] * params['candidates']):
        lp_file.write("x" + str(i) + "\n")
    for i in range(params['candidates']):
        lp_file.write("y" + str(i) + "\n")

    lp_file.write("End\n")
    lp_file.close()


    cp_lp = cplex.Cplex(lp_file_name)
    cp_lp.parameters.threads.set(1)
    cp_lp.set_results_stream(None)

    start = cp_lp.get_time()
    try:
        cp_lp.solve()
    except cplex.CplexSolverError:
        print("Exception raised during solve")
        return
    stop = cp_lp.get_time()

    result = [0.] * params['candidates']
    for i in range(params['candidates']):
        result[i] = cp_lp.solution.get_values('y' + str(i))
    # print(result)

    params['pure'] = True

    winner_id = 0
    winners = [0.] * params['orders']
    for i in range(params['candidates']):
        if math.isclose(result[i], 1.):
            if params['pure']:
                winners[winner_id] = i
            winner_id += 1
    winners = sorted(winners)

    os.remove(lp_file_name)

    return winners, stop-start


def generate_lp_file_matching_matrix(matrix_1, matrix_2, length, inner_distance):

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


def solve_lp_matrix(lp_file_name):
    cp_lp = cplex.Cplex(lp_file_name)
    cp_lp.set_results_stream(None)

    # cp_lp.parameters.threads.set(1)
    # cp_lp.parameters.mip.tolerances.mipgap = 0.0001
    # cp_lp.parameters.mip.strategy.probe.set(3)

    try:
        cp_lp.solve()

    except:
        print("Exception raised during solve")
        return

    return cp_lp.solution.get_objective_value()


# SPEARMAN - old
def generate_ilp_distance(lp_file_name, votes_1, votes_2, params, metric_name):
    lp_file = open(lp_file_name, 'w')
    lp_file.write("Minimize\n")  # obj: ")
    #
    # first = True
    # for k in range(params['voters']):
    #     for l in range(params['voters']):
    #
    #         vote_1 = votes_1[k]
    #         vote_2 = votes_2[l]
    #
    #         if metric_name == 'spearman':
    #             pote_1 = [0] * params['candidates']
    #             pote_2 = [0] * params['candidates']
    #
    #             for i in range(params['candidates']):
    #                 pote_1[vote_1[i]] = i
    #                 pote_2[vote_2[i]] = i
    #
    #         for i in range(params['candidates']):
    #             for j in range(params['candidates']):
    #                 if not first:
    #                     lp_file.write(" + ")
    #                 first = False
    #
    #                 if metric_name == "spearman":
    #                     weight = abs(pote_1[i] - pote_2[j])
    #                 elif metric_name == "alt":
    #                     weight = float(abs(pote_1[i] - pote_2[j]) ** (2)) / float(
    #                         1. + min(pote_1[i], pote_2[j]))
    #                 elif metric_name == 'hamming':
    #                     weight = hamming(vote_1, vote_2)
    #                 else:
    #                     weight = 0
    #
    #                 lp_file.write(
    #                     str(weight) + " P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(
    #                         j))
    # lp_file.write("\n")
    #
    # lp_file.write("Subject To\n")
    #
    # for k in range(params['voters']):
    #     for l in range(params['voters']):
    #         for i in range(params['candidates']):
    #             for j in range(params['candidates']):
    #                 lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
    #                 lp_file.write(" - " + "M" + "i" + str(i) + "j" + str(j) + " <= 0" + "\n")
    #
    #                 lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
    #                 lp_file.write(" - " + "N" + "k" + str(k) + "l" + str(l) + " <= 0" + "\n")
    #
    # for k in range(params['voters']):
    #     first = True
    #     for l in range(params['voters']):
    #         if not first:
    #             lp_file.write(" + ")
    #         first = False
    #         lp_file.write("N" + "k" + str(k) + "l" + str(l))
    #     lp_file.write(" = 1" + "\n")
    #
    # for l in range(params['voters']):
    #     first = True
    #     for k in range(params['voters']):
    #         if not first:
    #             lp_file.write(" + ")
    #         first = False
    #         lp_file.write("N" + "k" + str(k) + "l" + str(l))
    #     lp_file.write(" = 1" + "\n")
    #
    # for i in range(params['candidates']):
    #     first = True
    #     for j in range(params['candidates']):
    #         if not first:
    #             lp_file.write(" + ")
    #         first = False
    #         lp_file.write("M" + "i" + str(i) + "j" + str(j))
    #     lp_file.write(" = 1" + "\n")
    #
    # for j in range(params['candidates']):
    #     first = True
    #     for i in range(params['candidates']):
    #         if not first:
    #             lp_file.write(" + ")
    #         first = False
    #         lp_file.write("M" + "i" + str(i) + "j" + str(j))
    #     lp_file.write(" = 1" + "\n")
    #
    # # IMPORTANT #
    # for k in range(params['voters']):
    #     for i in range(params['candidates']):
    #         first = True
    #         for l in range(params['voters']):
    #             for j in range(params['candidates']):
    #                 if not first:
    #                     lp_file.write(" + ")
    #                 first = False
    #                 lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
    #         lp_file.write(" = 1" + "\n")
    #
    # # IMPORTANT #
    # for l in range(params['voters']):
    #     for j in range(params['candidates']):
    #         first = True
    #         for k in range(params['voters']):
    #             for i in range(params['candidates']):
    #                 if not first:
    #                     lp_file.write(" + ")
    #                 first = False
    #                 lp_file.write("P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j))
    #         lp_file.write(" = 1" + "\n")
    #
    # lp_file.write("Binary\n")
    #
    # for k in range(params['voters']):
    #     for l in range(params['voters']):
    #         for i in range(params['candidates']):
    #             for j in range(params['candidates']):
    #                 lp_file.write(
    #                     "P" + "k" + str(k) + "l" + str(l) + "i" + str(i) + "j" + str(j) + "\n")
    #
    # for k in range(params['voters']):
    #     for l in range(params['voters']):
    #         lp_file.write("N" + "k" + str(k) + "l" + str(l) + "\n")
    #
    # for i in range(params['candidates']):
    #     for j in range(params['candidates']):
    #         lp_file.write("M" + "i" + str(i) + "j" + str(j) + "\n")

    lp_file.write("End\n")


def solve_ilp_distance(lp_file_name, votes_1, votes_2, params, metric_name):
    cp_lp = cplex.Cplex(lp_file_name)
    cp_lp.set_results_stream(None)

    # cp_lp.parameters.threads.set(1)
    # cp_lp.parameters.timelimit.set(60)

    try:
        cp_lp.solve()

    except cplex.CplexSolverError:
        print("Exception raised during solve")
        return

    total = cp_lp.solution.get_objective_value()
    return total


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


# COPY
def generate_lp_file_borda_owa(owa, lp_file_name, params, votes):
    """ this function generates lp file"""

    with open(lp_file_name, 'w') as lp_file:
        lp_file.write("Maximize\nobj: ")
        pos = 0
        first = True
        for i in range(params['voters']):
            for j in range(params['orders']):
                for k in range(params['candidates']):
                    if not first and owa[j] >= 0.:
                        lp_file.write(" + ")
                    first = False
                    lp_file.write(str(owa[j]) + " x" + str(pos))
                    pos += 1
        lp_file.write("\n")

        lp_file.write("Subject To\n")
        lp_file.write("c0:")
        first = True
        for i in range(params['candidates']):
            if not first:
                lp_file.write(" +")
            first = False
            lp_file.write(" y" + str(i))
        lp_file.write(' = ' + str(params['orders']) + '\n')

        for i in range(params['voters']):
            for j in range(params['candidates']):
                lp_file.write("c" + str(i * params['candidates'] + j + 1) + ": ")
                pos = i * params['orders'] * params['candidates'] + j
                first = True
                for k in range(params['orders']):
                    if not first:
                        lp_file.write(" +")
                    first = False
                    lp_file.write(" x" + str(pos + params['candidates'] * k))

                for k in range(0, j + 1):
                    lp_file.write(" - y" + str(int(votes[i][k])))

                lp_file.write(" <= 0 \n")

        lp_file.write("Binary\n")
        for i in range(params['voters'] * params['orders'] * params['candidates']):
            lp_file.write("x" + str(i) + "\n")
        for i in range(params['candidates']):
            lp_file.write("y" + str(i) + "\n")

        lp_file.write("End\n")


# TMP - PRINT

def print_results(election, committee_size, cp):

    result = [0.] * election.num_candidates
    for i in range(election.num_candidates):
        result[i] = cp.solution.get_values('y' + str(i))
    # print(result)

    fixed = []
    for i, r in enumerate(result):
        if r >= 0.5:
            fixed.append(i)


    winner_id = 0
    winners = [-1] * committee_size
    for i in range(election.num_candidates):
        if result[i] == 1.:
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

    print(score)
    return score, fixed