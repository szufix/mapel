#!/usr/bin/env python

import cplex


# FOR METRICS
def solve_lp_matching_vector(cost_table, length):
    #print(cost_table)
    c = cplex.Cplex()

    # OBJECTIVE FUNCTION
    c.objective.set_sense(c.objective.sense.minimize)
    objective = []
    names = []
    pos = 0
    for i in range(length):
        for j in range(length):
            names.append('x' + str(pos))
            objective.append(cost_table[i][j])
            pos += 1
    c.variables.add(obj=objective,
                    names=names,
                    types=[c.variables.type.binary] * length**2)

    # FIRST GROUP OF CONSTRAINTS
    lin_expr = []
    for i in range(length):
        ind = []
        for j in range(length):
            pos = i * length + j
            ind.append('x' + str(pos))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * length))
    c.linear_constraints.add(lin_expr=lin_expr,
                             senses=['E'] * length,
                             rhs=[1.0] * length)

    # SECOND GROUP OF CONSTRAINTS
    lin_expr = []
    for j in range(length):
        ind = []
        for i in range(length):
            pos = i * length + j
            ind.append('x' + str(pos))
        lin_expr.append(cplex.SparsePair(ind=ind, val=[1.0] * length))
    c.linear_constraints.add(lin_expr=lin_expr,
                             senses=['E'] * length,
                             rhs=[1.0] * length)

    #c.write('new.lp')

    # SOLVE THE ILP
    c.set_results_stream(None)
    try:
        c.solve()
    except:  # cplex.CplexSolverError:
        print("Exception raised while solving")
        return

    # UNPACK THE RESULTS
    """   
    result = [0.] * length ** 2
    for i in range(len(result)):
        result[i] = c.solution.get_values('x' + str(i))
    matching = [0] * length
    ctr = 0
    for i in range(len(result)):
        if result[i] == 1:
            matching[ctr] = i % length
            ctr += 1
     """

    objective_value = c.solution.get_objective_value()
    return objective_value


def solve_lp_matching_interval(cost_table, length_1, length_2):

    precision = length_1 * length_2
    #print(cost_table)

    c = cplex.Cplex()

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
def generate_lp_file_dodgson_score(lp_file_name, N=None, e=None, D=None):

    lp_file = open(lp_file_name, 'w')
    lp_file.write("Minimize\nobj: ")

    first = True
    for i in range(len(N)):
        for j in range(1, len(D)):
            if not first:
                lp_file.write(" + ")
            first = False
            lp_file.write(str(j) + " y" + str(i) + "_" + str(j))
    lp_file.write("\n")

    lp_file.write("Subject To\n")
    ctr_c = 0

    for i in range(len(N)):
        lp_file.write("c" + str(ctr_c) + ":")
        lp_file.write(" y" + str(i) + "_" + str(0) + " = " + str(N[i]) + "\n")
        ctr_c += 1

    #"""
    for k in range(len(D)):
        lp_file.write("c" + str(ctr_c) + ":")
        first = True
        for i in range(len(N)):
            for j in range(1, len(D)):
                #print(i,j,k)
                #print(e[i][j][k], e[i][j-1][k])
                if not first:
                    lp_file.write(" +")
                first = False
                lp_file.write(" " + str(e[i][j][k] - e[i][j-1][k]) + " y" + str(i) + "_" + str(j))
        lp_file.write(" >= " + str(D[k]) + "\n")
        ctr_c += 1
    #"""
    #"""
    for i in range(len(N)):
        for j in range(1, len(D)):
            lp_file.write("c" + str(ctr_c) + ":")
            lp_file.write(" y" + str(i) + "_" + str(j-1) + " - y" + str(i) + "_" + str(j) + " >= 0" + "\n")
            ctr_c += 1
    #"""
    #"""
    # chyba nie potrzeba bo integer zalatwia sprawe...
    for i in range(len(N)):
        for j in range(len(D)):
            lp_file.write("c" + str(ctr_c) + ":")
            lp_file.write(" y" + str(i) + "_" + str(j) + " >= 0" + "\n")
            ctr_c += 1
    #"""
    #"""
    lp_file.write("General\n")
    for i in range(len(N)):
        for j in range(len(D)):
            lp_file.write("y" + str(i) + "_" + str(j) + "\n")
        ctr_c += 1
    #"""
    lp_file.write("End\n")


def solve_lp_dodgson_score(lp_file_name):
    """ this function ..."""

    cp_lp = cplex.Cplex(lp_file_name)
    cp_lp.set_results_stream(None)

    try:
        cp_lp.solve()
    except: #cplex.CplexSolverError:
        print("Exception raised during solve")
        return

    """
    import numpy as np
    result = np.zeros([len(N), len(D)])
    for i in range(len(N)):
        for j in range(len(D)):
            result[i] = cp_lp.solution.get_values('y' + str(i) + '_' + str(j))
    """

    return cp_lp.solution.get_objective_value()



# FOR WINNERS - needs update
def generate_lp_file_borda_owa(owa, lp_file_name, params, votes):
    """ this function generates lp file"""

    lp_file = open(lp_file_name, 'w')
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


def generate_lp_file_bloc_owa(owa, lp_file_name, params, votes, t_bloc):
    """ this function generates lp file"""

    lp_file = open(lp_file_name, 'w')
    lp_file.write("Maximize\nobj: ")
    pos = 0
    first = True
    for i in range(params['voters']):
        for j in range(params['orders']):
            for k in range(params['candidates']):

                if not first:
                    if k == t_bloc-1:
                        lp_file.write(" + ")
                first = False
                if k == t_bloc-1:
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


def get_winners_from_lp(tmp_file, params, candidates):
    """ this function ..."""

    cp_lp = cplex.Cplex(tmp_file)
    cp_lp.set_results_stream(None)

    try:
        cp_lp.solve()
    except cplex.CplexSolverError:
        print("Exception raised during solve")
        return

    result = [0.] * params['candidates']
    for i in range(params['candidates']):
        result[i] = cp_lp.solution.get_values('y' + str(i))
    # print(result)

    params['pure'] = True

    winner_id = 0
    winners = [0.] * params['orders']
    for i in range(params['candidates']):
        if result[i] == 1.:
            if params['pure']:
                winners[winner_id] = i
            else:
                winners[winner_id] = candidates[i]
            winner_id += 1
    winners = sorted(winners)
    return winners