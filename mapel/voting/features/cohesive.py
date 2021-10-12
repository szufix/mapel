#!/usr/bin/env python

from numpy import ceil
from pulp import *

from mapel.voting.objects.ApprovalElection import ApprovalElection


def count_largest_cohesiveness_level_l_of_cohesive_group(election: ApprovalElection):
    if election.model == 'approval_zeros':
        return 0
    elif election.model == 'approval_ones':
        return election.k

    l_ans = 0
    for l in range(1, election.num_voters + 1):
        if solve_ilp_instance(election, l):
            l_ans = l
        else:
            break
    return l_ans


def solve_ilp_instance(election: ApprovalElection, l: int = 1) -> bool:

    model = LpProblem("cohesiveness_level_l", LpMaximize)
    X = [LpVariable("x_" + str(i), cat='Binary') for i in
         range(election.num_voters)]  # X[i] = 1 if we select i-th voter, otherwise 0
    Y = [LpVariable("y_" + str(j), cat='Binary') for j in
         range(election.num_candidates)]  # Y[j] = 1 if we select j-th candidate, otherwise 0
    s = int(ceil(
        l * election.num_voters / election.k))  # If there is any valid l-cohesive group, then there is also at least one with minimum possible size

    objective = l
    model += objective  # We want to maximize cohesiveness level l (but l is constant, only convention)

    x_sum_eq = 0
    for x in X:
        x_sum_eq += x
    model += x_sum_eq == s  # We choose exactly s voters

    y_sum_ineq = 0
    for y in Y:
        y_sum_ineq += y
    model += y_sum_ineq >= l  # We choose at least l candidates (although l are sufficient in this case)

    cand_to_voters_variables_list = [[] for j in range(election.num_candidates)]
    for i, d in enumerate(election.votes):
        for j in d:
            cand_to_voters_variables_list[j].append(X[i])
    # We want to assert that the selected voters approve all the selected candidates.
    # For each candidate j,  we construct the following inequality:  a_{0,j} * x_0 + a_{1,j} * x_1 + ... + a_{n-1,j} * x_{n-1}  -   s * y_j    >=    0
    # We define a_{i, j} as the flag indicating whether i-th voter approves j-th candidate (1 if yes, otherwise 0)
    # Let us observe that if the j-th candidate is not selected, then s * y_j = 0 and the above inequality is naturally satisfied.
    # However, if j-th candidate is selected, then the above can be satisfied if and only if all s selected voters approve j-th candidate
    for j, y in enumerate(Y):
        y_ineq = 0
        for x in cand_to_voters_variables_list[j]:
            y_ineq += x
        y_ineq -= s * y
        model += y_ineq >= 0

    model.solve(PULP_CBC_CMD(msg=False))
    # print(model)
    # print(LpStatus[model.status])
    # print(int(value(model.objective)))    # prints the best objective value - in our case useless, but can be useful in the future
    # if LpStatus[model.status] == 'Optimal':
    #     print([var.name + "=" + str(var.varValue) for var in model.variables() if var.varValue is not None and var.varValue > 0], sep=" ")    # prints result variables which have value > 0
    return LpStatus[model.status] == 'Optimal'
