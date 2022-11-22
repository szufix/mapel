#!/usr/bin/env python

from numpy import ceil
import time

try:
    import pulp
except Exception:
    pulp = None


from mapel.elections.objects.ApprovalElection import ApprovalElection
from math import ceil
import itertools
from collections import defaultdict


def count_number_of_cohesive_groups_brute(election: ApprovalElection, l: int = 1,
                                          committee_size: int = 10):
    answer = 0
    min_size = int(ceil(l * election.num_voters / committee_size))
    voters = [i for i in range(0, election.num_voters)]
    for s in powerset(voters, min_size=min_size):
        if len(s) < min_size:
            continue
        cands = set(election.votes[s[0]])
        for v in s:
            cands &= election.votes[v]
        if len(cands) >= l:
            # print(s, "  ", cands)
            answer += 1
    return answer


####################################################################################################
####################################################################################################
####################################################################################################

def powerset(iterable, min_size=0):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(min_size, len(s) + 1))


def newton(n: int, k: int):
    if k > n:
        return 0
    answer = 1
    for i in range(n - k + 1, n + 1):
        answer *= i
    for i in range(1, k + 1):
        answer //= i
    return answer


def count_number_of_cohesive_groups(election: ApprovalElection, l: int = 1,
                                    committee_size: int = 10):

    if l > 1:
        raise NotImplementedError()
    answer = 0
    d = defaultdict(lambda: 0)
    timeout = time.time() + 20 * 1  # 20s from now
    for v in range(election.num_voters):
        for s in powerset(election.votes[v], min_size=1):
            d[s] += 1
            if time.time() > timeout:
                return -1
    min_size = int(ceil(l * election.num_voters / committee_size))
    for s in d:
        for siz in range(min_size, d[s] + 1):
            sign = 2 * (len(s) % 2) - 1  # 1 for even, -1 for odd, comes from (-1) ^ (s-1)
            answer += newton(d[s], siz) * sign
            # print(s, d[s], siz, sign, newton(d[s], siz) * sign)
    return answer


####################################################################################################
####################################################################################################
####################################################################################################

def count_largest_cohesiveness_level_l_of_cohesive_group(election: ApprovalElection, feature_params):
    committee_size = feature_params['committee_size']

    if election.model == 'approval_zeros':
        return 0
    elif election.model == 'approval_ones':
        return min(committee_size, election.num_candidates)

    l_ans = 0
    for l in range(1, election.num_voters + 1):
        if solve_ilp_instance(election, committee_size, l):
            l_ans = l
        else:
            break
    return l_ans


def solve_ilp_instance(election: ApprovalElection, committee_size: int, l: int = 1) -> bool:
    pulp.getSolver('CPLEX_CMD')
    model = pulp.LpProblem("cohesiveness_level_l", pulp.LpMaximize)
    X = [pulp.LpVariable("x_" + str(i), cat='Binary') for i in
         range(election.num_voters)]  # X[i] = 1 if we select i-th voter, otherwise 0
    Y = [pulp.LpVariable("y_" + str(j), cat='Binary') for j in
         range(election.num_candidates)]  # Y[j] = 1 if we select j-th candidate, otherwise 0
    s = int(ceil(
        l * election.num_voters / committee_size))  # If there is any valid l-cohesive group, then there is also at least one with minimum possible size

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

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    # print(culture_id)
    # print(LpStatus[culture_id.status])
    # print(int(value(culture_id.objective)))    # prints the best objective value - in our case useless, but can be useful in the future
    # if LpStatus[culture_id.status] == 'Optimal':
    #     print([var.election_id + "=" + str(var.varValue) for var in culture_id.variables() if var.varValue is not None and var.varValue > 0], sep=" ")    # prints result variables which have value > 0
    return pulp.LpStatus[model.status] == 'Optimal'
