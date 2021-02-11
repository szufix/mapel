#!/usr/bin/env python

import math
from . import elections as el
from . import objects as obj
from . import lp as lp
import random as rand
import os
import numpy as np


# MAPPINGS
def map_distance(distance_name):
    return {'positionwise': compute_basic_distance,
            'positionwise_extension': compute_positionwise_extension_distance,
            'discreet': compute_basic_distance,
            }.get(distance_name)


def map_metric(metric_name):
    return {'l1': l1,
            'l2': l2,
            'chebyshev': chebyshev,
            'hellinger': hellinger,
            'emd': emd,
            'discreet': discreet,
            }.get(metric_name)


def map_matching_cost(distance_name):
    return {'positionwise': get_matching_cost_positionwise,
            'positionwise_extension': get_matching_cost_positionwise_extension,
            'discreet': get_matching_cost_discreet,
            }.get(distance_name)


# CONVERTERS
def votes_to_vectors(election):
    """ convert VOTES to VECTORS """
    vectors = np.zeros([election.num_candidates, election.num_candidates])

    for i in range(election.num_voters):
        pos = 0
        for j in range(election.num_candidates):
            vote = election.votes[0][i][j]
            if vote in election.blank_1:
                continue
            if vote == -1:
                continue
            vectors[vote][pos] += 1
            pos += 1

    for i in range(election.num_candidates):
        for j in range(election.num_candidates):
            vectors[i][j] /= float(election.num_voters)

    return vectors


# DISTANCES
"""
def get_distance(experiment_id, elections_ids, distance_name='positionwise', metric_name='emd'):
def get_distance(model, i, j):
    election_1 = model.elections[i]
    election_2 = model.elections[j]
    distance_func = map_distance(model.distance_name)
    return distance_func(election_1, election_2, model.distance_name, model.metric_name)
"""

def get_distance(election_1, election_2, distance_name='positionwise', metric_name='emd'):
    distance_func = map_distance(distance_name)
    return distance_func(election_1, election_2, distance_name=distance_name, metric_name=metric_name)


def compute_basic_distance(election_1, election_2, distance_name='positionwise', metric_name='emd'):
    length = election_1.num_candidates
    metric_name = map_metric(metric_name)
    matching_cost = map_matching_cost(distance_name)
    cost_table = matching_cost(election_1, election_2, metric_name)
    objective_value = lp.solve_lp_matching_vector(cost_table, length)
    return objective_value


def compute_positionwise_extension_distance(election_1, election_2, distance_name, metric_name):
    length_1 = election_1.num_candidates
    length_2 = election_2.num_candidates
    metric_name = map_metric(metric_name)
    matching_cost = map_matching_cost(distance_name)
    cost_table = matching_cost(election_1, election_2, metric_name)
    result = lp.solve_lp_matching_interval(cost_table, length_1, length_2)
    return result


# HELPER FUNCTIONS
def get_matching_cost_positionwise(ele_1, ele_2, metric_name):
    vectors_1 = ele_1.votes_to_positionwise_vectors()
    vectors_2 = ele_2.votes_to_positionwise_vectors()
    size = ele_1.num_candidates
    cost_table = [[metric_name(list(vectors_1[i]), list(vectors_2[j]), size) for i in range(size)] for j in range(size)]
    return cost_table


def get_matching_cost_positionwise_extension(ele_1, ele_2, metric_name):
    size_1 = ele_1.num_candidates
    size_2 = ele_2.num_candidates
    precision = 9*9 * 10*10 #size_1*size_2
    interval_1 = ele_1.votes_to_positionwise_intervals(precision=precision)
    interval_2 = ele_2.votes_to_positionwise_intervals(precision=precision)
    cost_table = [[metric_name(list(interval_2[i]), list(interval_1[j]), precision) for i in range(size_2)] for j in range(size_1)]
    return cost_table


def get_matching_cost_discreet(ele_1, ele_2, metric_name):
    size = ele_1.num_candidates
    cost_table = [[metric_name(ele_1.votes[i], ele_2.votes[j], ele_1.num_candidates) for i in range(size)] for j in range(size)]
    return cost_table


# METRICS
def discreet(vector_1, vector_2, num_candidates):
    """ compute DISCREET metric """
    for i in range(num_candidates):
        if vector_1[i] != vector_2[i]:
            return 1
    return 0


def l1(vector_1, vector_2, num_candidates):
    """ compute L1 metric """
    return sum([abs(vector_1[i] - vector_2[i]) for i in range(num_candidates)])


def l2(vector_1, vector_2, num_candidates):
    """ compute L2 metric """
    return sum([math.pow((vector_1[i] - vector_2[i]), 2) for i in range(num_candidates)])


def chebyshev(vector_1, vector_2, num_candidates):
    """ compute CHEBYSHEV metric """
    return max([abs(vector_1[i] - vector_2[i]) for i in range(num_candidates)])


def hellinger(vector_1, vector_2, num_candidates):
    """ compute HELLINGER metric """
    h1 = np.average(vector_1)
    h2 = np.average(vector_2)
    product = sum([math.sqrt(vector_1[i] * vector_2[i]) for i in range(num_candidates)])
    return math.sqrt(1 - (1 / math.sqrt(h1 * h2 * num_candidates * num_candidates)) * product)


def emd(vector_1, vector_2, num_candidates):
    """ compute EMD metric """
    dirt = 0.
    for i in range(num_candidates-1):
        surplus = vector_1[i] - vector_2[i]
        dirt += abs(surplus)
        vector_1[i+1] += surplus
    return dirt


# SCORING FUNCTIONS
def get_highest_plurality_score(election):
    """ compute highest PLURALITY score of a given election"""

    if election.fake:
        c = election.num_candidates
        v = election.num_voters
        vectors = election.votes_to_positionwise_vectors()
        first = []
        for i in range(c):
            first.append(vectors[i][0])
        return max(first)*v

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        scores[vote[0]] += 1

    return max(scores)


def get_highest_borda_score(election):
    """ compute highest BORDA score of a given election"""

    if election.fake:
        c = election.num_candidates
        v = election.num_voters
        vectors = election.votes_to_positionwise_vectors()
        return sum([vectors[0][i]*(c-i-1) for i in range(c)])*v

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        for i in range(election.num_candidates):
            scores[vote[i]] += election.num_candidates - i - 1

    return max(scores)


def get_highest_copeland_score(potes, num_voters, num_candidates):
    """ compute highest COPELAND score of a given election """

    scores = np.zeors([num_candidates])

    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            result = 0
            for k in range(num_voters):
                if potes[k][i] < potes[k][j]:
                    result += 1
            if result > num_voters / 2:
                scores[i] += 1
                scores[j] -= 1
            elif result < num_voters / 2:
                scores[i] -= 1
                scores[j] += 1

    return max(scores)


def get_lowest_dodgson_score(election):
    """ compute lowest DODGSON score of a given election """

    min_score = math.inf

    for target_id in range(election.num_candidates):

        # PREPARE N
        unique_potes, N = potes_to_unique_potes(election.potes)

        e = np.zeros([len(N), election.num_candidates, election.num_candidates])

        # PREPARE e
        for i, p in enumerate(unique_potes):
            for j in range(election.num_candidates):
                for k in range(election.num_candidates):
                    if p[target_id] <= p[k] + j:
                        e[i][j][k] = 1

        # PREPARE D
        D = [0 for _ in range(election.num_candidates)]
        threshold = math.ceil((election.num_voters+1)/2.)
        for k in range(election.num_candidates):
            diff = 0
            for i, p in enumerate(unique_potes):
                if p[target_id] < p[k]:
                    diff += N[i]
                if diff >= threshold:
                    D[k] = 0
                else:
                    D[k] = threshold-diff
        D[target_id] = 0    # always winning

        file_name = str(rand.random()) + '.lp'
        path = os.path.join(os.getcwd(), "trash", file_name)
        lp.generate_lp_file_dodgson_score(path, N=N, e=e, D=D)
        score = lp.solve_lp_dodgson_score(path, N=N, D=D)

        remove_lp_file(path)

        if score < min_score:
            min_score = score

    return min_score


# OTHER
def potes_to_unique_potes(potes):
    unique_potes = []
    N = []
    for pote in potes:
        flag_new = True
        for i, p in enumerate(unique_potes):
            if list(pote) == list(p):
                N[i] += 1
                flag_new = False
        if flag_new:
            unique_potes.append(pote)
            N.append(1)
    return unique_potes, N


def is_condorect_winner(election):

    for i in range(election.num_candidates):

        condocret_winner = True
        for j in range(election.num_candidates):

            diff = 0
            for k in range(election.num_voters):

                if election.potes[k][i] <= election.potes[k][j]:
                    diff += 1

            if diff < math.ceil((election.num_voters+1)/2.):
                condocret_winner = False
                break

        if condocret_winner:
            return True

    return False


def remove_lp_file(path):
    try:
        os.remove(path)
    except:
        pass


def map_diameter(c):
    return 1/3*(c+1)*(c-1)


# MAIN FUNCTION
def compute_distances(experiment_id, metric_name='emd', distance_name='positionwise', starting_from=0):

    if starting_from == 0:
        model = obj.Model(experiment_id, distance_name=distance_name, metric_name=metric_name)
    else:
        model = obj.Model_xd(experiment_id, distance_name=distance_name, metric_name=metric_name)

    results = []

    for i in range(0, model.num_elections):
        print(i)
        for j in range(i + 1, model.num_elections):

            if j < starting_from:
                old_result = model.distances[i][j]
                results.append(old_result)
            else:
                result = get_distance(model.elections[i], model.elections[j], distance_name=model.distance_name, metric_name=model.metric_name)
                results.append(result)

    ctr = 0
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", str(distance_name) + ".txt")
    with open(path, 'w') as txtfile:
        txtfile.write(str(model.num_elections) + '\n')
        txtfile.write(str(model.num_families) + '\n')
        txtfile.write(str(len(results)) + '\n')
        for i in range(model.num_elections):
            for j in range(i + 1, model.num_elections):
                txtfile.write(str(i) + ' ' + str(j) + ' ' + str(results[ctr]) + '\n')
                ctr += 1
