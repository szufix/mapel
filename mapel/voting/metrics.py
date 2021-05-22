#!/usr/bin/env python

import math
import os
import random as rand
import time

import numpy as np

from threading import Thread

from . import lp as lp
from . import objects as obj
from time import sleep


# SUBELECTIONS
def compute_voter_subelection(election_1, election_2, distance_name=None, metric_name=None):
    objective_value = lp.solve_lp_voter_subelection(election_1, election_2, metric_name)
    return objective_value


def compute_candidate_subelection(election_1, election_2, distance_name=None, metric_name=None):
    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    objective_value = lp.solve_lp_candidate_subelections(path, election_1, election_2, magic_param=1)
    remove_lp_file(path)
    return objective_value


# MAPPINGS
def map_distance(distance_name):
    return {'positionwise': compute_basic_distance,
            'bordawise': compute_bordawise_distance,
            'pairwise': compute_pairwise_distance,
            'voterlikeness': compute_voterlikeness_distance,
            'agg_voterlikeness': compute_agg_voterlikeness_distance,
            'discrete': compute_voter_subelection,
            'voter_subelection': compute_voter_subelection,
            'candidate_subelection': compute_candidate_subelection,
            'spearman': compute_spearman_distance,
            }.get(distance_name)


def map_metric(metric_name):
    return {'l1': l1,
            'l2': l2,
            'chebyshev': chebyshev,
            'hellinger': hellinger,
            'emd': emd,
            'discrete': discrete,
            }.get(metric_name)


def map_matching_cost(distance_name):
    return {'positionwise': get_matching_cost_positionwise,
            }.get(distance_name)


# CONVERTERS -- probably unnecessary here
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

def get_distance(election_1, election_2, distance_name='', metric_name=''):
    distance_func = map_distance(distance_name)
    return distance_func(election_1, election_2, distance_name=distance_name, metric_name=metric_name)


def compute_basic_distance(election_1, election_2, distance_name='', metric_name=''):
    if distance_name in {'positionwise', 'bordawise'}:
        length = election_1.num_candidates
    metric_name = map_metric(metric_name)
    matching_cost = map_matching_cost(distance_name)
    cost_table = matching_cost(election_1, election_2, metric_name)
    objective_value = lp.solve_lp_matching_vector(cost_table, length)
    return objective_value


def compute_agg_voterlikeness_distance(election_1, election_2, distance_name='agg_voterlikeness', metric_name='l1'):
    """ Compute Aggregated-Voterlikeness distance between elections """
    vector_1, num_possible_scores = election_1.votes_to_agg_voterlikeness_vector()
    vector_2, _ = election_2.votes_to_agg_voterlikeness_vector()
    metric_name = map_metric(metric_name)
    return metric_name(vector_1, vector_2, num_possible_scores)


def compute_bordawise_distance(election_1, election_2, distance_name='bordawise', metric_name='emd'):
    """ Compute Bordawise distance between elections """
    vector_1, num_possible_scores = election_1.votes_to_bordawise_vector()
    vector_2, _ = election_2.votes_to_bordawise_vector()
    metric_name = map_metric(metric_name)
    return metric_name(vector_1, vector_2, num_possible_scores)


def compute_pairwise_distance(election_1, election_2, distance_name='pairwise', metric_name='l1'):
    """ Compute Pairwise distance between elections """
    length = election_1.num_candidates

    matrix_1 = election_1.votes_to_pairwise_matrix()
    matrix_2 = election_2.votes_to_pairwise_matrix()

    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_lp_file_matching_matrix(path, matrix_1, matrix_2, length)
    matching_cost = lp.solve_lp_matrix(path, matrix_1, matrix_2, length)
    remove_lp_file(path)

    return matching_cost


def compute_voterlikeness_distance(election_1, election_2, distance_name='voterlikeness', metric_name='l1'):
    """ Compute Voterlikeness distance between elections """

    length = election_1.num_voters

    matrix_1 = election_1.votes_to_voterlikeness_matrix()
    matrix_2 = election_2.votes_to_voterlikeness_matrix()

    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_lp_file_matching_matrix(path, matrix_1, matrix_2, length)
    matching_cost = lp.solve_lp_matrix(path, matrix_1, matrix_2, length)

    remove_lp_file(path)
    return matching_cost


def compute_spearman_distance(election_1, election_2, distance_name='spearman', metric_name='l1'):
    """ Compute Spearman distance between elections """

    # length = election_1.num_candidates
    # metric_name = map_metric(metric_name)
    # matching_cost = map_matching_cost(distance_name)
    # cost_table = matching_cost(election_1, election_2, metric_name)
    # objective_value = lp.solve_lp_matching_vector(cost_table, length)

    votes_1 = election_1.votes
    votes_2 = election_2.votes
    params = {'voters': election_1.num_voters,
              'candidates': election_1.num_candidates}

    file_name = str(rand.random()) + '.lp'
    path = os.path.join(os.getcwd(), "trash", file_name)
    lp.generate_ilp_distance(path, votes_1, votes_2, params, 'spearman')
    objective_value = lp.solve_ilp_distance(path, votes_1, votes_2, params, 'spearman')
    remove_lp_file(path)
    return objective_value


# HELPER FUNCTIONS
def get_matching_cost_positionwise(ele_1, ele_2, metric_name):
    """ Get mathcing cost for positionwise distances """
    vectors_1 = ele_1.votes_to_positionwise_vectors()
    vectors_2 = ele_2.votes_to_positionwise_vectors()
    size = ele_1.num_candidates
    cost_table = [[metric_name(list(vectors_1[i]), list(vectors_2[j]), size) for i in range(size)] for j in range(size)]
    return cost_table


# METRICS
def discrete(vector_1, vector_2, length):
    """ compute DISCRETE metric """
    for i in range(length):
        if vector_1[i] != vector_2[i]:
            return 1
    return 0


def l1(vector_1, vector_2, length):
    """ compute L1 metric """
    return sum([abs(vector_1[i] - vector_2[i]) for i in range(length)])


def l2(vector_1, vector_2, length):
    """ compute L2 metric """
    return sum([math.pow((vector_1[i] - vector_2[i]), 2) for i in range(length)])


def chebyshev(vector_1, vector_2, length):
    """ compute CHEBYSHEV metric """
    return max([abs(vector_1[i] - vector_2[i]) for i in range(length)])


def hellinger(vector_1, vector_2, length):
    """ compute HELLINGER metric """
    h1 = np.average(vector_1)
    h2 = np.average(vector_2)
    product = sum([math.sqrt(vector_1[i] * vector_2[i]) for i in range(length)])
    return math.sqrt(1 - (1 / math.sqrt(h1 * h2 * length * length)) * product)


def emd(vector_1, vector_2, length):
    """ compute EMD metric """
    dirt = 0.
    for i in range(length - 1):
        surplus = vector_1[i] - vector_2[i]
        dirt += abs(surplus)
        vector_1[i + 1] += surplus
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
        return max(first) * v

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        scores[vote[0]] += 1

    return max(scores)


def get_highest_borda_score(election):
    """ compute highest BORDA score of a given election"""

    c = election.num_candidates
    v = election.num_voters
    vectors = election.votes_to_positionwise_vectors()
    return sum([vectors[0][i] * (c - i - 1) for i in range(c)]) * v


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
        threshold = math.ceil((election.num_voters + 1) / 2.)
        for k in range(election.num_candidates):
            diff = 0
            for i, p in enumerate(unique_potes):
                if p[target_id] < p[k]:
                    diff += N[i]
                if diff >= threshold:
                    D[k] = 0
                else:
                    D[k] = threshold - diff
        D[target_id] = 0  # always winning

        file_name = str(rand.random()) + '.lp'
        path = os.path.join(os.getcwd(), "trash", file_name)
        lp.generate_lp_file_dodgson_score(path, N=N, e=e, D=D)
        score = lp.solve_lp_dodgson_score(path)

        remove_lp_file(path)

        if score < min_score:
            min_score = score

    return min_score


# OTHER
def potes_to_unique_potes(potes):
    """ Remove repetitions from potes (positional votes) """
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
    """ Check if election witness Condorect winner"""
    for i in range(election.num_candidates):

        condocret_winner = True
        for j in range(election.num_candidates):

            diff = 0
            for k in range(election.num_voters):

                if election.potes[k][i] <= election.potes[k][j]:
                    diff += 1

            if diff < math.ceil((election.num_voters + 1) / 2.):
                condocret_winner = False
                break

        if condocret_winner:
            return True

    return False


def remove_lp_file(path):
    """ Safely remove lp file """
    try:
        os.remove(path)
    except:
        pass


def map_diameter(c):
    """ Compute the diameter """
    return 1 / 3 * (c + 1) * (c - 1)


def result_backup(experiment_id, result, i, j):
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", "backup.txt")
    with open(path, 'a') as txtfile:
        txtfile.write(str(i) + ' ' + str(j) + ' ' + str(result) + '\n')


# MAIN FUNCTION
def single_thread(model, results, thread_ids, t, testing):
    """ Single thread for computing distance """

    for i, j in thread_ids:
        # if t == 0 or (t < 10 and testing):
        #    print(t, ' : ', i, j)

        distance = get_distance(model.elections[i], model.elections[j], distance_name=model.distance_name,
                                metric_name=model.metric_name)
        # print(distance)
        results[i][j] = distance

    print("thread " + str(t) + " is ready :)")


def compute_distances(experiment_id, metric_name='', distance_name='',
                      testing=False, num_threads=1, starting_from=0, ending_at=10000):
    """ Compute distance using threads"""

    if starting_from == 0 and ending_at == 10000:
        model = obj.Model(experiment_id, distance_name=distance_name, metric_name=metric_name)
        results = np.zeros([model.num_elections, model.num_elections])
    else:
        model = obj.Model_xd(experiment_id, distance_name=distance_name, metric_name=metric_name)
        results = np.zeros([model.num_elections, model.num_elections])
        for i in range(model.num_elections):
            for j in range(i + 1, model.num_elections):
                try:
                    results[i][j] = model.distances[i][j]
                except:
                    pass

    threads = [None for _ in range(num_threads)]

    ids = []
    for i in range(model.num_elections):
        for j in range(i + 1, model.num_elections):
            if (starting_from <= i < ending_at) or (starting_from <= j < ending_at):
                ids.append((i, j))

    num_distances = len(ids)

    for t in range(num_threads):
        print('thread: ', t)
        sleep(0.1)
        start = int(t * num_distances / num_threads)
        stop = int((t + 1) * num_distances / num_threads)
        thread_ids = ids[start:stop]

        threads[t] = Thread(target=single_thread, args=(model, results, thread_ids, t, testing))
        threads[t].start()

    for t in range(num_threads):
        threads[t].join()

    ctr = 0
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances",
                        str(metric_name) + '-' + str(distance_name) + ".txt")
    with open(path, 'w') as txtfile:
        txtfile.write(str(model.num_elections) + '\n')
        txtfile.write(str(model.num_families) + '\n')
        txtfile.write(str(int(len(results))) + '\n')
        for i in range(model.num_elections):
            for j in range(i + 1, model.num_elections):
                txtfile.write(str(i) + ' ' + str(j) + ' ' + str(results[i][j]) + '\n')
                ctr += 1


def compute_distances_classic(experiment_id, metric_name='emd', distance_name='positionwise',
                              starting_from=0, show=False, backup=False, run_from=0):
    """ Compute distance without using threads"""

    if starting_from == 0:
        model = obj.Model(experiment_id, distance_name=distance_name, metric_name=metric_name)
    else:
        model = obj.Model_xd(experiment_id, distance_name=distance_name, metric_name=metric_name)

    results = []

    for i in range(run_from, model.num_elections):
        print(i)
        for j in range(i + 1, model.num_elections):

            start = time.time()

            if j < starting_from:
                old_result = model.distances[i][j]
                results.append(old_result)
            else:
                result = get_distance(model.elections[i], model.elections[j], distance_name=model.distance_name,
                                      metric_name=model.metric_name)
                results.append(result)

                if backup:
                    result_backup(experiment_id, result, i, j)

                if show:
                    print(result)

            stop = time.time()
            if stop - start > 1:
                print(i, j, stop - start)

    ctr = 0
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances",
                        str(distance_name) + ".txt")
    with open(path, 'w') as txt_file:
        txt_file.write(str(model.num_elections) + '\n')
        txt_file.write(str(model.num_families) + '\n')
        txt_file.write(str(len(results)) + '\n')
        for i in range(model.num_elections):
            for j in range(i + 1, model.num_elections):
                txt_file.write(str(i) + ' ' + str(j) + ' ' + str(results[ctr]) + '\n')
                ctr += 1


def get_effective_num_candidates(election, mode='Borda'):
    """ Compute effective number of candidates """

    c = election.num_candidates
    vectors = election.votes_to_positionwise_vectors()

    if mode == 'Borda':
        scores = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) / (c * (c - 1) / 2) for j in range(c)]
    elif mode == 'Plurality':
        scores = [sum([vectors[j][i] for i in range(1)]) for j in range(c)]

    return 1. / sum([x * x for x in scores])
