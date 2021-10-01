#!/usr/bin/env python

import os
import numpy as np
from time import time
from threading import Thread


from mapel.voting.objects.Election import Election
from mapel.voting.objects.ApprovalElection import ApprovalElection
from mapel.voting.objects.OrdinalElection import OrdinalElection
from mapel.voting.objects.Graph import Graph
import mapel.voting._elections as el

import csv

from mapel.voting.metrics import main_ordinal_distances as mod
from mapel.voting.metrics import main_approval_distances as mad
from mapel.voting.metrics import main_graph_distances as mgd

from mapel.voting.metrics.inner_distances import l1

import networkx as nx


# MAIN FUNCTIONS
def get_distance(election_1, election_2, distance_name=None):
    """ Main function """
    if type(election_1) is Graph:
        return get_graph_distance(election_1.graph, election_2.graph, distance_name=distance_name)
    elif type(election_1) is ApprovalElection:
        return get_approval_distance(election_1, election_2, distance_name=distance_name)
    elif type(election_1) is OrdinalElection:
        return get_ordinal_distance(election_1, election_2, distance_name=distance_name)
    else:
        print('No such ballot!')


def get_approval_distance(election_1, election_2, distance_name=None):

    inner_distance, main_distance = distance_name.split('-')

    metrics_without_params = {
    }

    metrics_with_inner_distance = {
        'approval_frequency': mad.compute_approval_frequency,
        'coapproval_frequency_vectors': mad.compute_cooparoval_frequency_vectors,
    }

    if main_distance in metrics_without_params:
        return metrics_without_params.get(main_distance)(election_1, election_2)

    elif main_distance in metrics_with_inner_distance:
        return metrics_with_inner_distance.get(main_distance)(election_1, election_2,
                                                              inner_distance)


def get_ordinal_distance(election_1, election_2, distance_name=None):

    inner_distance, main_distance = distance_name.split('-')

    metrics_without_params = {
        'discrete': mod.compute_voter_subelection,
        'voter_subelection': mod.compute_voter_subelection,
        'candidate_subelection': mod.compute_candidate_subelection,
        'spearman': mod.compute_spearman_distance,
    }

    metrics_with_inner_distance = {
        'positionwise': mod.compute_positionwise_distance,
        'bordawise': mod.compute_bordawise_distance,
        'pairwise': mod.compute_pairwise_distance,
        'voterlikeness': mod.compute_voterlikeness_distance,
        'agg_voterlikeness': mod.compute_agg_voterlikeness_distance,
    }

    if main_distance in metrics_without_params:
        return metrics_without_params.get(main_distance)(election_1, election_2)

    elif main_distance in metrics_with_inner_distance:
        return metrics_with_inner_distance.get(main_distance)(election_1, election_2,
                                                              inner_distance)


def get_graph_distance(graph_1, graph_2, distance_name=''):

    graph_simple_metrics = {'closeness_centrality': nx.closeness_centrality,
                            'degree_centrality': nx.degree_centrality,
                            'betweenness_centrality': nx.betweenness_centrality,
                            'eigenvector_centrality': nx.eigenvector_centrality,
                            }

    graph_advanced_metrics = {
        'graph_edit_distance': mgd.compute_graph_edit_distance,
        'graph_histogram': mgd.compute_graph_histogram,
    }

    if distance_name in graph_simple_metrics:
        return mgd.compute_graph_simple_metrics(graph_1, graph_2, graph_simple_metrics[distance_name])

    if distance_name in graph_advanced_metrics:
        return graph_advanced_metrics.get(distance_name)(graph_1, graph_2)


def _minus_one(vector):
    if vector is None:
        return None
    new_vector = [0 for _ in range(len(vector))]
    for i in range(len(vector)):
        new_vector[vector[i]] = i
    return new_vector


def single_thread(experiment, distances, times, thread_ids, t, matchings):
    """ Single thread for computing distance """

    for election_id_1, election_id_2 in thread_ids:

        start_time = time()
        distance, matching = get_distance(experiment.instances[election_id_1],
                                experiment.instances[election_id_2],
                                distance_name=experiment.distance_name)

        matchings[election_id_1][election_id_2] = matching
        matchings[election_id_2][election_id_1] = _minus_one(matching)
        distances[election_id_1][election_id_2] = distance
        distances[election_id_2][election_id_1] = distances[election_id_1][election_id_2]
        times[election_id_1][election_id_2] = time() - start_time
        times[election_id_2][election_id_1] = times[election_id_1][election_id_2]

    print("thread " + str(t) + " is ready :)")


# deprecated
# def compute_distances(experiment_id, distance_name='emd-positionwise', num_threads=1):
#     """ Compute distances between instances (using threads)"""
#
#     experiment = Experiment(experiment_id, distance_name=distance_name, instances='import', with_matrices=True)
#     distances = {}
#     for election_id in experiment.instances:
#         distances[election_id] = {}
#
#     threads = [{} for _ in range(num_threads)]
#
#     ids = []
#     for i, election_1 in enumerate(experiment.instances):
#         for j, election_2 in enumerate(experiment.instances):
#             if i < j:
#                 ids.append((election_1, election_2))
#
#     num_distances = len(ids)
#
#     for t in range(num_threads):
#         print('thread: ', t)
#         sleep(0.1)
#         start = int(t * num_distances / num_threads)
#         stop = int((t + 1) * num_distances / num_threads)
#         thread_ids = ids[start:stop]
#
#         threads[t] = Thread(target=single_thread, args=(experiment, distances, thread_ids, t))
#         threads[t].start()
#
#     for t in range(num_threads):
#         threads[t].join()
#
#     path = os.path.join(os.getcwd(), "experiments", experiment_id, "distances",
#                         str(distance_name) + ".csv")
#
#     with open(path, 'w', newline='') as csv_file:
#         writer = csv.writer(csv_file, delimiter=',')
#         writer.writerow(["election_id_1", "election_id_2", "distance"])
#
#         for i, election_1 in enumerate(experiment.instances):
#             for j, election_2 in enumerate(experiment.instances):
#                 if i < j:
#                     distance = str(distances[election_1][election_2])
#                     writer.writerow([election_1, election_2, distance])


# NEEDS UPDATE #

# deprecated
# def extend_distances(experiment_id, distance_name='emd-positionwise',
#                       num_threads=1, starting_from=0, ending_at=10000):
#     """ Compute distance using threads"""
#
#     if starting_from == 0 and ending_at == 10000:
#         experiment = obj..Experiment(experiment_id,
#         distance_name=distance_name)
#         distances = {}
#         for election_id in experiment.instances:
#             distances[election_id] = {}
#     else:
#         experiment = Experiment_xd(experiment_id,
#         distance_name=distance_name)
#         distances = {}
#         for election_id in experiment.instances:
#             distances[election_id] = {}
#         for i, election_id_1 in enumerate(experiment.instances):
#             for j, election_id_2 in enumerate(experiment.instances):
#                 if i < j:
#                     try:
#                         distances[election_id_1][election_id_2] =
#                         experiment.distances[election_id_1][election_id_2]
#                     except:
#                         pass
#
#     threads = [{} for _ in range(num_threads)]
#
#     ids = []
#     for i, election_1 in enumerate(experiment.instances):
#         for j, election_2 in enumerate(experiment.instances):
#             if i < j:
#                 ids.append((election_1, election_2))
#
#     num_distances = len(ids)
#
#     for t in range(num_threads):
#         print('thread: ', t)
#         sleep(0.1)
#         start = int(t * num_distances / num_threads)
#         stop = int((t + 1) * num_distances / num_threads)
#         thread_ids = ids[start:stop]
#
#         threads[t] = Thread(target=single_thread, args=(experiment,
#         distances, thread_ids, t))
#         threads[t].start()
#
#     for t in range(num_threads):
#         threads[t].join()
#
#     path = os.path.join(os.getcwd(), "experiments", experiment_id,
#     "distances",
#                         str(distance_name) + ".csv")
#
#     with open(path, 'w', newline='') as csv_file:
#         writer = csv.writer(csv_file, delimiter=',')
#         writer.writerow(["election_id_1", "election_id_2", "distance"])
#
#         for i, election_1 in enumerate(experiment.instances):
#             for j, election_2 in enumerate(experiment.instances):
#                 if i < j:
#                     distance = str(distances[election_1][election_2])
#                     writer.writerow([election_1, election_2, distance])

### NEW 13.07.2021 ###


def compute_distances_between_votes(dict_with_votes,
                                    distance_name='emd-positionwise'):
    instances = {}
    for election_id in dict_with_votes:
        instances[election_id] = Election("virtual", "virtual",
                                          votes=dict_with_votes[election_id])

    distances = {}
    for election_id in instances:
        distances[election_id] = {}

    for i, election_id_1 in enumerate(instances):
        for j, election_id_2 in enumerate(instances):
            if i < j:
                distance = get_distance(instances[election_id_1],
                                        instances[election_id_2],
                                        distance_name=distance_name)
                distances[election_id_1][election_id_2] = distance
                distances[election_id_2][election_id_1] = \
                    distances[election_id_1][election_id_2]
    return distances


def thread_function(experiment, distance_name, all_pairs,
                    election_models, params,
                    num_voters, num_candidates, thread_ids,
                    distances, t, precision,
                    matchings, times):

    for election_id_1, election_id_2 in thread_ids:
        # print('hello', election_id_1, election_id_2)
        result = 0
        local_ctr = 0

        total_time = 0

        for p in range(precision):
            # print('p', p)
            # print('local', t)
            if t < 5:
                print("ctr: ", local_ctr)
                local_ctr += 1

            # print(params)
            election_1 = el.generate_instances(
                experiment=experiment,
                election_model=election_models[election_id_1],
                election_id=election_id_1,
                num_candidates=num_candidates, num_voters=num_voters,
                params=params[election_id_1])
            # print('start')
            election_2 = el.generate_instances(
                experiment=experiment,
                election_model=election_models[election_id_2],
                election_id=election_id_2,
                num_candidates=num_candidates, num_voters=num_voters,
                params=params[election_id_2])

            start_time = time()
            distance, mapping = get_distance(election_1, election_2,
                                             distance_name=distance_name)
            total_time += (time() - start_time)
            # print(distance)
            # delete tmp files

            all_pairs[election_id_1][election_id_2][p] = round(distance, 5)
            result += distance

        distances[election_id_1][election_id_2] = result / precision
        distances[election_id_2][election_id_1] = \
            distances[election_id_1][election_id_2]

        # matchings[election_id_1][election_id_2] = mapping
        times[election_id_1][election_id_2] = total_time / precision
        times[election_id_2][election_id_1] = \
            times[election_id_1][election_id_2]

    print("thread " + str(t) + " is ready :)")


def compute_subelection_by_groups(
        experiment, distance_name='0-voter_subelection', t=0, num_threads=1,
        precision=10, self_distances=False, num_voters=None,
        num_candidates=None):

    if num_candidates is None:
        num_candidates = experiment.default_num_candidates
    if num_voters is None:
        num_voters = experiment.default_num_voters

    num_distances = experiment.num_families * (experiment.num_families+1) / 2

    threads = [None for _ in range(num_threads)]

    election_models = {}
    params = {}

    for family_id in experiment.families:
        # print(family_id)
        for i in range(experiment.families[family_id].size):
            election_id = family_id + '_' + str(i)
            election_models[election_id] = \
                experiment.families[family_id].model
            params[election_id] = experiment.families[family_id].params


    ids = []
    for i, election_1 in enumerate(experiment.instances):
        for j, election_2 in enumerate(experiment.instances):
            if i == j:
                if self_distances:
                    ids.append((election_1, election_2))
            elif i < j:
                ids.append((election_1, election_2))

    all_pairs = {}
    std = {}
    distances = {}
    matchings = {}
    times = {}
    for i, election_1 in enumerate(experiment.instances):
        all_pairs[election_1] = {}
        std[election_1] = {}
        distances[election_1] = {}
        matchings[election_1] = {}
        times[election_1] = {}
        for j, election_2 in enumerate(experiment.instances):
            if i == j:
                if self_distances:
                    all_pairs[election_1][election_2] = \
                        [0 for _ in range(precision)]
            elif i < j:
                all_pairs[election_1][election_2] = \
                    [0 for _ in range(precision)]

    for t in range(num_threads):

        start = int(t * num_distances / num_threads)
        stop = int((t+1) * num_distances / num_threads)
        thread_ids = ids[start:stop]
        print('t: ', t)

        threads[t] = Thread(target=thread_function,
                            args=(experiment, distance_name, all_pairs,
                                  election_models, params,
                                  num_voters, num_candidates,
                                  thread_ids, distances, t,
                                  precision, matchings, times))
        threads[t].start()

    #"""
    for t in range(num_threads):
        threads[t].join()
    #"""
    #print(results)
    # print(all_pairs)

    # COMPUTE STD
    for i, election_1 in enumerate(experiment.instances):
        for j, election_2 in enumerate(experiment.instances):
            if i == j:
                if self_distances:
                    value = float(np.std(np.array(all_pairs[election_1][election_2])))
                    std[election_1][election_2] = round(value, 5)
            elif i < j:
                value = float(np.std(np.array(all_pairs[election_1][election_2])))
                std[election_1][election_2] = round(value, 5)

    experiment.distances = distances
    experiment.times = times

    if experiment.store:
        path = os.path.join(os.getcwd(), "experiments",
                            experiment.experiment_id, "distances",
                            str(distance_name) + ".csv")

        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(
                ["election_id_1", "election_id_2", "distance", "time", "std"])

            for i, election_1 in enumerate(experiment.instances):
                for j, election_2 in enumerate(experiment.instances):
                    if (i == j and self_distances) or i < j:
                        distance = str(distances[election_1][election_2])
                        time = str(times[election_1][election_2])
                        std_value = str(std[election_1][election_2])
                        writer.writerow(
                            [election_1, election_2, distance, time, std_value])

    if experiment.store:

        ctr = 0
        path = os.path.join(os.getcwd(), "experiments",
                            experiment.experiment_id, "distances",
                            str(distance_name) + "_all_pairs.txt")
        with open(path, 'w') as txtfile:
            for i, election_1 in enumerate(experiment.instances):
                for j, election_2 in enumerate(experiment.instances):
                    if (i == j and self_distances) or i < j:
                        for p in range(precision):

                            txtfile.write(str(i) + ' ' + str(j) + ' ' + str(p) + ' ' +
                                          str(all_pairs[election_1][election_2][p]) + '\n')
                            ctr += 1
