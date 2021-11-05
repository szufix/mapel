#!/usr/bin/env python

from time import time
from typing import Union, Callable

import networkx as nx
import numpy as np

from mapel.voting.metrics import main_approval_distances as mad
from mapel.voting.metrics import main_graph_distances as mgd
from mapel.voting.metrics import main_ordinal_distances as mod
from mapel.voting.metrics.inner_distances import map_str_to_func
from mapel.voting.objects.ApprovalElection import ApprovalElection
from mapel.voting.objects.Election import Election
from mapel.voting.objects.Experiment import Experiment
from mapel.voting.objects.Graph import Graph
from mapel.voting.objects.OrdinalElection import OrdinalElection


def get_distance(election_1: Union[Election or Graph], election_2: Union[Election or Graph],
                 distance_id: str = None) -> float or (float, list):
    """ Return: distance between instances, (if applicable) optimal matching """

    if type(election_1) is Graph and type(election_2) is Graph:
        return get_graph_distance(election_1.graph, election_2.graph, distance_id=distance_id)
    elif type(election_1) is ApprovalElection and type(election_2) is ApprovalElection:
        return get_approval_distance(election_1, election_2, distance_id=distance_id)
    elif type(election_1) is OrdinalElection and type(election_2) is OrdinalElection:
        return get_ordinal_distance(election_1, election_2, distance_id=distance_id)
    else:
        print('No such instance!')


def get_approval_distance(election_1: ApprovalElection, election_2: ApprovalElection,
                          distance_id: str = None) -> float or (float, list):
    """ Return: distance between approval elections, (if applicable) optimal matching """

    inner_distance, main_distance = extract_distance_id(distance_id)

    metrics_without_params = {
        'flow': mad.compute_flow,
        'hamming': mad.compute_hamming,
    }

    metrics_with_inner_distance = {
        'approvalwise': mad.compute_approvalwise,
        'coapproval_frequency': mad.compute_coapproval_frequency_vectors,
        'pairwise': mad.compute_pairwise,
        'voterlikeness': mad.compute_voterlikeness,
        'candidatelikeness': mad.compute_candidatelikeness,
    }

    if main_distance in metrics_without_params:
        return metrics_without_params.get(main_distance)(election_1, election_2)

    elif main_distance in metrics_with_inner_distance:
        return metrics_with_inner_distance.get(main_distance)(election_1, election_2,
                                                              inner_distance)


def get_ordinal_distance(election_1: OrdinalElection, election_2: OrdinalElection,
                         distance_id: str = None) -> float or (float, list):
    """ Return: distance between ordinal elections, (if applicable) optimal matching """

    inner_distance, main_distance = extract_distance_id(distance_id)

    metrics_without_params = {
        'discrete': mod.compute_voter_subelection,
        'voter_subelection': mod.compute_voter_subelection,
        'candidate_subelection': mod.compute_candidate_subelection,
        'spearman': mod.compute_spearman_distance,
        'swap_bf': mod.compute_swap_bf_distance,
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


def get_graph_distance(graph_1, graph_2, distance_id: str = None) -> float or (float, list):
    """ Return: distance between graphs, (if applicable) optimal matching """

    graph_simple_metrics = {'closeness_centrality': nx.closeness_centrality,
                            'degree_centrality': nx.degree_centrality,
                            'betweenness_centrality': nx.betweenness_centrality,
                            'eigenvector_centrality': nx.eigenvector_centrality,
                            }

    graph_advanced_metrics = {
        'graph_edit_distance': mgd.compute_graph_edit_distance,
        'graph_histogram': mgd.compute_graph_histogram,
    }

    if distance_id in graph_simple_metrics:
        return mgd.compute_graph_simple_metrics(graph_1, graph_2,
                                                graph_simple_metrics[distance_id])

    if distance_id in graph_advanced_metrics:
        return graph_advanced_metrics.get(distance_id)(graph_1, graph_2)


def extract_distance_id(distance_id: str) -> (Callable, str):
    if '-' in distance_id:
        inner_distance, main_distance = distance_id.split('-')
        inner_distance = map_str_to_func(inner_distance)
    else:
        main_distance = distance_id
        inner_distance = None
    return inner_distance, main_distance


def run_single_thread(experiment: Experiment, thread_ids: list,
                      distances: dict, times: dict, matchings: dict,
                      printing: bool) -> None:
    """ Single thread for computing distances """
    for election_id_1, election_id_2 in thread_ids:
        if printing:
            print(election_id_1, election_id_2)
        start_time = time()
        distance = get_distance(experiment.elections[election_id_1],
                                experiment.elections[election_id_2],
                                distance_id=experiment.distance_id)
        if type(distance) is tuple:
            distance, matching = distance
            matching = np.array(matching)
            matchings[election_id_1][election_id_2] = matching
            matchings[election_id_2][election_id_1] = np.argsort(matching)
        distances[election_id_1][election_id_2] = distance
        distances[election_id_2][election_id_1] = distances[election_id_1][election_id_2]
        times[election_id_1][election_id_2] = time() - start_time
        times[election_id_2][election_id_1] = times[election_id_1][election_id_2]

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
