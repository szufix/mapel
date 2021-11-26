#!/usr/bin/env python

import networkx as nx

from mapel.elections.metrics import main_graph_distances as mgd
from mapel.elections.objects.Election import Election
from mapel.graphs.objects.Graph import Graph


def get_distance(election_1: Election, election_2: Election,
                 distance_id: str = None) -> float or (float, list):
    """ Return: distance between instances, (if applicable) optimal matching """

    if type(election_1) is Graph and type(election_2) is Graph:
        return get_graph_distance(election_1.graph, election_2.graph, distance_id=distance_id)
    else:
        print('No such instance!')

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


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 13.10.2021 #
# # # # # # # # # # # # # # # #
