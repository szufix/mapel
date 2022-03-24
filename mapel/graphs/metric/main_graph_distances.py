import networkx as nx
import numpy as np

from mapel.main._inner_distances import map_str_to_func


# MAIN GRAPH DISTANCES
def compute_graph_simple_metrics(graph_1, graph_2, graph_simple_metric, inner_distance='l1'):
    g1 = graph_simple_metric(graph_1)
    g2 = graph_simple_metric(graph_2)
    v1 = sorted(list(g1.values()))
    v2 = sorted(list(g2.values()))
    inner_distance = map_str_to_func(inner_distance)
    return inner_distance(v1, v2), None


def compute_graph_edit_distance(graph_1, graph_2):
    gen = nx.optimize_graph_edit_distance(graph_1, graph_2)
    min_dist = np.infty
    for g in gen:
        min_dist = g
    return min_dist, None


def compute_graph_histogram(graph_1, graph_2, inner_distance='emd'):
    deg_1 = [x for _, x in graph_1.degree]
    vec_1 = np.zeros([len(deg_1)])
    for value in deg_1:
        vec_1[value] += 1

    deg_2 = [x for _, x in graph_2.degree]
    vec_2 = np.zeros([len(deg_2)])
    for value in deg_2:
        vec_2[value] += 1

    # maybe we can normalize the vectors

    inner_distance = map_str_to_func(inner_distance)
    return inner_distance(vec_1, vec_2), None
