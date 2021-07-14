""" This module contains all the functions that user can use """

### COMMENT ON SERVER ###
#########################

import csv
import os

try:
    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
except:
    pass

import networkx as nx
import numpy as np

from .objects.Experiment import Experiment, Experiment_xD, Experiment_2D, Experiment_3D


def convert_xd_to_2d(experiment_id, num_iterations=1000, distance_name="emd-positionwise",
                     attraction_factor=1., algorithm='spring'):
    """ Convert multi-dimensional experiment to two-dimensional experiment """

    experiment = Experiment_xD(experiment_id, distance_name=distance_name)
    X = np.zeros((experiment.num_elections, experiment.num_elections))

    for i, election_1_id in enumerate(experiment.elections):
        for j, election_2_id in enumerate(experiment.elections):
            if i < j:
                if experiment.distances[election_1_id][election_2_id] == 0:
                    experiment.distances[election_1_id][election_2_id] = 0.01
                if algorithm == 'spring':
                    X[i][j] = 1. / experiment.distances[election_1_id][election_2_id]
                else:
                    X[i][j] = experiment.distances[election_1_id][election_2_id]
                X[i][j] = X[i][j] ** attraction_factor
                X[j][i] = X[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    print("start spring_layout")

    # ppp = {0: [0., 0.], 1: [0.01, 0.01], 2: [0., 0.01], 3: [0.01, 0.]}
    if algorithm == 'spring':
        my_pos = nx.spring_layout(G, iterations=num_iterations, dim=2)#, pos=ppp, fixed=[0,1,2,3])
    elif algorithm == 'mds':
        my_pos = MDS(n_components=2).fit_transform(X)
    elif algorithm == 'tsne':
        my_pos = TSNE(n_components=2).fit_transform(X)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id,
                         "coordinates", distance_name + "_2d_a" + str(float(attraction_factor)) + ".csv")

    with open(file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["election_id", "x", "y"])

        ctr = 0
        for election_model_id in experiment.families:
            for j in range(experiment.families[election_model_id].size):
                a = election_model_id + '_' + str(j)
                x = round(my_pos[ctr][0], 5)
                y = round(my_pos[ctr][1], 5)
                writer.writerow([a, x, y])
                ctr += 1

# def convert_xd_to_2d_old(experiment_id, num_iterations=1000, distance_name="emd-positionwise",
#                      random=True, attraction_factor=1.):
#     """ Convert multi-dimensional experiment to two-dimensional experiment """
#
#     experiment = Experiment_xd(experiment_id, distance_name=distance_name)
#     X = np.zeros((experiment.num_elections, experiment.num_elections))
#
#     if random:
#         perm = np.random.permutation(experiment.num_elections)
#     else:
#         perm = [i for i in range(experiment.num_elections)]
#
#     rev_perm = [0 for _ in range(experiment.num_elections)]
#     for i in range(experiment.num_elections):
#         rev_perm[perm[i]] = int(i)
#
#     for i in range(experiment.num_elections):
#         for j in range(i + 1, experiment.num_elections):
#             experiment.distances[j][i] = experiment.distances[i][j]
#
#     for i in range(experiment.num_elections):
#         for j in range(i + 1, experiment.num_elections):
#             if experiment.distances[perm[i]][perm[j]] == 0:
#                 experiment.distances[perm[i]][perm[j]] = 0.01
#             X[i][j] = 1. / experiment.distances[perm[i]][perm[j]]
#             # TMP ROCK IT
#             X[i][j] = X[i][j] ** attraction_factor
#             # END OF TMP
#             X[j][i] = X[i][j]
#
#
#     dt = [('weight', float)]
#     X = X.view(dt)
#     G = nx.from_numpy_matrix(X)
#
#     print("start spring_layout")
#
#     # ppp = {0: [0., 0.], 1: [0.01, 0.01], 2: [0., 0.01], 3: [0.01, 0.]}
#     my_pos = nx.spring_layout(G, iterations=num_iterations, dim=2)#, pos=ppp, fixed=[0,1,2,3])
#
#
#     file_name = os.path.join(os.getcwd(), "experiments", experiment_id,
#                          "coordinates", distance_name + "_2d_a" + str(attraction_factor) + ".csv")
#
#     with open(file_name, 'w', newline='') as csvfile:
#
#         writer = csv.writer(csvfile, delimiter=',')
#         writer.writerow(["election_id", "x", "y"])
#
#         ctr = 0
#         for family in experiment.families:
#             for j in range(family.size):
#                 a = family.election_model + '_' + str(j)
#                 x = round(my_pos[rev_perm[ctr]][0], 5)
#                 y = round(my_pos[rev_perm[ctr]][1], 5)
#                 writer.writerow([a, x, y])
#                 ctr += 1


def convert_xd_to_3d(experiment_id, num_iterations=1000, distance_name="emd-positionwise",
                                                                  attraction_factor=1.):
    """ Convert multi-dimensional experiment to three-dimensional experiment """

    experiment = Experiment_xD(experiment_id, distance_name=distance_name)
    X = np.zeros((experiment.num_elections, experiment.num_elections))
    perm = np.random.permutation(experiment.num_elections)

    rev_perm = [0 for _ in range(experiment.num_elections)]
    for i in range(experiment.num_elections):
        rev_perm[perm[i]] = int(i)
    for i in range(experiment.num_elections):
        for j in range(i + 1, experiment.num_elections):
            experiment.distances[j][i] = experiment.distances[i][j]

    for i in range(experiment.num_elections):
        for j in range(i + 1, experiment.num_elections):
            if experiment.distances[perm[i]][perm[j]] == 0:
                experiment.distances[perm[i]][perm[j]] = 0.01
            X[i][j] = 1. / experiment.distances[perm[i]][perm[j]]
            X[i][j] = X[i][j]**attraction_factor
            X[j][i] = X[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    my_pos = nx.spring_layout(G, iterations=num_iterations, dim=3)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id,
                             "points",  distance_name + "_3d_a" + str(attraction_factor) + ".csv")

    with open(file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "x", "y", "z"])

        for i in range(experiment.num_elections):
            x = round(my_pos[rev_perm[i]][0], 5)
            y = round(my_pos[rev_perm[i]][1], 5)
            z = round(my_pos[rev_perm[i]][2], 5)
            writer.writerow([i, x, y, z])


def embed(distances, attraction_factor=1, algorithm='spring', num_iterations=1000):
    num_elections = len(distances)

    X = np.zeros((num_elections, num_elections))

    for i, election_1_id in enumerate(distances):
        for j, election_2_id in enumerate(distances):
            if i < j:
                if distances[election_1_id][election_2_id] == 0:
                    distances[election_1_id][election_2_id] = 0.01
                if algorithm == 'spring':
                    X[i][j] = 1. / distances[election_1_id][election_2_id]
                else:
                    X[i][j] = distances[election_1_id][election_2_id]
                X[i][j] = X[i][j] ** attraction_factor
                X[j][i] = X[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    if algorithm == 'spring':
        my_pos = nx.spring_layout(G, iterations=num_iterations, dim=2)
    elif algorithm == 'mds':
        my_pos = MDS(n_components=2).fit_transform(X)
    elif algorithm == 'tsne':
        my_pos = TSNE(n_components=2).fit_transform(X)

    points = {}
    for i, election_id in enumerate(distances):
        points[election_id] = [my_pos[i][0], my_pos[i][1]]

    return points
