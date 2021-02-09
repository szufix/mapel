""" This module contains all the functions that user can use """

### COMMENT ON SERVER ###
import matplotlib.pyplot as plt
import matplotlib.lines as lines
#########################

from . import objects as obj
import numpy as np
import os
from PIL import Image
from shutil import copyfile
import scipy.stats as stats
import networkx as nx
import csv


def convert_xd_to_2d(experiment, num_iterations=1000, metric="positionwise", random=True, magic=1.):
    model = obj.Model_xd(experiment, distance_name=metric)
    X = np.zeros((model.num_elections, model.num_elections))

    if random:
        perm = np.random.permutation(model.num_elections)
    else:
        perm = [i for i in range(model.num_elections)]

    rev_perm = [0 for _ in range(model.num_elections)]
    for i in range(model.num_elections):
        rev_perm[perm[i]] = int(i)

    for i in range(model.num_elections):
        for j in range(i + 1, model.num_elections):
            model.distances[j][i] = model.distances[i][j]

    for i in range(model.num_elections):
        for j in range(i + 1, model.num_elections):
            if model.distances[perm[i]][perm[j]] == 0:
                model.distances[perm[i]][perm[j]] = 0.01
            X[i][j] = 1. / model.distances[perm[i]][perm[j]]
            # TMP ROCK IT
            X[i][j] = X[i][j]**magic
            # END OF TMP
            X[j][i] = X[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    print("start spring_layout")

    my_pos = nx.spring_layout(G, iterations=num_iterations, dim=2)
    if magic == 1:
        file_name = os.path.join(os.getcwd(), "experiments", experiment, "controllers",
                             "points", metric + "_2d.csv")
    else:
        file_name = os.path.join(os.getcwd(), "experiments", experiment, "controllers",
                             "points", metric + "_2d_p" + str(magic) + ".csv")

    with open(file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "x", "y"])

        for i in range(model.num_elections):
            x = round(my_pos[rev_perm[i]][0], 5)
            y = round(my_pos[rev_perm[i]][1], 5)
            writer.writerow([i, x, y])


def convert_xd_to_3d(experiment, num_iterations=1000, metric="positionwise", magic=1.):
    model = obj.Model_xd(experiment, distance_name=metric)
    X = np.zeros((model.num_elections, model.num_elections))
    perm = np.random.permutation(model.num_elections)

    rev_perm = [0 for _ in range(model.num_elections)]
    for i in range(model.num_elections):
        rev_perm[perm[i]] = int(i)
    for i in range(model.num_elections):
        for j in range(i + 1, model.num_elections):
            model.distances[j][i] = model.distances[i][j]

    for i in range(model.num_elections):
        for j in range(i + 1, model.num_elections):
            if model.distances[perm[i]][perm[j]] == 0:
                model.distances[perm[i]][perm[j]] = 0.01
            X[i][j] = 1. / model.distances[perm[i]][perm[j]]
            X[i][j] = X[i][j]**magic
            X[j][i] = X[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    my_pos = nx.spring_layout(G, iterations=num_iterations, dim=3)

    if magic == 1:
        file_name = os.path.join(os.getcwd(), "experiments", experiment, "controllers",
                             "points", metric + "_3d.csv")
    else:
        file_name = os.path.join(os.getcwd(), "experiments", experiment, "controllers",
                             "points", metric + "_3d_p" + str(magic) + ".csv")

    with open(file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "x", "y", "z"])

        for i in range(model.num_elections):
            x = round(my_pos[rev_perm[i]][0], 5)
            y = round(my_pos[rev_perm[i]][1], 5)
            z = round(my_pos[rev_perm[i]][2], 5)
            writer.writerow([i, x, y, z])

