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


def convert_xd_to_2d(experiment, num_iterations=10000, metric="positionwise", random=True):
    model = obj.Model_xd(experiment, metric=metric)
    X = np.zeros((model.num_points, model.num_points))

    if random:
        perm = np.random.permutation(model.num_points)
    else:
        perm = [i for i in range(model.num_points)]

    rev_perm = [0 for _ in range(model.num_points)]
    for i in range(model.num_points):
        rev_perm[perm[i]] = int(i)

    for i in range(model.num_points):
        for j in range(i + 1, model.num_points):
            model.distances[j][i] = model.distances[i][j]

    for i in range(model.num_points):
        for j in range(i + 1, model.num_points):
            if model.distances[perm[i]][perm[j]] == 0:
                model.distances[perm[i]][perm[j]] = 0.0001
            X[i][j] = 1. / model.distances[perm[i]][perm[j]]
            X[j][i] = X[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    print("start spring_layout")
    my_pos = nx.spring_layout(G, iterations=num_iterations, dim=2)

    file_name = os.path.join(os.getcwd(), "experiments", experiment, "results", "points", metric + "_2d.csv")
    with open(file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "x", "y"])

        for i in range(model.num_points):
            x = round(my_pos[rev_perm[i]][0], 5)
            y = round(my_pos[rev_perm[i]][1], 5)
            writer.writerow([i, x, y])


def convert_xd_to_3d(experiment, num_iterations=1000, metric="positionwise"):
    model = obj.Model_xd(experiment, metric=metric)
    X = np.zeros((model.num_points, model.num_points))
    perm = np.random.permutation(model.num_points)
    """
    for i in range(0,60):
        perm[i] = i+30
    for i in range(60,90):
        perm[i] = i+180
    for i in range(90,210):
        perm[i] = i
    for i in range(210,390):
        perm[i] = i+210
    for i in range(390,550):
        perm[i] = i+250
    for i in range(550,640):
        perm[i] = i-220
    for i in range(640,670):
        perm[i] = i-340
    for i in range(670,700):
        perm[i] = i-670
    for i in range(700,730):
        perm[i] = i-490
    for i in range(730,760):
        perm[i] = i-460
    for i in range(760,800):
        perm[i] = i-160
    """
    rev_perm = [0 for _ in range(model.num_points)]
    for i in range(model.num_points):
        rev_perm[perm[i]] = int(i)
    for i in range(model.num_points):
        for j in range(i + 1, model.num_points):
            model.distances[j][i] = model.distances[i][j]

    for i in range(model.num_points):
        for j in range(i + 1, model.num_points):
            if model.distances[perm[i]][perm[j]] == 0:
                model.distances[perm[i]][perm[j]] = 0.01
            X[i][j] = 1. / model.distances[perm[i]][perm[j]]
            X[j][i] = X[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    my_pos = nx.spring_layout(G, iterations=num_iterations, dim=3)

    file_name = os.path.join(os.getcwd(), "experiments", experiment, "results", "points", metric + "_3d.csv")
    with open(file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "x", "y", "z"])

        for i in range(model.num_points):
            x = round(my_pos[rev_perm[i]][0], 5)
            y = round(my_pos[rev_perm[i]][1], 5)
            z = round(my_pos[rev_perm[i]][2], 5)
            writer.writerow([i, x, y, z])

