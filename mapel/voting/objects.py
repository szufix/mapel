#!/usr/bin/env python

import math
import os


class Model_xd:

    def __init__(self, name):
        self.name = name
        self.num_points, self.num_distances, self.distances = self.import_distances(name)
        self.num_families, self.family_name, self.family_special, self.family_size, \
            self.labels, self.colors, self.alphas = self.import_controllers(name)

    @staticmethod
    def import_distances(name):

        file_name = "results/distances/" + str(name) + ".txt"
        file_ = open(file_name, 'r')
        num_points = int(file_.readline())
        num_families = int(file_.readline())
        num_distances = int(file_.readline())

        hist_data = [[0 for _ in range(num_points)] for _ in range(num_points)]

        for a in range(num_points):
            for b in range(a + 1, num_points):
                line = file_.readline()
                line = line.split(' ')
                hist_data[a][b] = float(line[2])

        return num_points, num_distances, hist_data

    @staticmethod
    def import_controllers(name):

        file_name = "controllers/models/" + name + ".txt"
        file_ = open(file_name, 'r')
        num_voters = int(file_.readline())
        num_candidates = int(file_.readline())
        num_families = int(file_.readline())
        family_name = [0 for _ in range(num_families)]
        family_special = [0 for _ in range(num_families)]
        family_size = [0 for _ in range(num_families)]
        labels = [0 for _ in range(num_families)]
        colors = [0 for _ in range(num_families)]
        alphas = [0 for _ in range(num_families)]

        for i in range(num_families):
            line = file_.readline().rstrip("\n").split(',')
            family_name[i] = str(line[1])
            family_special[i] = float(line[2])
            family_size[i] = int(line[0])
            labels[i] = str(line[5])
            colors[i] = str(line[3].replace(" ", ""))
            alphas[i] = float(line[4])

        file_.close()
        return num_families, family_name, family_special, family_size, labels, colors, alphas


class Model_2d:

    def __init__(self, name, num_winners=0, shades=False):

        self.name = name
        self.num_points, self.points, = self.import_points(name)
        self.num_families, self.family_name, self.family_special, self.family_size, \
            self.labels, self.colors, self.alphas = self.import_controllers(name)
        self.points_by_families = self.compute_points_by_families()

        if num_winners > 0:
            self.winners = self.import_winners(name, num_winners)
        else:
            self.winners = []

        if shades:
            print("shades")
        else:
            self.shades = [1. for _ in range(self.num_points)]

    @staticmethod
    def import_points(name):

        file_name = str(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
        file_name += "/results/points/" + str(name) + ".txt"
        file_ = open(file_name, 'r')
        num_winners = int(file_.readline())
        points = [[0, 0] for _ in range(num_winners)]

        for i in range(num_winners):
            line = file_.readline().replace("\n", '').split(',')
            points[i] = [float(line[0]), float(line[1])]

        file_.close()
        return len(points), points

    @staticmethod
    def import_controllers(name):

        file_name = str(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
        file_name += "controllers/models/" + name + ".txt"
        file_ = open(file_name, 'r')
        num_voters = int(file_.readline())
        num_candidates = int(file_.readline())
        num_families = int(file_.readline())
        family_name = [0 for _ in range(num_families)]
        family_special = [0 for _ in range(num_families)]
        family_size = [0 for _ in range(num_families)]
        labels = [0 for _ in range(num_families)]
        colors = [0 for _ in range(num_families)]
        alphas = [0 for _ in range(num_families)]

        for i in range(num_families):
            line = file_.readline().rstrip("\n").split(',')
            family_name[i] = str(line[1])
            family_special[i] = float(line[2])
            family_size[i] = int(line[0])
            labels[i] = str(line[5])
            colors[i] = str(line[3].replace(" ", ""))
            alphas[i] = float(line[4])

        file_.close()
        return num_families, family_name, family_special, family_size, labels, colors, alphas

    @staticmethod
    def import_winners(name, first_winners):

        file_name = str(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
        file_name += "results/winners/" + str(name) + "_cc.txt"
        file_ = open(file_name, 'r')
        number_of_elections = int(file_.readline())
        real_num_winners = int(file_.readline())
        real_num_winners = first_winners
        real_winners = []
        time = float(file_.readline())

        for w in range(real_num_winners):
            real_winners.append(int(file_.readline()))

        return real_winners

    #@staticmethod
    def compute_points_by_families(self):

        points_by_families = [[[] for _ in range(2)] for _ in range(self.num_points)]
        ctr = 0

        for i in range(self.num_families):
            for j in range(self.family_size[i]):
                points_by_families[i][0].append(self.points[ctr][0])
                points_by_families[i][1].append(self.points[ctr][1])
                ctr += 1

        return points_by_families

    #@staticmethod
    def get_distance(self, i, j):

        distance = 0.

        for d in range(2):
            distance += (self.points[i][d] - self.points[j][d]) ** 2

        return math.sqrt(distance)
