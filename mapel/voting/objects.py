""" This module contains all the objects """

import os
import math
import csv


class Model:
    """ Abstract model of elections """

    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.num_voters, self.num_candidates, self.num_families, \
            self.families = self.import_controllers(exp_name)

    @staticmethod
    def import_controllers(exp_name):
        """ Import from a file all the controllers"""

        file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "controllers", "basic", "map.txt")
        file_ = open(file_name, 'r')
        num_voters = int(file_.readline())
        num_candidates = int(file_.readline())
        num_families = int(file_.readline())
        families = []

        for i in range(num_families):
            line = file_.readline().rstrip("\n").split(',')
            show = True
            if line[0][0] == "#":
                line[0] = line[0].replace("#", "")
                show = False

            families.append(Family(name=str(line[1]),
                                   special_1=str(line[2]),
                                   special_2=str(line[3]),
                                   size=int(line[0]), label=str(line[6]),
                                   color=str(line[4].replace(" ", "")), alpha=float(line[5]),
                                   show=show))

        file_.close()
        return num_voters, num_candidates, num_families, families


class Model_xd(Model):
    """ Multi-dimensional model of elections """

    def __init__(self, exp_name, metric):
        Model.__init__(self, exp_name)
        self.num_points, self.num_distances, self.distances = self.import_distances(exp_name, metric)

    @staticmethod
    def import_distances(exp_name, metric):
        """Import from a file precomputed distances between each pair of elections  """

        file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "results", "distances", str(metric) + ".txt")
        file_ = open(file_name, 'r')
        num_points = int(file_.readline())
        file_.readline()  # skip this line
        num_distances = int(file_.readline())

        hist_data = [[0 for _ in range(num_points)] for _ in range(num_points)]

        for a in range(num_points):
            for b in range(a + 1, num_points):
                line = file_.readline()
                line = line.split(' ')
                hist_data[a][b] = float(line[2])

        return num_points, num_distances, hist_data


class Model_2d(Model):
    """ Two-dimensional model of elections """

    def __init__(self, exp_name, num_winners=0, num_elections="800", winners_order="positionwise_approx_cc", main_order="", metric="positionwise"):
        Model.__init__(self, exp_name)

        self.num_points, self.points, = self.import_points(exp_name, metric)
        self.points_by_families = self.compute_points_by_families()

        self.winners_order = self.import_order(exp_name, num_elections, winners_order)
        self.main_order = self.import_order(exp_name, num_elections, main_order)
        self.winners = self.winners_order[0:num_winners]

    @staticmethod
    def import_points(experiment, metric):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        points = []
        file_name = os.path.join(os.getcwd(), "experiments", experiment, "results", "points", metric + "_2d.csv")
        with open(file_name, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                points.append([float(row['x']), float(row['y'])])

        return len(points), points

    @staticmethod
    def import_order(exp_name, num_elections, order_name):
        """ Import from a file precomputed order of all the elections """

        if order_name == "":
            return [i for i in range(num_elections)]

        file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "results", "orders", str(order_name) + ".txt")
        file_ = open(file_name, 'r')
        file_.readline()  # skip this line
        file_.readline()  # skip this line
        file_.readline()  # skip this line
        order = []

        for w in range(num_elections):
            order.append(int(file_.readline()))

        return order

    def compute_points_by_families(self):
        """ Group all points by their families """

        points_by_families = [[[] for _ in range(2)] for _ in range(self.num_points)]
        ctr = 0

        for i in range(self.num_families):
            for j in range(self.families[i].size):
                points_by_families[i][0].append(self.points[ctr][0])
                points_by_families[i][1].append(self.points[ctr][1])
                ctr += 1

        return points_by_families

    def get_distance(self, i, j):
        """ Compute Euclidean distance in two-dimensional space"""

        distance = 0.
        for d in range(2):
            distance += (self.points[i][d] - self.points[j][d]) ** 2

        return math.sqrt(distance)

    def rotate(self, angle):
        """ Rotate all the points by a given angle """

        for i in range(self.num_points):
            self.points[i][0], self.points[i][1] = rotate_point(0.5, 0.5, angle, self.points[i][0], self.points[i][1])

        self.points_by_families = self.compute_points_by_families()

    def reverse(self, ):
        """ Reverse all the points"""

        for i in range(self.num_points):
            self.points[i][0] = self.points[i][0]
            self.points[i][1] = -self.points[i][1]

        self.points_by_families = self.compute_points_by_families()

    def update(self):
        """ Save current coordinates of all the points to the original file"""

        file_name = "results/points/" + str(self.name) + ".txt"
        file_ = open(file_name, 'w')
        file_.write(str(self.num_points) + "\n")

        for i in range(self.num_points):
            x = round(self.points[i][0], 5)
            y = round(self.points[i][1], 5)
            file_.write(str(x) + ', ' + str(y) + "\n")
        file_.close()

    @staticmethod
    def rotate_point(cx, cy, angle, px, py):
        """ Rotate two-dimensional point by angle """

        s = math.sin(angle)
        c = math.cos(angle)
        px -= cx
        py -= cy
        x_new = px * c - py * s
        y_new = px * s + py * c
        px = x_new + cx
        py = y_new + cy

        return px, py


class Family:
    """ Family of elections """

    def __init__(self, name="none", special_1=0., special_2=0., size=0, label="none", color="black", alpha=1., show=True):

        self.name = name
        self.special_1 = special_1
        self.special_2 = special_2
        self.size = size
        self.label = label
        self.color = color
        self.alpha = alpha
        self.show = show







