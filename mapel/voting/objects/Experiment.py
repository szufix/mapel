#!/usr/bin/env python

import csv
import math
import os
import numpy as np


from .Election import Election
from .Family import Family


class Experiment:
    """Abstract set of elections."""

    def __init__(self, experiment_id, ignore=None, raw=False, distance_name='emd-positionwise'):

        self.experiment_id = experiment_id

        self.distance_name = distance_name

        self.families = self.import_controllers(ignore=ignore)

        if not raw:
            self.elections = self.add_elections_to_experiment()

    def add_elections_to_experiment(self):
        """ Import elections from a file """

        elections = {}

        for family_id in self.families:
            for j in range(self.families[family_id].size):
                election_id = family_id + '_' + str(j)
                election = Election(self.experiment_id, election_id)
                elections[election_id] = election

        return elections

    def import_controllers(self, ignore=None):
        """ Import controllers from a file """

        families = {}

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, 'map.csv')
        file_ = open(path, 'r')

        header = [h.strip() for h in file_.readline().split(',')]
        reader = csv.DictReader(file_, fieldnames=header)

        starting_from = 0
        for row in reader:

            election_model = None
            color = None
            label = None
            param_1 = None
            param_2 = None
            alpha = None
            size = None
            marker = None
            num_candidates = None
            num_voters = None

            if 'election_model' in row.keys():
                election_model = str(row['election_model']).strip()

            if 'color' in row.keys():
                color = str(row['color']).strip()

            if 'label' in row.keys():
                label = str(row['label'])

            if 'param_1' in row.keys():
                param_1 = float(row['param_1'])

            if 'param_2' in row.keys():
                param_2 = float(row['param_2'])

            if 'alpha' in row.keys():
                alpha = float(row['alpha'])

            if 'family_size' in row.keys():
                size = int(row['family_size'])

            if 'marker' in row.keys():
                marker = str(row['marker']).strip()

            if 'num_candidates' in row.keys():
                num_candidates = int(row['num_candidates'])

            if 'num_voters' in row.keys():
                num_voters = int(row['num_voters'])

            show = True
            if row['show'].strip() != 't':
                show = False

            family_id = election_model + '_' + str(num_candidates) + '_' + str(num_voters)
            if election_model in {'urn_model', 'norm-mallows', 'mallows','norm-mallows_matrix'} and param_1 != 0:
                family_id += '_' + str(float(param_1))
            if election_model in {'norm-mallows', 'mallows'} and param_2 != 0:
                family_id += '__' + str(float(param_2))

            # families.append(Family(election_model=election_model, param_1=param_1, param_2=param_2, label=label,
            #                        color=color, alpha=alpha, show=show, size=size, marker=marker,
            #                        starting_from=starting_from,
            #                        num_candidates=num_candidates, num_voters=num_voters))
            families[family_id] = Family(election_model=election_model, family_id=family_id,
                                              param_1=param_1, param_2=param_2, label=label,
                                   color=color, alpha=alpha, show=show, size=size, marker=marker,
                                   starting_from=starting_from,
                                   num_candidates=num_candidates, num_voters=num_voters)
            starting_from += size

        self.num_families = len(families)
        self.num_elections = sum([families[family_id].size for family_id in families])
        self.main_order = [i for i in range(self.num_elections)]

        if ignore is None:
            ignore = []

        ctr = 0
        for family_id in families:
            resize = 0
            for j in range(families[family_id].size):
                if self.main_order[ctr] >= self.num_elections or self.main_order[ctr] in ignore:
                    resize += 1
                ctr += 1
            families[family_id].size -= resize

        file_.close()
        return families

    # def import_order(self, main_order_name):
    #     """Import precomputed order of all the elections from a file."""
    #
    #     if main_order_name == 'default':
    #         main_order = [i for i in range(self.num_elections)]
    #
    #     else:
    #         file_name = os.path.join(os.getcwd(), "experiments", self.experiment_id, "results", "orders", main_order_name + ".txt")
    #         file_ = open(file_name, 'r')
    #         file_.readline()  # skip this line
    #         all_elections = int(file_.readline())
    #         file_.readline()  # skip this line
    #         main_order = []
    #
    #         for w in range(all_elections):
    #             main_order.append(int(file_.readline()))
    #
    #     return main_order


class Experiment_xD(Experiment):
    """ Multi-dimensional map of elections """

    def __init__(self, experiment_id, distance_name='positionwise', raw=False, self_distances=False):

        Experiment.__init__(self, experiment_id, distance_name='emd-positionwise', raw=raw)

        #self.num_points, self.num_distances, self.distances = self.import_distances(experiment_id)
        self.num_distances, self.distances, self.std = self.import_distances(experiment_id,
                                                                   distance_name=distance_name, self_distances=self_distances)

    #@staticmethod
    def import_distances(self, experiment_id, distance_name=None, self_distances=False):
        """ Import precomputed distances between each pair of elections from a file """

        file_name = str(distance_name) + '.csv'
        path = os.path.join(os.getcwd(), "experiments", experiment_id, "distances", file_name)
        num_points = self.num_elections
        num_distances = int(num_points*(num_points-1)/2)

        hist_data = {}
        std = [[0. for _ in range(num_points)] for _ in range(num_points)]

        for family_id in self.families:
            for j in range(self.families[family_id].size):
                election_id = family_id + '_' + str(j)
                hist_data[election_id] = {}

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')

            for row in reader:
                election_id_1 = row['election_id_1']
                election_id_2 = row['election_id_2']
                hist_data[election_id_1][election_id_2] = float(row['distance'])
                hist_data[election_id_2][election_id_1] = hist_data[election_id_1][election_id_2]

        # todo: add self-distances
        # for a in range(num_points):
        #     limit = a+1
        #     if self_distances:
        #         limit = a
        #     for b in range(limit, num_points):
        #         line = file_.readline()
        #         line = line.split(' ')
        #         hist_data[a][b] = float(line[2])
        #
        #         # tmp correction for discrete distance
        #         if distance_name == 'discrete':
        #             hist_data[a][b] = self.families[0].size - hist_data[a][b]   # todo: correct this
        #
        #
        #         hist_data[b][a] = hist_data[a][b]
        #
        #         if distance_name == 'voter_subelection':
        #             std[a][b] = float(line[3])
        #             std[b][a] = std[a][b]

        return num_distances, hist_data, std


class Experiment_2D(Experiment_xD):
    """ Two-dimensional model of elections """

    def __init__(self, experiment_id, distance_name="emd-positionwise", ignore=None, attraction_factor=1):

        Experiment_xD.__init__(self, experiment_id, distance_name=distance_name)

        self.attraction_factor = float(attraction_factor)

        self.num_points, self.points, = self.import_points(ignore=ignore)
        self.points_by_families = self.compute_points_by_families()

    def import_points(self, ignore=None):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        if ignore is None:
            ignore = []

        points = {}
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", self.distance_name + "_2d_a" + str(float(self.attraction_factor)) + ".csv")

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            ctr = 0
            print(path)
            for row in reader:
                if self.main_order[ctr] < self.num_elections and self.main_order[ctr] not in ignore:
                    points[row['election_id']] = [float(row['x']), float(row['y'])]
                ctr += 1

        return len(points), points

    def compute_points_by_families(self):
        """ Group all points by their families """

        points_by_families = {}

        for family_id in self.families:
            points_by_families[family_id] = [[] for _ in range(2)]

            for i in range(self.families[family_id].size):
                election_id = family_id + '_' + str(i)
                points_by_families[family_id][0].append(self.points[election_id][0])
                points_by_families[family_id][1].append(self.points[election_id][1])

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
            self.points[i][0], self.points[i][1] = self.rotate_point(0.5, 0.5, angle, self.points[i][0], self.points[i][1])

        self.points_by_families = self.compute_points_by_families()

    def reverse(self):
        """ Reverse all the points"""

        for i in range(self.num_points):
            self.points[i][0] = self.points[i][0]
            self.points[i][1] = -self.points[i][1]

        self.points_by_families = self.compute_points_by_families()

    def update(self):
        """ Save current coordinates of all the points to the original file"""

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "points", self.distance_name + "_2d_a" + str(self.attraction_factor) + ".csv")

        with open(path, 'w', newline='') as csvfile:

            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["id", "x", "y"])

            for i in range(self.num_points):
                x = round(self.points[i][0], 5)
                y = round(self.points[i][1], 5)
                writer.writerow([i, x, y])

    @staticmethod
    def rotate_point(cx, cy, angle, px, py):
        """ Rotate two-dimensional point by an angle """

        s = math.sin(angle)
        c = math.cos(angle)
        px -= cx
        py -= cy
        x_new = px * c - py * s
        y_new = px * s + py * c
        px = x_new + cx
        py = y_new + cy

        return px, py


class Experiment_3D(Experiment):
    """ Two-dimensional model of elections """

    def __init__(self, experiment_id, distance_name="emd-positionwise", ignore=None,
                 attraction_factor=1):

        Experiment.__init__(self, experiment_id, ignore=ignore, distance_name=distance_name)

        self.attraction_factor = int(attraction_factor)

        self.num_points, self.points, = self.import_points(ignore=ignore)
        self.points_by_families = self.compute_points_by_families()

    def import_points(self, ignore=None):
        """ Import from a file precomputed coordinates of all the points -- each point refer to one election """

        if ignore is None:
            ignore = []

        points = []
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "points", str(self.distance_name) + "_3d_a" + str(self.attraction_factor) + ".csv")

        with open(path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            ctr = 0
            print(path)
            for row in reader:
                if self.main_order[ctr] < self.num_elections and self.main_order[ctr] not in ignore:
                    points.append([float(row['x']), float(row['y']), float(row['z'])])
                ctr += 1

        return len(points), points

    def compute_points_by_families(self):
        """ Group all points by their families """

        points_by_families = [[[] for _ in range(3)] for _ in range(self.num_families)]
        ctr = 0

        for i in range(self.num_families):
            for j in range(self.families[i].size):
                points_by_families[i][0].append(self.points[ctr][0])
                points_by_families[i][1].append(self.points[ctr][1])
                points_by_families[i][2].append(self.points[ctr][2])
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
            self.points[i][0], self.points[i][1] = self.rotate_point(0.5, 0.5, angle, self.points[i][0], self.points[i][1])

        self.points_by_families = self.compute_points_by_families()

    def reverse(self, ):
        """ Reverse all the points"""

        for i in range(self.num_points):
            self.points[i][0] = self.points[i][0]
            self.points[i][1] = -self.points[i][1]

        self.points_by_families = self.compute_points_by_families()

    def update(self):
        """ Save current coordinates of all the points to the original file"""

        file_name = self.experiment_id + ".txt"
        path = os.path.join(os.getcwd(), "results", "points", file_name)
        file_ = open(path, 'w')
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
