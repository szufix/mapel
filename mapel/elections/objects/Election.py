#!/usr/bin/env python

import copy
import csv
import os
import itertools
from abc import ABCMeta, abstractmethod
import math

import numpy as np
import matplotlib.pyplot as plt

from mapel.main._glossary import *
from mapel.main.objects.Instance import Instance
from mapel.elections.other.winners import compute_sntv_winners, compute_borda_winners, \
    compute_stv_winners
from mapel.elections.other.winners2 import generate_winners


from sklearn.manifold import MDS


class Election(Instance):

    def __init__(self, experiment_id, election_id, votes=None, alpha=None, model_id=None,
                 ballot: str = 'ordinal', num_voters: int = None, num_candidates: int = None,
                 label = None, fast_import=False):

        super().__init__(experiment_id, election_id, model_id=model_id, alpha=alpha)

        self.election_id = election_id
        self.ballot = ballot

        self.label = label
        self.store = True

        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.winners = None
        self.alternative_winners = {}

        self.fake = model_id in LIST_OF_FAKE_MODELS

        self.votes = votes
        self.model_id = model_id
        self.potes = None

        self.distances = None
        if not fast_import:
            try:
                self._import_distances()
            except:
                pass

        self.coordinates = None
        try:
            self._import_coordinates()
        except:
            pass


    def import_matrix(self) -> np.ndarray:

        file_name = f'{self.election_id}.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, 'matrices', file_name)
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for i, row in enumerate(reader):
                for j, candidate_id in enumerate(row):
                    matrix[i][j] = row[candidate_id]
        return matrix

    def compute_potes(self):
        """ Convert votes to positional votes """
        if self.potes is None:
            self.potes = np.array([[vote.index(i) for i, _ in enumerate(vote)]
                                   for vote in self.votes])

    def vector_to_interval(self, vector, precision=None) -> list:
        # discreet version for now
        interval = []
        w = int(precision / self.num_candidates)
        for i in range(self.num_candidates):
            for j in range(w):
                interval.append(vector[i] / w)
        return interval

    def compute_alternative_winners(self, method=None, party_id=None, num_winners=None):

        election_without_party_id = remove_candidate_from_election(copy.deepcopy(self),
                                                                   party_id, num_winners)
        election_without_party_id = map_the_votes(election_without_party_id, party_id, num_winners)

        if method == 'sntv':
            winners_without_party_id = compute_sntv_winners(election=election_without_party_id,
                                                            num_winners=num_winners)
        elif method == 'borda':
            winners_without_party_id = compute_borda_winners(election=election_without_party_id,
                                                             num_winners=num_winners)
        elif method == 'stv':
            winners_without_party_id = compute_stv_winners(election=election_without_party_id,
                                                           num_winners=num_winners)
        elif method in {'approx_cc', 'approx_hb', 'approx_pav'}:
            winners_without_party_id = generate_winners(election=election_without_party_id,
                                                        num_winners=num_winners, method=method)
        else:
            winners_without_party_id = []

        winners_without_party_id = unmap_the_winners(winners_without_party_id, party_id, num_winners)

        self.alternative_winners[party_id] = winners_without_party_id

    def print_map(self, show=True, radius=100, name=None, alpha=0.1, s=30, circles=False):
        plt.figure(figsize=(6.4, 6.4))

        X = []
        Y = []
        for elem in self.coordinates:
            X.append(round(elem[0], 0))
            Y.append(round(elem[1], 0))

        plt.scatter(X, Y, color='blue', s=s, alpha=alpha)

        if circles:
            weighted_points = {}
            Xs = {}
            Ys = {}
            for elem in self.coordinates:
                elem[0] = round(elem[0], 0)
                elem[1] = round(elem[1], 0)
                str_elem = str(elem)
                if str_elem in weighted_points:
                    weighted_points[str_elem] += 1
                else:
                    weighted_points[str_elem] = 0
                    Xs[str_elem] = elem[0]
                    Ys[str_elem] = elem[1]

            for str_elem in weighted_points:
                if weighted_points[str_elem] > 10:
                    plt.scatter(Xs[str_elem], Ys[str_elem],
                                color='purple',
                                s=10 * weighted_points[str_elem],
                                alpha=0.2)

        plt.xlim([-radius, radius])
        plt.ylim([-radius, radius])
        plt.title(self.label, size=38)
        plt.axis('off')

        file_name = os.path.join(os.getcwd(), "images", name, f'{self.label}.png')
        plt.savefig(file_name, bbox_inches='tight', dpi=100)
        if show:
            plt.show()
        else:
            plt.clf()

    def online_mini_map(self):

        self.compute_potes()

        distances = np.zeros([len(self.potes), len(self.potes)])
        for v1 in range(len(self.potes)):
            for v2 in range(len(self.potes)):
                swap_distance = 0
                for i, j in itertools.combinations(self.potes[0], 2):
                    if (self.potes[v1][i] > self.potes[v1][j] and
                        self.potes[v2][i] < self.potes[v2][j]) or \
                            (self.potes[v1][i] < self.potes[v1][j] and
                             self.potes[v2][i] > self.potes[v2][j]):
                        swap_distance += 1
                distances[v1][v2] = swap_distance

        # my_pos = KamadaKawai().embed(
        #     distances=distances,
        # )
        my_pos = MDS(n_components=2, dissimilarity='precomputed').fit_transform(distances)
        X = []
        Y = []
        for elem in my_pos:
            X.append(elem[0])
            Y.append(elem[1])
        plt.scatter(X, Y, color='blue', s=12, alpha=0.3)
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])
        plt.title(self.label, size=26)
        plt.axis('off')

        file_name = os.path.join(os.getcwd(), "images", "mini_maps", f'{self.label}.png')
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        # plt.clf()
        # plt.savefig(file_name, bbox_inches=bbox_inches, dpi=250)
        plt.show()

    @abstractmethod
    def compute_distances(self):
        pass

    def embed(self, algorithm='MDS'):

        # self.coordinates = KamadaKawai().embed(
        #     distances=self.distances,
        # )
        self.coordinates = MDS(n_components=2, dissimilarity='precomputed').fit_transform(
            self.distances)
        # ADJUST

        # find max dist
        if (not 'identity' in self.model_id.lower()) and (not 'approval_id' in self.model_id.lower()):
            dist = np.zeros([len(self.coordinates), len(self.coordinates)])
            for pos_1, pos_2 in itertools.combinations([i for i in range(len(self.coordinates))],
                                                       2):
                # print(pos_1, pos_2)
                dist[pos_1][pos_2] = np.linalg.norm(
                    self.coordinates[pos_1] - self.coordinates[pos_2])

            result = np.where(dist == np.amax(dist))
            id_1 = result[0][0]
            id_2 = result[1][0]

            # rotate
            left = id_1
            right = id_2

            try:
                d_x = self.coordinates[right][0] - self.coordinates[left][0]
                d_y = self.coordinates[right][1] - self.coordinates[left][1]
                alpha = math.atan(d_x / d_y)
                self.rotate(alpha - math.pi / 2.)
                self.rotate(math.pi / 4.)
            except Exception:
                pass

        if self.store:
            file_name = f'{self.election_id}.csv'
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "coordinates",
                                file_name)

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(["vote_id", "x", "y"])

                for vote_id in range(self.num_voters):
                    x = str(self.coordinates[vote_id][0])
                    y = str(self.coordinates[vote_id][1])
                    writer.writerow([vote_id, x, y])

        return self.coordinates

    def _store_distances(self):
        file_name = f'{self.election_id}.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances",
                            file_name)

        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(
                ["v1", "v2", "distance"])

            for v1 in range(self.num_voters):
                for v2 in range(self.num_voters):
                    distance = str(self.distances[v1][v2])
                    writer.writerow([v1, v2, distance])

    def _import_distances(self):

        file_name = f'{self.election_id}.csv'
        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'distances', file_name)

        distances = np.zeros([self.num_voters, self.num_voters])
        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                distances[int(row['v1'])][int(row['v2'])] = float(row['distance'])
                distances[int(row['v2'])][int(row['v1'])] = float(row['distance'])

        self.distances = distances

    def _import_coordinates(self):

        file_name = f'{self.election_id}.csv'
        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'coordinates',
                            file_name)

        coordinates_dict = {}
        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                coordinates_dict[row['vote_id']] = [float(row['x']), float(row['y'])]

        coordinates = []
        for i in range(len(coordinates_dict)):
            coordinates.append(coordinates_dict[str(i)])

        self.coordinates = coordinates

    @staticmethod
    def rotate_point(cx, cy, angle, px, py) -> (float, float):
        """ Rotate two-dimensional point by an angle """
        s, c = math.sin(angle), math.cos(angle)
        px -= cx
        py -= cy
        x_new, y_new = px * c - py * s, px * s + py * c
        px, py = x_new + cx, y_new + cy

        return px, py

    def rotate(self, angle) -> None:
        """ Rotate all the points by a given angle """
        for instance_id in range(len(self.coordinates)):
            self.coordinates[instance_id][0], self.coordinates[instance_id][1] = \
                self.rotate_point(0.5, 0.5, angle, self.coordinates[instance_id][0],
                                  self.coordinates[instance_id][1])


def map_the_votes(election, party_id, party_size) -> Election:
    new_votes = [[] for _ in range(election.num_voters)]
    for i in range(election.num_voters):
        for j in range(election.num_candidates):
            if election.votes[i][j] >= party_id * party_size:
                new_votes[i].append(election.votes[i][j]-party_size)
            else:
                new_votes[i].append(election.votes[i][j])
    election.votes = new_votes
    return election


def unmap_the_winners(winners, party_id, party_size):
    new_winners = []
    for j in range(len(winners)):
        if winners[j] >= party_id * party_size:
            new_winners.append(winners[j]+party_size)
        else:
            new_winners.append(winners[j])
    return new_winners


def remove_candidate_from_election(election, party_id, party_size) -> Election:
    for vote in election.votes:
        for i in range(party_size):
            _id = party_id*party_size + i
            vote.remove(_id)
    election.num_candidates -= party_size
    return election
