#!/usr/bin/env python

import copy
import csv
import itertools
import logging
import math
import os
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from sklearn.manifold import MDS

from mapel.elections.features_ import get_local_feature
from mapel.elections.other.winners import compute_sntv_winners, compute_borda_winners, \
    compute_stv_winners
from mapel.core.glossary import *
from mapel.core.inner_distances import l2
from mapel.core.objects.Instance import Instance

import mapel.elections.persistence.election_exports as exports
import mapel.elections.persistence.election_imports as imports

OBJECT_TYPES = ['vote', 'candidate']


class Election(Instance):

    def __init__(self,
                 experiment_id=None,
                 election_id=None,
                 culture_id=None,
                 votes=None,
                 ballot_type: str = 'ordinal',
                 num_voters: int = None,
                 num_candidates: int = None,
                 label=None,
                 fast_import=False,
                 is_shifted=False,
                 is_imported=False,
                 **kwargs):

        super().__init__(experiment_id=experiment_id,
                         instance_id=election_id,
                         culture_id=culture_id,
                         **kwargs)

        self.election_id = election_id
        self.ballot_type = ballot_type
        self.label = label
        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.votes = votes
        self.is_exported = True
        self.winners = None
        self.alternative_winners = {}
        self.fake = culture_id in LIST_OF_FAKE_MODELS
        self.potes = None
        self.features = {}
        self.object_type = 'vote'
        self.points = {}
        self.is_shifted = is_shifted
        self.is_imported = is_imported
        self.fast_import = fast_import

        self.import_distances()
        self.import_coordinates()

    def import_distances(self):
        self.distances = {}
        if not self.fast_import:
            for object_type in OBJECT_TYPES:
                try:
                    self.distances[object_type] = imports.import_distances(self, object_type)
                except:
                    pass

    def import_coordinates(self):
        self.coordinates = {}
        for object_type in OBJECT_TYPES:
            try:
                self.coordinates[object_type] = imports.import_coordinates(self, object_type)
            except:
                pass

    def get_distances(self, object_type):
        try:
            return self.distances[object_type]
        except:
            self.distances[object_type] = imports.import_distances(self, object_type)
            return self.distances[object_type]

    def get_coordiantes(self, object_type):
        try:
            return self.coordinates[object_type]
        except:
            self.coordinates[object_type] = imports.import_coordinates(self, object_type)
            return self.coordinates[object_type]

    def set_default_object_type(self, object_type):
        self.object_type = object_type

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

    def compute_potes(self, mapping=None):
        """ Convert votes to positional votes (called potes) """
        if not self.fake and (self.potes is None or mapping is not None):
            if mapping is None:
                self.potes = np.array([[list(vote).index(i) for i, _ in enumerate(vote)]
                                       for vote in self.votes])
            else:
                self.potes = np.array([[list(vote).index(mapping[i]) for i, _ in enumerate(vote)]
                                       for vote in self.votes])
            return self.potes

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
        else:
            winners_without_party_id = []

        self.alternative_winners[party_id] = unmap_the_winners(winners_without_party_id, party_id,
                                                               num_winners)

    def print_euclidean_voters_and_candidates_map(self, show=True, radius=None, name=None,
                                                  alpha=0.5, s=30, circles=False,
                                                  saveas=None,
                                                  object_type=None, double_gradient=False):

        plt.figure(figsize=(6.4, 6.4))

        X_voters = []
        Y_voters = []
        for i in range(self.num_voters):
            x = self.points['voters'][i][0]
            y = self.points['voters'][i][1]
            plt.scatter(x, y, color=[0, y, x], s=s, alpha=0.3)
            # X_voters.append(election.points['voters'][i][0])
            # Y_voters.append(election.points['voters'][i][1])
        # plt.scatter(X_voters, Y_voters, color='grey', s=s, alpha=0.1)

        X_candidates = []
        Y_candidates = []
        for i in range(self.num_candidates):
            X_candidates.append(self.points['candidates'][i][0])
            Y_candidates.append(self.points['candidates'][i][1])
        plt.scatter(X_candidates, Y_candidates, color='red', s=s * 2, alpha=0.9)

        if radius:
            plt.xlim([-radius, radius])
            plt.ylim([-radius, radius])
        plt.title(self.label, size=38)
        plt.axis('off')

        if saveas is None:
            saveas = f'{self.label}_euc.png'

        file_name = os.path.join(os.getcwd(), "images", name, f'{saveas}.png')
        plt.savefig(file_name, bbox_inches='tight', dpi=100)
        if show:
            plt.show()
        else:
            plt.clf()


    @abstractmethod
    def compute_distances(self):
        pass

    #DIV-MERGE
    def embed(self, algorithm='MDS', object_type=None, virtual=False):

        if object_type is None:
            object_type = self.object_type

        MDS_object = MDS(n_components=2, dissimilarity='precomputed',

            # max_iter=1000,
            # n_init=20,
            # eps=1e-6,
            )

        #DIV-MERGE
        if algorithm == 'PCA':
            self.coordinates[object_type] = pca(self.distances[object_type])
        else:
            self.coordinates[object_type] = MDS_object.fit_transform(self.distances[object_type])

        if object_type == 'vote':
            length = self.num_options
        elif object_type == 'candidate':
            # length = experiment_id.num_candidates
            pass
        else:
            logging.warning('No such type of object!')
            length = None

        # ADJUST
        # find max dist
        # if (not ('identity' in election.culture_id.lower() and object_type=='vote')) \
        #         and (not ('approval_id' in election.culture_id.lower() and object_type=='vote')):
        if not self.all_dist_zeros(object_type):
            dist = np.zeros(
                [len(self.coordinates[object_type]), len(self.coordinates[object_type])])
            for pos_1, pos_2 in itertools.combinations(
                    [i for i in range(len(self.coordinates[object_type]))],
                    2):
                dist[pos_1][pos_2] = l2(self.coordinates[object_type][pos_1],
                                        self.coordinates[object_type][pos_2])

            result = np.where(dist == np.amax(dist))
            id_1 = result[0][0]
            id_2 = result[1][0]

            # rotate
            a = id_1
            b = id_2

            try:
                d_x = self.coordinates[object_type][a][0] - self.coordinates[object_type][b][0]
                d_y = self.coordinates[object_type][a][1] - self.coordinates[object_type][b][1]
                alpha = math.atan(d_x / d_y)
                self.rotate(alpha - math.pi / 2., object_type)
                self.rotate(math.pi / 4., object_type)
            except Exception:
                pass

            # PUT heavier corner in the left lower part
            if self.coordinates[object_type][a][0] < self.coordinates[object_type][b][0]:
                left = a
                right = b
            else:
                left = b
                right = a
            try:
                left_ctr = 0
                right_ctr = 0
                for v in range(length):
                    d_left = l2(self.coordinates[object_type][left],
                                self.coordinates[object_type][v])
                    d_right = l2(self.coordinates[object_type][right],
                                 self.coordinates[object_type][v])
                    if d_left < d_right:
                        left_ctr += 1
                    else:
                        right_ctr += 1

                if left_ctr < right_ctr:
                    self.rotate(math.pi, object_type)

            except Exception:
                pass


        if object_type == 'vote':
            length = self.num_options
        elif object_type == 'candidate':
            length = self.num_candidates
            pass

        if self.is_exported and not virtual:
            exports.export_coordinates(self, object_type=object_type, length=length)

    def all_dist_zeros(self, object_type):
        if np.abs(self.distances[object_type]).sum():
            return False
        else:
            return True

    @staticmethod
    def rotate_point(cx, cy, angle, px, py) -> (float, float):
        """ Rotate two-dimensional point by an angle """
        s, c = math.sin(angle), math.cos(angle)
        px -= cx
        py -= cy
        x_new, y_new = px * c - py * s, px * s + py * c
        px, py = x_new + cx, y_new + cy

        return px, py

    def rotate(self, angle, object_type) -> None:
        """ Rotate all the points by a given angle """
        for instance_id in range(len(self.coordinates[object_type])):
            self.coordinates[object_type][instance_id][0], \
            self.coordinates[object_type][instance_id][1] = \
                self.rotate_point(0.5, 0.5, angle, self.coordinates[object_type][instance_id][0],
                                  self.coordinates[object_type][instance_id][1])

    def compute_feature(self, feature_id, feature_long_id=None, **kwargs):
        if feature_long_id is None:
            feature_long_id = feature_id
        feature = get_local_feature(feature_id)
        self.features[feature_long_id] = feature(self, **kwargs)

    def get_feature(self,
                    feature_id,
                    feature_long_id=None,
                    overwrite=False,
                    **kwargs):
        if feature_long_id is None:
            feature_long_id = feature_id
        if feature_id not in self.features or overwrite:
            self.compute_feature(feature_id, feature_long_id, **kwargs)
        return self.features[feature_long_id]


def map_the_votes(election, party_id, party_size) -> Election:
    new_votes = [[] for _ in range(election.num_voters)]
    for i in range(election.num_voters):
        for j in range(election.num_candidates):
            if election.votes[i][j] >= party_id * party_size:
                new_votes[i].append(election.votes[i][j] - party_size)
            else:
                new_votes[i].append(election.votes[i][j])
    election.votes = new_votes
    return election


def unmap_the_winners(winners, party_id, party_size):
    new_winners = []
    for j in range(len(winners)):
        if winners[j] >= party_id * party_size:
            new_winners.append(winners[j] + party_size)
        else:
            new_winners.append(winners[j])
    return new_winners


def remove_candidate_from_election(election, party_id, party_size) -> Election:
    for vote in election.votes:
        for i in range(party_size):
            _id = party_id * party_size + i
            vote.remove(_id)
    election.num_candidates -= party_size
    return election

#DIV-MERGE
def pca(distance_matrix):
    # print(distance_matrix)
    # df = pd.read_csv("http://rosetta.reltech.org/TC/v15/Mapping/data/dist-Aus.csv")
    # A = df.values.T[1:].astype(float)
    A = distance_matrix
    # square it
    A = A ** 2
    # centering matrix
    n = A.shape[0]
    # J_c = 1. / n * (np.eye(n) - 1 + (n - 1) * np.eye(n))
    J_c = np.eye(n) - 1./n

    # perform double centering
    B = -0.5 * np.matmul(np.matmul(J_c, A), J_c)

    # find eigenvalues and eigenvectors
    eigen_val = la.eig(B)[0]
    eigen_vec = la.eig(B)[1].T

    eigen_vec_real = np.round(np.real(eigen_vec), 5)
    # eigen_vec_imag = np.round(np.imag(eigen_vec), 5)
    # if np.abs(eigen_vec_imag).sum() > 1:
    #     print("Complex eigenvectors!")
    #     print(np.abs(eigen_vec_imag).sum())
        # print(eigen_vec_imag)
        # print(eigen_vec)

    eigen_val_real = np.round(np.real(eigen_val), 5)
    # eigen_val_imag = np.round(np.imag(eigen_val), 5)
    # print(eigen_val_real)
    # print(eigen_val_imag)
    # if np.abs(eigen_val_imag).sum() > 1:
    #     print("Complex eigenvalues!")
        # print(eigen_val_imag)
        # print(eigen_val)
    bests = np.argsort(-eigen_val_real)
    i = bests[0]
    j = bests[1]

    PC1 = np.sqrt(eigen_val_real[i]) * eigen_vec_real[i]
    PC2 = np.sqrt(eigen_val_real[j]) * eigen_vec_real[j]
    res = np.array([[x, y] for x, y in zip(PC1, PC2)])
    # print(PC1)
    # print(PC2)
    return res

