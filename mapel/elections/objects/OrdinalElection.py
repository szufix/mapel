import ast
import itertools
import os
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

import mapel.elections.models.mallows as mallows
from mapel.elections.glossary_ import *
from mapel.elections.models.group_separable import get_gs_caterpillar_vectors
from mapel.elections.models.mallows import get_mallows_vectors
from mapel.elections.models.preflib import get_sushi_vectors
from mapel.elections.models.single_crossing import get_single_crossing_vectors
from mapel.elections.models.single_peaked import get_walsh_vectors, get_conitzer_vectors
from mapel.elections.models_ import generate_ordinal_votes, store_votes_in_a_file
from mapel.elections.objects.Election import Election
from mapel.elections.other.winners import compute_sntv_winners, compute_borda_winners, \
    compute_stv_winners
from mapel.elections.other.winners2 import generate_winners
from mapel.main.embedding.kamada_kawai.kamada_kawai import KamadaKawai

from sklearn.manifold import MDS

class OrdinalElection(Election):

    def __init__(self, experiment_id, election_id, votes=None, with_matrix=False, alpha=None,
                 model_id=None, params=None, label=None,
                 ballot: str = 'ordinal', num_voters: int = None, num_candidates: int = None,
                 _import: bool = False, shift: bool = False, variable = None):

        super().__init__(experiment_id, election_id, votes=votes, alpha=alpha,
                         model_id=model_id, ballot=ballot,
                         num_voters=num_voters, num_candidates=num_candidates)

        self.params = params
        self.variable = variable
        self.label = label
        self.store = True

        self.vectors = []
        self.matrix = []

        if _import and experiment_id != 'virtual':
            try:
                if votes is not None:
                    self.model_id = model_id
                    if str(votes[0]) in LIST_OF_FAKE_MODELS:
                        self.fake = True
                        self.votes = votes[0]
                        self.num_candidates = votes[1]
                        self.num_voters = votes[2]
                    else:
                        self.votes = votes
                        self.num_candidates = len(votes[0])
                        self.num_voters = len(votes)
                        self.potes = self.votes_to_potes()
                else:

                    self.fake = check_if_fake(experiment_id, election_id)
                    if self.fake:
                        self.model_id, self.params, self.num_voters, \
                        self.num_candidates = import_fake_soc_election(experiment_id, election_id)
                    else:
                        self.votes, self.num_voters, self.num_candidates, self.params, \
                            self.model_id = import_real_soc_election(experiment_id, election_id, shift)
                        try:
                            self.alpha = 1
                            if self.params and 'alpha' in self.params:
                                self.alpha = self.params['alpha']
                        except KeyError:
                            print("Error")
                            pass
                        self.potes = self.votes_to_potes()

                self.candidatelikeness_original_vectors = {}

                if with_matrix:
                    self.matrix = self.import_matrix()
                    self.vectors = self.matrix.transpose()
                else:
                    self.votes_to_positionwise_vectors()

            except:
                pass


        if params is None:
            params = {}

        self.params = params

        if self.model_id == 'all_votes':
            alpha = 1
        else:
            params, alpha = update_params(params, self.variable, self.model_id, self.num_candidates)

        self.params = params

        self.borda_points = []


        self.distances = None
        try:
            self.distances = self._import_distances()
        except:
            pass

        self.coordinates = None
        try:
            self.coordinates = self._import_coordinates()
        except:
            pass

    def get_vectors(self):
        if self.vectors is not None and len(self.vectors) > 0:
            return self.vectors
        return self.votes_to_positionwise_vectors()

    def get_matrix(self):
        if self.matrix is not None and len(self.matrix) > 0:
            return self.matrix
        return self.votes_to_positionwise_matrix()

    def votes_to_positionwise_vectors(self):

        vectors = np.zeros([self.num_candidates, self.num_candidates])

        if self.model_id == 'conitzer_matrix':
            vectors = get_conitzer_vectors(self.num_candidates)
        elif self.model_id == 'walsh_matrix':
            vectors = get_walsh_vectors(self.num_candidates)
        elif self.model_id == 'single-crossing_matrix':
            vectors = get_single_crossing_vectors(self.num_candidates)
        elif self.model_id == 'gs_caterpillar_matrix':
            vectors = get_gs_caterpillar_vectors(self.num_candidates)
        elif self.model_id == 'sushi_matrix':
            vectors = get_sushi_vectors()
        elif self.model_id in {'norm-mallows_matrix', 'mallows_matrix_path'}:
            vectors = get_mallows_vectors(self.num_candidates, self.params)
        elif self.model_id in {'identity', 'uniformity', 'antagonism', 'stratification'}:
            vectors = get_fake_vectors_single(self.model_id, self.num_candidates, self.num_voters)
        elif self.model_id in {'walsh_path', 'conitzer_path'}:
            vectors = get_fake_multiplication(self.num_candidates, self.params,
                                              self.model_id)
        elif self.model_id in PATHS:
            vectors = get_fake_convex(self.model_id, self.num_candidates, self.num_voters,
                                      self.params, get_fake_vectors_single)
        elif self.model_id == 'crate':
            vectors = get_fake_vectors_crate(num_candidates=self.num_candidates,
                                             fake_param=self.params)
        else:
            for i in range(self.num_voters):
                pos = 0
                for j in range(self.num_candidates):
                    vote = self.votes[i][j]
                    if vote == -1:
                        continue
                    vectors[vote][pos] += 1
                    pos += 1
            for i in range(self.num_candidates):
                for j in range(self.num_candidates):
                    vectors[i][j] /= float(self.num_voters)

        self.vectors = vectors
        self.matrix = self.vectors.transpose()

        return vectors

    def votes_to_positionwise_matrix(self):
        return self.votes_to_positionwise_vectors().transpose()

    def votes_to_pairwise_matrix(self) -> np.ndarray:
        """ convert VOTES to pairwise MATRIX """
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        if self.fake:

            if self.model_id in {'identity', 'uniformity', 'antagonism', 'stratification'}:
                matrix = get_fake_matrix_single(self.model_id, self.num_candidates)
            elif self.model_id in PATHS:
                matrix = get_fake_convex(self.model_id, self.num_candidates, self.num_voters,
                                         self.fake_param, get_fake_matrix_single)

        else:

            for v in range(self.num_voters):
                for c1 in range(self.num_candidates):
                    for c2 in range(c1 + 1, self.num_candidates):
                        matrix[int(self.votes[v][c1])][
                            int(self.votes[v][c2])] += 1

            for i in range(self.num_candidates):
                for j in range(i + 1, self.num_candidates):
                    matrix[i][j] /= float(self.num_voters)
                    matrix[j][i] = 1. - matrix[i][j]

        return matrix

    def votes_to_bordawise_vector(self) -> np.ndarray:
        """ convert VOTES to Borda vector """

        borda_vector = np.zeros([self.num_candidates])

        if self.fake:

            if self.model_id in {'identity', 'uniformity', 'antagonism', 'stratification'}:
                borda_vector = get_fake_borda_vector(self.model_id, self.num_candidates,
                                                     self.num_voters)
            elif self.model_id in PATHS:
                borda_vector = get_fake_convex(self.model_id, self.num_candidates,
                                               self.num_voters, self.params,
                                               get_fake_borda_vector)

        else:
            c = self.num_candidates
            v = self.num_voters
            vectors = self.votes_to_positionwise_matrix()
            borda_vector = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) * v for j in
                            range(self.num_candidates)]
            borda_vector = sorted(borda_vector, reverse=True)

        return np.array(borda_vector)

    def votes_to_candidatelikeness_original_vectors(self) -> None:
        """ convert VOTES to candidate-likeness VECTORS """
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        for c_1 in range(self.num_candidates):
            for c_2 in range(self.num_candidates):
                for vote in self.approval_votes:
                    if (c_1 in vote and c_2 not in vote) or (c_1 not in vote and c_2 in vote):
                        matrix[c_1][c_2] += 1
        matrix = matrix / self.num_voters
        self.candidatelikeness_original_vectors = matrix

    def votes_to_positionwise_intervals(self, precision: int = None) -> list:

        vectors = self.votes_to_positionwise_matrix()
        return [self.vector_to_interval(vectors[i], precision=precision)
                for i in range(len(vectors))]

    def votes_to_voterlikeness_vectors(self) -> np.ndarray:
        return self.votes_to_voterlikeness_matrix()

    def votes_to_voterlikeness_matrix(self, vector_type=None) -> np.ndarray:
        """ convert VOTES to voter-likeness MATRIX """
        matrix = np.zeros([self.num_voters, self.num_voters])
        # print(self.votes)
        self.potes = self.votes_to_potes()

        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):
                # Spearman distance between votes
                # matrix[v1][v2] = sum([abs(experiment.potes[v1][c]
                # - experiment.potes[v2][c]) for c in range(experiment.num_candidates)])

                # Swap distance between votes

                # for i in range(self.num_candidates):
                #     for j in range(i + 1, self.num_candidates):
                swap_distance = 0
                for i, j in itertools.combinations(self.potes[0], 2):
                    if (self.potes[v1][i] > self.potes[v1][j] and
                        self.potes[v2][i] < self.potes[v2][j]) or \
                            (self.potes[v1][i] < self.potes[v1][j] and
                             self.potes[v2][i] > self.potes[v2][j]):
                        swap_distance += 1
                matrix[v1][v2] = swap_distance

        # VOTERLIKENESS IS SYMETRIC
        for i in range(self.num_voters):
            for j in range(i + 1, self.num_voters):
                # matrix[i][j] **= 0.5
                matrix[j][i] = matrix[i][j]

        return matrix

    def votes_to_agg_voterlikeness_vector(self):
        """ convert VOTES to Borda vector """

        vector = np.zeros([self.num_voters])

        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):

                swap_distance = 0
                for i in range(self.num_candidates):
                    for j in range(i + 1, self.num_candidates):
                        if (self.potes[v1][i] > self.potes[v1][j] and
                            self.potes[v2][i] < self.potes[v2][j]) or \
                                (self.potes[v1][i] < self.potes[v1][j] and
                                 self.potes[v2][i] > self.potes[v2][j]):
                            swap_distance += 1
                vector[v1] += swap_distance

        return vector, len(vector)

    def votes_to_bordawise_vector_long_empty(self):

        num_possible_scores = 1 + self.num_voters * (self.num_candidates - 1)

        if self.votes[0][0] == -1:

            vector = [0 for _ in range(num_possible_scores)]
            peak = sum([i for i in range(self.num_candidates)])
            peak *= float(self.num_voters) / float(self.num_candidates)
            vector[int(peak)] = self.num_candidates

        else:

            vector = [0 for _ in range(num_possible_scores)]
            points = get_borda_points(self.votes, self.num_voters,
                                      self.num_candidates)
            for i in range(self.num_candidates):
                vector[points[i]] += 1

        return vector, num_possible_scores

    def compute_winners(self, method=None, num_winners=None):

        self.borda_points = get_borda_points(self.votes, self.num_voters, self.num_candidates)

        if method == 'sntv':
            self.winners = compute_sntv_winners(election=self, num_winners=num_winners)
        if method == 'borda':
            self.winners = compute_borda_winners(election=self, num_winners=num_winners)
        if method == 'stv':
            self.winners = compute_stv_winners(election=self, num_winners=num_winners)
        if method in {'approx_cc', 'approx_hb', 'approx_pav'}:
            self.winners = generate_winners(election=self, num_winners=num_winners, method=method)

    # PREPARE INSTANCE
    def prepare_instance(self, store=None):

        self.votes = generate_ordinal_votes(model_id=self.model_id, num_candidates=self.num_candidates,
                                       num_voters=self.num_voters, params=self.params)

        if store:
            self.store_ordinal_election()

    # STORE
    def store_ordinal_election(self):
        """ Store ordinal election in a .soc file """

        if self.model_id in LIST_OF_FAKE_MODELS:
            path = os.path.join("experiments", str(self.experiment_id),
                                "elections", (str(self.election_id) + ".soc"))
            file_ = open(path, 'w')
            file_.write(f'$ {self.model_id} {self.params} \n')
            file_.write(str(self.num_candidates) + '\n')
            file_.write(str(self.num_voters) + '\n')
            file_.close()

        else:

            path = os.path.join("experiments", str(self.experiment_id), "elections",
                                (str(self.election_id) + ".soc"))

            store_votes_in_a_file(self, self.model_id, self.election_id,
                                  self.num_candidates, self.num_voters,
                                  self.params, path, self.ballot, votes=self.votes)

    def compute_distances(self):

        potes = self.votes_to_potes()

        distances = np.zeros([len(potes), len(potes)])
        for v1 in range(len(potes)):
            for v2 in range(len(potes)):
                swap_distance = 0
                for i, j in itertools.combinations(potes[0], 2):
                    if (potes[v1][i] > potes[v1][j] and
                        potes[v2][i] < potes[v2][j]) or \
                            (potes[v1][i] < potes[v1][j] and
                             potes[v2][i] > potes[v2][j]):
                        swap_distance += 1
                distances[v1][v2] = swap_distance

        if self.store:
            file_name = f'{self.election_id}.csv'
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances", file_name)

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(
                    ["v1", "v2", "distance"])

                for v1 in range(len(potes)):
                    for v2 in range(len(potes)):
                        distance = str(distances[v1][v2])
                        writer.writerow([v1, v2, distance])

        self.distances = distances
        return distances

    def _import_distances(self):

        file_name = f'{self.election_id}.csv'
        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'distances',
                            file_name)

        distances = np.zeros([self.num_voters, self.num_voters])
        with open(path, 'r', newline='') as csv_file:

            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                distances[int(row['v1'])][int(row['v2'])] = float(row['distance'])
                distances[int(row['v2'])][int(row['v1'])] = float(row['distance'])

        return distances

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

    def embed(self, algorithm='MDS'):

        # my_pos = KamadaKawai().embed(
        #     distances=distances,
        # )
        self.coordinates = MDS(n_components=2, dissimilarity='precomputed').fit_transform(self.distances)
        # ADJUST

        # find max dist
        print(self.model_id)
        if not 'identity' in self.model_id.lower():
            dist = np.zeros([len(self.coordinates), len(self.coordinates)])
            for pos_1, pos_2 in itertools.combinations([i for i in range(len(self.coordinates))], 2):
                # print(pos_1, pos_2)
                dist[pos_1][pos_2] = np.linalg.norm(self.coordinates[pos_1] - self.coordinates[pos_2])

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
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "coordinates", file_name)

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(["vote_id", "x", "y"])

                for vote_id in range(self.num_voters):
                    x = str(self.coordinates[vote_id][0])
                    y = str(self.coordinates[vote_id][1])
                    writer.writerow([vote_id, x, y])

        return self.coordinates

    def _import_coordinates(self):

        file_name = f'{self.election_id}.csv'
        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'coordinates',
                            file_name)
        coordinates = np.zeros([self.num_voters, 2])
        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                coordinates[int(row['vote_id'])] = [float(row['x']), float(row['y'])]
        return coordinates

    def print_map(self, show=True, radius=100):
        plt.figure(figsize=(6.4, 6.4))

        X=[]
        Y=[]
        for elem in self.coordinates:
            X.append(elem[0])
            Y.append(elem[1])
        plt.scatter(X, Y, color='blue', s=12, alpha=0.3)
        plt.xlim([-radius, radius])
        plt.ylim([-radius, radius])
        plt.title(self.label, size=26)
        plt.axis('off')



        file_name = os.path.join(os.getcwd(), "images", "mini_maps", f'{self.label}.png')
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        if show:
            plt.show()
        else:
            plt.clf()

    def online_mini_map(self):

        potes = self.votes_to_potes()

        distances = np.zeros([len(potes), len(potes)])
        for v1 in range(len(potes)):
            for v2 in range(len(potes)):
                swap_distance = 0
                for i, j in itertools.combinations(potes[0], 2):
                    if (potes[v1][i] > potes[v1][j] and
                        potes[v2][i] < potes[v2][j]) or \
                            (potes[v1][i] < potes[v1][j] and
                             potes[v2][i] > potes[v2][j]):
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
        plt.xlim([-100,100])
        plt.ylim([-100,100])
        plt.title(self.label, size=26)
        plt.axis('off')

        file_name = os.path.join(os.getcwd(), "images", "mini_maps", f'{self.label}.png')
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        # plt.clf()
        # plt.savefig(file_name, bbox_inches=bbox_inches, dpi=250)
        plt.show()














def get_fake_multiplication(num_candidates, params, model):
    params['weight'] = 0.
    params['norm-phi'] = params['alpha']
    main_matrix = []
    if model == 'conitzer_path':
        main_matrix = get_conitzer_vectors(num_candidates).transpose()
    elif model == 'walsh_path':
        main_matrix = get_walsh_vectors(num_candidates).transpose()
    mallows_matrix = get_mallows_vectors(num_candidates, params).transpose()
    output = np.matmul(main_matrix, mallows_matrix).transpose()
    return output


def get_fake_vectors_single(fake_model_name, num_candidates, num_voters):
    vectors = np.zeros([num_candidates, num_candidates])

    if fake_model_name == 'identity':
        for i in range(num_candidates):
            vectors[i][i] = 1

    elif fake_model_name == 'uniformity':
        for i in range(num_candidates):
            for j in range(num_candidates):
                vectors[i][j] = 1. / num_candidates

    elif fake_model_name == 'stratification':
        half = int(num_candidates / 2)
        for i in range(half):
            for j in range(half):
                vectors[i][j] = 1. / half
        for i in range(half, num_candidates):
            for j in range(half, num_candidates):
                vectors[i][j] = 1. / half

    elif fake_model_name == 'antagonism':
        for i in range(num_candidates):
            for _ in range(num_candidates):
                vectors[i][i] = 0.5
                vectors[i][num_candidates - i - 1] = 0.5

    return vectors


def get_fake_vectors_crate(num_candidates=None, fake_param=None):
    base_1 = get_fake_vectors_single('uniformity', num_candidates)
    base_2 = get_fake_vectors_single('identity', num_candidates)
    base_3 = get_fake_vectors_single('antagonism', num_candidates)
    base_4 = get_fake_vectors_single('stratification', num_candidates)

    return crate_combination(base_1, base_2, base_3, base_4, length=num_candidates,
                             alpha=fake_param)


def get_fake_convex(fake_model_name, num_candidates, num_voters, params, function_name):
    if fake_model_name == 'unid':
        base_1 = function_name('uniformity', num_candidates, num_voters)
        base_2 = function_name('identity', num_candidates, num_voters)
    elif fake_model_name == 'anid':
        base_1 = function_name('antagonism', num_candidates, num_voters)
        base_2 = function_name('identity', num_candidates, num_voters)
    elif fake_model_name == 'stid':
        base_1 = function_name('stratification', num_candidates, num_voters)
        base_2 = function_name('identity', num_candidates, num_voters)
    elif fake_model_name == 'anun':
        base_1 = function_name('antagonism', num_candidates, num_voters)
        base_2 = function_name('uniformity', num_candidates, num_voters)
    elif fake_model_name == 'stun':
        base_1 = function_name('stratification', num_candidates, num_voters)
        base_2 = function_name('uniformity', num_candidates, num_voters)
    elif fake_model_name == 'stan':
        base_1 = function_name('stratification', num_candidates, num_voters)
        base_2 = function_name('antagonism', num_candidates, num_voters)
    else:
        raise NameError('No such fake vectors/matrix!')

    return convex_combination(base_1, base_2, length=num_candidates, params=params)


def convex_combination(base_1, base_2, length=0, params=None):
    alpha = params['alpha']
    if base_1.ndim == 1:
        output = np.zeros([length])
        for i in range(length):
            output[i] = alpha * base_1[i] + (1 - alpha) * base_2[i]
    elif base_1.ndim == 2:
        output = np.zeros([length, length])
        for i in range(length):
            for j in range(length):
                output[i][j] = alpha * base_1[i][j] + (1 - alpha) * base_2[i][j]
    else:
        raise NameError('Unknown base!')
    return output


def crate_combination(base_1, base_2, base_3, base_4, length=0, alpha=None):
    alpha = alpha['alpha']
    vectors = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            vectors[i][j] = alpha[0] * base_1[i][j] + \
                            alpha[1] * base_2[i][j] + \
                            alpha[2] * base_3[i][j] + \
                            alpha[3] * base_4[i][j]

    return vectors


def get_fake_matrix_single(fake_model_name, num_candidates):
    matrix = np.zeros([num_candidates, num_candidates])

    if fake_model_name == 'identity':
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                matrix[i][j] = 1

    elif fake_model_name in {'uniformity', 'antagonism'}:
        for i in range(num_candidates):
            for j in range(num_candidates):
                if i != j:
                    matrix[i][j] = 0.5

    elif fake_model_name == 'stratification':
        for i in range(int(num_candidates / 2)):
            for j in range(int(num_candidates / 2), num_candidates):
                matrix[i][j] = 1
        for i in range(int(num_candidates / 2)):
            for j in range(int(num_candidates / 2)):
                if i != j:
                    matrix[i][j] = 0.5
        for i in range(int(num_candidates / 2), num_candidates):
            for j in range(int(num_candidates / 2), num_candidates):
                if i != j:
                    matrix[i][j] = 0.5

    return matrix


def get_fake_borda_vector(fake_model_name, num_candidates, num_voters):
    borda_vector = np.zeros([num_candidates])

    m = num_candidates
    n = num_voters

    if fake_model_name == 'identity':
        for i in range(m):
            borda_vector[i] = n * (m - 1 - i)

    elif fake_model_name in {'uniformity', 'antagonism'}:
        for i in range(m):
            borda_vector[i] = n * (m - 1) / 2

    elif fake_model_name == 'stratification':
        for i in range(int(m / 2)):
            borda_vector[i] = n * (m - 1) * 3 / 4
        for i in range(int(m / 2), m):
            borda_vector[i] = n * (m - 1) / 4

    return borda_vector


def get_borda_points(votes, num_voters, num_candidates):
    points = np.zeros([num_candidates])
    scoring = [1. for _ in range(num_candidates)]

    for i in range(len(scoring)):
        scoring[i] = len(scoring) - i - 1

    for i in range(num_voters):
        for j in range(num_candidates):
            points[int(votes[i][j])] += scoring[j]

    return points


def check_if_fake(experiment_id, name):
    file_name = f'{name}.soc'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    line = my_file.readline().strip()
    my_file.close()
    return line[0] == '$'


def import_fake_soc_election(experiment_id, name):
    """ Import fake ordinal election form .soc file """

    ### TMP ###

    # file_name = f'{name}.soc'
    # path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    # my_file = open(path, 'r')
    #
    # fake = my_file.readline()  # skip
    # num_voters = int(my_file.readline())
    # num_candidates = int(my_file.readline())
    # model_name = str(my_file.readline()).strip()
    #
    # params = {}
    # if any(item in model_name for item in ['anid', 'stid', 'anun', 'stun',
    #                   'mallows_matrix_path', 'walsh_path', 'conitzer_path']):
    #     params['alpha'] = float(my_file.readline())
    #     params['norm-phi'] = params['alpha']
    # if 'mallows_matrix_path' in model_name:
    #     params['weight'] = float(my_file.readline())
    #
    # return model_name, params, num_voters, num_candidates

    ############################################################

    file_name = f'{name}.soc'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')

    first_line = my_file.readline()
    first_line = first_line.strip().split()
    model_name = first_line[1]
    if len(first_line) <= 2:
        params = {}
    else:
        params = ast.literal_eval(" ".join(first_line[2:]))

    num_candidates = int(my_file.readline())
    num_voters = int(my_file.readline())

    my_file.close()

    return model_name, params, num_voters, num_candidates


def old_name_extractor(first_line):
    if len(first_line) == 4:
        model_name = f'{first_line[1]} {first_line[2]} {first_line[3]}'
    elif len(first_line) == 3:
        model_name = f'{first_line[1]} {first_line[2]}'
    elif len(first_line) == 2:
        model_name = first_line[1]
    else:
        model_name = 'noname'
    return model_name


def import_real_soc_election(experiment_id: str, election_id: str, shift=False):
    """ Import real ordinal election form .soc file """

    file_name = f'{election_id}.soc'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')

    params = 0
    first_line = my_file.readline()

    if first_line[0] != '#':
        model_name = 'empty'
        num_candidates = int(first_line)
    else:
        first_line = first_line.strip().split()
        model_name = first_line[1]
        if experiment_id == 'original_ordinal_map':
            params = {}
            model_name = old_name_extractor(first_line)
            # print(model_name)
        else:
            # params = {}
            if len(first_line) <= 2:
                params = {}
            else:
                params = ast.literal_eval(" ".join(first_line[2:]))

        num_candidates = int(my_file.readline())

    for _ in range(num_candidates):
        my_file.readline()

    line = my_file.readline().rstrip("\n").split(',')
    num_voters = int(line[0])
    num_options = int(line[2])
    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    it = 0
    for j in range(num_options):
        line = my_file.readline().rstrip("\n").split(',')
        quantity = int(line[0])

        for k in range(quantity):
            for el in range(num_candidates):
                votes[it][el] = int(line[el + 1])
            it += 1

    if shift:
        for i in range(num_voters):
            for j in range(num_candidates):
                votes[i][j] -= 1
    my_file.close()

    return votes, num_voters, num_candidates, params, model_name


def convert_ordinal_to_approval(votes):
    approval_votes = [{} for _ in range(len(votes))]
    for i, vote in enumerate(votes):
        k = int(np.random.beta(2, 6)*len(votes[0]))+1
        approval_votes[i] = set(vote[0:k])

    return approval_votes


# HELPER FUNCTIONS


def update_params(params, variable, model_id, num_candidates):

    if variable is not None:
        params['alpha'] = params[variable]
        params['variable'] = variable

        if model_id in APPROVAL_MODELS:
            if 'p' not in params:
                params['p'] = np.random.rand()
            elif type(params['p']) is list:
                params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])

    else:
        if model_id in ['approval_partylist']:
            return params, 1

        if model_id in APPROVAL_MODELS:
            if 'p' not in params:
                params['p'] = np.random.rand()
            elif type(params['p']) is list:
                params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])

        if 'phi' in params and type(params['phi']) is list:
            params['phi'] = np.random.uniform(low=params['phi'][0], high=params['phi'][1])

        if model_id == 'mallows' and params['phi'] is None:
            params['phi'] = np.random.random()
        elif model_id == 'norm-mallows' and 'norm-phi' not in params:
            params['norm-phi'] = np.random.random()
        elif model_id in ['urn_model', 'approval_urn'] and 'alpha' not in params:
            params['alpha'] = gamma.rvs(0.8)

        if model_id == 'norm-mallows':
            params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])
            if 'weight' not in params:
                params['weight'] = 0.

        if model_id == 'mallows_matrix_path':
            params['norm-phi'] = params['alpha']
            params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])

        if model_id == 'erdos_renyi_graph' and params['p'] is None:
            params['p'] = np.random.random()

        if 'alpha' not in params:
            if 'norm-phi' in params:
                params['alpha'] = params['norm-phi']
            elif 'phi' in params:
                params['alpha'] = params['phi']
            else:
                params['alpha'] = np.random.rand()
        elif type(params['alpha']) is list:
            params['alpha'] = np.random.uniform(low=params['alpha'][0], high=params['alpha'][1])

    return params, params['alpha']


# HELPER FUNCTIONS #
def prepare_parties(model_id=None, params=None):
    parties = []

    if model_id == '2d_gaussian_party':
        for i in range(params['num_parties']):
            point = np.random.rand(1, 2)
            parties.append(point)

    elif model_id in ['1d_gaussian_party', 'conitzer_party', 'walsh_party']:
        for i in range(params['num_parties']):
            point = np.random.rand(1, 1)
            parties.append(point)

    return parties

