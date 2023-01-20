import ast
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gamma
import random as rand
import csv
import time

import mapel.elections.cultures.mallows as mallows
from mapel.core.glossary import *
from mapel.elections.cultures.group_separable import get_gs_caterpillar_vectors
from mapel.elections.cultures.mallows import get_mallows_vectors
from mapel.elections.cultures.preflib import get_sushi_vectors
from mapel.elections.cultures.single_crossing import get_single_crossing_vectors
from mapel.elections.cultures.single_peaked import get_walsh_vectors, get_conitzer_vectors
from mapel.elections.cultures_ import generate_ordinal_votes, store_votes_in_a_file, from_approval
from mapel.elections.objects.Election import Election
from mapel.elections.other.winners import compute_sntv_winners, compute_borda_winners, \
    compute_stv_winners
from mapel.elections.other.winners2 import generate_winners
from mapel.core.inner_distances import swap_distance_between_potes, \
    spearman_distance_between_potes
from mapel.elections.features.other import is_condorcet
from mapel.core.utils import *


class OrdinalElection(Election):

    def __init__(self, experiment_id, election_id, votes=None, with_matrix=False, alpha=None,
                 culture_id=None, params=None, label=None,
                 ballot: str = 'ordinal', num_voters: int = None, num_candidates: int = None,
                 _import: bool = False, shift: bool = False, variable=None, fast_import=False):

        super().__init__(experiment_id, election_id, votes=votes, alpha=alpha,
                         culture_id=culture_id, ballot=ballot, label=label,
                         num_voters=num_voters, num_candidates=num_candidates,
                         fast_import=fast_import)

        self.params = params
        self.variable = variable

        self.vectors = []
        self.matrix = []
        self.borda_points = []
        self.potes = None
        self.condorcet = None
        self.points = {}

        if _import and experiment_id != 'virtual':
            try:
                if votes is not None:
                    self.culture_id = culture_id
                    if str(votes[0]) in LIST_OF_FAKE_MODELS:
                        self.fake = True
                        self.votes = votes[0]
                        self.num_candidates = votes[1]
                        self.num_voters = votes[2]
                    else:
                        self.votes = votes
                        self.num_candidates = len(votes[0])
                        self.num_voters = len(votes)
                        self.compute_potes()
                else:
                    self.fake = check_if_fake(experiment_id, election_id)
                    if self.fake:
                        self.culture_id, self.params, self.num_voters, \
                        self.num_candidates = import_fake_soc_election(experiment_id, election_id)
                    else:
                        self.votes, self.num_voters, self.num_candidates, self.params, \
                        self.culture_id = import_real_soc_election(experiment_id, election_id,
                                                                   shift, fast_import)
                        try:
                            self.points['voters'] = self.import_ideal_points('voters')
                            self.points['candidates'] = self.import_ideal_points('candidates')
                        except:
                            pass

                        try:
                            self.alpha = 1
                            if self.params and 'alpha' in self.params:
                                self.alpha = self.params['alpha']
                        except KeyError:
                            print("Error")
                            pass
                        # if not fast_import:
                        #     self.compute_potes()

                self.candidatelikeness_original_vectors = {}

                if with_matrix:
                    self.matrix = self.import_matrix()
                    self.vectors = self.matrix.transpose()
                else:
                    if not fast_import:
                        self.votes_to_positionwise_vectors()


            except:
                pass


        if self.params is None:
            self.params = {}

        if culture_id is not None:
            self.params, self.alpha = update_params_ordinal(self.params, self.variable, self.culture_id,
                                                            self.num_candidates)

    def get_vectors(self):
        if self.vectors is not None and len(self.vectors) > 0:
            return self.vectors
        return self.votes_to_positionwise_vectors()

    def get_matrix(self):
        if self.matrix is not None and len(self.matrix) > 0:
            return self.matrix
        return self.votes_to_positionwise_matrix()

    def get_potes(self):
        if self.potes is not None:
            return self.potes
        return self.compute_potes()

    def votes_to_positionwise_vectors(self):
        vectors = np.zeros([self.num_candidates, self.num_candidates])

        if self.culture_id == 'conitzer_matrix':
            vectors = get_conitzer_vectors(self.num_candidates)
        elif self.culture_id == 'walsh_matrix':
            vectors = get_walsh_vectors(self.num_candidates)
        elif self.culture_id == 'single-crossing_matrix':
            vectors = get_single_crossing_vectors(self.num_candidates)
        elif self.culture_id == 'gs_caterpillar_matrix':
            vectors = get_gs_caterpillar_vectors(self.num_candidates)
        elif self.culture_id == 'sushi_matrix':
            vectors = get_sushi_vectors()
        elif self.culture_id in {'norm-mallows_matrix', 'mallows_matrix_path'}:
            vectors = get_mallows_vectors(self.num_candidates, self.params)
        elif self.culture_id in {'identity', 'uniformity', 'antagonism', 'stratification'}:
            vectors = get_fake_vectors_single(self.culture_id, self.num_candidates)
        elif self.culture_id in {'walsh_path', 'conitzer_path'}:
            vectors = get_fake_multiplication(self.num_candidates, self.params,
                                              self.culture_id)
        elif self.culture_id in PATHS:
            vectors = get_fake_convex(self.culture_id, self.num_candidates, self.num_voters,
                                      self.params, get_fake_vectors_single)
        elif self.culture_id == 'crate':
            vectors = get_fake_vectors_crate(num_candidates=self.num_candidates,
                                             fake_param=self.params)
        elif self.culture_id in ['from_approval']:
            # print(self.culture_id)
            vectors = from_approval(num_candidates=self.num_candidates,
                                                num_voters=self.num_voters,
                                                params=self.params)
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
            if self.culture_id in {'identity', 'uniformity', 'antagonism', 'stratification'}:
                matrix = get_fake_matrix_single(self.culture_id, self.num_candidates)
            elif self.culture_id in PATHS:
                matrix = get_fake_convex(self.culture_id, self.num_candidates, self.num_voters,
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
            if self.culture_id in {'identity', 'uniformity', 'antagonism', 'stratification'}:
                borda_vector = get_fake_borda_vector(self.culture_id, self.num_candidates,
                                                     self.num_voters)
            elif self.culture_id in PATHS:
                borda_vector = get_fake_convex(self.culture_id, self.num_candidates,
                                               self.num_voters, self.params,
                                               get_fake_borda_vector)
        else:
            c = self.num_candidates
            v = self.num_voters
            vectors = self.votes_to_positionwise_vectors()
            borda_vector = [sum([vectors[i][j] * (c - j - 1) for j in range(c)]) * v for i in
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
        self.candidatelikeness_original_vectors = matrix / self.num_voters

    def votes_to_positionwise_intervals(self, precision: int = None) -> list:

        vectors = self.votes_to_positionwise_matrix()
        return [self.vector_to_interval(vectors[i], precision=precision)
                for i in range(len(vectors))]

    def votes_to_voterlikeness_vectors(self) -> np.ndarray:
        return self.votes_to_voterlikeness_matrix()

    def votes_to_voterlikeness_matrix(self, vector_type=None) -> np.ndarray:
        """ convert VOTES to voter-likeness MATRIX """
        matrix = np.zeros([self.num_voters, self.num_voters])
        self.compute_potes()

        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):
                matrix[v1][v2] = swap_distance_between_potes(self.potes[v1], self.potes[v2])
                # matrix[v1][v2] = spearman_distance_between_potes(self.potes[v1], self.potes[v2])

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

    def compute_winners(self, method=None, num_winners=None):

        self.borda_points = get_borda_points(self.votes, self.num_voters, self.num_candidates)

        if method == 'sntv':
            self.winners = compute_sntv_winners(election=self, num_winners=num_winners)
        if method == 'borda':
            self.winners = compute_borda_winners(election=self, num_winners=num_winners)
        if method == 'stv':
            self.winners = compute_stv_winners(election=self, num_winners=num_winners)
        if method in {'approx_cc', 'approx_hb', 'approx_pav'}:
            self.winners = generate_winners(election=self, num_winners=num_winners)

    # PREPARE INSTANCE
    def prepare_instance(self, store=None, aggregated=True):
        # self.params['exp_id'] = self.experiment_id
        # self.params['ele_id'] = self.election_id
        # self.params['aggregated'] = aggregated
        self.votes = generate_ordinal_votes(culture_id=self.culture_id,
                                                num_candidates=self.num_candidates,
                                                num_voters=self.num_voters,
                                                params=self.params)
        if store:
            self._store_ordinal_election(aggregated=aggregated)

    # STORE
    def _store_ordinal_election(self, aggregated=True):
        """ Store ordinal election in a .soc file """

        path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections")
        make_folder_if_do_not_exist(path_to_folder)
        path_to_file = os.path.join(path_to_folder,  f'{self.election_id}.soc')

        if self.culture_id in LIST_OF_FAKE_MODELS:
            file_ = open(path_to_file, 'w')
            file_.write(f'$ {self.culture_id} {self.params} \n')
            file_.write(str(self.num_candidates) + '\n')
            file_.write(str(self.num_voters) + '\n')
            file_.close()
        else:
            store_votes_in_a_file(self, self.culture_id, self.num_candidates, self.num_voters,
                                  self.params, path_to_file, self.ballot, votes=self.votes,
                                  aggregated=aggregated)

    def compute_distances(self, distance_id='swap', object_type=None):
        """ Return: distances between votes """
        if object_type is None:
            object_type = self.object_type

        if object_type == 'vote':
            self.compute_potes()
            distances = np.zeros([self.num_voters, self.num_voters])
            for v1 in range(self.num_voters):
                for v2 in range(self.num_voters):
                    if distance_id == 'swap':
                        distances[v1][v2] = swap_distance_between_potes(
                            self.potes[v1], self.potes[v2])
                    elif distance_id == 'spearman':
                        distances[v1][v2] = spearman_distance_between_potes(
                            self.potes[v1], self.potes[v2])
        elif object_type == 'candidate':
            self.compute_potes()
            if distance_id == 'domination':
                distances = self.votes_to_pairwise_matrix()
                distances = np.abs(distances - 0.5) * self.num_voters
                np.fill_diagonal(distances, 0)
            elif distance_id == 'position':
                distances = np.zeros([self.num_candidates, self.num_candidates])
                for c1 in range(self.num_candidates):
                    for c2 in range(self.num_candidates):
                        dist = 0
                        for pote in self.potes:
                            dist += abs(pote[c1] - pote[c2])
                        distances[c1][c2] = dist

        self.distances[object_type] = distances

        if self.store:
            self._store_distances(object_type=object_type)

    def is_condorcet(self):
        """ Check if election witness Condorcet winner"""
        if self.condorcet is None:
            self.condorcet = is_condorcet(self)
        return self.condorcet

    def import_ideal_points(self, name):
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections",
                            f'{self.election_id}_{name}.csv')
        points = []
        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for row in reader:
                points.append([float(row['x']), float(row['y'])])
        return points

    def texify_label(self, name):
        return name.replace('phi', '$\phi$'). \
            replace('alpha', '$\\ \\alpha$'). \
            replace('§', '\n', 1)
            # replace('0.005', '$\\frac{1}{200}$'). \
            # replace('0.025', '$\\frac{1}{40}$'). \
            # replace('0.75', '$\\frac{3}{4}$'). \
            # replace('0.25', '$\\frac{1}{4}$'). \
            # replace('0.01', '$\\frac{1}{100}$'). \
            # replace('0.05', '$\\frac{1}{20}$'). \
            # replace('0.5', '$\\frac{1}{2}$'). \
            # replace('0.1', '$\\frac{1}{10}$'). \
            # replace('0.2', '$\\frac{1}{5}$'). \
            # replace('0.4', '$\\frac{2}{5}$'). \
            # replace('0.8', '$\\frac{4}{5}$'). \
            # replace(' ', '\n', 1)

    def print_map(self, show=True, radius=None, name=None, alpha=0.1, s=30, circles=False,
                  object_type=None, double_gradient=False, saveas=None, color='blue',
                  marker='o', title_size=20):

        if object_type == 'vote':
            length = self.num_voters
        elif object_type == 'candidate':
            length = self.num_candidates

        if object_type is None:
            object_type = self.object_type

        plt.figure(figsize=(6.4, 6.4))

        X = []
        Y = []
        for elem in self.coordinates[object_type]:
            X.append(elem[0])
            Y.append(elem[1])

        start = False
        if start:
            plt.scatter(X[0], Y[0],
                        color='sienna',
                        s=1000,
                        alpha=1,
                        marker='X')

        if double_gradient:
            for i in range(length):
                x = float(self.points['voters'][i][0])
                y = float(self.points['voters'][i][1])
                plt.scatter(X[i], Y[i], color=[0,y,x], s=s, alpha=alpha)
        else:
            plt.scatter(X, Y, color=color, s=s, alpha=alpha, marker=marker)


        if circles: # works only for votes
            weighted_points = {}
            Xs = {}
            Ys = {}
            for i in range(length):
                str_elem = str(self.votes[i])
                # str_elem = f'{round(X[i])}_{round(Y[i])}'
                if str_elem in weighted_points:
                    weighted_points[str_elem] += 1
                else:
                    weighted_points[str_elem] = 1
                    Xs[str_elem] = X[i]
                    Ys[str_elem] = Y[i]
            # print(weighted_points)
            # print(len(weighted_points))
            for str_elem in weighted_points:
                if weighted_points[str_elem] > 10 and str_elem!='set()':
                    plt.scatter(Xs[str_elem], Ys[str_elem],
                                color='purple',
                                s=10 * weighted_points[str_elem],
                                alpha=0.2)

        # print(len(weighted_points))

        avg_x = np.mean(X)
        avg_y = np.mean(Y)

        if radius:
            plt.xlim([avg_x-radius, avg_x+radius])
            plt.ylim([avg_y-radius, avg_y+radius])
        # plt.title(self.label, size=38)
        plt.title(self.texify_label(self.label), size=38)
        # plt.title(self.texify_label(self.label), size=38, y=0.94)
        # plt.title(self.label, size=title_size)
        plt.axis('off')

        if saveas is None:
            saveas = f'{self.label}'

        file_name = os.path.join(os.getcwd(), "images", name, f'{saveas}.png')
        plt.savefig(file_name, bbox_inches='tight', dpi=100)
        if show:
            plt.show()
        else:
            plt.clf()


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


def get_fake_vectors_single(fake_model_name, num_candidates):
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
        base_1 = function_name('uniformity', num_candidates)
        base_2 = function_name('identity', num_candidates)
    elif fake_model_name == 'anid':
        base_1 = function_name('antagonism', num_candidates)
        base_2 = function_name('identity', num_candidates)
    elif fake_model_name == 'stid':
        base_1 = function_name('stratification', num_candidates)
        base_2 = function_name('identity', num_candidates)
    elif fake_model_name == 'anun':
        base_1 = function_name('antagonism', num_candidates)
        base_2 = function_name('uniformity', num_candidates)
    elif fake_model_name == 'stun':
        base_1 = function_name('stratification', num_candidates)
        base_2 = function_name('uniformity', num_candidates)
    elif fake_model_name == 'stan':
        base_1 = function_name('stratification', num_candidates)
        base_2 = function_name('antagonism', num_candidates)
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


def import_real_soc_election(experiment_id: str, election_id: str, shift=False,
                             fast_import=False):
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
        else:
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

    ## TMP
    # if fast_import:
    #     my_file.close()
    #     return votes, num_voters, num_candidates, params, model_name
    ## END OF TMP

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

    return np.array(votes), num_voters, num_candidates, params, model_name


def convert_ordinal_to_approval(votes):
    approval_votes = [{} for _ in range(len(votes))]
    for i, vote in enumerate(votes):
        k = int(np.random.beta(2, 6) * len(votes[0])) + 1
        approval_votes[i] = set(vote[0:k])

    return approval_votes


# HELPER FUNCTIONS
def update_params_ordinal_mallows(params):
    if 'phi' in params and type(params['phi']) is list:
        params['phi'] = np.random.uniform(low=params['phi'][0], high=params['phi'][1])
    elif 'phi' not in params:
        params['phi'] = np.random.random()
    params['alpha'] = params['phi']


def update_params_ordinal_norm_mallows(params, num_candidates):
    if 'norm-phi' not in params:
        params['norm-phi'] = np.random.random()
    params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])
    if 'weight' not in params:
        params['weight'] = 0.
    params['alpha'] = params['norm-phi']


def update_params_ordinal_urn_model(params):
    if 'alpha' not in params:
        params['alpha'] = gamma.rvs(0.8)


def update_params_ordinal_mallows_matrix_path(params, num_candidates):
    params['norm-phi'] = params['alpha']
    params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])

def update_params_ordinal_mallows_triangle(params, num_candidates):
    params['norm-phi'] = 1 - np.sqrt(np.random.uniform())
    params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=params['norm-phi'])
    params['weight'] = np.random.uniform(0,0.5)
    params['alpha'] = params['norm-phi']
    params['tint'] = params['weight'] # for tint on plots


def update_params_ordinal_alpha(params):
    if 'alpha' not in params:
        params['alpha'] = 1
    elif type(params['alpha']) is list:
        params['alpha'] = np.random.uniform(low=params['alpha'][0], high=params['alpha'][1])


def update_params_ordinal_preflib(params, model_id):
    # list of IDs larger than 10
    folder = ''
    if model_id == 'irish':
        folder = 'irish_s1'
        # folder = 'irish_f'
        ids = [1, 3]
    elif model_id == 'glasgow':
        folder = 'glasgow_s1'
        # folder = 'glasgow_f'
        ids = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 21]
    elif model_id == 'formula':
        folder = 'formula_s1'
        # 17 races or more
        ids = [17, 35, 37, 40, 41, 42, 44, 45, 46, 47, 48]
    elif model_id == 'skate':
        folder = 'skate_ic'
        # 9 judges
        ids = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48]
    elif model_id == 'sushi':
        folder = 'sushi_ff'
        ids = [1]
    elif model_id == 'grenoble':
        folder = 'grenoble_ff'
        ids = [1]
    elif model_id == 'tshirt':
        folder = 'tshirt_ff'
        ids = [1]
    elif model_id == 'cities_survey':
        folder = 'cities_survey_s1'
        ids = [1, 2]
    elif model_id == 'aspen':
        folder = 'aspen_s1'
        ids = [1]
    elif model_id == 'marble':
        folder = 'marble_ff'
        ids = [1, 2, 3, 4, 5]
    elif model_id == 'cycling_tdf':
        folder = 'cycling_tdf_s1'
        # ids = [e for e in range(1, 69+1)]
        selection_method = 'random'
        ids = [14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]
    elif model_id == 'cycling_gdi':
        folder = 'cycling_gdi_s1'
        ids = [i for i in range(2, 23 + 1)]
    elif model_id == 'ers':
        folder = 'ers_s1'
        # folder = 'ers_f'
        # 500 voters or more
        ids = [3, 9, 23, 31, 32, 33, 36, 38, 40, 68, 77, 79, 80]
    elif model_id == 'ice_races':
        folder = 'ice_races_s1'
        # 80 voters or more
        ids = [4, 5, 8, 9, 15, 20, 23, 24, 31, 34, 35, 37, 43, 44, 49]
    else:
        ids = []

    if 'id' not in params:
        params['id'] = rand.choices(ids, k=1)[0]

    params['folder'] = folder


def update_params_ordinal(params, variable, culture_id, num_candidates):
    if variable is not None:
        params['alpha'] = params[variable]
        params['variable'] = variable
    else:
        if culture_id.lower() == 'mallows':
            update_params_ordinal_mallows(params)
        elif culture_id.lower() == 'norm_mallows' or culture_id.lower() == 'norm-mallows':
            update_params_ordinal_norm_mallows(params, num_candidates)
        elif culture_id.lower() == 'urn_model' or culture_id.lower() == 'urn':
            update_params_ordinal_urn_model(params)
        elif culture_id.lower() == 'mallows_matrix_path':
            update_params_ordinal_mallows_matrix_path(params, num_candidates)
        elif culture_id.lower() == 'mallows_triangle':
            update_params_ordinal_mallows_triangle(params, num_candidates)
        elif culture_id.lower() in LIST_OF_PREFLIB_MODELS:
            update_params_ordinal_preflib(params, culture_id)
        update_params_ordinal_alpha(params)
    return params, params['alpha']


# HELPER FUNCTIONS #
def prepare_parties(culture_id=None, params=None):
    parties = []
    if culture_id == '2d_gaussian_party':
        for i in range(params['num_parties']):
            point = np.random.rand(1, 2)
            parties.append(point)
    elif culture_id in ['1d_gaussian_party', 'conitzer_party', 'walsh_party']:
        for i in range(params['num_parties']):
            point = np.random.rand(1, 1)
            parties.append(point)
    return parties
