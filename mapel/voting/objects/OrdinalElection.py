#!/usr/bin/env python

import ast
import os

import numpy as np

from mapel.voting._glossary import *
from mapel.voting.elections.group_separable import get_gs_caterpillar_vectors
from mapel.voting.elections.mallows import get_mallows_vectors
from mapel.voting.elections.preflib import get_sushi_vectors
from mapel.voting.elections.single_crossing import get_single_crossing_vectors
from mapel.voting.elections.single_peaked import get_walsh_vectors, get_conitzer_vectors
from mapel.voting.objects.Election import Election
from mapel.voting.other.winners import compute_sntv_winners, compute_borda_winners, \
    compute_stv_winners
from mapel.voting.other.winners2 import generate_winners


class OrdinalElection(Election):

    def __init__(self, experiment_id, election_id, votes=None, with_matrix=False, alpha=None,
                 model_id=None,
                 ballot: str = 'ordinal', num_voters: int = None, num_candidates: int = None,
                 _import: bool = False, shift: bool = False):

        super().__init__(experiment_id, election_id, votes=votes, alpha=alpha,
                         model_id=model_id, ballot=ballot,
                         num_voters=num_voters, num_candidates=num_candidates)

        if votes is not None:
            if str(votes[0]) in LIST_OF_FAKE_MODELS:
                self.fake = True
                self.votes = votes[0]
                self.model_id = votes[0]
                self.num_candidates = votes[1]
                self.num_voters = votes[2]
                self.fake_param = votes[3]
            else:
                self.votes = votes
                self.num_candidates = len(votes[0])
                self.num_voters = len(votes)
                self.model_id = model_id
                self.potes = self.votes_to_potes()
        else:
            self.fake = check_if_fake(experiment_id, election_id)
            if self.fake:
                self.model_id, self.fake_param, self.num_voters, \
                self.num_candidates = import_fake_soc_election(experiment_id, election_id)
            else:
                self.votes, self.num_voters, self.num_candidates, self.param, \
                    self.model_id = import_real_soc_election(experiment_id, election_id, shift)

                self.potes = self.votes_to_potes()

        if with_matrix:
            self.matrix = self.import_matrix()
            self.vectors = self.matrix.transpose()
        else:
            self.votes_to_positionwise_vectors()

        self.borda_points = []

    def get_vectors(self):
        if self.vectors is not None:
            return self.vectors
        else:
            return self.votes_to_positionwise_vectors()

    def get_matrix(self):
        if self.matrix is not None:
            return self.matrix
        else:
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
            vectors = get_mallows_vectors(self.num_candidates, self.fake_param)
        elif self.model_id in {'identity', 'uniformity', 'antagonism', 'stratification'}:
            vectors = get_fake_vectors_single(self.model_id, self.num_candidates)
        elif self.model_id in {'walsh_path', 'conitzer_path'}:
            vectors = get_fake_multiplication(self.num_candidates, self.fake_param,
                                              self.model_id)
        elif self.model_id in PATHS:
            vectors = get_fake_convex(self.model_id, self.num_candidates, self.num_voters,
                                      self.fake_param, get_fake_vectors_single)
        elif self.model_id == 'crate':
            vectors = get_fake_vectors_crate(num_candidates=self.num_candidates,
                                             fake_param=self.fake_param)
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

    def votes_to_bordawise_vector(self) -> (np.ndarray, int):
        """ convert VOTES to Borda vector """

        borda_vector = np.zeros([self.num_candidates])

        if self.fake:

            if self.model_id in {'identity', 'uniformity', 'antagonism', 'stratification'}:
                borda_vector = get_fake_borda_vector(self.model_id, self.num_candidates,
                                                     self.num_voters)
            elif self.model_id in PATHS:
                borda_vector = get_fake_convex(self.model_id, self.num_candidates,
                                               self.num_voters, self.fake_param,
                                               get_fake_borda_vector)

        else:
            c = self.num_candidates
            v = self.num_voters
            vectors = self.votes_to_positionwise_matrix()
            borda_vector = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) * v for j in
                            range(self.num_candidates)]
            borda_vector = sorted(borda_vector, reverse=True)

        return borda_vector, len(borda_vector)

    def votes_to_positionwise_intervals(self, precision: int = None) -> list:

        vectors = self.votes_to_positionwise_matrix()
        return [self.vector_to_interval(vectors[i], precision=precision)
                for i in range(len(vectors))]

    def votes_to_voterlikeness_matrix(self) -> np.ndarray:
        """ convert VOTES to voter-likeness MATRIX """
        matrix = np.zeros([self.num_voters, self.num_voters])

        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):
                # Spearman distance between votes
                # matrix[v1][v2] = sum([abs(experiment.potes[v1][c]
                # - experiment.potes[v2][c]) for c in range(experiment.num_candidates)])

                # Swap distance between votes
                swap_distance = 0
                for i in range(self.num_candidates):
                    for j in range(i + 1, self.num_candidates):
                        if (self.potes[v1][i] > self.potes[v1][j] and
                            self.potes[v2][i] < self.potes[v2][j]) or \
                                (self.potes[v1][i] < self.potes[v1][j] and
                                 self.potes[v2][i] > self.potes[v2][j]):
                            swap_distance += 1
                matrix[v1][v2] = swap_distance

        # VOTERLIKENESS IS SYMETRIC
        for i in range(self.num_voters):
            for j in range(i + 1, self.num_voters):
                # matrix[i][j] /= float(experiment.num_candidates)
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

    # def votes_to_viper_vectors(experiment):
    #
    #     vectors = [[0. for _ in range(experiment.num_voters)]
    #                for _ in range(experiment.num_voters)]
    #
    #     c = experiment.num_candidates
    #     v = experiment.num_voters
    #     borda_vector = [sum([vectors[j][i] * (c - i - 1)
    #                          for i in range(c)]) * v
    #                     for j in range(experiment.num_candidates)]
    #
    #     for i in range(experiment.num_candidates):
    #         for j in range(experiment.num_candidates):
    #             vectors[i][j] /= float(experiment.num_voters)
    #
    #     return vectors

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

    # elif fake_model_name == 'walsh_fake':
    #     vectors = _sp.walsh(num_candidates)
    #
    # elif fake_model_name == 'conitzer_fake':
    #     vectors = _sp.conitzer(num_candidates)

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
    return line[0] == '$'


def import_fake_soc_election(experiment_id, name):
    """ Import fake ordinal election form .soc file """

    file_name = f'{name}.soc'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    my_file.readline()  # line with $ fake

    num_voters = int(my_file.readline().strip())
    num_candidates = int(my_file.readline().strip())
    fake_model_name = str(my_file.readline().strip())
    params = {}
    if fake_model_name == 'crate':
        params = [float(my_file.readline().strip()), float(my_file.readline().strip()),
                  float(my_file.readline().strip()), float(my_file.readline().strip())]
    elif fake_model_name == 'norm-mallows_matrix':
        params['norm-phi'] = float(my_file.readline().strip())
        params['weight'] = float(my_file.readline().strip())
    elif fake_model_name in PATHS:
        params['alpha'] = float(my_file.readline().strip())
        if fake_model_name == 'mallows_matrix_path':
            params['norm-phi'] = params['alpha']
            params['weight'] = float(my_file.readline().strip())
    else:
        params = {}

    return fake_model_name, params, num_voters, num_candidates


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


def import_real_soc_election(experiment_id, election_id, shift=False):
    """ Import real ordinal election form .soc file """

    file_name = str(election_id) + ".soc"
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
            print(model_name)
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


    return votes, num_voters, num_candidates, params, model_name
