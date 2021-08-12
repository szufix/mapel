#!/usr/bin/env python

import csv
import math
import os
import numpy as np


from mapel.voting.elections.group_separable import get_gs_caterpillar_vectors
from mapel.voting.elections.single_peaked import get_walsh_vectors, get_conitzer_vectors
from mapel.voting.elections.single_crossing import get_single_crossing_vectors
from mapel.voting.elections.mallows import get_mallows_vectors
from mapel.voting.elections.preflib import get_sushi_vectors

from mapel.voting.glossary import LIST_OF_FAKE_MODELS, LIST_OF_PREFLIB_MODELS


class Election:

    def __init__(self, experiment_id, election_id, votes=None, with_matrix=False, election_model=None,
                 num_voters=None, num_candidates=None):

        self.experiment_id = experiment_id
        self.election_id = election_id

        self.num_voters = num_voters
        self.num_candidates = num_candidates

        if election_model in LIST_OF_FAKE_MODELS:
            self.fake = True
        else:
            self.fake = False

        if votes is not None:
            if str(votes[0]) in LIST_OF_FAKE_MODELS:
                self.fake = True
                self.votes = votes[0]
                self.election_model = votes[0]
                self.num_candidates = votes[1]
                self.num_voters = votes[2]
                self.fake_param = votes[3]

            else:
                self.votes = votes
                self.num_candidates = len(votes[0])
                self.num_voters = len(votes)
                self.election_model = election_model
                self.potes = self.votes_to_potes()
        else:
            self.fake = check_if_fake(experiment_id, election_id)
            if self.fake:
                self.election_model, self.fake_param, self.num_voters, self.num_candidates = import_fake_elections(
                    experiment_id, election_id)
            else:
                self.votes, self.num_voters, self.num_candidates, self.param, self.election_model = import_soc_elections(
                    experiment_id, election_id)

                self.potes = self.votes_to_potes()

        if with_matrix:
            self.matrix = self.import_matrix()
            self.vectors = self.matrix.transpose()
        else:
            self.votes_to_positionwise_vectors()

    def import_matrix(self):

        file_name = self.election_id + '.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, 'matrices', file_name)
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for i, row in enumerate(reader):
                for j, candidate_id in enumerate(row):
                    matrix[i][j] = row[candidate_id]
        return matrix

    def votes_to_potes(self):
        """ Prepare positional votes """
        potes = np.zeros([self.num_voters, self.num_candidates])
        for i in range(self.num_voters):
            for j in range(self.num_candidates):
                potes[i][self.votes[i][j]] = j
        return potes

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

        vectors = np.zeros([self.num_candidates,self.num_candidates])

        if self.election_model == 'conitzer_matrix':
            vectors = get_conitzer_vectors(self.num_candidates)
        elif self.election_model == 'walsh_matrix':
            vectors = get_walsh_vectors(self.num_candidates)
        elif self.election_model == 'single-crossing_matrix':
            vectors = get_single_crossing_vectors(self.num_candidates)
        elif self.election_model == 'gs_caterpillar_matrix':
            vectors=get_gs_caterpillar_vectors(self.num_candidates)
        elif self.election_model == 'sushi_matrix':
            vectors=get_sushi_vectors()
        elif self.election_model == 'norm-mallows_matrix':
            vectors=get_mallows_vectors(self.num_candidates, self.fake_param)
        elif self.election_model in {'identity', 'uniformity', 'antagonism', 'stratification'}:
            vectors = get_fake_vectors_single(self.election_model, self.num_candidates, self.num_voters)
        elif self.election_model in {'unid', 'anid', 'stid', 'anun', 'stun', 'stan'}:
            vectors = get_fake_convex(self.election_model, self.num_candidates, self.num_voters, self.fake_param,
                                      get_fake_vectors_single)
        elif self.election_model == 'crate':
            vectors = get_fake_vectors_crate(self.election_model, self.num_candidates, self.num_voters, self.fake_param)
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

            # # todo: change to original version
            # if not self.fake:
            #     vectors = [*zip(*vectors)]

        # return matrix.transpose()

        self.vectors = vectors
        self.matrix = self.vectors.transpose()

        return vectors

    def votes_to_positionwise_matrix(self):
        return self.votes_to_positionwise_vectors().transpose()

    def votes_to_viper_vectors(self):

        vectors = [[0. for _ in range(self.num_voters)] for _ in range(self.num_voters)]

        c = self.num_candidates
        v = self.num_voters
        borda_vector = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) * v for j in range(self.num_candidates)]

        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
                vectors[i][j] /= float(self.num_voters)

        return vectors

    def vector_to_interval(self, vector, precision=None):
        # discreet version for now
        interval = []
        w = int(precision / self.num_candidates)
        for i in range(self.num_candidates):
            for j in range(w):
                interval.append(vector[i] / w)
        return interval

    def votes_to_positionwise_intervals(self, precision=None):

        vectors = self.votes_to_positionwise_matrix()
        intervals = []

        for i in range(len(vectors)):
            intervals.append(self.vector_to_interval(vectors[i], precision=precision))

        return intervals

    def votes_to_pairwise_matrix(self):
        """ convert VOTES to pairwise MATRIX """
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        if self.fake:

            if self.election_model in {'identity', 'uniformity', 'antagonism', 'stratification'}:
                matrix = get_fake_matrix_single(self.election_model, self.num_candidates, self.num_voters)
            elif self.election_model in {'unid', 'anid', 'stid', 'anun', 'stun', 'stan'}:
                matrix = get_fake_convex(self.election_model, self.num_candidates, self.num_voters, self.fake_param,
                                         get_fake_matrix_single)

        else:

            for v in range(self.num_voters):
                for c1 in range(self.num_candidates):
                    for c2 in range(c1 + 1, self.num_candidates):
                        matrix[int(self.votes[v][c1])][int(self.votes[v][c2])] += 1

            for i in range(self.num_candidates):
                for j in range(i + 1, self.num_candidates):
                    matrix[i][j] /= float(self.num_voters)
                    matrix[j][i] = 1. - matrix[i][j]

        return matrix

    def votes_to_voterlikeness_matrix(self):
        """ convert VOTES to voter-likeness MATRIX """
        matrix = np.zeros([self.num_voters, self.num_voters])

        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):
                # Spearman distance between votes
                # matrix[v1][v2] = sum([abs(self.potes[v1][c] - self.potes[v2][c]) for c in range(self.num_candidates)])

                # Swap distance between votes
                swap_distance = 0
                for i in range(self.num_candidates):
                    for j in range(i + 1, self.num_candidates):
                        if (self.potes[v1][i] > self.potes[v1][j] and self.potes[v2][i] < self.potes[v2][j]) or \
                                (self.potes[v1][i] < self.potes[v1][j] and self.potes[v2][i] > self.potes[v2][j]):
                            swap_distance += 1
                matrix[v1][v2] = swap_distance

        # VOTERLIKENESS IS SYMETRIC
        for i in range(self.num_voters):
            for j in range(i + 1, self.num_voters):
                # matrix[i][j] /= float(self.num_candidates)
                matrix[j][i] = matrix[i][j]

        return matrix

    def votes_to_bordawise_vector(self):
        """ convert VOTES to Borda vector """

        borda_vector = np.zeros([self.num_candidates])

        if self.fake:

            if self.election_model in {'identity', 'uniformity', 'antagonism', 'stratification'}:
                borda_vector = get_fake_borda_vector(self.election_model, self.num_candidates, self.num_voters)
            elif self.election_model in {'unid', 'anid', 'stid', 'anun', 'stun', 'stan'}:
                borda_vector = get_fake_convex(self.election_model, self.num_candidates, self.num_voters,
                                               self.fake_param,
                                               get_fake_borda_vector)

        else:
            c = self.num_candidates
            v = self.num_voters
            vectors = self.votes_to_positionwise_matrix()
            borda_vector = [sum([vectors[j][i] * (c - i - 1) for i in range(c)]) * v for j in
                            range(self.num_candidates)]
            borda_vector = sorted(borda_vector, reverse=True)

        return borda_vector, len(borda_vector)

    def votes_to_agg_voterlikeness_vector(self):
        """ convert VOTES to Borda vector """

        vector = np.zeros([self.num_voters])

        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):

                swap_distance = 0
                for i in range(self.num_candidates):
                    for j in range(i + 1, self.num_candidates):
                        if (self.potes[v1][i] > self.potes[v1][j] and self.potes[v2][i] < self.potes[v2][j]) or \
                                (self.potes[v1][i] < self.potes[v1][j] and self.potes[v2][i] > self.potes[v2][j]):
                            swap_distance += 1
                vector[v1] += swap_distance

        return vector, len(vector)

    def votes_to_bordawise_vector_long_empty(self):

        num_possible_scores = 1 + self.num_voters * (self.num_candidates - 1)

        if self.votes[0][0] == -1:

            vector = [0 for _ in range(num_possible_scores)]
            peak = sum([i for i in range(self.num_candidates)]) * float(self.num_voters) / float(self.num_candidates)
            vector[int(peak)] = self.num_candidates

        else:

            vector = [0 for _ in range(num_possible_scores)]
            points = get_borda_points(self.votes, self.num_voters, self.num_candidates)
            for i in range(self.num_candidates):
                vector[points[i]] += 1

        return vector, num_possible_scores


def check_if_fake(experiment_id, election_id):
    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    line = my_file.readline().strip()
    if line[0] == '$':
        return True
    return False


def get_borda_points(votes, num_voters, num_candidates):
    points = [0 for _ in range(num_candidates)]
    scoring = [1. for _ in range(num_candidates)]

    for i in range(len(scoring)):
        scoring[i] = len(scoring) - i - 1

    for i in range(num_voters):
        for j in range(num_candidates):
            points[int(votes[i][j])] += scoring[j]

    return points


def import_fake_elections(experiment_id, election_id):
    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    my_file.readline()  # line with $ fake

    num_voters = int(my_file.readline().strip())
    num_candidates = int(my_file.readline().strip())
    fake_model_name = str(my_file.readline().strip())
    if fake_model_name == 'crate':
        fake_param = [float(my_file.readline().strip()), float(my_file.readline().strip()),
                      float(my_file.readline().strip()), float(my_file.readline().strip())]
    elif fake_model_name == 'norm-mallows_matrix':
        fake_param = [float(my_file.readline().strip()), float(my_file.readline().strip())]
    else:
        fake_param = float(my_file.readline().strip())

    return fake_model_name, fake_param, num_voters, num_candidates


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

    # elif fake_model_name == 'walsh_fake':
    #     vectors = _sp.walsh(num_candidates)
    #
    # elif fake_model_name == 'conitzer_fake':
    #     vectors = _sp.conitzer(num_candidates)

    return vectors


def get_fake_matrix_single(fake_model_name, num_candidates, num_voters):
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


def get_fake_vectors_crate(fake_model_name, num_candidates, num_voters, fake_param):
    base_1 = get_fake_vectors_single('uniformity', num_candidates, num_voters)
    base_2 = get_fake_vectors_single('identity', num_candidates, num_voters)
    base_3 = get_fake_vectors_single('antagonism', num_candidates, num_voters)
    base_4 = get_fake_vectors_single('stratification', num_candidates, num_voters)

    return crate_combination(base_1, base_2, base_3, base_4, length=num_candidates, alpha=fake_param)


def get_fake_convex(fake_model_name, num_candidates, num_voters, fake_param, function_name):
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

    return convex_combination(base_1, base_2, length=num_candidates, alpha=fake_param)


def convex_combination(base_1, base_2, length=0, alpha=0):
    alpha = alpha['param_1']
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
            vectors[i][j] = alpha[0] * base_1[i][j] + alpha[1] * base_2[i][j] + \
                            alpha[2] * base_3[i][j] + alpha[3] * base_4[i][j]

    return vectors


def import_soc_elections(experiment_id, election_id):
    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')

    param = 0
    first_line = my_file.readline()
    if first_line[0] != '#':
        model_name = 'empty'
        num_candidates = int(first_line)
    else:
        first_line = first_line.strip().split()
        model_name = first_line[1]
        if any(map(str.isdigit, first_line[len(first_line) - 1])):
            param = first_line[len(first_line) - 1]
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
            for l in range(num_candidates):
                votes[it][l] = int(line[l + 1])
            it += 1

    # Shift by -1
    if model_name in LIST_OF_PREFLIB_MODELS:
        for i in range(num_voters):
            for j in range(num_candidates):
                votes[i][j] -= 1

    return votes, num_voters, num_candidates, param, model_name
