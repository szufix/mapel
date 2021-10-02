#!/usr/bin/env python

import numpy as np

from mapel.voting.elections.group_separable import get_gs_caterpillar_vectors
from mapel.voting.elections.single_peaked import get_walsh_vectors, get_conitzer_vectors
from mapel.voting.elections.single_crossing import get_single_crossing_vectors
from mapel.voting.elections.mallows import get_mallows_vectors
from mapel.voting.elections.preflib import get_sushi_vectors

from mapel.voting.winners import compute_sntv_winners, compute_borda_winners, compute_stv_winners
from mapel.voting.glossary import PATHS
from mapel.voting.not_in_the_package.__winners import generate_winners
from mapel.voting.objects.Election import Election


class OrdinalElection(Election):

    def __init__(self, experiment_id, name, votes=None, with_matrix=False, alpha=None, model=None,
                 ballot='ordinal', num_voters=None, num_candidates=None):

        super().__init__(experiment_id, name, votes=votes, with_matrix=with_matrix, alpha=alpha,
                         model=model, ballot=ballot,
                         num_voters=num_voters, num_candidates=num_candidates)

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

        if self.election_model == 'conitzer_matrix':
            vectors = get_conitzer_vectors(self.num_candidates)
        elif self.election_model == 'walsh_matrix':
            vectors = get_walsh_vectors(self.num_candidates)
        elif self.election_model == 'single-crossing_matrix':
            vectors = get_single_crossing_vectors(self.num_candidates)
        elif self.election_model == 'gs_caterpillar_matrix':
            vectors = get_gs_caterpillar_vectors(self.num_candidates)
        elif self.election_model == 'sushi_matrix':
            vectors = get_sushi_vectors()
        elif self.election_model in {'norm-mallows_matrix', 'mallows_matrix_path'}:
            vectors = get_mallows_vectors(self.num_candidates, self.fake_param)
        elif self.election_model in {'identity', 'uniformity', 'antagonism', 'stratification'}:
            vectors = get_fake_vectors_single(self.election_model, self.num_candidates,
                                              self.num_voters)
        elif self.election_model in {'walsh_path', 'conitzer_path'}:
            vectors = get_fake_multiplication(self.num_candidates, self.fake_param, self.election_model)
        elif self.election_model in PATHS:
            vectors = get_fake_convex(self.election_model, self.num_candidates, self.num_voters,
                                      self.fake_param, get_fake_vectors_single)
        elif self.election_model == 'crate':
            vectors = get_fake_vectors_crate(self.election_model, self.num_candidates,
                                             self.num_voters, self.fake_param)
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

    def votes_to_pairwise_matrix(self):
        """ convert VOTES to pairwise MATRIX """
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        if self.fake:

            if self.election_model in {'identity', 'uniformity',
                                       'antagonism', 'stratification'}:
                matrix = get_fake_matrix_single(self.election_model, self.num_candidates,
                                                self.num_voters)
            elif self.election_model in PATHS:
                matrix = get_fake_convex(self.election_model, self.num_candidates, self.num_voters,
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

    def votes_to_bordawise_vector(self):
        """ convert VOTES to Borda vector """

        borda_vector = np.zeros([self.num_candidates])

        if self.fake:

            if self.election_model in {'identity', 'uniformity',
                                       'antagonism', 'stratification'}:
                borda_vector = get_fake_borda_vector(self.election_model, self.num_candidates,
                                                     self.num_voters)
            elif self.election_model in PATHS:
                borda_vector = get_fake_convex(self.election_model, self.num_candidates,
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

    def votes_to_positionwise_intervals(self, precision=None):

        vectors = self.votes_to_positionwise_matrix()
        intervals = []

        for i in range(len(vectors)):
            intervals.append(self.vector_to_interval(vectors[i], precision=precision))

        return intervals

    def votes_to_voterlikeness_matrix(self):
        """ convert VOTES to voter-likeness MATRIX """
        matrix = np.zeros([self.num_voters, self.num_voters])

        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):
                # Spearman distance between votes
                # matrix[v1][v2] = sum([abs(self.potes[v1][c]
                # - self.potes[v2][c]) for c in range(self.num_candidates)])

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
                # matrix[i][j] /= float(self.num_candidates)
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
            peak = sum([i for i in range(self.num_candidates)]) * \
                       float(self.num_voters) / float(self.num_candidates)
            vector[int(peak)] = self.num_candidates

        else:

            vector = [0 for _ in range(num_possible_scores)]
            points = get_borda_points(self.votes, self.num_voters,
                                      self.num_candidates)
            for i in range(self.num_candidates):
                vector[points[i]] += 1

        return vector, num_possible_scores

    def votes_to_viper_vectors(self):

        vectors = [[0. for _ in range(self.num_voters)]
                   for _ in range(self.num_voters)]

        c = self.num_candidates
        v = self.num_voters
        borda_vector = [sum([vectors[j][i] * (c - i - 1)
                             for i in range(c)]) * v
                        for j in range(self.num_candidates)]

        for i in range(self.num_candidates):
            for j in range(self.num_candidates):
                vectors[i][j] /= float(self.num_voters)

        return vectors

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


def get_fake_multiplication(num_candidates, params, election_model):
    params['weight'] = 0.
    params['norm-phi'] = params['alpha']
    main_matrix = []
    if election_model == 'conitzer_path':
        main_matrix = get_conitzer_vectors(num_candidates).transpose()
    elif election_model == 'walsh_path':
        main_matrix = get_walsh_vectors(num_candidates).transpose()
    mallows_matrix = get_mallows_vectors(num_candidates, params).transpose()
    # mallows_matrix = np.linalg.inv(mallows_matrix) # JUST FOR FUN
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

    # elif fake_model_name == 'walsh_fake':
    #     vectors = _sp.walsh(num_candidates)
    #
    # elif fake_model_name == 'conitzer_fake':
    #     vectors = _sp.conitzer(num_candidates)

    return vectors


def get_fake_vectors_crate(fake_model_name, num_candidates, num_voters, fake_param):
    base_1 = get_fake_vectors_single('uniformity', num_candidates, num_voters)
    base_2 = get_fake_vectors_single('identity', num_candidates, num_voters)
    base_3 = get_fake_vectors_single('antagonism', num_candidates, num_voters)
    base_4 = get_fake_vectors_single('stratification', num_candidates, num_voters)

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
                output[i][j] = alpha * base_1[i][j] + (1 - alpha) * base_2[i][
                    j]
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


def get_borda_points(votes, num_voters, num_candidates):
    points = [0 for _ in range(num_candidates)]
    scoring = [1. for _ in range(num_candidates)]

    for i in range(len(scoring)):
        scoring[i] = len(scoring) - i - 1

    for i in range(num_voters):
        for j in range(num_candidates):
            points[int(votes[i][j])] += scoring[j]

    return points
