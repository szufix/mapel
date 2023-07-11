import logging
from collections import Counter

from matplotlib import pyplot as plt

import csv

from mapel.elections.cultures.group_separable import get_gs_caterpillar_vectors
from mapel.elections.cultures.preflib import get_sushi_vectors
from mapel.elections.cultures.single_crossing import get_single_crossing_vectors
from mapel.elections.cultures_ import generate_ordinal_votes, \
    from_approval, generate_ordinal_alliance_votes
from mapel.elections.objects.Election import Election
from mapel.elections.other.winners import compute_sntv_winners, compute_borda_winners, \
    compute_stv_winners
from mapel.elections.other.winners import generate_winners
from mapel.core.inner_distances import swap_distance_between_potes, \
    spearman_distance_between_potes
from mapel.elections.features.other import is_condorcet
from mapel.core.utils import *
import mapel.elections.persistence.election_exports as exports
import mapel.elections.persistence.election_imports as imports
from mapel.elections.cultures.fake import *
from mapel.elections.other.winners import get_borda_points


class OrdinalElection(Election):

    def __init__(self,
                 experiment_id=None,
                 election_id=None,
                 culture_id=None,
                 votes=None,
                 params=None,
                 label=None,
                 num_voters: int = None,
                 num_candidates: int = None,
                 variable=None,
                 fast_import=False,
                 **kwargs):

        super().__init__(experiment_id=experiment_id,
                         election_id=election_id,
                         culture_id=culture_id,
                         votes=votes,
                         label=label,
                         num_voters=num_voters,
                         num_candidates=num_candidates,
                         fast_import=fast_import,
                         **kwargs)

        self.params = params
        self.variable = variable
        self.vectors = []
        self.matrix = []
        self.potes = None
        self.condorcet = None
        self.points = {}
        self.alliances = {}

        self.import_ordinal_election()

    def import_ordinal_election(self):
        if self.is_imported and self.experiment_id != 'virtual':
            try:
                if self.votes is not None:
                    self.culture_id = self.culture_id
                    if str(self.votes[0]) in LIST_OF_FAKE_MODELS:
                        self.fake = True
                        self.votes = self.votes[0]
                        self.num_candidates = self.votes[1]
                        self.num_voters = self.votes[2]
                    else:
                        self.votes = self.votes
                        self.num_candidates = len(self.votes[0])
                        self.num_voters = len(self.votes)
                        self.compute_potes()
                else:
                    self.fake = imports.check_if_fake(self.experiment_id, self.election_id, 'soc')
                    if self.fake:
                        self.culture_id, self.params, self.num_voters, \
                        self.num_candidates = imports.import_fake_soc_election(self.experiment_id,
                                                                               self.election_id)
                    else:

                        self.votes, self.num_voters, self.num_candidates, self.params, \
                        self.culture_id, self.alliances, \
                        self.num_options, self.quantites = imports.import_real_soc_election(
                            self.experiment_id, self.election_id, self.is_shifted)

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

                self.candidatelikeness_original_vectors = {}

                if not self.fast_import:
                    self.votes_to_positionwise_vectors()
            except:
                self.is_correct = False
                pass

        if self.params is None:
            self.params = {}

        try:
            self.params, self.printing_params = update_params_ordinal(
                self.params,
                self.printing_params,
                self.variable,
                self.culture_id,
                self.num_candidates)
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
                matrix = get_fake_convex(self.culture_id,
                                         self.num_candidates,
                                         self.num_voters,
                                         self.fake_param,
                                         get_fake_matrix_single)

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
                borda_vector = get_fake_convex(self.culture_id,
                                               self.num_candidates,
                                               self.num_voters,
                                               self.params,
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

    def votes_to_voterlikeness_matrix(self, vote_distance='swap') -> np.ndarray:
        """ convert VOTES to voter-likeness MATRIX """
        matrix = np.zeros([self.num_voters, self.num_voters])
        self.compute_potes()

        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):
                if vote_distance == 'swap':
                    matrix[v1][v2] = swap_distance_between_potes(self.potes[v1], self.potes[v2])
                elif vote_distance == 'spearman':
                    matrix[v1][v2] = spearman_distance_between_potes(self.potes[v1], self.potes[v2])

        for i in range(self.num_voters):
            for j in range(i + 1, self.num_voters):
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

    def prepare_instance(self, is_exported=None, is_aggregated=True):
        if 'num_alliances' in self.params:
            self.votes, self.alliances = generate_ordinal_alliance_votes(
                culture_id=self.culture_id,
                num_candidates=self.num_candidates,
                num_voters=self.num_voters,
                params=self.params)
        else:
            self.votes = generate_ordinal_votes(culture_id=self.culture_id,
                                                num_candidates=self.num_candidates,
                                                num_voters=self.num_voters,
                                                params=self.params)

        c = Counter(map(tuple, self.votes))
        counted_votes = [[count, list(row)] for row, count in c.items()]
        counted_votes = sorted(counted_votes, reverse=True)
        self.quantites = [a[0] for a in counted_votes]
        self.num_options = len(counted_votes)

        if is_exported:
            exports.export_ordinal_election(self, is_aggregated=is_aggregated)

    def get_distinct_votes(self):
        import itertools
        votes = self.votes
        votes = votes.tolist()
        votes.sort()
        return list(k for k, _ in itertools.groupby(votes))

    def compute_distances(self, distance_id='swap', object_type=None):
        """ Return: distances between votes """
        if object_type is None:
            object_type = self.object_type

        self.distinct_votes = self.get_distinct_votes()
        self.distinct_potes = convert_votes_to_potes(self.distinct_votes)
        self.num_dist_votes = len(self.distinct_votes)

        if object_type == 'vote':
            distances = np.zeros([self.num_dist_votes, self.num_dist_votes])
            for v1 in range(self.num_dist_votes):
                for v2 in range(self.num_dist_votes):
                    if distance_id == 'swap':
                        distances[v1][v2] = swap_distance_between_potes(
                            self.distinct_potes[v1], self.distinct_potes[v2])
                    elif distance_id == 'spearman':
                        distances[v1][v2] = spearman_distance_between_potes(
                            self.distinct_potes[v1], self.distinct_potes[v2])
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
        else:
            logging.warning('incorrect object_type')

        self.distances[object_type] = distances

        if object_type == 'vote':
            length = self.num_dist_votes
        elif object_type == 'candidate':
            length = self.num_candidates

        if self.is_exported:
            exports.export_distances(self, object_type=object_type, length=length)

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

    @staticmethod
    def texify_label(name):
        return name.replace('phi', '$\phi$'). \
            replace('alpha', '$\\ \\alpha$'). \
            replace('omega', '$\\ \\omega$'). \
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
        else:
            logging.warning(f'Incorrect object type: {object_type}')

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

        if object_type == 'vote':
            if double_gradient:
                for i in range(length):
                    x = float(self.points['voters'][i][0])
                    y = float(self.points['voters'][i][1])
                    plt.scatter(X[i], Y[i], color=[0, y, x], s=s, alpha=alpha)
            else:
                for i in range(len(X)):
                    plt.scatter(X[i], Y[i], color=color, alpha=alpha, marker=marker,
                                s=self.quantites[i] * s)

        elif object_type == 'candidate':
            for i in range(len(X)):
                plt.scatter(X[i], Y[i], color=color, alpha=alpha, marker=marker,
                            s=s)

        #
        # if circles:  # works only for votes
        #     weighted_points = {}
        #     Xs = {}
        #     Ys = {}
        #     for i in range(length):
        #         str_elem = str(experiment_id.votes[i])
        #         if str_elem in weighted_points:
        #             weighted_points[str_elem] += 1
        #         else:
        #             weighted_points[str_elem] = 1
        #             Xs[str_elem] = X[i]
        #             Ys[str_elem] = Y[i]
        #     for str_elem in weighted_points:
        #         if weighted_points[str_elem] > 10 and str_elem != 'set()':
        #             plt.scatter(Xs[str_elem], Ys[str_elem],
        #                         color='purple',
        #                         s=10 * weighted_points[str_elem],
        #                         alpha=0.2)

        avg_x = np.mean(X)
        avg_y = np.mean(Y)

        if radius:
            plt.xlim([avg_x - radius, avg_x + radius])
            plt.ylim([avg_y - radius, avg_y + radius])
        # plt.title(election.label, size=38)

        plt.title(self.texify_label(self.label), size=title_size)

        # plt.title(election.texify_label(election.label), size=38, y=0.94)
        # plt.title(election.label, size=title_size)
        plt.axis('off')

        if saveas is None:
            saveas = f'{self.label}_{object_type}'

        file_name = os.path.join(os.getcwd(), "images", name, f'{saveas}.png')
        plt.savefig(file_name, bbox_inches='tight', dpi=100)
        if show:
            plt.show()
        else:
            plt.clf()


def convert_votes_to_potes(votes):
    """ Convert votes to positional votes (called potes) """
    return np.array([[list(vote).index(i) for i, _ in enumerate(vote)]
                     for vote in votes])
