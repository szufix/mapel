#!/usr/bin/env python

from matplotlib import pyplot as plt
from mapel.elections.cultures_ import generate_approval_votes
from mapel.elections.objects.Election import Election
from mapel.core.inner_distances import hamming
from mapel.core.utils import *
import mapel.elections.persistence.election_imports as imports
import mapel.elections.persistence.election_exports as exports
from mapel.elections.cultures.params import *


class ApprovalElection(Election):

    def __init__(self,
                 experiment_id,
                 election_id,
                 culture_id=None,
                 num_candidates=None,
                 is_imported: bool = False,
                 is_shifted: bool = False,
                 params=None,
                 variable=None,
                 **kwargs):

        super().__init__(experiment_id,
                         election_id,
                         culture_id=culture_id,
                         num_candidates=num_candidates,
                         **kwargs)

        self.params = params
        self.variable = variable

        self.approvalwise_vector = []
        self.coapproval_frequency_vectors = []
        self.voterlikeness_vectors = []
        self.pairwise_matrix = []
        self.candidatelikeness_original_vectors = []
        self.candidatelikeness_sorted_vectors = []
        self.hamming_candidates = []
        self.reverse_approvals = []

        if is_imported and experiment_id != 'virtual':
            try:
                fake = imports.check_if_fake(experiment_id, election_id, 'app')
                if fake:
                    self.culture_id, self.params, self.num_voters, self.num_candidates = \
                        imports.import_fake_app_election(experiment_id, election_id)
                else:
                    self.votes, self.num_voters, self.num_candidates, self.params, \
                    self.culture_id = imports.import_real_app_election(experiment_id,
                                                                       election_id,
                                                                       is_shifted)
            except:
                pass

        if self.params is None:
            self.params = {}

        if culture_id is not None:
            self.params, self.printing_params = update_params_approval(self.params,
                                                                                   self.printing_params,
                                                                                   self.variable,
                                                                                   self.culture_id,
                                                                                   self.num_candidates)

    def votes_to_approvalwise_vector(self) -> None:
        """ Convert votes to approvalwise vectors """
        approvalwise_vector = np.zeros([self.num_candidates])
        for vote in self.votes:
            for c in vote:
                approvalwise_vector[c] += 1
        approvalwise_vector = approvalwise_vector / self.num_voters
        self.approvalwise_vector = np.sort(approvalwise_vector)

    def votes_to_coapproval_frequency_vectors(self, vector_type='A') -> None:
        """ Convert votes to frequency vectors """
        vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
        for vote in self.votes:
            size = len(vote)
            for c in range(self.num_candidates):
                if c in vote:
                    if vector_type in ['A', 'B']:
                        vectors[c][size - 1] += 1
                    elif vector_type == 'C':
                        vectors[c][2 * size - 1] += 1
                    elif vector_type in ['D', 'E']:
                        vectors[c][self.num_candidates + size - 1] += 1
                else:
                    if vector_type == 'A':
                        vectors[c][self.num_candidates + size] += 1
                    elif vector_type == 'B':
                        vectors[c][2 * self.num_candidates - size - 1] += 1
                    elif vector_type == 'C':
                        vectors[c][2 * size] += 1
                    elif vector_type == 'D':
                        vectors[c][size] += 1
                    elif vector_type == 'E':
                        vectors[c][self.num_candidates - size - 1] += 1
        vectors = vectors / self.num_voters
        self.coapproval_frequency_vectors = vectors

    def votes_to_pairwise_matrix(self) -> None:
        """ Convert votes to pairwise matrix """
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        for c_1 in range(self.num_candidates):
            for c_2 in range(self.num_candidates):
                for vote in self.votes:
                    if (c_1 in vote and c_2 in vote) or (c_1 not in vote and c_2 not in vote):
                        matrix[c_1][c_2] += 1
        matrix = matrix / self.num_voters
        self.pairwise_matrix = matrix

    def votes_to_candidatelikeness_original_vectors(self) -> None:
        """ Convert votes to ... """
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        for c_1 in range(self.num_candidates):
            for c_2 in range(self.num_candidates):
                for vote in self.votes:
                    if (c_1 in vote and c_2 not in vote) or (c_1 not in vote and c_2 in vote):
                        matrix[c_1][c_2] += 1
        matrix = matrix / self.num_voters
        self.candidatelikeness_original_vectors = matrix

    def votes_to_candidatelikeness_sorted_vectors(self) -> None:
        """ Convert votes to ... """
        self.votes_to_candidatelikeness_original_vectors()
        self.candidatelikeness_sorted_vectors = np.sort(self.candidatelikeness_original_vectors)

    def votes_to_voterlikeness_vectors(self, vector_type='hamming') -> None:
        """ Convert votes to ... """

        vectors = np.zeros([self.num_voters, self.num_voters])

        for i in range(self.num_voters):
            for j in range(self.num_voters):
                set_a = self.votes[i]
                set_b = self.votes[j]
                if vector_type == 'hamming':
                    vectors[i][j] = hamming(set_a, set_b)
                elif vector_type == 'to_be_deleted':
                    vectors[i][j] = len(set_a.intersection(set_b)) - len(set_a)
            vectors[i] = sorted(vectors[i])

        self.voterlikeness_vectors = vectors

    def compute_reverse_approvals(self):
        reverse_approvals = [set() for _ in range(self.num_candidates)]
        for i, vote in enumerate(self.votes):
            for c in vote:
                reverse_approvals[int(c)].add(i)

        self.reverse_approvals = reverse_approvals

    def prepare_instance(self, is_exported=None, is_aggregated=True):

        self.votes = generate_approval_votes(culture_id=self.culture_id,
                                             num_candidates=self.num_candidates,
                                             num_voters=self.num_voters,
                                             params=self.params)
        if is_exported:
            exports.export_approval_election(self, is_aggregated=is_aggregated)

    def compute_distances_between_votes(self, distance_id='hamming'):
        distances = np.zeros([self.num_voters, self.num_voters])
        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):
                if distance_id == 'hamming':
                    distances[v1][v2] = hamming(self.votes[v1], self.votes[v2])
                elif distance_id == 'jaccard':
                    if len(self.votes[v1].union(self.votes[v2])) == 0:
                        distances[v1][v2] = 1
                    else:
                        distances[v1][v2] = 1 - len(
                            self.votes[v1].intersection(self.votes[v2])) / len(
                            self.votes[v1].union(self.votes[v2]))

        self.distances['vote'] = distances

        if self.is_exported:
            exports.export_distances(self, object_type='vote')

        return distances

    def compute_distances_between_candidates(self, distance_id='hamming'):
        #
        distances = np.zeros([self.num_candidates, self.num_candidates])
        for c1 in range(self.num_candidates):
            for c2 in range(self.num_candidates):
                if distance_id == 'hamming':
                    distances[c1][c2] = hamming(self.reverse_approvals[c1],
                                                self.reverse_approvals[c2])
                elif distance_id == 'jaccard':
                    if len(self.reverse_approvals[c1].union(self.reverse_approvals[c2])) == 0:
                        distances[c1][c2] = 1
                    else:
                        distances[c1][c2] = 1 - len(
                            self.reverse_approvals[c1].intersection(self.reverse_approvals[c2])) \
                                            / len(
                            self.reverse_approvals[c1].union(self.reverse_approvals[c2]))

        self.distances['candidate'] = distances

        if self.is_exported:
            self.export_distances(object_type='candidate')

    def print_map(self, show=True, radius=None, name=None, alpha=0.1, s=30, circles=False,
                  object_type=None, double_gradient=False, saveas=None, color='blue',
                  marker='o', title_size=20, annotate=False):

        if object_type == 'vote':
            length = self.num_voters
            votes = self.votes
        elif object_type == 'candidate':
            length = self.num_candidates
            self.compute_reverse_approvals()
            votes = self.reverse_approvals

        if object_type is None:
            object_type = self.object_type

        plt.figure(figsize=(6.4, 6.4))
        # = plt.subplots()

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

        if circles:  # works only for votes
            weighted_points = {}
            Xs = {}
            Ys = {}
            for i in range(length):
                str_elem = str(votes[i])
                if str_elem in weighted_points:
                    weighted_points[str_elem] += 1
                else:
                    weighted_points[str_elem] = 1
                    Xs[str_elem] = X[i]
                    Ys[str_elem] = Y[i]

            for str_elem in weighted_points:
                if object_type == 'vote':
                    if weighted_points[str_elem] > 10 and str_elem != 'set()':
                        print(str_elem)
                        plt.scatter(Xs[str_elem], Ys[str_elem],
                                    color='purple',
                                    s=10 * weighted_points[str_elem],
                                    alpha=0.2)
                if object_type == 'candidate':
                    if weighted_points[str_elem] > 5 and str_elem != 'set()':
                        print(str_elem)
                        plt.scatter(Xs[str_elem], Ys[str_elem],
                                    color='purple',
                                    s=50 * weighted_points[str_elem],
                                    alpha=0.2)

        if double_gradient:
            for i in range(length):
                x = float(self.points['voters'][i][0])
                y = float(self.points['voters'][i][1])
                plt.scatter(X[i], Y[i], color=[0, y, x], s=s, alpha=alpha)
        else:
            plt.scatter(X, Y, color=color, s=s, alpha=alpha, marker=marker)

        if annotate:
            for i in range(len(X)):
                plt.annotate(i, (X[i], Y[i]), color='black')

        avg_x = np.mean(X)
        avg_y = np.mean(Y)

        if radius:
            plt.xlim([avg_x - radius, avg_x + radius])
            plt.ylim([avg_y - radius, avg_y + radius])
        # plt.title(election.label, size=38)
        plt.title(self.texify_label(self.label), size=title_size)
        plt.axis('off')

        if saveas is None:
            saveas = f'{self.label}'

        file_name = os.path.join(os.getcwd(), "images", name, f'{saveas}.png')
        plt.savefig(file_name, bbox_inches='tight', dpi=100)
        if show:
            plt.show()
        else:
            plt.clf()
