#!/usr/bin/env python

import ast
import os

import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import gamma

import mapel.elections.cultures.mallows as mallows
from mapel.core.glossary import *
from mapel.elections.cultures_ import generate_approval_votes, store_votes_in_a_file
from mapel.elections.objects.Election import Election
from mapel.core.inner_distances import hamming
from mapel.core.utils import *
import time

class ApprovalElection(Election):

    def __init__(self, experiment_id, election_id, votes=None, alpha=1, culture_id=None,
                 ballot='approval', num_voters=None, num_candidates=None, _import=False,
                 shift: bool = False, params=None, variable=None, label=None, fast_import=False):

        super().__init__(experiment_id, election_id, votes=votes, alpha=alpha,
                         culture_id=culture_id, ballot=ballot, num_voters=num_voters,
                         num_candidates=num_candidates, label=label, fast_import=fast_import)
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

        if _import and experiment_id != 'virtual':
            try:
                fake = check_if_fake(experiment_id, election_id)
                if fake:
                    self.culture_id, self.params, self.num_voters, self.num_candidates = \
                        import_fake_app_election(experiment_id, election_id)
                else:
                    self.votes, self.num_voters, self.num_candidates, self.params, \
                    self.culture_id = import_real_app_election(experiment_id, election_id, shift)
                    try:
                        self.alpha = self.params['alpha']
                    except:
                        self.alpha = 1
                        pass
            except:
                pass

        if self.params is None:
            self.params = {}

        if culture_id is not None:
            self.params, self.alpha = update_params_approval(self.params, self.variable, self.culture_id,
                                                            self.num_candidates)


    def votes_to_approvalwise_vector(self) -> None:
        """ Convert votes to approvalwise vectors """

        if self.culture_id == 'approval_half_1':
            self.approvalwise_vector = np.sort(np.array([0.75 for _ in
                                                         range(int(self.num_candidates / 2))] +
                                                        [0.25 for _ in
                                                         range(int(self.num_candidates / 2))]))
        elif self.culture_id == 'approval_half_2':
            self.approvalwise_vector = np.sort(np.array([i / (self.num_candidates - 1) for i in
                                                         range(self.num_candidates)]))
        elif self.culture_id == 'approval_skeleton':
            self.approvalwise_vector = np.sort(get_skeleton_approvalwise_vector(self))

        else:
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
        # vectors = vectors / experiment.num_candidates
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

    # PREPARE INSTANCE
    def prepare_instance(self, store=None, aggregated=True):

        self.votes = generate_approval_votes(culture_id=self.culture_id,
                                             num_candidates=self.num_candidates,
                                             num_voters=self.num_voters,
                                             params=self.params)
        avg = 0
        for vote in self.votes:
            avg += len(vote)

        if store:
            self.store_approval_election(aggregated=aggregated)

    # STORE
    def store_approval_election(self, aggregated=True):
        """ Store approval election in an .app file """

        path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections")
        make_folder_if_do_not_exist(path_to_folder)
        path_to_file = os.path.join(path_to_folder, f'{self.election_id}.app')

        if self.culture_id in APPROVAL_FAKE_MODELS:
            file_ = open(path_to_file, 'w')
            file_.write(f'$ {self.culture_id} {self.params} \n')
            file_.write(str(self.num_candidates) + '\n')
            file_.write(str(self.num_voters) + '\n')
            file_.close()

        else:
            store_votes_in_a_file(self, self.culture_id, self.num_candidates, self.num_voters,
                                  self.params, path_to_file, self.ballot, votes=self.votes,
                                  aggregated=aggregated)

    def compute_distances_between_votes(self, distance_id='hamming'):
        print(self.num_voters)
        distances = np.zeros([self.num_voters, self.num_voters])
        for v1 in range(self.num_voters):
            for v2 in range(self.num_voters):
                if distance_id == 'hamming':
                    distances[v1][v2] = hamming(self.votes[v1], self.votes[v2])
                elif distance_id == 'jaccard':
                    if len(self.votes[v1].union(self.votes[v2])) == 0:
                        distances[v1][v2] = 1
                    else:
                        distances[v1][v2] = 1 - len(self.votes[v1].intersection(self.votes[v2])) / len(self.votes[v1].union(self.votes[v2]))

        self.distances['vote'] = distances

        if self.store:
            self._store_distances(object_type='vote')

        return distances

    def compute_distances_between_candidates(self, distance_id='hamming'):
        #
        distances = np.zeros([self.num_candidates, self.num_candidates])
        for c1 in range(self.num_candidates):
            for c2 in range(self.num_candidates):
                if distance_id == 'hamming':
                    distances[c1][c2] = hamming(self.reverse_approvals[c1], self.reverse_approvals[c2])
                elif distance_id == 'jaccard':
                    if len(self.reverse_approvals[c1].union(self.reverse_approvals[c2])) == 0:
                        distances[c1][c2] = 1
                    else:
                        distances[c1][c2] = 1 - len(self.reverse_approvals[c1].intersection(self.reverse_approvals[c2])) \
                                            / len(self.reverse_approvals[c1].union(self.reverse_approvals[c2]))

        self.distances['candidate'] = distances

        if self.store:
            self._store_distances(object_type='candidate')

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

        if circles:     # works only for votes
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
                    if weighted_points[str_elem] > 10 and str_elem!='set()':
                        print(str_elem)
                        plt.scatter(Xs[str_elem], Ys[str_elem],
                                    color='purple',
                                    s=10 * weighted_points[str_elem],
                                    alpha=0.2)
                if object_type == 'candidate':
                    if weighted_points[str_elem] > 5 and str_elem!='set()':
                        print(str_elem)
                        plt.scatter(Xs[str_elem], Ys[str_elem],
                                    color='purple',
                                    s=50 * weighted_points[str_elem],
                                    alpha=0.2)


        # print(len(weighted_points))

        if double_gradient:
            for i in range(length):
                x = float(self.points['voters'][i][0])
                y = float(self.points['voters'][i][1])
                plt.scatter(X[i], Y[i], color=[0,y,x], s=s, alpha=alpha)
        else:
            plt.scatter(X, Y, color=color, s=s, alpha=alpha, marker=marker)

        if annotate:
            for i in range(len(X)):
                plt.annotate(i, (X[i], Y[i]), color='black')

        avg_x = np.mean(X)
        avg_y = np.mean(Y)

        if radius:
            plt.xlim([avg_x-radius, avg_x+radius])
            plt.ylim([avg_y-radius, avg_y+radius])
        # plt.title(self.label, size=38)
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

def import_real_app_election(experiment_id: str, election_id: str, shift=False):
    """ Import real approval election from .app file """

    file_name = f'{election_id}.app'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')

    params = 0
    first_line = my_file.readline()
    if first_line[0] != '#':
        culture_id = 'empty'
        num_candidates = int(first_line)
    else:
        first_line = first_line.strip().split()
        culture_id = first_line[1]
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
    votes = [set() for _ in range(num_voters)]

    it = 0
    for j in range(num_options):
        line = my_file.readline().rstrip("\n").replace("{", ''). \
            replace("}", '').replace(' ', '').split(',')
        if line[1] != '':
            quantity = int(line[0])
            for k in range(quantity):
                for el in range(len(line) - 1):
                    votes[it].add(int(line[el + 1]))
                it += 1

    if culture_id in NICE_NAME.values():
        rev_dict = dict(zip(NICE_NAME.values(), NICE_NAME.keys()))
        culture_id = rev_dict[culture_id]

    if shift:
        for i, vote in enumerate(votes):
            new_vote = set()
            for c in vote:
                new_vote.add(c - 1)
            votes[i] = new_vote
    my_file.close()

    return votes, num_voters, num_candidates, params, culture_id


def import_fake_app_election(experiment_id: str, name: str):
    """ Import fake approval election from .app file """

    file_name = f'{name}.app'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    first_line = my_file.readline()
    first_line = first_line.strip().split()
    fake_model_name = first_line[1]
    if len(first_line) <= 2:
        params = {}
    else:
        params = ast.literal_eval(" ".join(first_line[2:]))

    num_candidates = int(my_file.readline().strip())
    num_voters = int(my_file.readline().strip())

    return fake_model_name, params, num_voters, num_candidates


def check_if_fake(experiment_id: str, election_id: str) -> bool:
    file_name = f'{election_id}.app'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    line = my_file.readline().strip()
    my_file.close()
    return line[0] == '$'


def get_skeleton_approvalwise_vector(election):
    phi = election.params['phi']
    p = election.params['p']
    k = int(p * election.num_candidates)

    vector = [phi * p for _ in range(election.num_candidates)]
    for i in range(k):
        vector[i] += 1 - phi

    return np.array(vector)


# HELPER FUNCTIONS
def update_params_approval_alpha(params):
    if 'alpha' not in params:
        params['alpha'] = 1
    elif type(params['alpha']) is list:
        params['alpha'] = np.random.uniform(low=params['alpha'][0], high=params['alpha'][1])


def update_params_approval_p(params):
    if 'p' not in params:
        params['p'] = np.random.rand()
    elif type(params['p']) is list:
        params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])
        print(params)


def update_params_approval_resampling(params):
    if 'phi' in params and type(params['phi']) is list:
        params['phi'] = np.random.uniform(low=params['phi'][0], high=params['phi'][1])
    elif 'phi' not in params:
        params['phi'] = np.random.random()
    params['alpha'] = params['phi']

    if 'p' in params and type(params['p']) is list:
        params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])
    elif 'p' not in params:
        params['p'] = np.random.random()


def update_params_approval_disjoint(params):
    if 'phi' in params and type(params['phi']) is list:
        params['phi'] = np.random.uniform(low=params['phi'][0], high=params['phi'][1])
    elif 'phi' not in params:
        params['phi'] = np.random.random()
    params['alpha'] = params['phi']

    if 'p' in params and type(params['p']) is list:
        params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])
    elif 'p' not in params:
        params['p'] = np.random.random() / params['g']


def update_params_approval(params, variable, culture_id, num_candidates):

    if variable is not None:
        if culture_id in APPROVAL_MODELS:
            update_params_approval_p(params)
        params['alpha'] = params[variable]
        params['variable'] = variable
    else:
        if culture_id in APPROVAL_MODELS:
            update_params_approval_p(params)
        elif culture_id.lower() == 'resampling':
            update_params_approval_resampling(params)
        elif culture_id.lower() == 'disjoint':
            update_params_approval_disjoint(params)
        update_params_approval_alpha(params)

    return params, params['alpha']
