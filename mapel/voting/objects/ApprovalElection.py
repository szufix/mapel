#!/usr/bin/env python

import ast
import os

import numpy as np

from mapel.voting.metrics.inner_distances import hamming
from mapel.voting.objects.Election import Election

from mapel.voting._glossary import NICE_NAME


class ApprovalElection(Election):

    def __init__(self, experiment_id, name, votes=None, alpha=1, model=None,
                 ballot='approval', num_voters=None, num_candidates=None, _import=False):

        super().__init__(experiment_id, name, votes=votes, alpha=alpha,
                         model=model, ballot=ballot, num_voters=num_voters,
                         num_candidates=num_candidates)

        self.approvalwise_vector = []
        self.coapproval_frequency_vectors = []
        self.voterlikeness_vectors = []
        self.pairwise_matrix = []
        self.candidatelikeness_original_vectors = []
        self.candidatelikeness_sorted_vectors = []
        self.model = model
        self.hamming_candidates = []

        if _import:
            fake = check_if_fake(experiment_id, name)
            if fake:
                self.model, self.params, self.num_voters, self.num_candidates = \
                    import_fake_app_election(experiment_id, name)
            else:
                self.votes, self.num_voters, self.num_candidates, self.params, \
                    self.model = import_real_app_election(experiment_id, name)
                try:
                    self.alpha = self.params[self.params['variable']]
                except KeyError:
                    pass

    def votes_to_approvalwise_vector(self) -> None:
        """ Convert votes to ... """

        if self.model == 'approval_half_1':
            self.approvalwise_vector = np.sort(np.array([0.75 for _ in
                                                         range(int(self.num_candidates / 2))] +
                                                        [0.25 for _ in
                                                         range(int(self.num_candidates / 2))]))
        elif self.model == 'approval_half_2':
            self.approvalwise_vector = np.sort(np.array([i / (self.num_candidates - 1) for i in
                                                         range(self.num_candidates)]))
        else:
            approvalwise_vector = np.zeros([self.num_candidates])
            for vote in self.votes:
                for c in vote:
                    approvalwise_vector[c] += 1
            approvalwise_vector = approvalwise_vector / self.num_voters
            self.approvalwise_vector = np.sort(approvalwise_vector)
            # print(self.approvalwise_vector)

    def votes_to_coapproval_frequency_vectors(self, vector_type='A') -> None:
        """ Convert votes to ... """
        vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
        for vote in self.votes:
            size = len(vote)
            for c in range(self.num_candidates):
                if c in vote:
                    if vector_type in ['A', 'B']:
                        vectors[c][size - 1] += 1
                    elif vector_type == 'C':
                        vectors[c][2 * size - 1] += 1
                else:
                    if vector_type == 'A':
                        vectors[c][self.num_candidates + size] += 1
                    elif vector_type == 'B':
                        vectors[c][2 * self.num_candidates - size - 1] += 1
                    elif vector_type == 'C':
                        vectors[c][2 * size] += 1
        vectors = vectors / self.num_voters
        # vectors = vectors / experiment.num_candidates
        self.coapproval_frequency_vectors = vectors

    def votes_to_pairwise_matrix(self) -> None:
        """ Convert votes to ... """
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
                elif vector_type == 'martin':
                    vectors[i][j] = len(set_a.intersection(set_b)) - len(set_a)
            vectors[i] = sorted(vectors[i])

        self.voterlikeness_vectors = vectors



def import_real_app_election(experiment_id: str, election_id: str):
    """ Import real approval election from .app file """

    file_name = str(election_id) + ".app"
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

    # Shift by -1
    # if model_name in LIST_OF_PREFLIB_MODELS:
    #     for i in range(num_voters):
    #         for j in range(num_candidates):
    #             votes[i][j] -= 1
    if model_name in NICE_NAME.values():
        rev_dict = dict(zip(NICE_NAME.values(), NICE_NAME.keys()))
        model_name = rev_dict[model_name]
    # print(model_name)

    return votes, num_voters, num_candidates, params, model_name


def import_fake_app_election(experiment_id: str, election_id: str):
    """ Import fake approval election from .app file """

    file_name = f'{election_id}.app'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    my_file.readline()  # line with $ fake

    num_voters = int(my_file.readline().strip())
    num_candidates = int(my_file.readline().strip())
    fake_model_name = str(my_file.readline().strip())
    params = {}

    return fake_model_name, params, num_voters, num_candidates


def check_if_fake(experiment_id: str, election_id: str) -> bool:
    file_name = str(election_id) + ".app"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    line = my_file.readline().strip()
    return line[0] == '$'
