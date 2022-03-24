#!/usr/bin/env python

import ast
import os

import numpy as np

from mapel.elections._glossary import *
from mapel.main._inner_distances import hamming
from mapel.elections.objects.Election import Election


class ApprovalElection(Election):

    def __init__(self, experiment_id, election_id, votes=None, alpha=1, model_id=None,
                 ballot='approval', num_voters=None, num_candidates=None, _import=False,
                 shift: bool = False):

        super().__init__(experiment_id, election_id, votes=votes, alpha=alpha,
                         model_id=model_id, ballot=ballot, num_voters=num_voters,
                         num_candidates=num_candidates)

        self.approvalwise_vector = []
        self.coapproval_frequency_vectors = []
        self.voterlikeness_vectors = []
        self.pairwise_matrix = []
        self.candidatelikeness_original_vectors = []
        self.candidatelikeness_sorted_vectors = []
        self.model = model_id
        self.hamming_candidates = []
        self.reverse_approvals = []

        if _import:
            fake = check_if_fake(experiment_id, election_id)
            if fake:
                self.model, self.params, self.num_voters, self.num_candidates = \
                    import_fake_app_election(experiment_id, election_id)
            else:
                self.votes, self.num_voters, self.num_candidates, self.params, \
                    self.model = import_real_app_election(experiment_id, election_id, shift)
                try:
                    self.alpha = self.params['alpha']
                except:
                    self.alpha = 1
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
        elif self.model == 'approval_skeleton':
            self.approvalwise_vector = np.sort(get_skeleton_approvalwise_vector(self))

        else:
            approvalwise_vector = np.zeros([self.num_candidates])
            for vote in self.votes:
                for c in vote:
                    approvalwise_vector[c] += 1
            approvalwise_vector = approvalwise_vector / self.num_voters
            self.approvalwise_vector = np.sort(approvalwise_vector)

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

    def compute_reverse_approvals(self):
        reverse_approvals = [set() for _ in range(self.num_candidates)]
        for i, vote in enumerate(self.votes):
            for c in vote:
                reverse_approvals[c].add({i})

        self.reverse_approvals = reverse_approvals


def import_real_app_election(experiment_id: str, election_id: str, shift=False):
    """ Import real approval election from .app file """

    file_name = f'{election_id}.app'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')

    params = 0
    first_line = my_file.readline()
    if first_line[0] != '#':
        model_id = 'empty'
        num_candidates = int(first_line)
    else:
        first_line = first_line.strip().split()
        model_id = first_line[1]
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

    if model_id in NICE_NAME.values():
        rev_dict = dict(zip(NICE_NAME.values(), NICE_NAME.keys()))
        model_id = rev_dict[model_id]

    if shift:
        for i, vote in enumerate(votes):
            new_vote = set()
            for c in vote:
                new_vote.add(c-1)
            votes[i] = new_vote

    return votes, num_voters, num_candidates, params, model_id


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
    return line[0] == '$'


def get_skeleton_approvalwise_vector(election):

    phi = election.params['phi']
    p = election.params['p']
    k = int(p * election.num_candidates)

    vector = [phi*p for _ in range(election.num_candidates)]
    for i in range(k):
        vector[i] += 1-phi

    return np.array(vector)
