#!/usr/bin/env python

import numpy as np
import os
import ast

from mapel.voting.objects.Election import Election
from mapel.voting.metrics.inner_distances import hamming
from mapel.voting.glossary import APPROVAL_FAKE_MODELS


class ApprovalElection(Election):

    def __init__(self, experiment_id, name, votes=None, with_matrix=False, alpha=1, model=None,
                 ballot='approval', num_voters=None, num_candidates=None, _import=False):

        super().__init__(experiment_id, name, votes=votes, with_matrix=with_matrix, alpha=alpha,
                         model=model, ballot=ballot,
                         num_voters=num_voters, num_candidates=num_candidates)

        self.approval_frequency_vector = []
        self.coapproval_frequency_vectors = []
        self.voterlikeness_vectors = []
        self.coapproval_pairwise_matrix = []
        self.model = model

        if votes is not None and len(votes) > 0:
            self.k = max([len(x) for x in votes])
        else:
            self.k = 1

        if _import:
            fake = check_if_fake(experiment_id, name)
            if fake:
                self.model, self.params, self.num_voters, self.num_candidates = \
                    import_fake_app_election(experiment_id, name)
            else:
                self.votes, self.num_voters, self.num_candidates, self.params, \
                    self.election_model = import_real_app_election(experiment_id, name)
                try:
                    self.alpha = self.params[self.params['path_param']]
                except:
                    pass

            if votes is not None and len(votes) > 0:
                self.k = max([len(x) for x in votes])
            else:
                self.k = 1

    def get_approval_frequency_vector(self):
        return self.approval_frequency_vector

    def get_coapproval_frequency_vectors(self):
        return self.coapproval_frequency_vectors

    def get_voterlikeness_vectors(self):
        return self.voterlikeness_vectors

    def votes_to_approval_frequency_vector(self):
        """ Convert votes to ... """

        if self.model == 'approval_zeros':
            self.approval_frequency_vector = np.zeros([self.num_candidates])
        elif self.model == 'approval_ones':
            self.approval_frequency_vector = np.array([1 for _ in range(self.num_candidates)])
        elif self.model == 'approval_id_0.5':
            self.approval_frequency_vector = np.sort(np.array([1 for _ in range(int(self.num_candidates/2))] +
                            [0 for _ in range(int(self.num_candidates/2))]))
        elif self.model == 'approval_ic_0.5':
            self.approval_frequency_vector = np. array([0.5 for _ in range(self.num_candidates)])
        elif self.model == 'approval_half_1':
            self.approval_frequency_vector = np.sort(np.array([0.75 for _ in range(int(self.num_candidates / 2))] +
                                    [0.25 for _ in range(int(self.num_candidates / 2))]))
        elif self.model == 'approval_half_2':
            self.approval_frequency_vector = np.sort(np.array([i/(self.num_candidates-1) for i in range(self.num_candidates)]))
        else:
            approval_frequency_vector = np.zeros([self.num_candidates])
            for vote in self.votes:
                for c in vote:
                    approval_frequency_vector[c] += 1
            approval_frequency_vector = approval_frequency_vector / self.num_voters
            self.approval_frequency_vector =  np.sort(approval_frequency_vector)

    def votes_to_coapproval_frequency_vectors(self):
        """ Convert votes to ... """

        if self.model == 'approval_zeros':
            vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
            for i in range(self.num_candidates):
                vectors[i][self.num_candidates] = 1
                # vectors[i][2*self.num_candidates - 1] = 1
                # vectors[i][0] = 1
            self.coapproval_frequency_vectors = vectors
        elif self.model == 'approval_ones':
            vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
            for i in range(self.num_candidates):
                vectors[i][self.num_candidates-1] = 1
                # vectors[i][2*self.num_candidates - 1] = 1
            self.coapproval_frequency_vectors = vectors
        else:
            vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
            for vote in self.votes:
                size = len(vote)
                for c in range(self.num_candidates):
                    if c in vote:
                        vectors[c][2*size-1] += 1
                    else:
                        vectors[c][self.num_candidates + size] += 1
                        # vectors[c][2*self.num_candidates - size - 1] += 1
                        # vectors[c][2*size] += 1
            vectors = vectors / self.num_voters
            self.coapproval_frequency_vectors = vectors

    def votes_to_coapproval_pairwise_matrix(self):
        """ Convert votes to ... """
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        for c_1 in range(self.num_candidates):
            for c_2 in range(self.num_candidates):
                for vote in self.votes:
                    if (c_1 in vote and c_2 in vote) or (c_1 not in vote and c_2 not in vote):
                        matrix[c_1][c_2] += 1
        matrix = matrix / self.num_voters
        self.coapproval_pairwise_matrix = matrix

    def votes_to_voterlikeness_vectors(self):
        """ Convert votes to ... """

        vectors = np.zeros([self.num_voters, self.num_voters])

        for i in range(self.num_voters):
            for j in range(self.num_voters):
                vectors[i][j] = hamming(self.votes[i], self.votes[j])
            vectors[i] = sorted(vectors[i])

        self.voterlikeness_vectors = vectors


def import_real_app_election(experiment_id, election_id):
    """ Import real approval election from .app file """

    file_name = str(election_id) + ".app"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "instances", file_name)
    my_file = open(path, 'r')

    params = 0
    first_line = my_file.readline()
    if first_line[0] != '#':
        model_name = 'empty'
        num_candidates = int(first_line)
    else:
        first_line = first_line.strip().split()
        model_name = first_line[1]
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
        line = my_file.readline().rstrip("\n").replace("{", '').\
            replace("}", '').replace(' ', '').split(',')
        if line[1] != '':
            quantity = int(line[0])
            for k in range(quantity):
                for l in range(len(line)-1):
                    votes[it].add(int(line[l + 1]))
                it += 1

    # Shift by -1
    # if model_name in LIST_OF_PREFLIB_MODELS:
    #     for i in range(num_voters):
    #         for j in range(num_candidates):
    #             votes[i][j] -= 1

    return votes, num_voters, num_candidates, params, model_name


def import_fake_app_election(experiment_id, election_id):
    """ Import fake approval election from .app file """

    file_name = str(election_id) + ".app"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "instances", file_name)
    my_file = open(path, 'r')
    my_file.readline()  # line with $ fake

    num_voters = int(my_file.readline().strip())
    num_candidates = int(my_file.readline().strip())
    fake_model_name = str(my_file.readline().strip())
    params = {}

    return fake_model_name, params, num_voters, num_candidates


def check_if_fake(experiment_id, election_id):
    file_name = str(election_id) + ".app"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "instances", file_name)
    my_file = open(path, 'r')
    line = my_file.readline().strip()
    if line[0] == '$':
        return True
    return False