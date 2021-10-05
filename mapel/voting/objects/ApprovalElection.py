#!/usr/bin/env python

import numpy as np

from mapel.voting.objects.Election import Election
from mapel.voting.metrics.inner_distances import hamming


class ApprovalElection(Election):

    def __init__(self, experiment_id, name, votes=None, with_matrix=False, alpha=None, model=None,
                 ballot='approval', num_voters=None, num_candidates=None):

        super().__init__(experiment_id, name, votes=votes, with_matrix=with_matrix, alpha=alpha,
                         model=model, ballot=ballot,
                         num_voters=num_voters, num_candidates=num_candidates)

        self.approval_frequency_vector = []
        self.coapproval_frequency_vectors = []

    def get_approval_frequency_vector(self):
        return self.approval_frequency_vector

    def get_coapproval_frequency_vectors(self):
        return self.coapproval_frequency_vectors

    def votes_to_approval_frequency_vector(self):
        if self.model == 'approval_zeros':
            self.approval_frequency_vector = np.zeros([self.num_candidates])
        elif self.model == 'approval_ones':
            self.approval_frequency_vector =  np.array([1 for _ in range(self.num_candidates)])
        elif self.model == 'approval_id_0.5':
            self.approval_frequency_vector =  np.sort(np.array([1 for _ in range(int(self.num_candidates/2))] +
                            [0 for _ in range(int(self.num_candidates/2))]))
        elif self.model == 'approval_ic_0.5':
            self.approval_frequency_vector =  np. array([0.5 for _ in range(self.num_candidates)])
        elif self.model == 'approval_half_1':
            self.approval_frequency_vector =  np.sort(np.array([0.75 for _ in range(int(self.num_candidates / 2))] +
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
        if self.model == 'approval_zeros':
            vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
            for i in range(self.num_candidates):
                # vectors[i][self.num_candidates] = 1
                # vectors[i][2*self.num_candidates - 1] = 1
                vectors[i][0] = 1
            self.coapproval_frequency_vectors = vectors
        elif self.model == 'approval_ones':
            vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
            for i in range(self.num_candidates):
                # vectors[i][self.num_candidates-1] = 1
                vectors[i][2*self.num_candidates - 1] = 1
            self.coapproval_frequency_vectors = vectors
        else:
            vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
            for vote in self.votes:
                size = len(vote)
                for c in range(self.num_candidates):
                    if c in vote:
                        vectors[c][2*size-1] += 1
                    else:
                        # vectors[c][self.num_candidates + size] += 1
                        # vectors[c][2*self.num_candidates - size - 1] += 1
                        vectors[c][2*size] += 1
            vectors = vectors / self.num_voters
            self.coapproval_frequency_vectors = vectors

    def votes_to_coapproval_pairwise_matrix(self):
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        for c_1 in range(self.num_candidates):
            for c_2 in range(self.num_candidates):
                for vote in self.votes:
                    if (c_1 in vote and c_2 in vote) or (c_1 not in vote and c_2 not in vote):
                        matrix[c_1][c_2] += 1
        matrix = matrix / self.num_voters
        return matrix
