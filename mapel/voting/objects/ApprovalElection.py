#!/usr/bin/env python

import numpy as np

from mapel.voting.objects.Election import Election


class ApprovalElection(Election):

    def __init__(self, experiment_id, name, votes=None, with_matrix=False, alpha=None, model=None,
                 ballot='approval', num_voters=None, num_candidates=None):

        super().__init__(experiment_id, name, votes=votes, with_matrix=with_matrix, alpha=alpha,
                         model=model, ballot=ballot,
                         num_voters=num_voters, num_candidates=num_candidates)

    def votes_to_approval_frequency_vector(self):
        approval_frequency_vector = np.zeros([self.num_candidates])
        for vote in self.votes:
            for c in vote:
                approval_frequency_vector[c] += 1
        approval_frequency_vector = approval_frequency_vector / self.num_voters
        return np.sort(approval_frequency_vector)

    def votes_to_cooparoval_frequency_vectors(self):
        vectors = np.zeros([self.num_candidates, self.num_candidates * 2])
        for vote in self.votes:
            size = len(vote)
            for c in range(self.num_candidates):
                if c in vote:
                    vectors[c][size] += 1
                else:
                    vectors[c][self.num_candidates + size] += 1
        return vectors
