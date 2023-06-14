#!/usr/bin/env python
import numpy as np
import copy
from mapel.core.objects.Instance import Instance

from mapel.roommates.cultures_ import generate_votes

from mapel.roommates.persistence.instance_imports import import_real_instance
from mapel.roommates.persistence.instance_exports import export_instance_to_a_file

from mapel.roommates.features_ import get_local_feature

class Roommates(Instance):

    def __init__(self,
                 experiment_id,
                 instance_id,
                 alpha=1,
                 culture_id=None,
                 num_agents=None,
                 is_imported=True,
                 votes=None):

        super().__init__(experiment_id, instance_id, alpha=alpha, culture_id=culture_id)

        self.num_agents = num_agents
        self.votes = votes

        self.retrospetive_vectors = None
        self.positionwise_vectors = None

        if is_imported and experiment_id != 'virtual':
            try:
                self.votes, self.num_agents, self.params, self.culture_id = \
                    import_real_instance(experiment_id)
                self.alpha = self.params['alpha']
            except:
                pass

    def get_retrospective_vectors(self):
        if self.retrospetive_vectors is not None:
            return self.retrospetive_vectors
        return self.votes_to_retrospective_vectors()

    def get_positionwise_vectors(self):
        if self.positionwise_vectors is not None:
            return self.positionwise_vectors
        return self.votes_to_positionwise_vectors()

    def votes_to_retrospective_vectors(self):

        vectors = np.zeros([self.num_agents, self.num_agents - 1], dtype=int)

        order_votes = [[] for _ in range(self.num_agents)]

        for a in range(self.num_agents):
            (missing,) = set(range(self.num_agents)) - set(self.votes[a])
            order_votes[missing] = copy.deepcopy(self.votes[a])

        for a in range(self.num_agents):
            for i, b in enumerate(order_votes[a]):
                vectors[a][i] = int(list(order_votes[b]).index(a))

        self.retrospetive_vectors = vectors
        return vectors

    def votes_to_positionwise_vectors(self):

        vectors = np.zeros([self.num_agents, self.num_agents - 1])

        for i in range(self.num_agents):
            pos = 0
            for j in range(self.num_agents - 1):
                vote = self.votes[i][j]
                vectors[vote][pos] += 1
                pos += 1

        for i in range(self.num_agents):
            for j in range(self.num_agents - 1):
                vectors[i][j] /= float(self.num_agents)

        self.positionwise_vectors = vectors
        return vectors

    def votes_to_pairwise_matrix(self) -> np.ndarray:
        """ convert VOTES to pairwise MATRIX """
        matrix = np.zeros([self.num_agents, self.num_agents])

        for v in range(self.num_agents):
            for c1 in range(self.num_agents - 1):
                for c2 in range(c1 + 1, self.num_agents - 1):
                    matrix[int(self.votes[v][c1])][int(self.votes[v][c2])] += 1

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                matrix[i][j] /= float(self.num_agents)
                matrix[j][i] = 1. - matrix[i][j]

        return matrix

    def prepare_instance(self, is_exported=None, params: dict = None):

        if params is None:
            params = {}

        if self.culture_id == 'roommates_norm-mallows' and 'norm-phi' not in params:
            params['norm-phi'] = np.random.rand()
            params['alpha'] = params['norm-phi']

        elif self.culture_id == 'roommates_urn' and 'alpha' not in params:
            params['alpha'] = np.random.rand()

        elif 'alpha' not in params:
            params['alpha'] = 1

        if 'variable' in params:
            params['alpha'] = params[params['variable']]

        self.alpha = params['alpha']

        self.votes = generate_votes(culture_id=self.culture_id,
                                    num_agents=self.num_agents,
                                    params=params)
        self.params = params

        if is_exported:
            export_instance_to_a_file(self)

    def compute_feature(self, feature_id, feature_long_id=None, **kwargs):
        if feature_long_id is None:
            feature_long_id = feature_id
        feature = get_local_feature(feature_id)
        self.features[feature_long_id] = feature(self, **kwargs)

