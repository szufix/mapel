
import ast
from mapel.core.objects.Instance import Instance

from mapel.marriages.cultures_ import *
from mapel.core.glossary import *
from mapel.core.utils import *

from mapel.roommates.persistence.instance_imports import import_real_instance
from mapel.marriages.persistence.instance_exports import export_instance_to_a_file

from mapel.marriages.features_ import get_feature


class Marriages(Instance):

    def __init__(self,
                 experiment_id,
                 instance_id,
                 alpha=1,
                 culture_id=None,
                 num_agents=None,
                 is_imported=True, votes=None):

        super().__init__(experiment_id, instance_id, alpha=alpha, culture_id=culture_id)

        self.num_agents = num_agents
        self.votes = votes

        self.retrospetive_vectors = None
        self.positionwise_vectors = None

        if is_imported and experiment_id != 'virtual':
            try:
                self.votes, self.num_agents, self.params, self.model_id = import_real_instance(self)
                self.alpha = self.params['alpha']
            except:
                pass

    def get_retrospective_vectors(self):
        if self.retrospetive_vectors is not None:
            return self.retrospetive_vectors
        else:
            return self.votes_to_retrospective_vectors()

    def get_positionwise_vectors(self):
        if self.positionwise_vectors is not None:
            return self.positionwise_vectors
        else:
            return self.votes_to_positionwise_vectors()

    def votes_to_retrospective_vectors(self):

        vectors = np.zeros([2, self.num_agents, self.num_agents], dtype=int)

        for a in range(self.num_agents):
            for i, b in enumerate(self.votes[0][a]):
                vectors[0][a][i] = int(list(self.votes[1][b]).index(a))

        for a in range(self.num_agents):
            for i, b in enumerate(self.votes[1][a]):
                vectors[1][a][i] = int(list(self.votes[0][b]).index(a))

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

    # PREPARE INSTANCE
    def prepare_instance(self, is_exported=None, params: dict = None):
        if params is None:
            params = {}

        # if election.culture_id == 'norm-mallows' and 'norm-phi' not in params:
        #     params['norm-phi'] = np.random.rand()
        # elif election.culture_id == 'urn' and 'alpha' not in params:
        #     params['alpha'] = np.random.rand()
        if 'norm-phi' in params:
            params['alpha'] = params['norm-phi']
        else:
            params['alpha'] = 1

        self.params = params
        self.alpha = params['alpha']
        self.votes = generate_votes(culture_id=self.culture_id, num_agents=self.num_agents, params=params)

        if is_exported:
            export_instance_to_a_file(self)


    def compute_feature(self, feature_id, feature_long_id=None, **kwargs):
        if feature_long_id is None:
            feature_long_id = feature_id
        feature = get_feature(feature_id)
        self.features[feature_long_id] = feature(self, **kwargs)
