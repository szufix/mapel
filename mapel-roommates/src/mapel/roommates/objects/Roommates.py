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
        # print(self.votes_to_retrospective_vectors())
        if self.retrospetive_vectors is not None:
            return self.retrospetive_vectors
        return self.votes_to_retrospective_vectors()

    def get_positionwise_vectors(self):
        if self.positionwise_vectors is not None:
            return self.positionwise_vectors
        return self.votes_to_positionwise_vectors()


    '''
        Na teoria:
        0: 1=2>3
        1: 2>3
        2: 3>0
        3: 2>1=0

        No python:
        0: [[1,2],[3], []]
        1: [[2], [3], []]
        2: [[3], [0], []]
        3: [[2], [1,0], []]

        Se a não classifica b, então MA[a][idx(b)] = -1
        Se a classifica b, mas b não classifica a, então MA[a][idx(b)] = num_agents (pior rank possível)
        Matriz de mutualidade:
        0: [2, 1, -1]
        1: [3, 1, -1]
        2: [0, 0, -1]
        3: [0, 1, -1]

        0: 1=2
        1: 3
        2: 1
        3: 0=1

        0: [[1,2], [], []]
        1: [[3], [], []]
        2: [[1], [], []]
        3: [[0,1], [], []]

        0: [(4+4 /2), -1, -1] = [4, -1, -1]
        1: [0, -1, -1]
        2: [4, -1, -1]
        3: [(4+0 / 2), -1, -1] = [2, -1, -1]


        0: [2, 1, -1] = [4, -1, -1]
        1: [3, 1, -1] = [0, -1, -1]
        2: [0, 0, -1] = [4, -1, -1]
        3: [0, 1, -1] = [2, -1, -1]
        mapeamento: 0->0, 1->2, 2->1, 3->3
        distancia: 4 + 3 + 1 + 4
        
        OBS: fully incomplete gera MA só com -1, fully tied gera MA só com num_agents. Provavelmente é distancia maxima
    '''

    def votes_to_retrospective_vectors(self):
        '''
            Retorna a matriz de mutualidade
        '''
        def rank_of(a1, a2):
            for i in range(len(self.votes[a1])):
                if a2 in self.votes[a1][i]:
                    return i
            return self.num_agents-1

        vectors = np.zeros([self.num_agents, self.num_agents - 1], dtype=int)
        for agent in range(self.num_agents):
            next_idx = 0
            for j in range(self.num_agents-1):

                rankings = self.votes[agent][j]
                if len(rankings) == 0:
                    if j == next_idx:
                        vectors[agent][j] = self.num_agents-1
                        next_idx += 1
                    continue

                idx_of = [rank_of(agent2, agent) for agent2 in rankings]
                idx_of.sort()
                for k in range(len(idx_of)):
                    vectors[agent][next_idx+k] = idx_of[k]
                next_idx = next_idx + k +1

        # if self.culture_id == 'expectation':
        #     for i in range(len(self.votes)):
        #         print(self.votes[i])
        #     print('*')
        #     print(vectors)
        #     10/0
                
        self.retrospetive_vectors = vectors
        return vectors
    
    '''
        vectors = np.zeros([self.num_agents, self.num_agents - 1], dtype=int)

        order_votes = [[] for _ in range(self.num_agents)]

        for a in range(self.num_agents):
            # Nao sei oq é order_votes
            # print('votes[' + str(a) + ']: ' + str(list(self.votes[a])))
            (missing,) = set(range(self.num_agents)) - set(self.votes[a])
            order_votes[missing] = copy.deepcopy(self.votes[a])

        for a in range(self.num_agents):
            for i, b in enumerate(order_votes[a]):
                # vectors[agent][i] = indice de 'agent' na lista de 'b'
                # 'b' é o i-ésimo agente da lista de 'agent'
                vectors[a][i] = int(list(order_votes[b]).index(a))
            # print('vectors[' + str(a) + ']: ' + str(list(vectors[a])))

        self.retrospetive_vectors = vectors
        return vectors
    '''

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

