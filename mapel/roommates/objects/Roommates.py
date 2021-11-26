
import numpy as np

from mapel.main.objects.Instance import Instance


class Roommates(Instance):

    def __init__(self, experiment_id, instance_id, alpha=1, model_id=None, votes=None):

        super().__init__(experiment_id, instance_id, alpha=alpha, model_id=model_id)

        self.votes = votes
        self.num_agents = len(votes)

        self.retrospetive_vectors = None
        self.positionwise_vectors = None

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

        vectors = np.zeros([self.num_agents, self.num_agents-1])

        for a in range(self.num_agents):
            for i, b in enumerate(self.votes[a]):
                vectors[a][i] = list(self.votes[b]).index(a)

        self.retrospetive_vectors = vectors
        return vectors

    def votes_to_positionwise_vectors(self):

        vectors = np.zeros([self.num_agents, self.num_agents-1])

        for i in range(self.num_agents):
            pos = 0
            for j in range(self.num_agents-1):
                vote = self.votes[i][j]
                vectors[vote][pos] += 1
                pos += 1
        for i in range(self.num_agents):
            for j in range(self.num_agents-1):
                vectors[i][j] /= float(self.num_agents)

        self.positionwise_vectors = vectors
        return vectors



