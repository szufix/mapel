import numpy as np
import os
import ast
import copy

from mapel.main.objects.Instance import Instance
from mapel.roommates.models._utils import convert


class Roommates(Instance):

    def __init__(self, experiment_id, instance_id, alpha=1, model_id=None, num_agents=None,
                 _import=True, votes=None):

        super().__init__(experiment_id, instance_id, alpha=alpha, model_id=model_id)

        self.num_agents = num_agents
        self.votes = votes

        self.retrospetive_vectors = None
        self.positionwise_vectors = None

        if experiment_id != 'virtual':
            self.votes, self.num_agents, self.params, self.model_id = \
                import_real_roommates_instance(experiment_id, instance_id)
            self.alpha = self.params['alpha']

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

        # print(self.model_id)

        if self.model_id == 'roommates_test':

            def rotate(vector, shift):
                shift = shift % len(vector)
                return vector[shift:] + vector[:shift]

            votes = [list(range(self.num_agents)) for _ in range(self.num_agents)]
            votes = [rotate(vote, shift) for shift, vote in enumerate(votes)]
            return convert(votes)
        #
        # elif self.model_id == 'roommates_teo':
        #     # return np.array([
        #     #     [i] * (self.num_agents-1) for i in range(self.num_agents)
        #     # ])
        #     tmp_teo = np.array([[1, 2, 3, 4, 5, 6, 7],
        #                         [1, 3, 5, 7, 2, 4, 6],
        #                         [2, 4, 6, 1, 3, 5, 7],
        #                         [3, 5, 7, 2, 4, 6, 1],
        #                         [4, 6, 1, 3, 5, 7, 2],
        #                         [5, 7, 2, 4, 6, 1, 3],
        #                         [6, 1, 3, 5, 7, 2, 4],
        #                         [7, 2, 4, 6, 1, 3, 5]])
        #
        #     return tmp_teo - 1

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


def old_name_extractor(first_line):
    if len(first_line) == 4:
        model_name = f'{first_line[1]} {first_line[2]} {first_line[3]}'
    elif len(first_line) == 3:
        model_name = f'{first_line[1]} {first_line[2]}'
    elif len(first_line) == 2:
        model_name = first_line[1]
    else:
        model_name = 'noname'
    return model_name


def import_real_roommates_instance(experiment_id, election_id, shift=False):
    """ Import real ordinal election form .soc file """

    file_name = f'{election_id}.ri'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "instances", file_name)
    with open(path, 'r') as my_file:

        params = 0
        first_line = my_file.readline()

        if first_line[0] != '#':
            model_id = 'empty'
            num_agents = int(first_line)
        else:
            first_line = first_line.strip().split()
            model_id = first_line[1]

            if len(first_line) <= 2:
                params = {}
            else:
                params = ast.literal_eval(" ".join(first_line[2:]))

            num_agents = int(my_file.readline())

        num_candidates = num_agents - 1

        for _ in range(num_agents):
            my_file.readline()

        line = my_file.readline().rstrip("\n").split(',')
        num_voters = int(line[0])
        num_options = int(line[2])
        votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

        it = 0
        for j in range(num_options):
            line = my_file.readline().rstrip("\n").split(',')
            quantity = int(line[0])

            for k in range(quantity):
                for el in range(num_candidates):
                    votes[it][el] = int(line[el + 1])
                it += 1

        if shift:
            for i in range(num_voters):
                for j in range(num_candidates):
                    votes[i][j] -= 1

    return votes, num_agents, params, model_id
