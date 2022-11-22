import numpy as np
import os
from collections import Counter
import copy
import ast
from mapel.core.objects.Instance import Instance

from mapel.marriages.models_main import generate_votes
from mapel.core.glossary import *
from mapel.core.utils import *


class Marriages(Instance):

    def __init__(self, experiment_id, instance_id, alpha=1, model_id=None, num_agents=None,
                 _import=True, votes=None):

        super().__init__(experiment_id, instance_id, alpha=alpha, model_id=model_id)

        self.num_agents = num_agents
        self.votes = votes

        self.retrospetive_vectors = None
        self.positionwise_vectors = None

        if _import and experiment_id != 'virtual':
            try:
                self.votes, self.num_agents, self.params, self.model_id = \
                    self.import_real_instance()
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
    def prepare_instance(self, store=None, params: dict = None):
        if params is None:
            params = {}

        # if self.culture_id == 'norm-mallows' and 'norm-phi' not in params:
        #     params['norm-phi'] = np.random.rand()
        # elif self.culture_id == 'urn' and 'alpha' not in params:
        #     params['alpha'] = np.random.rand()
        if 'norm-phi' in params:
            params['alpha'] = params['norm-phi']
        else:
            params['alpha'] = 1

        self.params = params
        self.alpha = params['alpha']
        self.votes = generate_votes(model_id=self.model_id, num_agents=self.num_agents, params=params)

        if store:
            self.store_instance_in_a_file()

    def store_instance_in_a_file(self):
        """ Store votes in a file """

        path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id, "instances")
        make_folder_if_do_not_exist(path_to_folder)
        path_to_file = os.path.join(path_to_folder, f'{self.instance_id}.mi')

        with open(path_to_file, 'w') as file_:

            if self.model_id in NICE_NAME:
                file_.write("# " + NICE_NAME[self.model_id] + " " + str(self.params) + "\n")
            else:
                file_.write("# " + self.model_id + " " + str(self.params) + "\n")

            for s in range(2):

                file_.write(str(self.num_agents) + "\n")

                for i in range(self.num_agents):
                    if s == 0:
                        file_.write(str(i) + ', a' + str(i) + "\n")
                    else:
                        file_.write(str(i) + ', b' + str(i) + "\n")

                # c = Counter(map(tuple, self.votes[s]))
                # counted_votes = [[count, list(row)] for row, count in c.items()]
                # counted_votes = sorted(counted_votes, reverse=True)

                votes = self.votes[s]
                # print(votes)

                file_.write(str(self.num_agents) + ', ' + str(self.num_agents) + ', ' +
                            str(len(votes)) + "\n")

                for i in range(len(votes)):
                    file_.write(str(i) + ', ')
                    for j in range(self.num_agents):
                        file_.write(str(int(votes[i][j])))
                        if j < len(votes[i]) - 1:
                            file_.write(", ")
                    file_.write("\n")

    # def old_name_extractor(first_line):
    #     if len(first_line) == 4:
    #         model_name = f'{first_line[1]} {first_line[2]} {first_line[3]}'
    #     elif len(first_line) == 3:
    #         model_name = f'{first_line[1]} {first_line[2]}'
    #     elif len(first_line) == 2:
    #         model_name = first_line[1]
    #     else:
    #         model_name = 'noname'
    #     return model_name

    def import_real_instance(self, shift=False):
        """ Import real ordinal election form .soc file """

        votes = [0, 0]

        file_name = f'{self.instance_id}.mi'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "instances", file_name)
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


            for s in range(2):

                if s == 1:
                    num_agents = int(my_file.readline())

                num_candidates = num_agents

                for _ in range(num_agents):
                    my_file.readline()

                line = my_file.readline().rstrip("\n").split(',')
                num_voters = int(line[0])
                num_options = int(line[2])
                votes[s] = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

                for j in range(num_options):
                    line = my_file.readline().rstrip("\n").split(',')
                    _id = int(line[0])

                    for k in range(1):
                        for el in range(num_candidates):
                            votes[s][_id][el] = int(line[el + 1])

                if shift:
                    for i in range(num_voters):
                        for j in range(num_candidates):
                            votes[s][i][j] -= 1

        rev_NICE_NAME = {v: k for k, v in NICE_NAME.items()}

        if model_id in rev_NICE_NAME:
            model_id = rev_NICE_NAME[model_id]

        return votes, num_agents, params, model_id