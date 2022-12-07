import numpy as np
import os
from collections import Counter
import copy
import ast
from mapel.core.objects.Instance import Instance
from mapel.roommates.cultures._utils import convert

from mapel.roommates.cultures_ import generate_votes
from mapel.core.glossary import *
from mapel.core.utils import *

#

class Roommates(Instance):

    def __init__(self, experiment_id, instance_id, alpha=1, culture_id=None, num_agents=None,
                 _import=True, votes=None):

        super().__init__(experiment_id, instance_id, alpha=alpha, culture_id=culture_id)

        self.num_agents = num_agents
        self.votes = votes

        self.retrospetive_vectors = None
        self.positionwise_vectors = None

        if _import and experiment_id != 'virtual':
            try:
                self.votes, self.num_agents, self.params, self.culture_id = \
                    self.import_real_instance()
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

        if self.culture_id == 'roommates_tmp': # A-S-SYM

            n = self.num_agents // 2

            vector = [n-i-1 for i in range(n)] + [2*n-i-2 for i in range(n-1)]

            vectors = [vector for _ in range(self.num_agents)]
            vectors = np.array(vectors)
            self.retrospetive_vectors = vectors
            return vectors

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

    # PREPARE INSTANCE
    def prepare_instance(self, store=None, params: dict = None):

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

        self.votes = generate_votes(culture_id=self.culture_id, num_agents=self.num_agents, params=params)
        self.params = params

        if store:
            self.store_instance_in_a_file()

    def store_instance_in_a_file(self):
        """ Store votes in a file """

        path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id, "instances")
        make_folder_if_do_not_exist(path_to_folder)
        path_to_file = os.path.join(path_to_folder, f'{self.instance_id}.ri')

        with open(path_to_file, 'w') as file_:

            if self.culture_id in NICE_NAME:
                file_.write("# " + NICE_NAME[self.culture_id] + " " + str(self.params) + "\n")
            else:
                file_.write("# " + self.culture_id + " " + str(self.params) + "\n")

            file_.write(str(self.num_agents) + "\n")

            for i in range(self.num_agents):
                file_.write(str(i) + ', a' + str(i) + "\n")

            c = Counter(map(tuple, self.votes))
            counted_votes = [[count, list(row)] for row, count in c.items()]
            # counted_votes = sorted(counted_votes, reverse=True)

            file_.write(str(self.num_agents) + ', ' + str(self.num_agents) + ', ' +
                        str(len(counted_votes)) + "\n")

            for i in range(len(counted_votes)):
                file_.write(str(counted_votes[i][0]) + ', ')
                for j in range(len(counted_votes[i][1])):
                    file_.write(str(int(counted_votes[i][1][j])))
                    if j < len(counted_votes[i][1]) - 1:
                        file_.write(", ")
                file_.write("\n")

    def import_real_instance(self, shift=False):
        """ Import real ordinal election form .soc file """

        file_name = f'{self.instance_id}.ri'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "instances", file_name)
        print(path)
        with open(path, 'r') as my_file:
            params = 0
            first_line = my_file.readline()

            if first_line[0] != '#':
                culture_id = 'empty'
                num_agents = int(first_line)
            else:
                first_line = first_line.strip().split()
                culture_id = first_line[1]

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
                    print(k, votes)
                    for el in range(num_candidates):
                        votes[it][el] = int(line[el + 1])
                    it += 1



            if shift:
                for i in range(num_voters):
                    for j in range(num_candidates):
                        votes[i][j] -= 1

        rev_NICE_NAME = {v: k for k, v in NICE_NAME.items()}

        if culture_id in rev_NICE_NAME:
            culture_id = rev_NICE_NAME[culture_id]


        return votes, num_agents, params, culture_id