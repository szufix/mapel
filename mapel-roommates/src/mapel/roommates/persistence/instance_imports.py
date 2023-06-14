
import os
import ast

from mapel.core.glossary import NICE_NAME


def import_real_instance(self, shift=False):
    """ Import real ordinal election form .soc file """

    file_name = f'{self.instance_id}.ri'
    path = os.path.join(os.getcwd(), "election", self.experiment_id, "instances", file_name)
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
