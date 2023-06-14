
import os
import ast

from mapel.core.glossary import NICE_NAME


def import_real_instance(experiment, shift=False):
    """ Import real ordinal election form .soc file """

    votes = [0, 0]

    file_name = f'{experiment.instance_id}.mi'
    path = os.path.join(os.getcwd(), "election", experiment.experiment_id, "instances", file_name)
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