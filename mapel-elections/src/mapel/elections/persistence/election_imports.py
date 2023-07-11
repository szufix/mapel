import ast
import csv
import os

import numpy as np
from mapel.core.glossary import *


def import_real_soc_election(experiment_id: str, election_id: str, is_shifted=False):
    """ Import real ordinal election form .soc file """

    file_name = f'{election_id}.soc'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')

    params = 0
    first_line = my_file.readline()

    if first_line[0] != '#':
        model_name = 'empty'
        num_candidates = int(first_line)
    else:
        first_line = first_line.strip().split()
        model_name = first_line[1]
        if experiment_id == 'original_ordinal_map':
            params = {}
            model_name = _old_name_extractor(first_line)
        else:
            if len(first_line) <= 2:
                params = {}
            else:
                params = ast.literal_eval(" ".join(first_line[2:]))

        num_candidates = int(my_file.readline())

    alliances = {}
    for i in range(num_candidates):
        line = my_file.readline().strip().split()
        if len(line) > 2:
            alliances[i] = int(line[2])

    line = my_file.readline().rstrip("\n").split(',')
    num_voters = int(line[0])
    num_options = int(line[2])
    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    it = 0
    quantites = []
    for j in range(num_options):
        line = list(map(int, my_file.readline().rstrip("\n").split(',')))
        quantity = line[0]
        quantites.append(quantity)

        for k in range(quantity):
            votes[it] = line[1:num_candidates + 1]
            it += 1

    if is_shifted:
        votes = [[vote - 1 for vote in voter] for voter in votes]
    my_file.close()

    return np.array(votes), num_voters, num_candidates, params, model_name, alliances, \
           num_options, quantites


def import_fake_soc_election(experiment_id, name):
    """ Import fake ordinal election form .soc file """

    file_name = f'{name}.soc'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')

    first_line = my_file.readline()
    first_line = first_line.strip().split()
    model_name = first_line[1]
    if len(first_line) <= 2:
        params = {}
    else:
        params = ast.literal_eval(" ".join(first_line[2:]))

    num_candidates = int(my_file.readline())
    num_voters = int(my_file.readline())

    my_file.close()

    return model_name, params, num_voters, num_candidates


def import_real_app_election(experiment_id: str, election_id: str, is_shifted=False):
    """ Import real approval election from .app file """

    file_name = f'{election_id}.app'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')

    params = 0
    first_line = my_file.readline()
    if first_line[0] != '#':
        culture_id = 'empty'
        num_candidates = int(first_line)
    else:
        first_line = first_line.strip().split()
        culture_id = first_line[1]
        if len(first_line) <= 2:
            params = {}
        else:
            params = ast.literal_eval(" ".join(first_line[2:]))

        num_candidates = int(my_file.readline())

    for _ in range(num_candidates):
        my_file.readline()

    line = my_file.readline().rstrip("\n").split(',')
    num_voters = int(line[0])
    num_options = int(line[2])
    votes = [set() for _ in range(num_voters)]

    it = 0
    for j in range(num_options):
        line = my_file.readline().rstrip("\n").replace("{", ''). \
            replace("}", '').replace(' ', '').split(',')
        if line[1] != '':
            quantity = int(line[0])
            for k in range(quantity):
                for el in range(len(line) - 1):
                    votes[it].add(int(line[el + 1]))
                it += 1

    if culture_id in NICE_NAME.values():
        rev_dict = dict(zip(NICE_NAME.values(), NICE_NAME.keys()))
        culture_id = rev_dict[culture_id]

    if is_shifted:
        votes = [{c - 1 for c in vote} for vote in votes]
    my_file.close()

    return votes, num_voters, num_candidates, params, culture_id


def import_fake_app_election(experiment_id: str, name: str):
    """ Import fake approval election from .app file """

    file_name = f'{name}.app'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    first_line = my_file.readline()
    first_line = first_line.strip().split()
    fake_model_name = first_line[1]
    if len(first_line) <= 2:
        params = {}
    else:
        params = ast.literal_eval(" ".join(first_line[2:]))

    num_candidates = int(my_file.readline().strip())
    num_voters = int(my_file.readline().strip())

    return fake_model_name, params, num_voters, num_candidates


def check_if_fake(experiment_id, name, extention):
    file_name = f'{name}.{extention}'
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    line = my_file.readline().strip()
    my_file.close()
    return line[0] == '$'


def _old_name_extractor(first_line):
    if len(first_line) == 4:
        model_name = f'{first_line[1]} {first_line[2]} {first_line[3]}'
    elif len(first_line) == 3:
        model_name = f'{first_line[1]} {first_line[2]}'
    elif len(first_line) == 2:
        model_name = first_line[1]
    else:
        model_name = 'noname'
    return model_name


def import_distances(experiment, object_type='vote'):

    file_name = f'{experiment.election_id}_{object_type}.csv'
    path = os.path.join(os.getcwd(), 'experiments', experiment.experiment_id, 'distances',
                        file_name)

    with open(path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        length = int(len(list(reader))**0.5)

    with open(path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        distances = np.zeros([length, length])
        for row in reader:
            distances[int(row['v1'])][int(row['v2'])] = float(row['distance'])
            distances[int(row['v2'])][int(row['v1'])] = float(row['distance'])

    return distances


def import_coordinates(self, object_type='vote'):
    file_name = f'{self.election_id}_{object_type}.csv'
    path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'coordinates',
                        file_name)
    with open(path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        length = len(list(reader))

    with open(path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        coordinates = np.zeros([length, 2])
        for row in reader:
            coordinates[int(row['vote_id'])] = [float(row['x']), float(row['y'])]

    return coordinates
