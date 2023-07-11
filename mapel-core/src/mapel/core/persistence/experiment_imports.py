import ast
import csv
import logging
import os

import numpy as np
from mapel.core.glossary import *


# Distances
def import_distances_from_file(experiment, distance_id):
    """ Import distances between each pair of instances from a file """

    distances = {}

    file_name = f'{distance_id}.csv'
    path = os.path.join(os.getcwd(), 'experiments', experiment.experiment_id,
                        'distances', file_name)

    with open(path, 'r', newline='') as csv_file:

        reader = csv.DictReader(csv_file, delimiter=';')

        for row in reader:
            try:
                instance_id_1 = row['election_id_1']
                instance_id_2 = row['election_id_2']
            except:
                try:
                    instance_id_1 = row['instance_id_1']
                    instance_id_2 = row['instance_id_2']
                except:
                    pass
            if instance_id_1 not in experiment.instances \
                    or instance_id_2 not in experiment.instances:
                continue

            if instance_id_1 not in distances:
                distances[instance_id_1] = {}

            if instance_id_2 not in distances:
                distances[instance_id_2] = {}

            try:
                distances[instance_id_1][instance_id_2] = float(row['distance'])
                distances[instance_id_2][instance_id_1] = distances[instance_id_1][
                    instance_id_2]
            except KeyError:
                pass

    return distances


def add_distances_to_experiment(experiment) -> (dict, dict, dict):
    """ Import precomputed distances between each pair of instances from a file
        while preparing an experiment """
    try:
        file_name = f'{experiment.distance_id}.csv'
        path = os.path.join(os.getcwd(),
                            'experiments',
                            experiment.experiment_id,
                            'distances',
                            file_name)

        distances = {}
        times = {}
        stds = {}
        mappings = {}
        with open(path, 'r', newline='') as csv_file:

            reader = csv.DictReader(csv_file, delimiter=';')
            warn = False

            for row in reader:
                try:
                    instance_id_1 = row['election_id_1']
                    instance_id_2 = row['election_id_2']
                except:
                    try:
                        instance_id_1 = row['instance_id_1']
                        instance_id_2 = row['instance_id_2']
                    except:
                        pass

                if instance_id_1 not in experiment.instances or \
                        instance_id_2 not in experiment.instances:
                    continue

                if instance_id_1 not in distances:
                    distances[instance_id_1] = {}
                if instance_id_1 not in times:
                    times[instance_id_1] = {}
                if instance_id_1 not in stds:
                    stds[instance_id_1] = {}
                if instance_id_1 not in mappings:
                    mappings[instance_id_1] = {}

                if instance_id_2 not in distances:
                    distances[instance_id_2] = {}
                if instance_id_2 not in times:
                    times[instance_id_2] = {}
                if instance_id_2 not in stds:
                    stds[instance_id_2] = {}
                if instance_id_2 not in mappings:
                    mappings[instance_id_2] = {}

                try:
                    distances[instance_id_1][instance_id_2] = float(row['distance'])
                    distances[instance_id_2][instance_id_1] = distances[instance_id_1][
                        instance_id_2]
                except:
                    pass

                try:
                    times[instance_id_1][instance_id_2] = float(row['time'])
                    times[instance_id_2][instance_id_1] = times[instance_id_1][instance_id_2]
                except:
                    pass

                try:
                    stds[instance_id_1][instance_id_2] = float(row['std'])
                    stds[instance_id_2][instance_id_1] = stds[instance_id_1][instance_id_2]
                except:
                    pass

                try:
                    mappings[instance_id_1][instance_id_2] = ast.literal_eval(str(row['mapping']))
                    mappings[instance_id_2][instance_id_1] = np.argsort(mappings[instance_id_1][instance_id_2])
                except:
                    pass

                if instance_id_1 not in experiment.instances:
                    warn = True

            if warn:
                text = f'Possibly outdated distances are imported!'
                logging.warning(text)

        return distances, times, stds, mappings

    except FileNotFoundError:
        return None, None, None, None


# Features
def get_values_from_csv_file(experiment, feature_id, feature_long_id=None,
                             upper_limit=np.infty,
                             lower_limit=-np.infty,
                             column_id='value') -> dict:
    """ Import values for a feature_id from a .csv file """

    feature_long_id = feature_id if feature_long_id is None else feature_long_id

    path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "features",
                        f'{feature_long_id}.csv')

    values = {}
    with open(path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')

        for row in reader:
            election_id = row.get('instance_id', row.get('election_id'))
            value = row[column_id]

            if value in {'None', 'Blank'} or column_id == 'time' and float(value) == 0.:
                values[election_id] = None
                continue

            value = float(value)
            values[election_id] = min(max(value, lower_limit), upper_limit)

    return values


# Coordinates
def add_coordinates_to_experiment(experiment, dim=2, file_name=None) -> dict:
    """ Import from a file precomputed coordinates of all the points --
        each point refer to one instance """

    coordinates = {}
    if file_name is None:
        file_name = f'{experiment.embedding_id}_{experiment.distance_id}_{dim}d.csv'
    path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id,
                        "coordinates", file_name)

    with open(path, 'r', newline='') as csv_file:

        reader = csv.DictReader(csv_file, delimiter=';')

        warn = False

        for row in reader:
            try:
                instance_id = row['instance_id']
            except KeyError:
                try:
                    instance_id = row['election_id']
                except KeyError:
                    pass

            if dim == 1:
                coordinates[instance_id] = [float(row['x'])]
            elif dim == 2:
                coordinates[instance_id] = [float(row['x']), float(row['y'])]
            elif dim == 3:
                coordinates[instance_id] = [float(row['x']), float(row['y']), float(row['z'])]

            if instance_id not in experiment.instances:
                warn = True

        if warn:
            text = f'Possibly outdated coordinates are imported!'
            logging.warning(text)

    return coordinates

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 11.07.2023 #
# # # # # # # # # # # # # # # #
