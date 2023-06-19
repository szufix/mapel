
import os
import csv
from mapel.core.glossary import *
from mapel.core.utils import make_folder_if_do_not_exist

# Features
def export_instance_feature(experiment, feature_id, feature_dict):
    path_to_folder = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "features")
    make_folder_if_do_not_exist(path_to_folder)
    path_to_file = os.path.join(path_to_folder, f'{feature_id}_{experiment.embedding_id}.csv')

    with open(path_to_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["instance_id", "value", "time"])
        for key in feature_dict['value']:
            writer.writerow([key, feature_dict['value'][key], feature_dict['time'][key]])


def export_feature(experiment, feature_dict=None, saveas=None):
    path_to_folder = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "features")
    make_folder_if_do_not_exist(path_to_folder)
    path_to_file = os.path.join(path_to_folder, f'{saveas}.csv')

    with open(path_to_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["election_id", "value"])
        for key in feature_dict:
            writer.writerow([key, str(feature_dict[key])])


# Embeddings
def export_embedding(experiment, embedding_id, saveas, dim, my_pos):
    if saveas is None:
        file_name = f'{embedding_id}_{experiment.distance_id}_{str(dim)}d.csv'
    else:
        file_name = f'{saveas}.csv'
    path_to_folder = os.path.join(os.getcwd(), "experiments", experiment.experiment_id,
                                  "coordinates")
    make_folder_if_do_not_exist(path_to_folder)
    path_to_file = os.path.join(path_to_folder, file_name)

    with open(path_to_file, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=';')
        if dim == 1:
            writer.writerow(["instance_id", "x"])
        elif dim == 2:
            writer.writerow(["instance_id", "x", "y"])
        elif dim == 3:
            writer.writerow(["instance_id", "x", "y", "z"])

        ctr = 0
        for instance_id in experiment.instances:
            x = round(experiment.coordinates[instance_id][0], 5)
            if dim == 1:
                writer.writerow([instance_id, x])
            else:
                y = round(experiment.coordinates[instance_id][1], 5)
                if dim == 2:
                    writer.writerow([instance_id, x, y])
                else:
                    z = round(my_pos[ctr][2], 5)
                    writer.writerow([instance_id, x, y, z])
            ctr += 1


# Distances
def export_distances_to_file(experiment,
                             distance_id,
                             distances,
                             times,
                             self_distances=False):

    path_to_folder = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "distances")
    make_folder_if_do_not_exist(path_to_folder)
    path = os.path.join(path_to_folder, f'{distance_id}.csv')

    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["instance_id_1", "instance_id_2", "distance", "time"])

        for i, instance_1 in enumerate(experiment.elections):
            for j, instance_2 in enumerate(experiment.elections):
                if i < j or (i == j and self_distances):
                    distance = str(distances[instance_1][instance_2])
                    time_ = str(times[instance_1][instance_2])
                    writer.writerow([instance_1, instance_2, distance, time_])






# Temporary
def export_election_feature_to_file(experiment,
                                    feature_id,
                                    saveas,
                                    feature_dict,):

    path_to_folder = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "features")
    make_folder_if_do_not_exist(path_to_folder)

    if feature_id in EMBEDDING_RELATED_FEATURE:
        path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id,
                            "features", f'{feature_id}_{experiment.embedding_id}.csv')
    else:
        path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id,
                            "features", f'{saveas}.csv')

    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        if feature_id == 'ejr':
            writer.writerow(["instance_id", "ejr", "pjr", "jr", "pareto", "time"])
            for key in feature_dict['ejr']:
                writer.writerow([key, feature_dict['ejr'][key],
                                 feature_dict['pjr'][key],
                                 feature_dict['jr'][key],
                                 feature_dict['pareto'][key],
                                 feature_dict['time'][key]])
        elif feature_id in {'partylist'}:
            writer.writerow(["instance_id", "value", "bound", "num_large_parties"])
            for key in feature_dict:
                writer.writerow([key, feature_dict[key][0], feature_dict[key][1],
                                 feature_dict[key][2]])
        elif feature_id in FEATURES_WITH_DISSAT:
            writer.writerow(["instance_id", "value", 'time', 'dissat'])
            for key in feature_dict['value']:
                writer.writerow(
                    [key, feature_dict['value'][key], feature_dict['time'][key],
                     feature_dict['dissat'][key]])
        else:
            writer.writerow(["instance_id", "value", "time"])
            for key in feature_dict['value']:
                writer.writerow([key, feature_dict['value'][key], feature_dict['time'][key]])