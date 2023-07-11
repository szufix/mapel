import os
import csv
from mapel.core.glossary import *
from mapel.core.utils import make_folder_if_do_not_exist


# Features
def export_feature_to_file(experiment,
                           feature_id,
                           saveas,
                           feature_dict):

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
        header = ['instance_id'] + list(feature_dict.keys())
        writer.writerow(header)
        for instance_id in feature_dict['instance_id']:
            row = [instance_id] + [feature_dict[key][instance_id] for key in feature_dict.keys()]
            writer.writerow(row)


def export_feature_from_function_to_file(experiment, feature_id, feature_dict):
    path_to_file = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "features",
                                f'{feature_id}_{experiment.embedding_id}.csv')

    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

    with open(path_to_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        header = ['instance_id'] + list(feature_dict.keys())
        writer.writerow(header)
        for instance_id in feature_dict['instance_id']:
            row = [instance_id] + [feature_dict[key][instance_id] for key in feature_dict.keys()]
            writer.writerow(row)


def export_normalized_feature_to_file(experiment, feature_dict=None, saveas=None):
    path_to_file = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "features",
                                f'{saveas}.csv')

    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

    with open(path_to_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        header = ['instance_id'] + list(feature_dict.keys())
        writer.writerow(header)
        for instance_id in feature_dict['instance_id']:
            row = [instance_id] + [feature_dict[key][instance_id] for key in feature_dict.keys()]
            writer.writerow(row)


# Embeddings
def export_embedding_to_file(experiment, embedding_id, saveas, dim, my_pos):
    """ Export coordinates of all instances to a file """
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
    """ Export distances between each pair of instances to a file """
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


def export_distances_helper(exp, instances_ids, distances, times, t):
    """ Store distances to csv file """
    file_name = f'{exp.distance_id}_p{t}.csv'
    path = os.path.join(os.getcwd(), "experiments", exp.experiment_id, "distances", file_name)
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["instance_id_1", "instance_id_2", "distance", "time"])
        for election_id_1, election_id_2 in instances_ids:
            distance = float(distances[election_id_1][election_id_2])
            time_ = float(times[election_id_1][election_id_2])
            writer.writerow([election_id_1, election_id_2, distance, time_])


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 11.07.2023 #
# # # # # # # # # # # # # # # #
