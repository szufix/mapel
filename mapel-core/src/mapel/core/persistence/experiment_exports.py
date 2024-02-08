import csv
import os

from mapel.core.glossary import *
from mapel.core.utils import make_folder_if_do_not_exist


def export_feature_to_file(experiment,
                           feature_id,
                           feature_dict,
                           saveas):
    """
    Exports feature to .csv file.

    Parameters
    ----------
        experiment : Experiment
           Experiment object.
        feature_id : str
            Name of the feature.
        feature_dict : dict
            Dictionary with feature values.
        saveas : str
            Name of the file to save the feature to.
    """

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
        any_key = [key for key in feature_dict.keys()][0]
        for instance_id in feature_dict[any_key]:
            row = [instance_id] + [feature_dict[key][instance_id] for key in feature_dict.keys()]
            writer.writerow(row)


def export_normalized_feature_to_file(experiment,
                                      feature_dict=None,
                                      saveas=None):
    """
    Exports normalized feature to .csv file.

    Parameters
    ----------
        experiment : Experiment
           Experiment object.
        feature_dict : dict
            Dictionary with feature values.
        saveas : str
            Name of the file to save the feature to.
    """

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
def export_embedding_to_file(experiment,
                             embedding_id: str,
                             saveas: str,
                             dim: int,
                             my_pos):
    """
    Exports coordinates of all instances to a .csv file.

    Parameters
    ----------
        experiment : Experiment
           Experiment object.
        embedding_id : str
            Name of the embedding.
        saveas : str
            Name of the file to save the feature to.
        dim : int
            Dimension of the embedding.
        my_pos:
            list of coordinates
    """

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
                             self_distances=False,
                             ids=None):
    """
    Exports distances between each pair of instances to a .csv file.

    Parameters
    ----------
        experiment : Experiment
           Experiment object.
        distance_id : str
            Name of the distance.
        distances :  dict
            Dictionary with distances between each pair of instances
        times : dict
            Dictionary with time of calculation of each distance.
        self_distances : bool
            Flag indicating whether to export self-distances.
        ids:
            List of the Ids.
    """

    path_to_folder = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "distances")
    make_folder_if_do_not_exist(path_to_folder)
    path = os.path.join(path_to_folder, f'{distance_id}.csv')

    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["instance_id_1", "instance_id_2", "distance", "time"])

        for pair in ids:
            instance_1, instance_2 = pair
            distance = str(distances[instance_1][instance_2])
            time_ = str(times[instance_1][instance_2])
            writer.writerow([instance_1, instance_2, distance, time_])


def export_distances_multiple_processes(
        experiment,
        instances_ids,
        distances,
        times,
        process_id
):
    """
    Exports distances between each pair of instances to a .csv file.

    Parameters
    ----------
        experiment : Experiment
           Experiment object.
        instances_ids:
            List of the Ids.
        distances :  dict
            Dictionary with distances between each pair of instances
        times : dict
            Dictionary with time of calculation of each distance.
        process_id : bool
            Id of the process.
    """

    file_name = f'{experiment.distance_id}_p{process_id}.csv'
    path = os.path.join(os.getcwd(),
                        "experiments",
                        experiment.experiment_id,
                        "distances",
                        file_name)
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["instance_id_1", "instance_id_2", "distance", "time"])
        for election_id_1, election_id_2 in instances_ids:
            distance = float(distances[election_id_1][election_id_2])
            time_ = float(times[election_id_1][election_id_2])
            writer.writerow([election_id_1, election_id_2, distance, time_])
