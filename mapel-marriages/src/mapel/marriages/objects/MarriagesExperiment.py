#!/usr/bin/env python
import ast
import copy
import csv
import itertools
import os
from multiprocessing import Process
from time import sleep

from mapel.core.objects.Experiment import Experiment
from mapel.marriages.objects.MarriagesFamily import MarriagesFamily
from mapel.marriages.objects.Marriages import Marriages
import mapel.marriages.models_main as models_main
import mapel.marriages.metrics_main as metr
import mapel.marriages.features.basic_features as basic
import mapel.marriages.features_main as features
from mapel.core.printing import get_values_from_csv_file
from mapel.core.utils import make_folder_if_do_not_exist
try:
    from sklearn.manifold import MDS
    from sklearn.manifold import TSNE
    from sklearn.manifold import SpectralEmbedding
    from sklearn.manifold import LocallyLinearEmbedding
    from sklearn.manifold import Isomap
except ImportError as error:
    MDS = None
    TSNE = None
    SpectralEmbedding = None
    LocallyLinearEmbedding = None
    Isomap = None
    print(error)


class MarriagesExperiment(Experiment):
    """Abstract set of elections."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.default_num_agents = 10

        self.matchings = {}

        # self.import_matchings()

    def import_matchings(self):
        matchings = {}

        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'features',
                            'stable_sr.csv')
        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                election_id = row['instance_id']
                value = row['matching']
                if value == '':
                    matchings[election_id] = None
                else:
                    matchings[election_id] = value

        self.matchings = matchings

    def add_instances_to_experiment(self):

        instances = {}

        for family_id in self.families:

            ids = []
            if self.families[family_id].single:
                election_id = family_id
                election = Marriages(self.experiment_id, election_id)
                instances[election_id] = election
                ids.append(str(election_id))
            else:
                for j in range(self.families[family_id].size):
                    election_id = family_id + '_' + str(j)
                    election = Marriages(self.experiment_id, election_id)
                    instances[election_id] = election
                    ids.append(str(election_id))

            self.families[family_id].instance_ids = ids

        return instances

    def compute_distances(self, distance_id: str = 'emd-positionwise', num_processes: int = 1,
                          self_distances: bool = False, vector_type: str = 'A',
                          printing: bool = False) -> None:
        """ Compute distances between elections (using processes) """

        self.distance_id = distance_id

        if '-pairwise' in distance_id:
            for instance in self.instances.values():
                instance.votes_to_pairwise_matrix()

        matchings = {election_id: {} for election_id in self.instances}
        distances = {election_id: {} for election_id in self.instances}
        times = {election_id: {} for election_id in self.instances}

        ids = []
        for i, election_1 in enumerate(self.instances):
            for j, election_2 in enumerate(self.instances):
                if i == j:
                    if self_distances:
                        ids.append((election_1, election_2))
                elif i < j:
                    ids.append((election_1, election_2))

        num_distances = len(ids)

        if self.experiment_id == 'virtual' or num_processes == 1:
            metr.run_single_process(self, ids, distances, times, matchings, printing)

        else:
            processes = []
            for t in range(num_processes):
                print(f'Starting thread: {t}')
                sleep(0.1)
                start = int(t * num_distances / num_processes)
                stop = int((t + 1) * num_distances / num_processes)
                instances_ids = ids[start:stop]

                process = Process(target=metr.run_multiple_processes, args=(self, instances_ids,
                                                                            distances, times,
                                                                            matchings,
                                                                            printing, t))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            distances = {instance_id: {} for instance_id in self.instances}
            times = {instance_id: {} for instance_id in self.instances}
            for t in range(num_processes):

                file_name = f'{distance_id}_p{t}.csv'
                path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances",
                                    file_name)

                with open(path, 'r', newline='') as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=';')

                    for row in reader:
                        distances[row['instance_id_1']][row['instance_id_2']] = float(
                            row['distance'])
                        times[row['instance_id_1']][row['instance_id_2']] = float(row['time'])

        if self.store:
            self._store_distances_to_file(distance_id, distances, times)

        self.distances = distances
        self.times = times
        self.matchings = matchings

    def _store_distances_to_file(self, distance_id, distances, times):
        path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances")
        make_folder_if_do_not_exist(path_to_folder)
        path_to_file = os.path.join(path_to_folder, f'{distance_id}.csv')

        with open(path_to_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(["instance_id_1", "instance_id_2", "distance", "time"])

            for election_1, election_2 in itertools.combinations(self.instances, 2):
                distance = str(distances[election_1][election_2])
                time_ = str(times[election_1][election_2])
                writer.writerow([election_1, election_2, distance, time_])

    def import_controllers(self):
        """ Import controllers from a file """

        families = {}

        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'map.csv')
        file_ = open(path, 'r')

        header = [h.strip() for h in file_.readline().split(';')]
        reader = csv.DictReader(file_, fieldnames=header, delimiter=';')

        starting_from = 0
        for row in reader:

            model_id = None
            color = None
            label = None
            params = None
            alpha = None
            size = None
            marker = None
            num_agents = None
            family_id = None
            show = True


            if 'culture_id' in row.keys():
                model_id = str(row['culture_id']).strip()

            if 'color' in row.keys():
                color = str(row['color']).strip()

            if 'label' in row.keys():
                label = str(row['label'])

            if 'family_id' in row.keys():
                family_id = str(row['family_id'])

            if 'params' in row.keys():
                params = ast.literal_eval(str(row['params']))

            if 'alpha' in row.keys():
                alpha = float(row['alpha'])

            if 'size' in row.keys():
                size = int(row['size'])

            if 'marker' in row.keys():
                marker = str(row['marker']).strip()

            if 'num_agents' in row.keys():
                num_agents = int(row['num_agents'])

            if 'path' in row.keys():
                path = ast.literal_eval(str(row['path']))

            if 'show' in row.keys():
                show = row['show'].strip() == 't'

            single_instance = size == 1

            families[family_id] = MarriagesFamily(model_id=model_id,
                                                  family_id=family_id,
                                                  params=params, label=label,
                                                  color=color, alpha=alpha, show=show,
                                                  size=size, marker=marker,
                                                  starting_from=starting_from,
                                                  num_agents=num_agents, path=path,
                                                  single=single_instance)
            starting_from += size

        self.num_families = len(families)
        self.num_instances = sum([families[family_id].size for family_id in families])
        self.main_order = [i for i in range(self.num_instances)]

        file_.close()
        return families

    def prepare_instances(self):

        if self.instances is None:
            self.instances = {}

        for family_id in self.families:

            new_instances = self.families[family_id].prepare_family(
                store=self.store,
                experiment_id=self.experiment_id)

            for instance_id in new_instances:
                self.instances[instance_id] = new_instances[instance_id]

    def compute_stable_sr(self):
        for instance_id in self.instances:
            print(instance_id)
            if instance_id in ['roommates_test']:
                self.matchings[instance_id] = 'None'
            else:
                usable_matching = basic.compute_stable_SR(self.instances[instance_id].votes)
                self.matchings[instance_id] = usable_matching

        if self.store:

            path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                          "features")
            make_folder_if_do_not_exist(path_to_folder)
            path_to_file = os.path.join(path_to_folder, f'stable_sr.csv')

            with open(path_to_file, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(
                    ["instance_id", "matching"])

                for instance_id in self.instances:
                    usable_matching = self.matchings[instance_id]
                    writer.writerow([instance_id, usable_matching])

    def compute_feature(self, feature_id: str = None, feature_params=None) -> dict:

        if feature_params is None:
            feature_params = {}

        feature_dict = {'value': {}, 'time': {}, 'std': {}}

        features_with_time = {}
        features_with_std = {'avg_num_of_bps_for_rand_matching',
                             'avg_number_of_bps_for_random_matching'}

        # print(self.matchings)

        if feature_id == 'summed_rank_difference':
            minimal = get_values_from_csv_file(self, feature_id='summed_rank_minimal_matching')
            maximal = get_values_from_csv_file(self, feature_id='summed_rank_maximal_matching')

            for instance_id in self.instances:
                if minimal[instance_id] is None:
                    value = 'None'
                else:
                    value = abs(maximal[instance_id] - minimal[instance_id])
                feature_dict['value'][instance_id] = value
                feature_dict['time'][instance_id] = 0

        else:

            for instance_id in self.instances:
                print(instance_id)
                feature = features.get_feature(feature_id)
                instance = self.instances[instance_id]
                # if feature_id in ['summed_rank_minimal_matching'] \
                #         and self.matchings[instance_id] is None:
                #     value = 'None'
                # else:
                value = feature(instance.votes)
                print(value)
                #
                # elif feature_id in ['largest_cohesive_group', 'number_of_cohesive_groups',
                #                     'number_of_cohesive_groups_brute',
                #                     'proportionality_degree_pav',
                #                     'proportionality_degree_av',
                #                     'proportionality_degree_cc',
                #                     'justified_ratio',
                #                     'cohesiveness',
                #                     'partylist',
                #                     'highest_cc_score',
                #                     'highest_hb_score']:
                #     value = feature(election, feature_params)
                #
                # elif feature_id in {'avg_distortion_from_guardians',
                #                     'worst_distortion_from_guardians',
                #                     'distortion_from_all',
                #                     'distortion_from_top_100'}:
                #     value = feature(self, election_id)
                # else:
                #     value = feature(election)

                if feature_id in features_with_time:
                    feature_dict['value'][instance_id] = value[0]
                    feature_dict['time'][instance_id] = value[1]
                elif feature_id in features_with_std:
                    feature_dict['value'][instance_id] = value[0]
                    feature_dict['std'][instance_id] = value[1]
                else:
                    feature_dict['value'][instance_id] = value

        if self.store:

            path_to_folder = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                          "features")
            make_folder_if_do_not_exist(path_to_folder)
            path_to_file = os.path.join(path_to_folder, f'{feature_id}.csv')

            with open(path_to_file, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')

                if feature_id in features_with_time:
                    writer.writerow(["instance_id", "value", 'time'])
                    for key in feature_dict['value']:
                        writer.writerow([key, feature_dict['value'][key], round(feature_dict['time'][key],3)])
                elif feature_id in features_with_std:
                    writer.writerow(["instance_id", "value", 'std'])
                    for key in feature_dict['value']:
                        writer.writerow([key, feature_dict['value'][key], round(feature_dict['std'][key],3)])
                else:
                    writer.writerow(["instance_id", "value"])
                    for key in feature_dict['value']:
                        writer.writerow([key, feature_dict['value'][key]])

        self.features[feature_id] = feature_dict
        return feature_dict

    def create_structure(self) -> None:

        if not os.path.isdir("experiments/"):
            os.mkdir(os.path.join(os.getcwd(), "experiments"))

        if not os.path.isdir("images/"):
            os.mkdir(os.path.join(os.getcwd(), "images"))

        if not os.path.isdir("trash/"):
            os.mkdir(os.path.join(os.getcwd(), "trash"))

        try:
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "features"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "coordinates"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "instances"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "matrices"))

            # PREPARE MAP.CSV FILE
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "map.csv")

            with open(path, 'w') as file_csv:
                file_csv.write(
                    "size;num_agents;culture_id;params;color;alpha;family_id;label;marker;show\n")
                file_csv.write("10;16;ic;{};black;1;IC;IC;o;t\n")
        except FileExistsError:
            print("Experiment already exists!")


    def get_election_id_from_model_name(self, model_id: str) -> str:
        for family_id in self.families:
            if self.families[family_id].culture_id == model_id:
                return family_id
