#!/usr/bin/env python
import ast
import copy
import csv
import itertools
import os
from threading import Thread
from time import sleep

from mapel.main.objects.Experiment import Experiment
from mapel.roommates.objects.RoommatesFamily import RoommatesFamily
from mapel.roommates.objects.Roommates import Roommates
import mapel.roommates.models_main as models_main
import mapel.roommates.metrics_main as metr
import mapel.roommates.features.basic_features as basic
import mapel.roommates.features_main as features

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


class RoommatesExperiment(Experiment):
    """Abstract set of elections."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_num_agents = 10

        self.stable_sr = {}

    def add_instances_to_experiment(self):

        instances = {}

        for family_id in self.families:

            ids = []
            if self.families[family_id].single_instance:
                election_id = family_id
                election = Roommates(self.experiment_id, election_id)
                instances[election_id] = election
                ids.append(str(election_id))
            else:
                for j in range(self.families[family_id].size):
                    election_id = family_id + '_' + str(j)
                    election = Roommates(self.experiment_id, election_id)
                    instances[election_id] = election
                    ids.append(str(election_id))

            self.families[family_id].instance_ids = ids

        return instances


    def add_instance(self, model_id="none", params=None, label=None,
                     color="black", alpha=1., show=True, marker='x', starting_from=0, size=1,
                     num_agents=None, instance_id=None):

        if num_agents is None:
            num_agents = self.default_num_agents

        return self.add_family(model_id=model_id, params=params, size=size, label=label,
                               color=color, alpha=alpha, show=show, marker=marker,
                               starting_from=starting_from, family_id=instance_id,
                               num_agents=num_agents, single_instance=True)

    def add_family(self, model_id: str = "none", params: dict = None, size: int = 1,
                   label: str = None, color: str = "black", alpha: float = 1.,
                   show: bool = True, marker: str = 'o', starting_from: int = 0,
                   family_id: str = None, single_instance: bool = False,
                   num_agents: int = None, path: dict = None):

        if num_agents is None:
            num_agents = self.default_num_agents

        if self.families is None:
            self.families = {}

        if label is None:
            label = family_id

        self.families[family_id] = RoommatesFamily(model_id=model_id, family_id=family_id,
                                                   params=params, label=label, color=color,
                                                   alpha=alpha, single_instance=single_instance,
                                                   show=show, size=size, marker=marker,
                                                   starting_from=starting_from,
                                                   num_agents=num_agents, path=path)

        self.num_families = len(self.families)
        self.num_instances = sum([self.families[family_id].size for family_id in self.families])

        models_main.prepare_roommates_instances(experiment=self,
                                    model_id=self.families[family_id].model_id,
                                    family_id=family_id,
                                    params=copy.deepcopy(self.families[family_id].params))


    def compute_distances(self, distance_id: str = 'emd-positionwise', num_threads: int = 1,
                          self_distances: bool = False, vector_type: str = 'A',
                          printing: bool = False) -> None:

        self.distance_id = distance_id

        matchings = {instance_id: {} for instance_id in self.instances}
        distances = {instance_id: {} for instance_id in self.instances}
        times = {instance_id: {} for instance_id in self.instances}

        threads = [{} for _ in range(num_threads)]

        ids = []
        for i,instance_1 in enumerate(self.instances):
            for j, instance_2 in enumerate(self.instances):
                if i == j:
                    if self_distances:
                        ids.append((instance_1, instance_2))
                elif i < j:
                    ids.append((instance_1, instance_2))

        num_distances = len(ids)

        for t in range(num_threads):
            print(f'Starting thread: {t}')
            sleep(0.1)
            start = int(t * num_distances / num_threads)
            stop = int((t + 1) * num_distances / num_threads)
            thread_ids = ids[start:stop]

            threads[t] = Thread(target=metr.run_single_thread, args=(self, thread_ids,
                                                                     distances, times, matchings,
                                                                     printing))
            threads[t].start()

        for t in range(num_threads):
            threads[t].join()

        if self.store:

            file_name = f'{distance_id}.csv'
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances",
                                file_name)

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(
                    ["instance_id_1", "instance_id_2", "distance", "time"])

                for election_1, election_2 in itertools.combinations(self.instances, 2):
                    distance = str(distances[election_1][election_2])
                    time = str(times[election_1][election_2])
                    writer.writerow([election_1, election_2, distance, time])

        self.distances = distances
        self.times = times
        self.matchings = matchings

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


            if 'model_id' in row.keys():
                model_id = str(row['model_id']).strip()

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

            families[family_id] = RoommatesFamily(model_id=model_id,
                                                 family_id=family_id,
                                                 params=params, label=label,
                                                 color=color, alpha=alpha, show=show,
                                                 size=size, marker=marker,
                                                 starting_from=starting_from,
                                                 num_agents=num_agents, path=path,
                                                  single_instance=single_instance)
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

            models_main.prepare_roommates_instances(
                experiment=self,
                model_id=self.families[family_id].model_id,
                family_id=family_id,
                params=self.families[family_id].params)

    def compute_stable_sr(self):
        for instance_id in self.instances:
            # print(instance_id)
            usable_matching = basic.compute_stable_SR(self.instances[instance_id].votes)
            # print(usable_matching)
            self.stable_sr[instance_id] = usable_matching

        if self.store:

            file_name = f'stable_sr.csv'
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "features",
                                file_name)

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(
                    ["instance_id", "matching"])

                for instance_id in self.instances:
                    usable_matching = self.stable_sr[instance_id]
                    writer.writerow([instance_id, usable_matching])

    def compute_feature(self, feature_id: str = None, feature_params=None) -> dict:

        if feature_params is None:
            feature_params = {}

        feature_dict = {'value': {}, 'time': {}, 'std': {}}

        features_with_time = {}
        features_with_std = {'avg_num_of_bps_for_rand_matching'}

        for instance_id in self.instances:
            print(instance_id)
            feature = features.get_feature(feature_id)
            instance = self.instances[instance_id]
            # if feature_id in ['monotonicity_1', 'monotonicity_triplets']:
            value = feature(instance)
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

            # if feature_id in EMBEDDING_RELATED_FEATURE:
            #     path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
            #                         "features", f'{feature_id}__{self.distance_id}.csv')
            # else:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "features", f'{feature_id}.csv')

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')

                if feature_id in features_with_time:
                    writer.writerow(["election_id", "value", 'time'])
                    for key in feature_dict['value']:
                        writer.writerow([key, feature_dict['value'][key], round(feature_dict['time'][key],3)])
                elif feature_id in features_with_std:
                    writer.writerow(["election_id", "value", 'std'])
                    for key in feature_dict['value']:
                        writer.writerow([key, feature_dict['value'][key], round(feature_dict['std'][key],3)])
                else:
                    writer.writerow(["election_id", "value"])
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
                    "size;num_agents;model_id;params;color;alpha;family_id;label;marker;show\n")
                file_csv.write("10;20;roommates_ic;{};black;1;IC;IC;o;t\n")
        except FileExistsError:
            print("Experiment already exists!")
