#!/usr/bin/env python
import ast
import copy
import csv
import math
import os
from abc import ABCMeta, abstractmethod
from threading import Thread
from time import sleep

import networkx as nx
import numpy as np

import mapel.voting.elections_main as _elections
import mapel.voting.metrics_main as metr
import mapel.voting.elections.preflib as preflib
import mapel.voting.features_main as features
import mapel.voting._print as pr
from mapel.voting.objects.Family import Family

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

COLORS = ['blue', 'green', 'black', 'red', 'orange', 'purple', 'brown', 'lime', 'cyan', 'grey']


class Experiment:
    __metaclass__ = ABCMeta
    """Abstract set of elections."""

    def __init__(self, elections=None, distances=None,
                 coordinates=None, distance_name='emd-positionwise', experiment_id=None,
                 election_type='ordinal', _import=True):

        self._import = _import

        self.distance_name = distance_name
        self.elections = {}

        self.families = None
        self.distances = None
        self.times = None
        self.stds = None
        self.points_by_families = None
        self.matchings = None

        self.num_families = None
        self.num_elections = None
        self.main_order = None
        self.election_type = election_type

        if experiment_id is None:
            self.experiment_id = 'virtual'
            self.store = False
        else:
            self.store = True
            self.experiment_id = experiment_id
            self.create_structure()
            self.families = self.import_controllers()

        if elections is not None:
            if elections == 'import':
                self.elections = self.add_elections_to_experiment()
            else:
                self.elections = elections

        if distances is not None:
            if distances == 'import':
                self.distances, self.times, self.stds = \
                    self.add_distances_to_experiment(distance_name=distance_name)
            else:
                self.distances = distances

        if coordinates is not None:
            if coordinates == 'import':
                self.coordinates = self.add_coordinates_to_experiment()
            else:
                self.coordinates = coordinates

        self.features = {}

    def add_family(self, model="none", params=None, size=1, label=None, color="black",
                   alpha=1., show=True, marker='o', starting_from=0, num_candidates=None,
                   family_id=None, single_election=False, num_nodes=None,
                   path=None, name=None):
        """ Add family of elections to the experiment """

        if name is not None:
            family_id = name

        if self.families is None:
            self.families = {}

        elif label is None:
            label = family_id

        self.families[family_id] = Family(model=model, family_id=family_id,
                                          params=params, label=label, color=color, alpha=alpha,
                                          show=show, size=size, marker=marker,
                                          starting_from=starting_from, num_nodes=num_nodes,
                                          num_candidates=num_candidates,
                                          path=path,
                                          single_election=single_election)

        self.num_families = len(self.families)
        self.num_elections = sum([self.families[family_id].size for family_id in self.families])
        self.main_order = [i for i in range(self.num_elections)]

        params = self.families[family_id].params
        model = self.families[family_id].model

        ids = _elections.prepare_statistical_culture_family(experiment=self,
                                                            model=model,
                                                            family_id=family_id,
                                                            params=copy.deepcopy(params))

        self.families[family_id].election_ids = ids

        return ids

    def add_election(self, model="none", params=None, label=None,
                     color="black", alpha=1., show=True, marker='x', starting_from=0, size=1,
                     num_candidates=None, num_voters=None, name=None, num_nodes=None):
        """ Add election to the experiment """

        return self.add_family(model=model, params=params, size=size, label=label, color=color,
                               alpha=alpha, show=show, marker=marker, starting_from=starting_from,
                               num_candidates=num_candidates,
                               family_id=name, num_nodes=num_nodes, single_election=True)[0]

    def add_graph(self, **kwargs):
        return self.add_election(**kwargs)

    def prepare_elections(self):
        """ Prepare elections for a given experiment """

        if self.elections is None:
            self.elections = {}

        if self.store:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections")
            for file_name in os.listdir(path):
                os.remove(os.path.join(path, file_name))

            for family_id in self.families:
                params = self.families[family_id].params
                model = self.families[family_id].model

                if model in preflib.LIST_OF_PREFLIB_MODELS:
                    ids = _elections.prepare_preflib_family(
                        experiment=self, model=model, params=params)
                else:
                    ids = _elections.prepare_statistical_culture_family(
                        experiment=self, model=model,
                        family_id=family_id, params=params)

                self.families[family_id].election_ids = ids

    def compute_winners(self, method=None, num_winners=1):
        for election_id in self.elections:
            self.elections[election_id].compute_winners(method=method, num_winners=num_winners)

    def compute_alternative_winners(self, method=None, num_winners=None, num_parties=None):
        for election_id in self.elections:
            for party_id in range(num_parties):
                self.elections[election_id].compute_alternative_winners(
                    method=method, party_id=party_id, num_winners=num_winners)

    def compute_distances(self, distance_name='emd-positionwise', num_threads=1,
                          self_distances=False, vector_type='A', printing=False) -> None:
        """ Compute distances between elections (using threads) """

        self.distance_name = distance_name

        # precompute vectors, matrices, etc...
        if '-approvalwise' in distance_name:
            for election in self.elections.values():
                election.votes_to_approvalwise_vector()
        elif '-coapproval_frequency' in distance_name or '-flow' in distance_name:
            for election in self.elections.values():
                election.votes_to_coapproval_frequency_vectors(vector_type=vector_type)
        elif '-voterlikeness' in distance_name:
            for election in self.elections.values():
                election.votes_to_voterlikeness_vectors(vector_type=vector_type)
        elif '-candidatelikeness' in distance_name:
            for election in self.elections.values():
                election.votes_to_candidatelikeness_sorted_vectors()
        elif '-pairwise' in distance_name:
            for election in self.elections.values():
                election.votes_to_pairwise_matrix()
        # continue with normal code

        matchings = {}
        distances = {}
        times = {}
        for election_id in self.elections:
            distances[election_id] = {}
            times[election_id] = {}
            matchings[election_id] = {}

        threads = [{} for _ in range(num_threads)]

        ids = []
        for i, election_1 in enumerate(self.elections):
            for j, election_2 in enumerate(self.elections):
                if i == j:
                    if self_distances:
                        ids.append((election_1, election_2))
                elif i < j:
                    ids.append((election_1, election_2))

        num_distances = len(ids)

        for t in range(num_threads):
            print('Starting thread: ', t)
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
            path = os.path.join(os.getcwd(), "experiments",
                                self.experiment_id, "distances",
                                str(distance_name) + ".csv")

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(
                    ["election_id_1", "election_id_2", "distance", "time"])

                for i, election_1 in enumerate(self.elections):
                    for j, election_2 in enumerate(self.elections):
                        if i < j:
                            distance = str(distances[election_1][election_2])
                            time = str(times[election_1][election_2])
                            writer.writerow([election_1, election_2, distance, time])

        self.distances = distances
        self.times = times
        self.matchings = matchings

    @abstractmethod
    def add_elections_to_experiment(self):
        pass

    @abstractmethod
    def create_structure(self):
        pass

    def add_distances_to_experiment(self, distance_name=None):
        distances, times, stds = self.import_my_distances(distance_name=distance_name)
        return distances, times, stds

    def add_coordinates_to_experiment(self):
        return self.import_cooridnates()

    def embed(self, attraction_factor=None, algorithm='spring',
              num_iterations=1000, radius=np.infty, dim=2, num_neighbors=None,
              method='standard') -> None:

        if algorithm == 'spring':
            attraction_factor = 2
        else:
            attraction_factor = 1

        num_elections = len(self.distances)

        x = np.zeros((num_elections, num_elections))

        for i, election_1_id in enumerate(self.distances):
            for j, election_2_id in enumerate(self.distances):
                if i < j:
                    if self.distances[election_1_id][election_2_id] == 0:
                        self.distances[election_1_id][election_2_id] = 0.01
                    if algorithm in {'spring'}:
                        normal = True
                        if self.distances[election_1_id][election_2_id] > radius:
                            x[i][j] = 0.
                            normal = False
                        if num_neighbors is not None:
                            tmp = self.distances[election_1_id]
                            sorted_list_1 = list((dict(sorted(tmp.items(),
                                                              key=lambda item: item[1]))).keys())
                            tmp = self.distances[election_2_id]
                            sorted_list_2 = list((dict(sorted(tmp.items(),
                                                              key=lambda item: item[1]))).keys())
                            if (election_1_id not in sorted_list_2[0:num_neighbors]) and (
                                    election_2_id not in sorted_list_1[0:num_neighbors]):
                                x[i][j] = 0.
                                normal = False
                        if normal:
                            x[i][j] = 1. / self.distances[election_1_id][
                                election_2_id]
                    else:
                        x[i][j] = self.distances[election_1_id][election_2_id]
                    x[i][j] = x[i][j] ** attraction_factor
                    x[j][i] = x[i][j]

        dt = [('weight', float)]
        x = x.view(dt)
        graph = nx.from_numpy_matrix(x)

        if num_neighbors is None:
            num_neighbors = 100

        if algorithm == 'spring':
            my_pos = nx.spring_layout(graph, iterations=num_iterations,
                                      dim=dim)
        elif algorithm in {'mds', 'MDS'}:
            my_pos = MDS(n_components=dim).fit_transform(x)
        elif algorithm in {'tsne', 'TSNE'}:
            my_pos = TSNE(n_components=dim).fit_transform(x)
        elif algorithm in {'se', 'SE'}:
            my_pos = SpectralEmbedding(n_components=dim).fit_transform(x)
        elif algorithm in {'isomap', 'ISOMAP'}:
            my_pos = Isomap(n_components=dim,
                            n_neighbors=num_neighbors).fit_transform(x)
        elif algorithm in {'lle', 'LLE'}:
            my_pos = LocallyLinearEmbedding(n_components=dim,
                                            n_neighbors=num_neighbors,
                                            max_iter=num_iterations,
                                            method=method).fit_transform(x)
        else:
            my_pos = []
            print("Unknown method!")

        points = {}
        for i, election_id in enumerate(self.distances):
            points[election_id] = [my_pos[i][d] for d in range(dim)]

        if self.store:
            file_name = f'{self.distance_name}_{str(dim)}d.csv'
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "coordinates", file_name)
            with open(path, 'w', newline='') as csvfile:

                writer = csv.writer(csvfile, delimiter=';')
                if dim == 2:
                    writer.writerow(["election_id", "x", "y"])
                elif dim == 3:
                    writer.writerow(["election_id", "x", "y", "z"])

                ctr = 0
                for election_id in self.elections:
                    x = round(points[election_id][0], 5)
                    y = round(points[election_id][1], 5)
                    if dim == 2:
                        writer.writerow([election_id, x, y])
                    elif dim == 3:
                        z = round(my_pos[ctr][2], 5)
                        writer.writerow([election_id, x, y, z])
                    ctr += 1

        self.coordinates = points

    def get_election_id_from_model_name(self, model: str) -> str:
        for election_id in self.elections:
            if self.elections[election_id].model == model:
                return election_id

    def print_map(self, dim: int = 2, **kwargs) -> None:
        """ Print the two-dimensional embedding of multi-dimensional
        map of the elections """
        if dim == 2:
            pr.print_map_2d(self, **kwargs)
        elif dim == 3:
            pr.print_map_3d(self, **kwargs)

    # def add_matrices_to_experiment(experiment):
    #     """ Import elections from a file """
    #
    #     matrices = {}
    #     vectors = {}
    #
    #     for family_id in experiment.families:
    #         for j in range(experiment.families[family_id].size):
    #             election_id = family_id + '_' + str(j)
    #             matrix = experiment.import_matrix(election_id)
    #             matrices[election_id] = matrix
    #             vectors[election_id] = matrix.transpose()
    #
    #     return matrices, vectors

    # def import_matrix(experiment, election_id):
    #
    #     file_name = election_id + '.csv'
    #     path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id,
    #     'matrices', file_name)
    #     num_candidates = experiment.elections[election_id].num_candidates
    #     matrix = np.zeros([num_candidates, num_candidates])
    #
    #     with open(path, 'r', newline='') as csv_file:
    #         reader = csv.DictReader(csv_file, delimiter=',')
    #         for i, row in enumerate(reader):
    #             for j, candidate_id in enumerate(row):
    #                 matrix[i][j] = row[candidate_id]
    #     return matrix

    def print_matrix(self, **kwargs):
        pr.print_matrix(experiment=self, **kwargs)

    def import_controllers(self):
        """ Import controllers from a file """

        families = {}

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            'map.csv')
        file_ = open(path, 'r')

        header = [h.strip() for h in file_.readline().split(';')]
        reader = csv.DictReader(file_, fieldnames=header, delimiter=';')

        starting_from = 0
        for row in reader:

            model = None
            color = None
            label = None
            params = None
            alpha = None
            size = None
            marker = None
            num_candidates = None
            num_voters = None

            if 'model' in row.keys():
                model = str(row['model']).strip()

            if 'color' in row.keys():
                color = str(row['color']).strip()

            if 'label' in row.keys():
                label = str(row['label'])

            if 'params' in row.keys():
                params = ast.literal_eval(str(row['params']))

            if 'alpha' in row.keys():
                alpha = float(row['alpha'])

            if 'size' in row.keys():
                size = int(row['size'])

            if 'marker' in row.keys():
                marker = str(row['marker']).strip()

            if 'num_candidates' in row.keys():
                num_candidates = int(row['num_candidates'])

            if 'num_voters' in row.keys():
                num_voters = int(row['num_voters'])

            if 'path' in row.keys():
                path = ast.literal_eval(str(row['path']))

            family_id = str(row['label'])
            # family_id = model

            show = True
            if row['show'].strip() != 't':
                show = False

            if model in {'urn_model'} and params['alpha'] is not None:
                family_id += '_' + str(float(params['alpha']))
            elif model in {'mallows'} and params['phi'] is not None:
                family_id += '_' + str(float(params['phi']))
            elif model in {'norm-mallows', 'norm-mallows_matrix'} \
                    and params['norm-phi'] is not None:
                family_id += '_' + str(float(params['norm-phi']))

            if size == 1:
                single_election = True
            else:
                single_election = False

            families[family_id] = Family(model=model,
                                         family_id=family_id,
                                         params=params, label=label,
                                         color=color, alpha=alpha, show=show,
                                         size=size, marker=marker,
                                         starting_from=starting_from,
                                         num_candidates=num_candidates,
                                         num_voters=num_voters, path=path,
                                         single_election=single_election)
            starting_from += size

        self.num_families = len(families)
        self.num_elections = sum(
            [families[family_id].size for family_id in families])
        self.main_order = [i for i in range(self.num_elections)]

        file_.close()
        return families

    def import_cooridnates(self, ignore=None) -> dict:
        """ Import from a file precomputed coordinates of all the points --
        each point refer to one election """

        if ignore is None:
            ignore = []

        points = {}
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", f'{self.distance_name}_2d.csv')

        with open(path, 'r', newline='') as csv_file:

            # ORIGINAL

            reader = csv.DictReader(csv_file, delimiter=';')
            ctr = 0

            for row in reader:
                if self.main_order[ctr] < self.num_elections and \
                        self.main_order[ctr] not in ignore:
                    points[row['election_id']] = [float(row['x']), float(row['y'])]
                ctr += 1

        return points

    def compute_points_by_families(self) -> None:
        """ Group all points by their families """

        points_by_families = {}

        if self.families is None:
            self.families = {}
            for i, election_id in enumerate(self.elections):
                ele = self.elections[election_id]
                # print(ele)
                model = ele.model
                family_id = model
                # param_1 = 0
                # param_2 = 0
                label = election_id
                color = COLORS[int(i % len(COLORS))]

                alpha = 1.
                # show = True
                # size = 1
                # marker = 'o'
                # starting_from = 0
                num_candidates = ele.num_candidates
                num_voters = ele.num_voters

                self.families[election_id] = Family(
                    model=model, family_id=family_id,
                    label=label, alpha=alpha,
                    color=color,
                    num_candidates=num_candidates, num_voters=num_voters)

            for family_id in self.families:

                points_by_families[family_id] = [[] for _ in range(3)]
                points_by_families[family_id][0].append(self.coordinates[family_id][0])
                points_by_families[family_id][1].append(self.coordinates[family_id][1])
                try:
                    points_by_families[family_id][2].append(self.coordinates[family_id][2])
                except Exception:
                    pass
        else:

            for family_id in self.families:

                points_by_families[family_id] = [[] for _ in range(3)]

                for i in range(self.families[family_id].size):
                    if self.families[family_id].size == 1:
                        election_id = family_id
                    else:
                        election_id = family_id + '_' + str(i)
                    points_by_families[family_id][0].append(self.coordinates[election_id][0])
                    points_by_families[family_id][1].append(self.coordinates[election_id][1])
                    try:
                        points_by_families[family_id][2].append(self.coordinates[election_id][2])
                    except Exception:
                        pass

        self.points_by_families = points_by_families

    def compute_feature(self, name: str = None, committee_size: int = 10) -> dict:

        feature_dict = {}

        for election_id in self.elections:
            print(election_id)
            feature = features.get_feature(name)
            election = self.elections[election_id]
            # print(election_id, election)
            if name in ['largest_cohesive_group']:
                value = feature(election, committee_size)

            elif name in {'avg_distortion_from_guardians',
                        'worst_distortion_from_guardians'}:
                value = feature(self, election_id)
            else:
                value = feature(election)
            # values.append(value)
            feature_dict[election_id] = value

        if self.store:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "features", str(name) + '.csv')
            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(["election_id", "value"])
                for key in feature_dict:
                    writer.writerow([key, feature_dict[key]])

        self.features[name] = feature_dict
        return feature_dict

    def get_distance(self, i, j):
        """ Compute Euclidean distance in two-dimensional space"""

        distance = 0.
        for d in range(2):
            distance += (self.coordinates[i][d] - self.coordinates[j][d]) ** 2

        return math.sqrt(distance)

    def rotate(self, angle) -> None:
        """ Rotate all the points by a given angle """

        for election_id in self.elections:
            self.coordinates[election_id][0], self.coordinates[election_id][1] = \
                self.rotate_point(0.5, 0.5, angle, self.coordinates[election_id][0],
                                  self.coordinates[election_id][1])

        self.compute_points_by_families()

    def reverse(self) -> None:
        """ Reverse all the points"""

        for election_id in self.elections:
            self.coordinates[election_id][0] = self.coordinates[election_id][0]
            self.coordinates[election_id][1] = -self.coordinates[election_id][1]

        self.compute_points_by_families()

    def update(self) -> None:
        """ Save current coordinates of all the points to the original file"""

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", f'{self.distance_name}_2d.csv')

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(["election_id", "x", "y"])

            for election_id in self.elections:
                x = round(self.coordinates[election_id][0], 5)
                y = round(self.coordinates[election_id][1], 5)
                writer.writerow([election_id, x, y])

    @staticmethod
    def rotate_point(cx, cy, angle, px, py) -> (float, float):
        """ Rotate two-dimensional point by an angle """

        s = math.sin(angle)
        c = math.cos(angle)
        px -= cx
        py -= cy
        x_new = px * c - py * s
        y_new = px * s + py * c
        px = x_new + cx
        py = y_new + cy

        return px, py

    def import_distances(self, self_distances=False, distance_name='emd-positionwise'):
        """ Import precomputed distances between each pair of elections from a file """

        file_name = str(distance_name) + '.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "distances", file_name)
        num_points = self.num_elections
        num_distances = int(num_points * (num_points - 1) / 2)

        hist_data = {}
        std = [[0. for _ in range(num_points)] for _ in range(num_points)]

        for family_id in self.families:
            for j in range(self.families[family_id].size):
                election_id = family_id + '_' + str(j)
                hist_data[election_id] = {}

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                election_id_1 = row['election_id_1']
                election_id_2 = row['election_id_2']
                hist_data[election_id_1][election_id_2] = float(
                    row['distance'])
                hist_data[election_id_2][election_id_1] = \
                    hist_data[election_id_1][election_id_2]

        return num_distances, hist_data, std

    def import_my_distances(self, distance_name: str = 'emd-positionwise') -> (dict, dict, dict):
        """ Import precomputed distances between each pair of elections from a file """
        file_name = str(distance_name) + '.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances", file_name)
        distances = {}
        times = {}
        stds = {}

        for election_id in self.elections:
            distances[election_id] = {}
            times[election_id] = {}
            stds[election_id] = {}

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                election_id_1 = row['election_id_1']
                election_id_2 = row['election_id_2']

                distances[election_id_1][election_id_2] = float(row['distance'])
                distances[election_id_2][election_id_1] = distances[election_id_1][election_id_2]
                try:
                    times[election_id_1][election_id_2] = float(row['time'])
                    times[election_id_2][election_id_1] = times[election_id_1][election_id_2]
                except KeyError:
                    pass

                try:
                    stds[election_id_1][election_id_2] = float(row['std'])
                    stds[election_id_2][election_id_1] = stds[election_id_1][election_id_2]
                except KeyError:
                    pass

        return distances, times, stds
