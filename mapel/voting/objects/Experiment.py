#!/usr/bin/env python
import ast
import copy
import csv
import itertools
import math
import os
import warnings
from abc import ABCMeta, abstractmethod
from threading import Thread
from time import sleep

import networkx as nx
import numpy as np

import mapel.voting._print as pr
import mapel.voting.elections.preflib as preflib
import mapel.voting.elections_main as _elections
import mapel.voting.features_main as features
import mapel.voting.metrics_main as metr
from mapel.voting._glossary import *
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


class Experiment:
    __metaclass__ = ABCMeta
    """Abstract set of elections."""

    def __init__(self, elections=None, distances=None, dim=2, store=True,
                 coordinates=None, distance_id='emd-positionwise', experiment_id=None,
                 election_type='ordinal', _import=True, clean=False):

        self._import = _import
        self.clean = clean
        self.experiment_id = experiment_id
        if clean:
            self.clean_elections()

        self.distance_id = distance_id

        self.elections = None
        self.distances = None
        self.coordinates = None

        self.families = {}
        self.times = {}
        self.stds = {}
        self.matchings = {}
        self.features = {}
        self.coordinates_by_families = {}

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
            self.store = store

        if isinstance(elections, dict):
            self.elections = elections
            print('=== Omitting import! ===')
        elif self.experiment_id != 'virtual':
            try:
                self.elections = self.add_elections_to_experiment()
                print('=== Elections imported successfully! ===')
            except FileNotFoundError:
                print('=== Elections not found! ===')
                self.elections = {}
        else:
            self.elections = {}

        if isinstance(distances, dict):
            self.distances = distances
            print('=== Omitting import! ===')
        elif self.experiment_id != 'virtual':
            try:
                self.distances, self.times, self.stds = self.add_distances_to_experiment()
                print('=== Distances imported successfully! ===')
            except FileNotFoundError:
                print('=== Distances not found! ===')
        else:
            self.distances = {}

        if isinstance(coordinates, dict):
            self.coordinates = coordinates
            print('=== Omitting import! ===')
        elif self.experiment_id != 'virtual':
            try:
                self.coordinates = self.add_coordinates_to_experiment(dim=dim)
                print('=== Coordinates imported successfully! ===')
            except FileNotFoundError:
                print('=== Coordinates not found! ===')
        else:
            self.coordinates = {}


    def add_family(self, model_id: str = None, params: dict = None, size=1, label=None, color="black",
                   alpha=1., show=True, marker='o', starting_from=0, num_candidates=None,
                   family_id=None, single_election=False, num_nodes=None,
                   path=None, election_id=None):
        """ Add family of elections to the experiment """

        if election_id is not None:
            family_id = election_id

        if self.families is None:
            self.families = {}

        elif label is None:
            label = family_id

        self.families[family_id] = Family(model_id=model_id, family_id=family_id,
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
        model_id = self.families[family_id].model_id

        ids = _elections.prepare_statistical_culture_family(experiment=self,
                                                            model_id=model_id,
                                                            family_id=family_id,
                                                            params=copy.deepcopy(params))

        self.families[family_id].election_ids = ids

        return ids

    def add_election(self, model="none", params=None, label=None,
                     color="black", alpha=1., show=True, marker='x', starting_from=0, size=1,
                     num_candidates=None, num_voters=None, election_id=None, num_nodes=None):
        """ Add election to the experiment """

        return self.add_family(model_id=model, params=params, size=size, label=label, color=color,
                               alpha=alpha, show=show, marker=marker, starting_from=starting_from,
                               num_candidates=num_candidates,
                               family_id=election_id, num_nodes=num_nodes, single_election=True)[0]

    def add_graph(self, **kwargs):
        return self.add_election(**kwargs)

    def prepare_elections(self):
        """ Prepare elections for a given experiment """

        if self.elections is None:
            self.elections = {}

        elections = {}
        if self.store:

            for family_id in self.families:
                params = self.families[family_id].params
                model_id = self.families[family_id].model_id

                # if self.clean:
                #     path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections")
                #     for file_name in os.listdir(path):
                #         os.remove(os.path.join(path, file_name))

                if model_id in LIST_OF_PREFLIB_MODELS:
                    ids = preflib.prepare_preflib_family(
                        experiment=self, model=model_id, family_id=family_id, params=params)

                    self.families[family_id].election_ids = ids

                elif model_id not in ['core', 'pabulib'] and (model_id in NICE_NAME or
                                                           model_id in LIST_OF_FAKE_MODELS or
                                                           model_id in APPROVAL_MODELS):
                    tmp_elections = _elections.prepare_statistical_culture_family(
                        experiment=self, model_id=model_id, family_id=family_id, params=params)

                    self.families[family_id].election_ids = tmp_elections.keys()

                    for election_id in tmp_elections:
                        elections[election_id] = tmp_elections[election_id]

                    # print(tmp_elections)

        # self.elections = self.add_elections_to_experiment()
        self.elections = elections

    def compute_winners(self, method=None, num_winners=1):
        for election_id in self.elections:
            self.elections[election_id].compute_winners(method=method, num_winners=num_winners)

    def compute_alternative_winners(self, method=None, num_winners=None, num_parties=None):
        for election_id in self.elections:
            for party_id in range(num_parties):
                self.elections[election_id].compute_alternative_winners(
                    method=method, party_id=party_id, num_winners=num_winners)

    def compute_distances(self, distance_id: str = 'emd-positionwise', num_threads: int = 1,
                          self_distances: bool = False, vector_type: str = 'A',
                          printing: bool = False) -> None:
        """ Compute distances between elections (using threads) """

        self.distance_id = distance_id

        # precompute vectors, matrices, etc...
        if '-approvalwise' in distance_id:
            for election in self.elections.values():
                election.votes_to_approvalwise_vector()
        elif '-coapproval_frequency' in distance_id or '-flow' in distance_id:
            for election in self.elections.values():
                election.votes_to_coapproval_frequency_vectors(vector_type=vector_type)
        elif '-voterlikeness' in distance_id:
            for election in self.elections.values():
                election.votes_to_voterlikeness_vectors(vector_type=vector_type)
        elif '-candidatelikeness' in distance_id:
            for election in self.elections.values():
                # print(election)
                election.votes_to_candidatelikeness_sorted_vectors()
        elif '-pairwise' in distance_id:
            for election in self.elections.values():
                election.votes_to_pairwise_matrix()
        # continue with normal code

        matchings = {election_id: {} for election_id in self.elections}
        distances = {election_id: {} for election_id in self.elections}
        times = {election_id: {} for election_id in self.elections}

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
                    ["election_id_1", "election_id_2", "distance", "time"])

                for election_1, election_2 in itertools.combinations(self.elections, 2):
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
    def add_folders_to_experiment(self):
        pass

    @abstractmethod
    def create_structure(self):
        pass

    def embed(self, algorithm: str = 'spring', num_iterations: int = 1000, radius: float = np.infty,
              dim: int = 2, num_neighbors: int = None, method: str = 'standard',
              zero_distance: int = 0.01) -> None:

        if algorithm == 'spring':
            attraction_factor = 2
        else:
            attraction_factor = 1

        num_elections = len(self.distances)

        x = np.zeros((num_elections, num_elections))

        for i, election_id_1 in enumerate(self.distances):
            for j, election_id_2 in enumerate(self.distances):
                if i < j:
                    if self.distances[election_id_1][election_id_2] == 0.:
                        self.distances[election_id_1][election_id_2] = zero_distance
                    if algorithm in {'spring'}:
                        normal = True
                        if self.distances[election_id_1][election_id_2] > radius:
                            x[i][j] = 0.
                            normal = False
                        if num_neighbors is not None:
                            tmp = self.distances[election_id_1]
                            sorted_list_1 = list((dict(sorted(tmp.items(),
                                                              key=lambda item: item[1]))).keys())
                            tmp = self.distances[election_id_2]
                            sorted_list_2 = list((dict(sorted(tmp.items(),
                                                              key=lambda item: item[1]))).keys())
                            if (election_id_1 not in sorted_list_2[0:num_neighbors]) and (
                                    election_id_2 not in sorted_list_1[0:num_neighbors]):
                                x[i][j] = 0.
                                normal = False
                        if normal:
                            x[i][j] = 1. / self.distances[election_id_1][
                                election_id_2]
                    else:
                        x[i][j] = self.distances[election_id_1][election_id_2]
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
            my_pos = Isomap(n_components=dim, n_neighbors=num_neighbors).fit_transform(x)
        elif algorithm in {'lle', 'LLE'}:
            my_pos = LocallyLinearEmbedding(n_components=dim,
                                            n_neighbors=num_neighbors,
                                            max_iter=num_iterations,
                                            method=method).fit_transform(x)
        else:
            my_pos = []
            print("Unknown method!")

        coordinates = {}
        for i, election_id in enumerate(self.distances):
            coordinates[election_id] = [my_pos[i][d] for d in range(dim)]

        if self.store:
            file_name = f'{self.distance_id}_{str(dim)}d.csv'
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
                    x = round(coordinates[election_id][0], 5)
                    y = round(coordinates[election_id][1], 5)
                    if dim == 2:
                        writer.writerow([election_id, x, y])
                    elif dim == 3:
                        z = round(my_pos[ctr][2], 5)
                        writer.writerow([election_id, x, y, z])
                    ctr += 1

        self.coordinates = coordinates

    def get_election_id_from_model_name(self, model_id: str) -> str:
        for family_id in self.families:
            if self.families[family_id].model_id == model_id:
                return family_id

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

        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'map.csv')
        file_ = open(path, 'r')

        header = [h.strip() for h in file_.readline().split(';')]
        reader = csv.DictReader(file_, fieldnames=header, delimiter=';')

        all_num_candidates = []
        all_num_voters = []

        starting_from = 0
        for row in reader:

            model_id = None
            color = None
            label = None
            params = None
            alpha = None
            size = None
            marker = None
            num_candidates = None
            num_voters = None

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

            if 'num_candidates' in row.keys():
                num_candidates = int(row['num_candidates'])

            if 'num_voters' in row.keys():
                num_voters = int(row['num_voters'])

            if 'path' in row.keys():
                path = ast.literal_eval(str(row['path']))


            show = row['show'].strip() == 't'

            if model_id in {'urn_model'} and params['alpha'] is not None:
                family_id += '_' + str(float(params['alpha']))
            elif model_id in {'mallows'} and params['phi'] is not None:
                family_id += '_' + str(float(params['phi']))
            elif model_id in {'norm-mallows', 'norm-mallows_matrix'} \
                    and params['norm-phi'] is not None:
                family_id += '_' + str(float(params['norm-phi']))

            single_election = size == 1

            families[family_id] = Family(model_id=model_id,
                                         family_id=family_id,
                                         params=params, label=label,
                                         color=color, alpha=alpha, show=show,
                                         size=size, marker=marker,
                                         starting_from=starting_from,
                                         num_candidates=num_candidates,
                                         num_voters=num_voters, path=path,
                                         single_election=single_election)
            starting_from += size

            all_num_candidates.append(num_candidates)
            all_num_voters.append(num_voters)

        check_if_all_equal(all_num_candidates, 'num_candidates')
        check_if_all_equal(all_num_voters, 'num_voters')

        self.num_families = len(families)
        self.num_elections = sum([families[family_id].size for family_id in families])
        self.main_order = [i for i in range(self.num_elections)]

        file_.close()
        return families

    def add_coordinates_to_experiment(self, ignore=None, dim=2) -> dict:
        """ Import from a file precomputed coordinates of all the points --
        each point refer to one election """

        if ignore is None:
            ignore = []

        coordinates = {}
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", f'{self.distance_id}_{dim}d.csv')

        with open(path, 'r', newline='') as csv_file:

            # ORIGINAL

            reader = csv.DictReader(csv_file, delimiter=';')

            warn = False

            for row in reader:
                election_id = row['election_id']
                if dim == 2:
                    coordinates[election_id] = [float(row['x']), float(row['y'])]
                elif dim == 3:
                    coordinates[election_id] = [float(row['x']), float(row['y']), float(row['z'])]

                if election_id not in self.elections:
                    warn = True

            if warn:
                text = f'Possibly outdated coordinates are imported!'
                warnings.warn(text)

        return coordinates

    def compute_coordinates_by_families(self) -> None:
        """ Group all points by their families """

        coordinates_by_families = {}

        if self.families is None:
            self.families = {}
            for i, election_id in enumerate(self.elections):
                ele = self.elections[election_id]
                model = ele.model_id
                family_id = model
                label = election_id
                color = COLORS[int(i % len(COLORS))]

                alpha = 1.
                num_candidates = ele.num_candidates
                num_voters = ele.num_voters

                self.families[election_id] = Family(
                    model_id=model, family_id=family_id,
                    label=label, alpha=alpha,
                    color=color,
                    num_candidates=num_candidates, num_voters=num_voters)

            for family_id in self.families:

                coordinates_by_families[family_id] = [[] for _ in range(3)]
                coordinates_by_families[family_id][0].append(self.coordinates[family_id][0])
                coordinates_by_families[family_id][1].append(self.coordinates[family_id][1])
                try:
                    coordinates_by_families[family_id][2].append(self.coordinates[family_id][2])
                except Exception:
                    pass
        else:

            for family_id in self.families:

                coordinates_by_families[family_id] = [[] for _ in range(3)]

                for election_id in self.families[family_id].election_ids:
                    coordinates_by_families[family_id][0].append(self.coordinates[election_id][0])
                    coordinates_by_families[family_id][1].append(self.coordinates[election_id][1])
                    try:
                        coordinates_by_families[family_id][2].append(
                            self.coordinates[election_id][2])
                    except Exception:
                        pass

        self.coordinates_by_families = coordinates_by_families

    def compute_feature(self, feature_id: str = None,
                        feature_params=None) -> dict:
        if feature_params is None:
            feature_params = {}

        feature_dict = {}

        for election_id in self.elections:
            print(election_id)
            feature = features.get_feature(feature_id)
            election = self.elections[election_id]
            if feature_id in ['monotonicity_1', 'monotonicity_2']:
                value = feature(self, election)

            elif feature_id in ['largest_cohesive_group', 'number_of_cohesive_groups',
                                'number_of_cohesive_groups_brute',
                                'proportionality_degree_pav',
                                'proportionality_degree_av',
                                'proportionality_degree_cc',]:
                value = feature(election, feature_params)

            elif feature_id in {'avg_distortion_from_guardians',
                                'worst_distortion_from_guardians',
                                'distortion_from_all',
                                'distortion_from_top_100'}:
                value = feature(self, election_id)
            elif feature_id in {'partylist'}:
                value = feature(election, feature_params)
            else:
                value = feature(election)
            feature_dict[election_id] = value

        if self.store:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "features", f'{feature_id}.csv')
            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(["election_id", "value", "bound", "num_large_parties"])
                if feature_id in {'partylist'}:
                    for key in feature_dict:
                        writer.writerow([key, feature_dict[key][0], feature_dict[key][1],
                                         feature_dict[key][2]])
                else:
                    for key in feature_dict:
                        writer.writerow([key, feature_dict[key]])

        self.features[feature_id] = feature_dict
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

        self.compute_coordinates_by_families()

    def reverse(self) -> None:
        """ Reverse all the points"""

        for election_id in self.elections:
            self.coordinates[election_id][0] = self.coordinates[election_id][0]
            self.coordinates[election_id][1] = -self.coordinates[election_id][1]

        self.compute_coordinates_by_families()

    def update(self) -> None:
        """ Save current coordinates of all the points to the original file"""

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", f'{self.distance_id}_2d.csv')

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

        s, c = math.sin(angle), math.cos(angle)
        px -= cx
        py -= cy
        x_new, y_new = px * c - py * s, px * s + py * c
        px, py = x_new + cx,  y_new + cy

        return px, py

    # def import_distances(self, self_distances=False, distance_id='emd-positionwise'):
    #     """ Import precomputed distances between each pair of elections from a file """
    #
    #     file_name = str(distance_id) + '.csv'
    #     path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
    #                         "distances", file_name)
    #     num_points = self.num_elections
    #     num_distances = int(num_points * (num_points - 1) / 2)
    #
    #     hist_data = {}
    #     std = [[0. for _ in range(num_points)] for _ in range(num_points)]
    #
    #     for family_id in self.families:
    #         for j in range(self.families[family_id].size):
    #             election_id = family_id + '_' + str(j)
    #             hist_data[election_id] = {}
    #
    #     with open(path, 'r', newline='') as csv_file:
    #         reader = csv.DictReader(csv_file, delimiter=';')
    #
    #         for row in reader:
    #             election_id_1 = row['election_id_1']
    #             election_id_2 = row['election_id_2']
    #             hist_data[election_id_1][election_id_2] = float(
    #                 row['distance'])
    #             hist_data[election_id_2][election_id_1] = \
    #                 hist_data[election_id_1][election_id_2]
    #
    #     return num_distances, hist_data, std

    def add_distances_to_experiment(self) -> (dict, dict, dict):
        """ Import precomputed distances between each pair of elections from a file """
        file_name = f'{self.distance_id}.csv'
        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'distances', file_name)

        distances = {}
        times = {}
        stds = {}

        with open(path, 'r', newline='') as csv_file:

            reader = csv.DictReader(csv_file, delimiter=';')
            warn = False

            for row in reader:
                election_id_1 = row['election_id_1']
                election_id_2 = row['election_id_2']

                if election_id_1 not in distances:
                    distances[election_id_1] = {}
                if election_id_1 not in times:
                    times[election_id_1] = {}
                if election_id_1 not in stds:
                    stds[election_id_1] = {}

                if election_id_2 not in distances:
                    distances[election_id_2] = {}
                if election_id_2 not in times:
                    times[election_id_2] = {}
                if election_id_2 not in stds:
                    stds[election_id_2] = {}

                try:
                    distances[election_id_1][election_id_2] = float(row['distance'])
                    distances[election_id_2][election_id_1] = distances[election_id_1][election_id_2]
                except KeyError:
                    pass

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

                if election_id_1 not in self.elections:
                    warn = True

            if warn:
                text = f'Possibly outdated distances are imported!'
                warnings.warn(text)


        return distances, times, stds

    def clean_elections(self):
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections")
        for file_name in os.listdir(path):
            os.remove(os.path.join(path, file_name))


def check_if_all_equal(values, subject):
    if any(x != values[0] for x in values):
        text = f'Not all {subject} values are equal!'
        warnings.warn(text)
