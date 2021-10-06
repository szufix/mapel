#!/usr/bin/env python
import copy
from abc import ABCMeta, abstractmethod

from mapel.voting.objects.Election import Election
from mapel.voting.objects.Family import Family

import mapel.voting._elections as _elections
import mapel.voting.features as features

from threading import Thread
from time import sleep

import mapel.voting._metrics as metr

import mapel.voting.print as pr

import mapel.voting.elections.preflib as preflib

import math
import csv
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ast

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
    """Abstract set of instances."""

    def __init__(self, ignore=None, instances=None, distances=None, with_matrices=False,
                 coordinates=None, distance_name='emd-positionwise', experiment_id=None,
                 instance_type='ordinal'):

        self.distance_name = distance_name
        self.instances = {}
        self.default_num_candidates = 10
        self.default_num_voters = 100

        self.families = None
        self.distances = None
        self.times = None
        self.stds = None
        self.points_by_families = None
        self.matchings = None

        self.num_families = None
        self.num_instances = None
        self.main_order = None
        self.instance_type = instance_type

        if experiment_id is None:
            self.store = False
        else:
            self.store = True
            self.experiment_id = experiment_id
            self.create_structure()
            self.families = self.import_controllers(ignore=ignore)
            self.attraction_factor = 1

        if instances is not None:
            if instances == 'import':
                self.instances = self.add_instances_to_experiment()
            else:
                self.instances = instances

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

    def set_default_num_candidates(self, num_candidates):
        self.default_num_candidates = num_candidates

    def set_default_num_voters(self, num_voters):
        self.default_num_voters = num_voters

    def add_instance(self, model="none", params=None, label=None,
                     color="black", alpha=1., show=True, marker='x', starting_from=0, size=1,
                     num_candidates=None, num_voters=None, election_id=None, num_nodes=None):
        """ Add election to the experiment """

        if num_candidates is None:
            num_candidates = self.default_num_candidates

        if num_voters is None:
            num_voters = self.default_num_voters

        return self.add_family(model=model, params=params, size=size, label=label, color=color,
                               alpha=alpha, show=show,  marker=marker, starting_from=starting_from,
                               num_candidates=num_candidates, num_voters=num_voters,
                               family_id=election_id, num_nodes=num_nodes, single_election=True)[0]

    def add_family(self, model="none", params=None, size=1, label=None, color="black",
                   alpha=1., show=True, marker='o', starting_from=0, num_candidates=None,
                   num_voters=None, family_id=None, single_election=False, num_nodes=None,
                   path=None, name=None):
        """ Add family of instances to the experiment """
        if name is not None:
            family_id = name

        if num_candidates is None:
            num_candidates = self.default_num_candidates

        if num_voters is None:
            num_voters = self.default_num_voters

        if self.families is None:
            self.families = {}

        if family_id is None:
            family_id = model + '_' + str(num_candidates) + '_' + str(num_voters)
            if model in {'urn_model'} and params['alpha'] is not None:
                family_id += '_' + str(float(params['alpha']))
            elif model in {'mallows'} and params['phi'] is not None:
                family_id += '_' + str(float(params['phi']))
            elif model in {'norm-mallows', 'norm-mallows_matrix'} \
                    and params['norm-phi'] is not None:
                family_id += '_' + str(float(params['norm-phi']))

        if label in ["UN", "ID", "AN", "ST", "CON", "WAL", "CAT",
                     "SHI", "MID"]:
            single_election = True
            family_id = label

        elif label is None:
            label = family_id

        self.families[family_id] = Family(model=model, family_id=family_id,
                                          params=params, label=label, color=color, alpha=alpha,
                                          show=show, size=size, marker=marker,
                                          starting_from=starting_from, num_nodes=num_nodes,
                                          num_candidates=num_candidates,
                                          num_voters=num_voters, path=path,
                                          single_election=single_election)

        self.num_families = len(self.families)
        self.num_instances = sum([self.families[family_id].size for family_id in self.families])
        self.main_order = [i for i in range(self.num_instances)]

        params = self.families[family_id].params
        model = self.families[family_id].model

        ids = _elections.prepare_statistical_culture_family(experiment=self,
                                                            model=model,
                                                            family_id=family_id,
                                                            params=copy.deepcopy(params))

        self.families[family_id].election_ids = ids

        return ids

    def add_election(self, name=None, **kwargs):
        return self.add_instance(election_id=name, **kwargs)

    def add_graph(self, name=None, **kwargs):
        return self.add_instance(election_id=name, **kwargs)

    def prepare_elections(self):
        self.prepare_instances()

    def prepare_instances(self):
        """ Prepare instances for a given experiment """

        if self.instances is None:
            self.instances = {}

        if self.store:
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "instances")
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
        for election_id in self.instances:
            self.instances[election_id].compute_winners(method=method, num_winners=num_winners)

    def compute_alternative_winners(self, method=None, num_winners=None, num_parties=None):
        for election_id in self.instances:
            for party_id in range(num_parties):
                self.instances[election_id].compute_alternative_winners(
                    method=method, party_id=party_id, num_winners=num_winners)

    def compute_distances(self, distance_name='emd-positionwise', num_threads=1,
                          self_distances=False):
        """ Compute distances between instances (using threads) """

        self.distance_name = distance_name

        # precompute vectors, matrices, etc...
        if '-approval_frequency' in distance_name:
            for instance in self.instances.values():
                instance.votes_to_approval_frequency_vector()
        elif '-coapproval_frequency_vectors' in distance_name:
            for instance in self.instances.values():
                instance.votes_to_coapproval_frequency_vectors()
        # continue with normal code

        matchings = {}
        distances = {}
        times = {}
        for election_id in self.instances:
            distances[election_id] = {}
            times[election_id] = {}
            matchings[election_id] = {}

        threads = [{} for _ in range(num_threads)]

        ids = []
        for i, election_1 in enumerate(self.instances):
            for j, election_2 in enumerate(self.instances):
                if i == j:
                    if self_distances:
                        ids.append((election_1, election_2))
                elif i < j:
                    ids.append((election_1, election_2))

        num_distances = len(ids)

        for t in range(num_threads):
            print('thread: ', t)
            sleep(0.1)
            start = int(t * num_distances / num_threads)
            stop = int((t + 1) * num_distances / num_threads)
            thread_ids = ids[start:stop]
            copy_t = t

            threads[t] = Thread(target=metr.single_thread, args=(
                self, distances, times, thread_ids, copy_t, matchings))
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

                for i, election_1 in enumerate(self.instances):
                    for j, election_2 in enumerate(self.instances):
                        if i < j:
                            distance = str(distances[election_1][election_2])
                            time = str(distances[election_1][election_2])
                            writer.writerow([election_1, election_2, distance, time])

        self.distances = distances
        self.times = times
        self.matchings = matchings

    @abstractmethod
    def add_instances_to_experiment(self):
        pass

    def create_structure(self):

        # PREPARE STRUCTURE

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

            path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "map.csv")
            with open(path, 'w') as file_csv:
                file_csv.write("size;num_candidates;num_voters;model;params;"
                    "color;alpha;label;marker;show\n")
                file_csv.write(
                    "3;10;100;impartial_culture;{};black;1;"
                    "Impartial Culture;o;t\n")
                file_csv.write("3;10;100;iac;{};black;0.7;IAC;o;t\n")
                file_csv.write("3;10;100;conitzer;{};limegreen;1;SP by Conitzer;o;t\n")
                file_csv.write("3;10;100;walsh;{};olivedrab;1;SP by Walsh;o;t\n")
                file_csv.write("3;10;100;spoc_conitzer;{};DarkRed;0.7;SPOC;o;t\n")
                file_csv.write("3;10;100;group-separable;{};blue;1;Group-Separable;o;t\n")
                file_csv.write("3;10;100;single-crossing;{};purple;0.6;Single-Crossing;o;t\n")
                file_csv.write("3;10;100;1d_interval;{};DarkGreen;1;1D Interval;o;t\n")
                file_csv.write("3;10;100;2d_disc;{};Green;1;2D Disc;o;t\n")
                file_csv.write("3;10;100;3d_cube;{};ForestGreen;0.7;3D Cube;o;t\n")
                file_csv.write("3;10;100;2d_sphere;{};black;0.2;2D Sphere;o;t\n")
                file_csv.write("3;10;100;3d_sphere;{};black;0.4;3D Sphere;o;t\n")
                file_csv.write("3;10;100;urn_model;{'alpha':0.1};yellow;1;"
                    "Urn model 0.1;o;t\n")
                file_csv.write(
                    "3;10;100;norm-mallows;{'norm-phi':0.5};blue;1;"
                    "Norm-Mallows 0.5;o;t\n")
                file_csv.write(
                    "3;10;100;urn_model;{'alpha':0};orange;1;"
                    "Urn model (gamma);o;t\n")
                file_csv.write(
                    "3;10;100;norm-mallows;{'norm-phi':0};cyan;1;"
                    "Norm-Mallows (uniform);o;t\n")
                file_csv.write("1;10;100;identity;{};blue;1;Identity;x;t\n")
                file_csv.write("1;10;100;uniformity;{};black;1;Uniformity;x;t\n")
                file_csv.write("1;10;100;antagonism;{};red;1;Antagonism;x;t\n")
                file_csv.write("1;10;100;stratification;{};green;1;Stratification;x;t\n")
                file_csv.write("1;10;100;walsh_matrix;{};olivedrab;1;Walsh Matrix;x;t\n")
                file_csv.write("1;10;100;conitzer_matrix;{};limegreen;1;"
                    "Conitzer Matrix;x;t\n")
                file_csv.write("1;10;100;single-crossing_matrix;{};purple;0.6;"
                    "Single-Crossing Matrix;x;t\n")
                file_csv.write("1;10;100;gs_caterpillar_matrix;{};green;1;"
                    "GS Caterpillar Matrix;x;t\n")
                file_csv.write("3;10;100;unid;{};blue;1;UNID;3;f\n")
                file_csv.write("3;10;100;anid;{};black;1;ANID;3;f\n")
                file_csv.write("3;10;100;stid;{};black;1;STID;3;f\n")
                file_csv.write("3;10;100;anun;{};black;1;ANUN;3;f\n")
                file_csv.write("3;10;100;stun;{};black;1;STUN;3;f\n")
                file_csv.write("3;10;100;stan;{};red;1;STAN;3;f\n")
        except FileExistsError:
            print("Experiment already exists!")


    def add_distances_to_experiment(self, distance_name=None):
        distances, times, stds = self.import_my_distances(distance_name=distance_name)
        return distances, times, stds

    def add_coordinates_to_experiment(self):
        coordinates = self.import_cooridnates()
        return coordinates

    def embed(self, attraction_factor=1, algorithm='spring',
              num_iterations=1000, radius=np.infty, dim=2,
              distance_name='emd-positionwise', num_neighbors=None,
              method='standard'):
        num_instances = len(self.distances)

        x = np.zeros((num_instances, num_instances))

        for i, election_1_id in enumerate(self.distances):
            for j, election_2_id in enumerate(self.distances):
                if i < j:
                    if self.distances[election_1_id][election_2_id] == 0:
                        self.distances[election_1_id][election_2_id] = 0.01
                    if algorithm in {'spring'}:
                        normal = True
                        if self.distances[election_1_id][election_2_id] \
                                > radius:
                            x[i][j] = 0.
                            normal = False
                        if num_neighbors is not None:
                            tmp = self.distances[election_1_id]
                            sorted_list_1 = list((dict(sorted(tmp.items(), key=lambda item:
                                                              item[1]))).keys())
                            tmp = self.distances[election_2_id]
                            sorted_list_2 = list((dict(sorted(tmp.items(),key=lambda item:
                                                              item[1]))).keys())
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
            file_name = os.path.join(os.getcwd(), "experiments",
                                     self.experiment_id,
                                     "coordinates",
                                     distance_name + "_" + str(dim) + "d_a" +
                                     str(float(attraction_factor)) + ".csv")

            with open(file_name, 'w', newline='') as csvfile:

                writer = csv.writer(csvfile, delimiter=';')
                if dim == 2:
                    writer.writerow(["election_id", "x", "y"])
                elif dim == 3:
                    writer.writerow(["election_id", "x", "y", "z"])

                ctr = 0
                for model_id in self.families:
                    for election_id in \
                            self.families[model_id].election_ids:
                        x = round(points[election_id][0], 5)
                        y = round(points[election_id][1], 5)
                        if dim == 2:
                            writer.writerow([election_id, x, y])
                        elif dim == 3:
                            z = round(my_pos[ctr][2], 5)
                            writer.writerow([election_id, x, y, z])
                        ctr += 1

        self.coordinates = points

    def print_map_tmp(self, group_by=None, saveas=None):
        if group_by is None:

            for election_id in self.coordinates:
                plt.scatter(self.coordinates[election_id][0], self.coordinates[election_id][1],
                            label=election_id)
            plt.legend()
            plt.show()

        else:

            for color in group_by:
                x_array = []
                y_array = []
                for election_id in group_by[color]:
                    x_array.append(self.coordinates[election_id][0])
                    y_array.append(self.coordinates[election_id][1])
                plt.scatter(x_array, y_array, label=group_by[color][0],color=color)
            plt.legend()
            if saveas is not None:
                file_name = saveas + ".png"
                path = os.path.join(os.getcwd(), file_name)
                plt.savefig(path, bbox_inches='tight')
            plt.show()

    def get_election_id_from_model_name(self, model_name):
        for election_id in self.instances:
            if self.instances[election_id].model == model_name:
                return election_id

    def print_map(self, dim=2, **kwargs):
        """ Print the two-dimensional embedding of multi-dimensional
        map of the instances """
        if dim == 2:
            self.print_map_2d(**kwargs)
        elif dim == 3:
            self.print_map_3d(**kwargs)

    def print_map_2d(self, mask=False, mixed=False, fuzzy_paths=True,
                     xlabel=None, shading=False,
                     angle=0, reverse=False, update=False, feature=None,
                     attraction_factor=2, axis=False,
                     distance_name="emd-positionwise", guardians=False,
                     ticks=None, skeleton=None, roads=None,
                     title=None, dim=2, event='none',
                     saveas=None, show=True, ms=20, normalizing_func=None,
                     xticklabels=None, cmap=None,
                     ignore=None, marker_func=None, tex=False, black=False,
                     legend=True, levels=False, tmp=False, adjust=False):

        if skeleton is None:
            skeleton = []

        if roads is None:
            roads = []

        self.compute_points_by_families()

        if adjust:

            try:
                uniformity = self.get_election_id_from_model_name('uniformity')
                identity = self.get_election_id_from_model_name('identity')
                antagonism = self.get_election_id_from_model_name('antagonism')
                stratification = self.get_election_id_from_model_name('stratification')

                d_x = self.coordinates[identity][0] - self.coordinates[uniformity][0]
                d_y = self.coordinates[identity][1] - self.coordinates[uniformity][1]
                alpha = math.atan(d_x / d_y)
                self.rotate(alpha - math.pi / 2.)
                if self.coordinates[uniformity][0] > self.coordinates[identity][0]:
                    self.rotate(math.pi)

                if self.coordinates[antagonism][1] < self.coordinates[stratification][1]:
                    self.reverse()
            except:
                pass

            try:
                uniformity = self.get_election_id_from_model_name('approval_ic_0.5')
                identity = self.get_election_id_from_model_name('approval_id_0.5')
                antagonism = self.get_election_id_from_model_name('approval_ones')
                stratification = self.get_election_id_from_model_name('approval_zeros')

                d_x = self.coordinates[identity][0] - self.coordinates[uniformity][0]
                d_y = self.coordinates[identity][1] - self.coordinates[uniformity][1]
                alpha = math.atan(d_x / d_y)
                self.rotate(alpha - math.pi / 2.)
                if self.coordinates[uniformity][0] > self.coordinates[identity][0]:
                    self.rotate(math.pi)

                if self.coordinates[antagonism][1] < self.coordinates[stratification][1]:
                    self.reverse()
            except:
                pass

        if angle != 0:
            self.rotate(angle)

        if reverse:
            self.reverse()

        if adjust or update:
            self.update()

        if cmap is None:
            cmap = pr.custom_div_cmap()

        if feature is not None:
            fig = plt.figure(figsize=(6.4, 6.4 + 0.48))
        else:
            fig = plt.figure()

        ax = fig.add_subplot()

        plt.axis('equal')

        if not axis:
            plt.axis('off')


        # COLORING
        if feature is not None:
            pr.color_map_by_feature(experiment=self, fig=fig, ax=ax,
                                    feature=feature,
                                    normalizing_func=normalizing_func,
                                    marker_func=marker_func,
                                    xticklabels=xticklabels, ms=ms, cmap=cmap,
                                    ticks=ticks)
        else:
            if event in {'skeleton'}:
                pr.skeleton_coloring(experiment=self, ax=ax, ms=ms, dim=dim)
                pr.add_skeleton(experiment=self, skeleton=skeleton, ax=ax)
                pr.add_roads(experiment=self, roads=roads, ax=ax)
            else:
                if shading:
                    pr.basic_coloring_with_shading(experiment=self, ax=ax, ms=ms, dim=dim)
                else:
                    pr.basic_coloring(experiment=self, ax=ax, ms=ms, dim=dim)


        # BACKGROUND
        pr.basic_background(ax=ax, values=feature, legend=legend,
                            saveas=saveas, xlabel=xlabel,
                            title=title)

        if tex:
            pr.saveas_tex(saveas=saveas)

        if show:
            plt.show()

    def print_map_3d(self, mask=False, mixed=False, fuzzy_paths=True,
                     xlabel=None,
                     angle=0, reverse=False, update=False, feature=None,
                     attraction_factor=1, axis=False,
                     distance_name="emd-positionwise", guardians=False,
                     ticks=None, dim=3,
                     title=None, shading=False,
                     saveas="map_2d", show=True, ms=20, normalizing_func=None,
                     xticklabels=None, cmap=None,
                     ignore=None, marker_func=None, tex=False, black=False,
                     legend=True, levels=False, tmp=False):

        self.compute_points_by_families()

        # if angle != 0:
        #     self.rotate(angle)

        # if reverse:
        #     self.reverse()

        # if update:
        #     self.update()

        if cmap is None:
            cmap = pr.custom_div_cmap()

        if feature is not None:
            fig = plt.figure(figsize=(6.4, 4.8 + 0.48))
        else:
            fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        if not axis:
            plt.axis('off')

        # COLORING
        if feature is not None:
            pr.color_map_by_feature(experiment=self, fig=fig, ax=ax,
                                    feature=feature,
                                    normalizing_func=normalizing_func,
                                    marker_func=marker_func,
                                    xticklabels=xticklabels, ms=ms, cmap=cmap,
                                    ticks=ticks, dim=dim)
        else:
            pr.basic_coloring(experiment=self, ax=ax, ms=ms, dim=dim)

        # BACKGROUND
        pr.basic_background(ax=ax, values=feature, legend=legend,
                            saveas=saveas, xlabel=xlabel,
                            title=title)

        if tex:
            pr.saveas_tex(saveas=saveas)

        if show:
            plt.show()

    # def add_matrices_to_experiment(self):
    #     """ Import instances from a file """
    #
    #     matrices = {}
    #     vectors = {}
    #
    #     for family_id in self.families:
    #         for j in range(self.families[family_id].size):
    #             election_id = family_id + '_' + str(j)
    #             matrix = self.import_matrix(election_id)
    #             matrices[election_id] = matrix
    #             vectors[election_id] = matrix.transpose()
    #
    #     return matrices, vectors

    # def import_matrix(self, election_id):
    #
    #     file_name = election_id + '.csv'
    #     path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
    #     'matrices', file_name)
    #     num_candidates = self.instances[election_id].num_candidates
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

    def import_controllers(self, ignore=None):
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

            single_election = False
            # if label in ["UN", "ID", "AN", "ST", "CON", "WAL", "CAT",
            #              "SHI", "MID"]:
            #     single_election = True
            #     family_id = label

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
        self.num_instances = sum(
            [families[family_id].size for family_id in families])
        self.main_order = [i for i in range(self.num_instances)]

        if ignore is None:
            ignore = []

        file_.close()
        return families

    def import_points(self, ignore=None):
        """ Import from a file precomputed coordinates of all the points --
        each point refer to one election """

        return self.import_cooridnates(ignore=ignore)

    def import_cooridnates(self, ignore=None):
        """ Import from a file precomputed coordinates of all the points --
        each point refer to one election """

        if ignore is None:
            ignore = []

        points = {}
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", self.distance_name + "_2d_a" + str(
                float(self.attraction_factor)) + ".csv")

        with open(path, 'r', newline='') as csv_file:

            # ORIGINAL

            reader = csv.DictReader(csv_file, delimiter=';')
            ctr = 0

            for row in reader:
                if self.main_order[ctr] < self.num_instances and \
                        self.main_order[ctr] not in ignore:
                    points[row['election_id']] = [float(row['x']), float(row['y'])]
                ctr += 1

        return points

    def compute_points_by_families(self):
        """ Group all points by their families """

        points_by_families = {}

        if self.families is None:
            self.families = {}
            for i, election_id in enumerate(self.instances):
                ele = self.instances[election_id]
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
                points_by_families[family_id][0].append(
                    self.coordinates[family_id][0])
                points_by_families[family_id][1].append(
                    self.coordinates[family_id][1])
                try:
                    points_by_families[family_id][2].append(
                        self.coordinates[family_id][2])
                except:
                    pass
        else:

            for family_id in self.families:
                points_by_families[family_id] = [[] for _ in range(3)]

                for i in range(self.families[family_id].size):
                    # print(i)
                    # print(self.coordinates)
                    if self.families[family_id].single_election:
                        election_id = family_id
                    else:
                        election_id = family_id + '_' + str(i)
                    points_by_families[family_id][0].append(
                        self.coordinates[election_id][0])
                    points_by_families[family_id][1].append(
                        self.coordinates[election_id][1])
                    try:
                        points_by_families[family_id][2].append(
                            self.coordinates[election_id][2])
                    except:
                        pass

        self.points_by_families = points_by_families

    def compute_feature(self, name=None):

        # values = []
        feature_dict = {}

        for election_id in self.instances:
            feature = features.get_feature(name)
            election = self.instances[election_id]
            # print(election_id, election)
            if name in {'avg_distortion_from_guardians',
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

    def rotate(self, angle):
        """ Rotate all the points by a given angle """

        for election_id in self.instances:
            self.coordinates[election_id][0], self.coordinates[election_id][
                1] = self.rotate_point(
                0.5, 0.5, angle, self.coordinates[election_id][0],
                self.coordinates[election_id][1])

        self.compute_points_by_families()

    def reverse(self):
        """ Reverse all the points"""

        for election_id in self.instances:
            self.coordinates[election_id][0] = self.coordinates[election_id][0]
            self.coordinates[election_id][1] = -self.coordinates[election_id][
                1]

        self.compute_points_by_families()

    def update(self):
        """ Save current coordinates of all the points to the original file"""

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", self.distance_name + "_2d_a" +
                            str(float(self.attraction_factor)) + ".csv")

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(["election_id", "x", "y"])

            for election_id in self.instances:
                x = round(self.coordinates[election_id][0], 5)
                y = round(self.coordinates[election_id][1], 5)
                writer.writerow([election_id, x, y])

    @staticmethod
    def rotate_point(cx, cy, angle, px, py):
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

    def import_distances(self, self_distances=False,
                         distance_name='emd-positionwise'):
        """ Import precomputed distances between each pair of instances
        from a file """

        file_name = str(distance_name) + '.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "distances", file_name)
        num_points = self.num_instances
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

    def import_my_distances(self, distance_name='emd-positionwise'):
        """ Import precomputed distances between each pair of instances
        from a file """

        file_name = str(distance_name) + '.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "distances", file_name)
        distances = {}
        times = {}
        stds = {}

        for family_id in self.families:
            for election_id in self.families[family_id].election_ids:
                distances[election_id] = {}
                times[election_id] = {}
                stds[election_id] = {}

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')

            for row in reader:
                election_id_1 = row['election_id_1']
                election_id_2 = row['election_id_2']

                distances[election_id_1][election_id_2] = float(row['distance'])
                distances[election_id_2][election_id_1] = \
                    distances[election_id_1][election_id_2]
                try:
                    times[election_id_1][election_id_2] = float(row['time'])
                    times[election_id_2][election_id_1] = \
                        times[election_id_1][election_id_2]
                except:
                    pass

                try:
                    stds[election_id_1][election_id_2] = float(row['std'])
                    stds[election_id_2][election_id_1] = \
                        stds[election_id_1][election_id_2]
                except:
                    pass

        return distances, times, stds


