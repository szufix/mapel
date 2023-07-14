#!/usr/bin/env python
import csv
import itertools
import logging
import math
import os
from abc import ABCMeta, abstractmethod
import time

from scipy.stats import stats
from tqdm import tqdm

from PIL import Image
from mapel.core.objects.Family import Family
import mapel.core.printing as pr
import matplotlib.pyplot as plt

import mapel.core.persistence.experiment_imports as imports
import mapel.core.persistence.experiment_exports as exports
import mapel.core.embedding.embed as embed

COLORS = []


class Experiment:
    __metaclass__ = ABCMeta
    """Abstract set of instances."""

    def __init__(self,
                 experiment_id=None,
                 instances=None,
                 distances=None,
                 coordinates=None,
                 distance_id=None,
                 embedding_id=None,
                 is_exported=True,
                 is_imported=True,
                 clean=False,
                 coordinates_names=None,
                 fast_import=False,
                 with_matrix=False):

        self.is_imported = is_imported
        self.is_exported = is_exported
        self.fast_import = fast_import
        self.with_matrix = with_matrix
        self.distance_id = distance_id
        self.embedding_id = embedding_id
        self.clean = clean
        self.dim = 2
        self.coordinates_lists = {}
        self.features = {}
        self.cultures = {}
        self.families = {}
        self.times = {}
        self.stds = {}
        self.matchings = {}
        self.coordinates_by_families = {}
        self.experiment_id = None
        self.instances = None
        self.distances = None
        self.coordinates = None
        self.num_families = None
        self.num_instances = None
        self.main_order = None

        if clean:
            self.clean_instances()

        if experiment_id is not None:
            self.experiment_id = experiment_id
            self.add_folders_to_experiment()
            self.families = self.import_controllers()

            self.import_instances(instances)
            self.import_distances(distances)
            self.import_coordinates(coordinates, coordinates_names)

    def import_instances(self, instances):
        if isinstance(instances, dict):
            self.instances = instances
        elif self.is_imported and self.experiment_id is not None:

            try:
                self.instances = self.add_instances_to_experiment()
                self.num_instances = len(self.instances)
            except FileNotFoundError:
                self.instances = {}
        else:
            self.instances = {}

    def import_distances(self, distances):
        if isinstance(distances, dict):
            self.distances = distances
        elif self.is_imported and self.experiment_id is not None:
            self.distances, self.times, self.stds, self.mappings = \
                imports.add_distances_to_experiment(self)
        else:
            self.distances = {}

    def import_coordinates(self, coordinates, coordinates_names):

        if isinstance(coordinates, dict):
            self.coordinates = coordinates
        elif self.is_imported and self.experiment_id is not None:
            try:
                if coordinates_names is not None:
                    for file_name in coordinates_names:
                        self.coordinates_lists[file_name] = \
                            imports.add_coordinates_to_experiment(self,
                                                                  dim=self.dim,
                                                                  file_name=file_name)
                    self.coordinates = self.coordinates_lists[coordinates_names[0]]
                else:
                    self.coordinates = imports.add_coordinates_to_experiment(self, dim=self.dim)
            except FileNotFoundError:
                pass
        else:
            self.coordinates = {}

    @abstractmethod
    def prepare_instances(self):
        pass

    @abstractmethod
    def add_instance(self):
        pass

    @abstractmethod
    def add_family(self):
        pass

    @abstractmethod
    def add_instances_to_experiment(self):
        pass

    @abstractmethod
    def add_folders_to_experiment(self):
        pass

    @abstractmethod
    def import_controllers(self):
        pass

    @abstractmethod
    def add_culture(self, name, function):
        pass

    @abstractmethod
    def add_feature(self, name, function):
        pass

    def embed_2d(self, **kwargs) -> None:
        embed.embed(self, dim=2, **kwargs)

    def print_map_1d(self, **kwargs) -> None:
        pr.print_map_1d(self, **kwargs)

    def print_map_2d(self, **kwargs) -> None:
        pr.print_map_2d(self, **kwargs)

    def print_map_2d_colored_by_feature(self, **kwargs) -> None:
        pr.print_map_2d_colored_by_feature(self, **kwargs)

    def print_map_2d_colored_by_features(self, **kwargs) -> None:
        pr.print_map_2d_colored_by_features(self, **kwargs)

    def print_map_3d(self, **kwargs) -> None:
        pr.print_map_3d(self, **kwargs)

    def print_map(self, dim: int = 2, **kwargs) -> None:
        if dim == 1:
            pr.print_map_1d(self, **kwargs)
        elif dim == 2:
            pr.print_map_2d(self, **kwargs)
        elif dim == 3:
            pr.print_map_3d(self, **kwargs)

    def print_matrix(self, **kwargs):
        pr.print_matrix(experiment=self, **kwargs)

    def compute_feature_from_function(self, function, feature_id='my_feature'):
        feature_dict = {'value': {}, 'time': {}}
        for instance_id in tqdm(self.instances):
            start = time.time()
            instance = self.instances[instance_id]
            value = function(instance)
            total_time = time.time() - start
            feature_dict['value'][instance_id] = value
            feature_dict['time'][instance_id] = total_time
        if self.is_exported:
            exports.export_feature_from_function_to_file(self, feature_id, feature_dict)
        self.features[feature_id] = feature_dict
        return feature_dict

    def compute_coordinates_by_families(self, dim=2) -> None:
        """ Group all points by their families """

        coordinates_by_families = {}

        if self.families is None:
            self.families = {}
            for i, instance_id in enumerate(self.instances):
                ele = self.instances[instance_id]
                model = ele.culture_id
                family_id = model
                label = instance_id
                color = COLORS[int(i % len(COLORS))]

                alpha = 1.

                self.families[instance_id] = Family(
                    culture_id=model, family_id=family_id,
                    label=label, alpha=alpha,
                    color=color)

            for family_id in self.families:

                coordinates_by_families[family_id] = [[] for _ in range(dim)]
                coordinates_by_families[family_id][0].append(self.coordinates[family_id][0])
                try:
                    coordinates_by_families[family_id][1].append(self.coordinates[family_id][1])
                finally:
                    pass

                try:
                    coordinates_by_families[family_id][2].append(self.coordinates[family_id][2])
                finally:
                    pass
        else:

            for family_id in self.families:

                coordinates_by_families[family_id] = [[] for _ in range(3)]

                if dim == 2:
                    for instance_id in self.families[family_id].instance_ids:
                        coordinates_by_families[family_id][0].append(
                            self.coordinates[instance_id][0])
                        coordinates_by_families[family_id][1].append(
                            self.coordinates[instance_id][1])

        self.coordinates_by_families = coordinates_by_families

    def get_distance(self, i, j):
        """ Compute Euclidean distance in two-dimensional space"""

        distance = 0.
        for d in range(2):
            distance += (self.coordinates[i][d] - self.coordinates[j][d]) ** 2

        return math.sqrt(distance)

    def rotate(self, angle) -> None:
        """ Rotate all the points by a given angle """

        for instance_id in self.instances:
            self.coordinates[instance_id][0], self.coordinates[instance_id][1] = \
                self.rotate_point(0.5, 0.5, angle, self.coordinates[instance_id][0],
                                  self.coordinates[instance_id][1])

        self.compute_coordinates_by_families()

    def reverse(self) -> None:
        """ Reverse all the points"""

        for instance_id in self.instances:
            self.coordinates[instance_id][0] = self.coordinates[instance_id][0]
            self.coordinates[instance_id][1] = -self.coordinates[instance_id][1]

        self.compute_coordinates_by_families()

    def update(self) -> None:
        """ Save current coordinates of all the points to the original file"""

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                            "coordinates", f'{self.distance_id}_2d.csv')

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(["instance_id", "x", "y"])

            for election_id in self.instances:
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
        px, py = x_new + cx, y_new + cy

        return px, py

    def clean_instances(self):
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "instances")
        for file_name in os.listdir(path):
            os.remove(os.path.join(path, file_name))

    def get_feature(self, feature_id, column_id='value'):

        # if feature_id not in election.features:
        #     election.features[feature_id] = election.import_feature(feature_id)

        self.features[feature_id] = self.import_feature(feature_id, column_id=column_id)

        return self.features[feature_id]

    def import_feature(self, feature_id, column_id='value', rule=None):
        if rule is None:
            feature_long_id = feature_id
        else:
            feature_long_id = f'{feature_id}_{rule}'
        return imports.get_values_from_csv_file(self, feature_id=feature_id,
                                                column_id=column_id,
                                                feature_long_id=feature_long_id)

    def normalize_feature_by_feature(self, nom=None, denom=None, saveas=None, column_id='value'):

        f1 = self.get_feature(nom, column_id=column_id)
        f2 = self.get_feature(denom, column_id=column_id)
        f3 = {}

        for election_id in f1:
            if f1[election_id] is None:
                f3[election_id] = None
            else:
                if f2[election_id] == 0:
                    f3[election_id] = 'Blank'
                else:
                    f3[election_id] = f1[election_id] / f2[election_id]

        exports.export_normalized_feature_to_file(self, feature_dict=f3, saveas=saveas)

    def print_correlation_between_distances(self,
                                            distance_id_1=None,
                                            distance_id_2=None,
                                            title=None,
                                            s=12,
                                            alpha=0.25,
                                            color='purple',
                                            title_size=24,
                                            label_size=20,
                                            ticks_size=10):
        if distance_id_1 is None:
            logging.warning('distance_id_1 is not defined')
        if distance_id_2 is None:
            logging.warning('distance_id_2 is not defined')

        all_distances = {}

        all_distances[distance_id_1] = imports.import_distances_from_file(self, distance_id_1)
        all_distances[distance_id_2] = imports.import_distances_from_file(self, distance_id_2)

        names = list(all_distances.keys())

        def nice(name):
            return {
                'spearman': 'Spearman',
                'l1-mutual_attraction': '$\ell_1$ Mutual Attraction',
                'hamming': "Hamming",
                "jaccard": "Jaccard",
                'discrete': 'Discrete',
                'swap': 'Swap',
                'emd-bordawise': "EMD-Bordawise",
                'emd-positionwise': 'EMD-Positionwise',
                'l1-positionwise': "$\ell_1$-Positionwise",
                'l1-pairwise': "$\ell_1$-Pairwise",
            }.get(name, name)

        for name_1, name_2 in itertools.combinations(names, 2):

            values_x = []
            values_y = []
            empty_x = []
            empty_y = []
            for e1, e2 in itertools.combinations(all_distances[name_1], 2):
                if e1 in ['AN', 'UN', 'ID', 'ST'] or e2 in ['AN', 'UN', 'ID', 'ST']:
                    empty_x.append(all_distances[name_1][e1][e2])
                    empty_y.append(all_distances[name_2][e1][e2])
                else:
                    values_x.append(all_distances[name_1][e1][e2])
                    values_y.append(all_distances[name_2][e1][e2])

            fig = plt.figure(figsize=[6.4, 4.8])
            plt.gcf().subplots_adjust(left=0.2)
            plt.gcf().subplots_adjust(bottom=0.2)
            ax = fig.add_subplot()

            ax.scatter(values_x, values_y, s=s, alpha=alpha, color=color)

            PCC = round(stats.pearsonr(values_x, values_y)[0], 3)
            print('PCC', PCC)
            SCC = round(stats.spearmanr(values_x, values_y)[0], 3)
            print('SCC', SCC)

            plt.xlim(left=0)
            plt.ylim(bottom=0)

            plt.xticks(fontsize=ticks_size)
            plt.yticks(fontsize=ticks_size)

            plt.xlabel(nice(name_1), size=label_size)
            plt.ylabel(nice(name_2), size=label_size)

            if title:
                plt.title(title, size=title_size)

            path = f'images/correlation'
            is_exist = os.path.exists(path)

            if not is_exist:
                os.makedirs(path)

            saveas = f'images/correlation/corr_{name_1}_{name_2}'
            plt.savefig(saveas, pad_inches=1)
            plt.show()

    def merge_election_images(self, size=250, name=None, show=False, ncol=1, nrow=1,
                              object_type=None):

        images = []
        for i, election in enumerate(self.instances.values()):
            images.append(Image.open(f'images/{name}/{election.label}_{object_type}.png'))

        image1_size = images[0].size

        new_image = Image.new('RGB', (ncol * image1_size[0], nrow * image1_size[1]),
                              (size, size, size))

        for i in range(ncol):
            for j in range(nrow):
                new_image.paste(images[i + j * ncol], (image1_size[0] * i, image1_size[1] * j))

        new_image.save(f'images/{name}_{object_type}.png', "PNG", quality=85)
        if show:
            new_image.show()

    def merge_election_images_in_parts(self, size=250, name=None, show=False, ncol=1, nrow=1,
                                       distance_id='hamming'):
        pass

    def merge_election_images_double(self, size=250, name=None,
                                     distance_ids=None,
                                     show=False, ncol=1, nrow=1):
        images = []
        for i, election in enumerate(self.instances.values()):
            images.append(Image.open(f'images/{name}/{election.label}_{distance_ids[0]}.png'))
            images.append(Image.open(f'images/{name}/{election.label}_{distance_ids[1]}.png'))
        image1_size = images[0].size

        new_image = Image.new('RGB', (ncol * image1_size[0], nrow * image1_size[1]),
                              (size, size, size))

        for i in range(ncol):
            for j in range(nrow):
                new_image.paste(images[i + j * ncol], (image1_size[0] * i, image1_size[1] * j))

        new_image.save(f'images/microscope/{name}.png', "PNG", quality=85)
        if show:
            new_image.show()
