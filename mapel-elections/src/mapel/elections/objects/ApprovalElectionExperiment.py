#!/usr/bin/env python
from abc import ABC

from mapel.core.matchings import solve_matching_vectors
from mapel.elections.objects.ElectionExperiment import ElectionExperiment
from mapel.elections.other import pabulib
from mapel.core.utils import *
import numpy as np
import csv

import mapel.elections.cultures_ as cultures
import mapel.elections.features_ as features
import mapel.elections.distances_ as distances
from tqdm import tqdm

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


class ApprovalElectionExperiment(ElectionExperiment, ABC):
    """ Abstract set of approval elections."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_culture(self, name, function):
        cultures.add_approval_culture(name, function)

    def add_feature(self, name, function):
        features.add_approval_feature(name, function)

    def add_distance(self, name, function):
        distances.add_approval_distance(name, function)

    def compute_distance_between_rules(self, list_of_rules=None, printing=False,
                                       distance_id=None, committee_size=10):

        self.import_committees(list_of_rules=list_of_rules)

        path = os.path.join(os.getcwd(), "experiments", f'{self.experiment_id}', '..',
                            'rules_output', 'distances', f'{distance_id}.csv')

        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(["election_id_1", "election_id_2", "distance", "time"])

            for i, r1 in enumerate(list_of_rules):
                for j, r2 in enumerate(list_of_rules):
                    if i < j:
                        if printing:
                            print(r1, r2)
                        all_distance = []
                        for election_id in self.elections:
                            com1 = self.all_winning_committees[r1][
                                election_id][0]
                            com2 = self.all_winning_committees[r2][
                                election_id][0]

                            if distance_id == 'discrete':
                                distance = len(com1.symmetric_difference(com2))
                            elif distance_id == 'wrong':
                                distance = 1 - len(com1.intersection(com2)) / len(com1.union(com2))
                            elif distance_id in ['hamming', 'jaccard']:
                                cand_dist = np.zeros([committee_size, committee_size])

                                self.elections[election_id].compute_reverse_approvals()
                                for k1, c1 in enumerate(com1):
                                    for k2, c2 in enumerate(com2):

                                        ac1 = self.elections[election_id].reverse_approvals[c1]
                                        ac2 = self.elections[election_id].reverse_approvals[c2]
                                        if distance_id == 'hamming':
                                            cand_dist[k1][k2] = len(ac1.symmetric_difference(ac2))
                                        elif distance_id == 'jaccard':
                                            if len(ac1.union(ac2)) != 0:
                                                cand_dist[k1][k2] = 1 - len(
                                                    ac1.intersection(ac2)) / len(ac1.union(ac2))
                                distance, _ = solve_matching_vectors(cand_dist)
                                distance /= committee_size
                                if printing:
                                    print(distance)
                            all_distance.append(distance)
                        mean = sum(all_distance) / self.num_elections
                        writer.writerow([r1, r2, mean, 0.])

    def compute_rule_features(self,
                              feature_id=None,
                              list_of_rules=None,
                              feature_params=None,
                              **kwargs):
        if feature_params is None:
            feature_params = {}

        self.import_committees(list_of_rules=list_of_rules)

        for election_id in self.elections:
            self.elections[election_id].winning_committee = {}

        for r in list_of_rules:
            for election_id in self.elections:
                self.elections[election_id].winning_committee[r] = \
                    self.all_winning_committees[r][election_id][0]

        for rule in tqdm(list_of_rules):
            feature_params['rule'] = rule
            self.compute_feature(feature_id=feature_id, feature_params=feature_params, **kwargs)

    def print_latex_table(self,
                          feature_id=None,
                          column_id='value',
                          list_of_rules=None,
                          list_of_models=None):

        features = {}
        for rule in list_of_rules:
            features[rule] = self.import_feature(feature_id, column_id=column_id, rule=rule)

        results = {}
        for model in list_of_models:
            results[model] = {}
            for rule in list_of_rules:
                feature = features[rule]
                total_value = 0
                ctr = 0
                for instance in feature:
                    if model in instance:
                        total_value += feature[instance]
                        ctr += 1
                avg_value = round(total_value / ctr, 2)
                results[model][rule] = avg_value

        print("\\toprule")
        print("rule", end=" ")
        for model in list_of_models:
            print(f'& {model}', end=" ")
        print("\\\\ \\midrule")

        for rule in list_of_rules:
            print(rule, end=" ")
            for model in list_of_models:
                print(f'& {results[model][rule]}', end=" ")
            # print("")
            print("\\\\ \\midrule")

    def print_latex_multitable(self,
                               features_id=None,
                               columns_id=None,
                               list_of_rules=None,
                               list_of_models=None):

        all_results = {}
        for feature_id, column_id in zip(features_id, columns_id):
            features = {}
            for rule in list_of_rules:
                features[rule] = self.import_feature(feature_id, column_id=column_id, rule=rule)

            results = {}
            for model in list_of_models:
                results[model] = {}
                for rule in list_of_rules:
                    feature = features[rule]
                    total_value = 0
                    ctr = 0
                    for instance in feature:
                        if model.lower() in instance.lower():
                            total_value += feature[instance]
                            ctr += 1
                    if ctr == 0:
                        avg_value = -1
                    else:
                        avg_value = round(total_value / ctr, 2)
                    results[model][rule] = avg_value
            all_results[f'{feature_id}_{column_id}'] = results

        for rule in list_of_rules:
            print(rule, end=" ")
            for model in list_of_models:
                print(f'&', end=" ")
                for feature_id, column_id in zip(features_id, columns_id):
                    print(f'{all_results[f"{feature_id}_{column_id}"][model][rule]}', end=" \ ")
            print("\\\\ \\midrule")

    def add_folders_to_experiment(self) -> None:

        dirs = ["experiments", "images", "trash"]
        for dir in dirs:
            if not os.path.isdir(dir):
                os.mkdir(os.path.join(os.getcwd(), dir))

        if not os.path.isdir(os.path.join(os.getcwd(), "experiments", self.experiment_id)):
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id))

        list_of_folders = ['distances',
                           'features',
                           'coordinates',
                           'elections',
                           'matrices']

        for folder_name in list_of_folders:
            if not os.path.isdir(os.path.join(os.getcwd(), "experiments",
                                              self.experiment_id, folder_name)):
                os.mkdir(os.path.join(os.getcwd(), "experiments",
                                      self.experiment_id, folder_name))

        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "map.csv")
        if not os.path.exists(path):
            with open(path, 'w') as file_csv:
                file_csv.write("size;num_candidates;num_voters;culture_id;params;color;alpha;"
                               "label;marker;show;path")
                file_csv.write("1;50;200;ic;{'p': 0.5};black;0.75;IC 0.5;*;t;{}")
        else:
            print("Experiment already exists!")

    def convert_pb_to_app(self, **kwargs):
        pabulib.convert_pb_to_app(self, **kwargs)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
