#!/usr/bin/env python
import os

from mapel.elections.objects.ApprovalElection import ApprovalElection
from mapel.elections.objects.ElectionExperiment import ElectionExperiment
from mapel.elections.other import pabulib
from mapel.core.glossary import *
from mapel.core.utils import *
import numpy as np
import csv

# from mapel.core.matchings import solve_matching_vectors

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


class ApprovalElectionExperiment(ElectionExperiment):
    """ Abstract set of approval elections."""

    def __init__(self, instances=None, distances=None, _import=True, shift=False,
                 coordinates=None, distance_id='emd-positionwise', experiment_id=None,
                 instance_type='approval', dim=2, store=True,
                 coordinates_names=None,
                 embedding_id='spring',
                 fast_import=False):
        self.shift = shift
        super().__init__(instances=instances, distances=distances,
                         coordinates=coordinates, distance_id=distance_id,
                         experiment_id=experiment_id, dim=dim, store=store,
                         instance_type=instance_type, _import=_import,
                         coordinates_names=coordinates_names,
                         embedding_id=embedding_id,
                         fast_import=fast_import)

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

                                        ac1 = self.elections[election_id].reverse_approvals[
                                            c1]
                                        ac2 = self.elections[election_id].reverse_approvals[
                                            c2]
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

    def compute_rule_features(self, feature_id=None, list_of_rules=None, printing=False,
                              feature_params=None, **kwargs):
        if feature_params is None:
            feature_params = {}

        self.import_committees(list_of_rules=list_of_rules)

        for election_id in self.elections:
            self.elections[election_id].winning_committee = {}

        for r in list_of_rules:
            for election_id in self.elections:
                self.elections[election_id].winning_committee[r] = \
                    self.all_winning_committees[r][election_id][0]

        for rule in list_of_rules:
            if printing:
                print(rule)
            feature_params['rule'] = rule
            self.compute_feature(feature_id=feature_id, feature_params=feature_params, **kwargs)

    def print_latex_table(self, feature_id=None, column_id='value',
                          list_of_rules=None, list_of_models=None):

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
            # print(f'& {nice_model[model]}', end=" ")
            print(f'& {model}', end=" ")
        # print("")
        print("\\\\ \\midrule")

        for rule in list_of_rules:
            print(rule, end=" ")
            for model in list_of_models:
                print(f'& {results[model][rule]}', end=" ")
            # print("")
            print("\\\\ \\midrule")

    def print_latex_multitable(self, features_id=None, columns_id=None,
                               list_of_rules=None, list_of_models=None):

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

        # print("\\toprule")
        # print("rule", end=" ")
        # for model in list_of_models:
        #     # print(f'& {nice_model[model]}', end=" ")
        #     print(f'& {model}', end=" ")
        # # print("")
        # print("\\\\ \\midrule")

        for rule in list_of_rules:
            print(rule, end=" ")
            for model in list_of_models:
                print(f'&', end=" ")
                for feature_id, column_id in zip(features_id, columns_id):
                    print(f'{all_results[f"{feature_id}_{column_id}"][model][rule]}', end=" \ ")
            # print("")
            print("\\\\ \\midrule")

    def create_structure(self) -> None:

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
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "matrices"))

            # PREPARE MAP.CSV FILE

            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "map.csv")

            with open(path, 'w') as file_csv:
                file_csv.write("size;num_candidates;num_voters;culture_id;params;color;alpha;"
                               "label;marker;show;path")
                file_csv.write("1;50;200;approval_id;{'p': 0.5};brown;0.75;ID 0.5;*;t;{}")
                file_csv.write("1;50;200;approval_ic;{'p': 0.5};black;0.75;IC 0.5;*;t;{}")
                file_csv.write("1;50;200;approval_full;{};red;0.75;full;*;t;{}")
                file_csv.write("1;50;200;approval_empty;{};green;0.75;empty;*;t;{}")
                file_csv.write("30;50;200;approval_id;{};brown;0.7;ID_path;o;t;{'variable': 'p'}")
                file_csv.write("30;50;200;approval_ic;{};black;0.7;IC_path;o;t;{'variable': 'p'}")
                file_csv.write("20;50;200;approval_mallows;{'p': 0.1};cyan;1;"
                               "shumal_p01_path;o;t;{'variable': 'phi'}")
                file_csv.write("25;50;200;approval_mallows;{'p': 0.2};deepskyblue;1;"
                               "shumal_p02_path;o;t;{'variable': 'phi'}")
                file_csv.write("30;50;200;approval_mallows;{'p': 0.3};blue;1;"
                               "shumal_p03_path;o;t;{'variable': 'phi'}")
                file_csv.write("35;50;200;approval_mallows;{'p': 0.4};darkblue;1;"
                               "shumal_p04_path;o;t;{'variable': 'phi'}")
                file_csv.write("40;50;200;approval_mallows;{'p': 0.5};purple;1;"
                               "shumal_p05_path;s;t;{'variable': 'phi'}")
                file_csv.write("35;50;200;approval_mallows;{'p': 0.6};darkblue;1;"
                               "shumal_p06_path;x;t;{'variable': 'phi'}")
                file_csv.write("30;50;200;approval_mallows;{'p': 0.7};blue;1;"
                               "shumal_p07_path;x;t;{'variable': 'phi'}")
                file_csv.write("25;50;200;approval_mallows;{'p': 0.8};deepskyblue;1;"
                               "shumal_p08_path;x;t;{'variable': 'phi'}")
                file_csv.write("20;50;200;approval_mallows;{'p': 0.9};cyan;1;"
                               "shumal_p09_path;x;t;{'variable': 'phi'}")
        except FileExistsError:
            print("Experiment already exists!")

    def convert_pb_to_app(self, **kwargs):
        pabulib.convert_pb_to_app(self, **kwargs)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
