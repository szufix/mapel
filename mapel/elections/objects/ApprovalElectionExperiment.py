#!/usr/bin/env python
import os

from mapel.elections.objects.ApprovalElection import ApprovalElection
from mapel.elections.objects.ElectionExperiment import ElectionExperiment
from mapel.elections.other import pabulib
from mapel.elections._glossary import *

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
                 instance_type='approval', dim=2, store=True):
        self.shift = shift
        super().__init__(instances=instances, distances=distances,
                         coordinates=coordinates, distance_id=distance_id,
                         experiment_id=experiment_id, dim=dim, store=store,
                         instance_type=instance_type, _import=_import)

    def add_elections_to_experiment(self) -> dict:
        """ Return: elections imported from files """

        elections = {}

        for family_id in self.families:
            print(family_id)
            ids = []

            if self.families[family_id].model_id in APPROVAL_MODELS or \
                    self.families[family_id].model_id in APPROVAL_FAKE_MODELS or \
                    self.families[family_id].model_id in ['pabulib']:

                if self.families[family_id].single_election:
                    election_id = family_id
                    election = ApprovalElection(self.experiment_id, election_id,
                                                _import=self._import)
                    elections[election_id] = election
                    ids.append(str(election_id))
                else:
                    for j in range(self.families[family_id].size):
                        election_id = family_id + '_' + str(j)
                        election = ApprovalElection(self.experiment_id, election_id,
                                                    _import=self._import)
                        elections[election_id] = election
                        ids.append(str(election_id))
            else:

                path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                    "elections", self.families[family_id].model_id)
                for i, name in enumerate(os.listdir(path)):
                    if i >= self.families[family_id].size:
                        break
                    name = os.path.splitext(name)[0]
                    name = f'{self.families[family_id].model_id}/{name}'
                    election = ApprovalElection(self.experiment_id, name,
                                                _import=self._import, shift=self.shift)
                    elections[name] = election
                    ids.append(str(name))

            self.families[family_id].instance_ids = ids

        return elections

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
                file_csv.write("size;num_candidates;num_voters;model_id;params;color;alpha;"
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
