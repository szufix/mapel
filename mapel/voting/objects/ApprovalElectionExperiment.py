#!/usr/bin/env python
import os
from mapel.voting.objects.ApprovalElection import ApprovalElection
from mapel.voting.objects.ElectionExperiment import ElectionExperiment

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
    """Abstract set of elections."""

    def __init__(self, elections=None, distances=None,
                 coordinates=None, distance_name='emd-positionwise', experiment_id=None,
                 election_type='ordinal', _import=True):

        super().__init__(elections=elections, distances=distances,
                         coordinates=coordinates, distance_name=distance_name,
                         experiment_id=experiment_id, _import=_import,
                         election_type=election_type)

    def add_elections_to_experiment(self) -> dict:
        """ Return: elections imported from files"""

        elections = {}

        for family_id in self.families:

            ids = []
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

            self.families[family_id].election_ids = ids

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
                file_csv.write("size;num_candidates;num_voters;model;params;color;alpha;"
                               "label;marker;show;path")
                file_csv.write("1;50;200;approval_id_0.5;{};brown;0.75;ID 0.5;*;t;{}")
                file_csv.write("1;50;200;approval_ic_0.5;{};black;0.75;IC 0.5;*;t;{}")
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
