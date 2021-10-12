#!/usr/bin/env python
import copy
import os

import mapel.voting._elections as _elections
from mapel.voting.objects.Experiment import Experiment
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


class ElectionExperiment(Experiment):

    def __init__(self, ignore=None, elections=None, distances=None, with_matrices=False,
                 coordinates=None, distance_name='emd-positionwise', experiment_id=None,
                 election_type='ordinal', attraction_factor=1):

        super().__init__(ignore=ignore, elections=elections, distances=distances,
                         coordinates=coordinates, distance_name=distance_name,
                         experiment_id=experiment_id,
                         election_type=election_type, attraction_factor=attraction_factor)

        self.default_num_candidates = 10
        self.default_num_voters = 100

    def set_default_num_candidates(self, num_candidates):
        """ Set default number of candidates """
        self.default_num_candidates = num_candidates

    def set_default_num_voters(self, num_voters):
        """ Set default number of voters """
        self.default_num_voters = num_voters

    def add_election(self, model="none", params=None, label=None,
                     color="black", alpha=1., show=True, marker='x', starting_from=0, size=1,
                     num_candidates=None, num_voters=None, name=None, num_nodes=None):
        """ Add election to the experiment """

        if num_candidates is None:
            num_candidates = self.default_num_candidates

        if num_voters is None:
            num_voters = self.default_num_voters

        return self.add_election_family(model=model, params=params, size=size, label=label, color=color,
                               alpha=alpha, show=show,  marker=marker, starting_from=starting_from,
                               num_candidates=num_candidates, num_voters=num_voters,
                               family_id=name, num_nodes=num_nodes, single_election=True)[0]

    def add_election_family(self, model="none", params=None, size=1, label=None, color="black",
                   alpha=1., show=True, marker='o', starting_from=0, num_candidates=None,
                   num_voters=None, family_id=None, single_election=False, num_nodes=None,
                   path=None, name=None):
        """ Add family of elections to the experiment """

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
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections"))
            os.mkdir(os.path.join(os.getcwd(), "experiments", self.experiment_id, "matrices"))

            # PREPARE MAP.CSV FILE

            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "map.csv")

            with open(path, 'w') as file_csv:
                file_csv.write(
                    "size;num_candidates;num_voters;model;params;color;alpha;label;marker;show\n")
                file_csv.write("3;10;100;impartial_culture;{};black;1;Impartial Culture;o;t\n")
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
                file_csv.write("3;10;100;urn_model;{'alpha':0.1};yellow;1;Urn model 0.1;o;t\n")
                file_csv.write(
                    "3;10;100;norm-mallows;{'norm-phi':0.5};blue;1;Norm-Mallows 0.5;o;t\n")
                file_csv.write("3;10;100;urn_model;{'alpha':0};orange;1;Urn model (gamma);o;t\n")
                file_csv.write(
                    "3;10;100;norm-mallows;{'norm-phi':0};cyan;1;Norm-Mallows (uniform);o;t\n")
                file_csv.write("1;10;100;identity;{};blue;1;Identity;x;t\n")
                file_csv.write("1;10;100;uniformity;{};black;1;Uniformity;x;t\n")
                file_csv.write("1;10;100;antagonism;{};red;1;Antagonism;x;t\n")
                file_csv.write("1;10;100;stratification;{};green;1;Stratification;x;t\n")
                file_csv.write("1;10;100;walsh_matrix;{};olivedrab;1;Walsh Matrix;x;t\n")
                file_csv.write("1;10;100;conitzer_matrix;{};limegreen;1;Conitzer Matrix;x;t\n")
                file_csv.write(
                    "1;10;100;single-crossing_matrix;{};purple;0.6;Single-Crossing Matrix;x;t\n")
                file_csv.write(
                    "1;10;100;gs_caterpillar_matrix;{};green;1;GS Caterpillar Matrix;x;t\n")
                file_csv.write("3;10;100;unid;{};blue;1;UNID;3;f\n")
                file_csv.write("3;10;100;anid;{};black;1;ANID;3;f\n")
                file_csv.write("3;10;100;stid;{};black;1;STID;3;f\n")
                file_csv.write("3;10;100;anun;{};black;1;ANUN;3;f\n")
                file_csv.write("3;10;100;stun;{};black;1;STUN;3;f\n")
                file_csv.write("3;10;100;stan;{};red;1;STAN;3;f\n")
        except FileExistsError:
            print("Experiment already exists!")

