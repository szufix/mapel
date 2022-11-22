#!/usr/bin/env python
import os

from mapel.elections.objects.ElectionExperiment import ElectionExperiment
from mapel.elections.cultures_ import LIST_OF_ORDINAL_MODELS_WITH_PARAMS, \
    LIST_OF_ORDINAL_MODELS_WITHOUT_PARAMS

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


class OrdinalElectionExperiment(ElectionExperiment):
    """Abstract set of elections."""

    def __init__(self, instances=None, distances=None, _import=True, shift=False,
                 coordinates=None, distance_id='emd-positionwise', experiment_id=None,
                 instance_type='ordinal', dim=2, store=True, coordinates_names=None,
                 embedding_id='kamada', fast_import=False, with_matrix=False):
        self.shift = shift
        super().__init__(instances=instances, distances=distances,
                         coordinates=coordinates, distance_id=distance_id,
                         experiment_id=experiment_id, dim=dim, store=store,
                         instance_type=instance_type, _import=_import,
                         coordinates_names=coordinates_names,
                         embedding_id=embedding_id,
                         fast_import=fast_import,
                         with_matrix=with_matrix)

    def add_culture_with_params(self, name, func):
        LIST_OF_ORDINAL_MODELS_WITH_PARAMS[name] = func

    def add_culture_without_params(self, name, func):
        LIST_OF_ORDINAL_MODELS_WITHOUT_PARAMS[name] = func

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
                file_csv.write(
                    "size;num_candidates;num_voters;culture_id;params;color;alpha;family_id;marker;show\n")
                file_csv.write("3;10;100;impartial_culture;{};black;1;Impartial Culture;o;t\n")
                file_csv.write("3;10;100;iac;{};black;0.7;IAC;o;t\n")
                file_csv.write("3;10;100;conitzer;{};limegreen;1;SP by Conitzer;o;t\n")
                file_csv.write("3;10;100;walsh;{};olivedrab;1;SP by Walsh;o;t\n")
                file_csv.write("3;10;100;spoc_conitzer;{};DarkRed;0.7;SPOC;o;t\n")
                file_csv.write("3;10;100;group-separable;{};blue;1;Group-Separable;o;t\n")
                file_csv.write("3;10;100;single-crossing;{};purple;0.6;Single-Crossing;o;t\n")
                file_csv.write("3;10;100;1d_interval;{'dim': 1};DarkGreen;1;1D Interval;o;t\n")
                file_csv.write("3;10;100;2d_disc;{'dim': 2};Green;1;2D Disc;o;t\n")
                file_csv.write("3;10;100;3d_cube;{'dim': 3};ForestGreen;0.7;3D Cube;o;t\n")
                file_csv.write("3;10;100;2d_sphere;{'dim': 2};black;0.2;2D Sphere;o;t\n")
                file_csv.write("3;10;100;3d_sphere;{'dim': 3};black;0.4;3D Sphere;o;t\n")
                file_csv.write("3;10;100;urn_model;{'alpha':0.1};yellow;1;Urn culture_id 0.1;o;t\n")
                file_csv.write(
                    "3;10;100;norm-mallows;{'norm-phi':0.5};blue;1;Norm-Mallows 0.5;o;t\n")
                file_csv.write("3;10;100;urn_model;{'alpha':0};orange;1;Urn culture_id (gamma);o;t\n")
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
                # file_csv.write("3;10;100;unid;{};blue;1;UNID;3;f\n")
                # file_csv.write("3;10;100;anid;{};black;1;ANID;3;f\n")
                # file_csv.write("3;10;100;stid;{};black;1;STID;3;f\n")
                # file_csv.write("3;10;100;anun;{};black;1;ANUN;3;f\n")
                # file_csv.write("3;10;100;stun;{};black;1;STUN;3;f\n")
                # file_csv.write("3;10;100;stan;{};red;1;STAN;3;f\n")
        except FileExistsError:
            print("Experiment already exists!")
