"""
Implementation of the feature distance.

"""


import mapel.elections as mapel
from mapel.elections.objects import Election
import numpy as np
from collections import defaultdict
import math

"""
L1 distance between two feature vectors
"""
def features_vector_l1(e1: Election, e2: Election):
    vector1 = np.array(e1.election_features.features_vector)
    vector2 = np.array(e2.election_features.features_vector)
    return np.linalg.norm(vector1 - vector2, ord=1)


"""
L2 distance between two feature vectors
"""
def features_vector_l2(e1: Election, e2: Election):
    vector1 = np.array(e1.election_features.features_vector)
    vector2 = np.array(e2.election_features.features_vector)
    return np.linalg.norm(vector1 - vector2, ord=2)


"""
Sample driver function of the feature distance.

For successful calculation the features you decide to use need to be loaded into memory for example as follows:

experiment.features["Agreement"] = experiment.import_feature("AgreementApprox")

Then the prepare_feature_vectors function allows to pick which features will be used - codes of the features need to be passed as a list.
Currently supported feature codes are the following:
d := Diversity
a := Agreement
p := Polarization
e := Entropy
e2 := Entropy squared

Example usage which selects DAP features:
experiment.prepare_feature_vectors(['d', 'a', 'p'])
Any number of features can be used.
To add a new feature to the feature vector either adjust the ElectionExperiment.py file or contact jita-mertlova.

Full example usage of this distance as the DAP distance, assuming 'myExperiment' is an experiment folder containing calculated
AgreementApprox, PolarizationApprox and DiversityApprox features.

experiment = mapel.prepare_offline_ordinal_experiment(experiment_id="myExperiment")
experiment.features["Agreement"] = experiment.import_feature("AgreementApprox")
experiment.features["Diversity"] = experiment.import_feature("DiversityApprox")
experiment.features["Polarization"] = experiment.import_feature("PolarizationApprox")
experiment.prepare_feature_vectors(['d', 'a', 'p'])
experiment.compute_distances(distance_id='feature_l2')
experiment.embed_2d(embedding_id='mds')
experiment.print_map_2d()
experiment.print_map_2d(legend=False)

"""


