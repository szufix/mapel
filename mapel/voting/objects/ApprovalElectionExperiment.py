#!/usr/bin/env python
import copy

from mapel.voting.objects.Election import Election
from mapel.voting.objects.ApprovalElection import ApprovalElection
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

from mapel.voting.objects.ElectionExperiment import ElectionExperiment


class ApprovalExperiment(ElectionExperiment):
    """Abstract set of instances."""

    def __init__(self, ignore=None, instances=None, distances=None, with_matrices=False,
                 coordinates=None, distance_name='emd-positionwise', experiment_id=None,
                 instance_type='ordinal'):

        super().__init__(ignore=ignore, instances=instances, distances=distances,
                         with_matrices=with_matrices,
                         coordinates=coordinates, distance_name=distance_name,
                         experiment_id=experiment_id,
                         instance_type=instance_type)

    def add_instances_to_experiment(self):
        """ Import instances from a file """

        instances = {}

        for family_id in self.families:

            ids = []
            if self.families[family_id].single_election:
                election_id = family_id
                election = ApprovalElection(self.experiment_id, election_id, _import=True)
                instances[election_id] = election
                ids.append(str(election_id))
            else:
                for j in range(self.families[family_id].size):
                    election_id = family_id + '_' + str(j)
                    election = ApprovalElection(self.experiment_id, election_id, _import=True)
                    instances[election_id] = election
                    ids.append(str(election_id))

            self.families[family_id].election_ids = ids

        return instances
