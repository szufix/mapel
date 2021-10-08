#!/usr/bin/env python
import copy
from abc import ABCMeta, abstractmethod, ABC

from mapel.voting.objects.Family import Family
from mapel.voting.objects.Experiment import Experiment

import mapel.voting._elections as _elections
import mapel.voting.features as features
import mapel.voting._metrics as metr
import mapel.voting.print as pr
import mapel.voting.elections.preflib as preflib

import math
import csv
import os

from threading import Thread
from time import sleep

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


class ElectionExperiment(Experiment):

    def __init__(self, ignore=None, instances=None, distances=None, with_matrices=False,
                 coordinates=None, distance_name='emd-positionwise', experiment_id=None,
                 instance_type='ordinal'):

        super().__init__(ignore=ignore, instances=instances, distances=distances,
                         coordinates=coordinates, distance_name=distance_name,
                         experiment_id=experiment_id,
                         instance_type=instance_type)

        self.default_num_candidates = 10
        self.default_num_voters = 100

    def set_default_num_candidates(self, num_candidates):
        self.default_num_candidates = num_candidates

    def set_default_num_voters(self, num_voters):
        self.default_num_voters = num_voters

    def add_election(self, model="none", params=None, label=None,
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