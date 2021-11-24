#!/usr/bin/env python
import os
import csv
import copy
import copy
import csv
import itertools
import os
from abc import abstractmethod
from threading import Thread
from time import sleep
import numpy as np

from mapel.main.objects.Experiment import Experiment
from mapel.roommates.objects.RoommatesFamily import RoommatesFamily
import mapel.roommates.models_main as models_main
import mapel.roommates.metrics_main as metr

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


class RoommatesExperiment(Experiment):
    """Abstract set of elections."""

    def __init__(self, experiment_id=None):
        super().__init__(experiment_id=experiment_id)
        self.default_num_agents = 10

    def add_instance(self, model_id="none", params=None, label=None,
                     color="black", alpha=1., show=True, marker='x', starting_from=0, size=1,
                     num_agents=None, instance_id=None):

        if num_agents is None:
            num_agents = self.default_num_agents

        return self.add_family(model_id=model_id, params=params, size=size, label=label,
                               color=color, alpha=alpha, show=show, marker=marker,
                               starting_from=starting_from, family_id=instance_id,
                               num_agents=num_agents, single_instance=True)

    def add_family(self, model_id: str = "none", params: dict = None, size: int = 1,
                   label: str = None, color: str = "black", alpha: float = 1.,
                   show: bool = True, marker: str = 'o', starting_from: int = 0,
                   family_id: str = None, single_instance: bool = False,
                   num_agents: int = None, path: dict = None) -> list:

        if num_agents is None:
            num_agents = self.default_num_agents

        if self.families is None:
            self.families = {}

        if label is None:
            label = family_id

        self.families[family_id] = RoommatesFamily(model_id=model_id, family_id=family_id,
                                                   params=params, label=label, color=color,
                                                   alpha=alpha,
                                                   show=show, size=size, marker=marker,
                                                   starting_from=starting_from,
                                                   num_agents=num_agents, path=path)

        self.num_families = len(self.families)
        self.num_instances = sum([self.families[family_id].size for family_id in self.families])

        instances = models_main.prepare_roommates_instances(experiment=self,
                                    model_id=self.families[family_id].model_id,
                                    family_id=family_id,
                                    params=copy.deepcopy(self.families[family_id].params))

        self.families[family_id].instances_ids = list(instances.keys())

        return list(instances.keys())

    def compute_distances(self, distance_id: str = 'emd-positionwise', num_threads: int = 1,
                          self_distances: bool = False, vector_type: str = 'A',
                          printing: bool = False) -> None:

        self.distance_id = distance_id

        matchings = {instance_id: {} for instance_id in self.instances}
        distances = {instance_id: {} for instance_id in self.instances}
        times = {instance_id: {} for instance_id in self.instances}

        threads = [{} for _ in range(num_threads)]

        ids = []
        for i,instance_1 in enumerate(self.instances):
            for j, instance_2 in enumerate(self.instances):
                if i == j:
                    if self_distances:
                        ids.append((instance_1, instance_2))
                elif i < j:
                    ids.append((instance_1, instance_2))

        num_distances = len(ids)

        for t in range(num_threads):
            print(f'Starting thread: {t}')
            sleep(0.1)
            start = int(t * num_distances / num_threads)
            stop = int((t + 1) * num_distances / num_threads)
            thread_ids = ids[start:stop]

            threads[t] = Thread(target=metr.run_single_thread, args=(self, thread_ids,
                                                                     distances, times, matchings,
                                                                     printing))
            threads[t].start()

        for t in range(num_threads):
            threads[t].join()
        if self.store:

            file_name = f'{distance_id}.csv'
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "distances",
                                file_name)

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(
                    ["election_id_1", "election_id_2", "distance", "time"])

                for election_1, election_2 in itertools.combinations(self.elections, 2):
                    distance = str(distances[election_1][election_2])
                    time = str(times[election_1][election_2])
                    writer.writerow([election_1, election_2, distance, time])

        self.distances = distances
        self.times = times
        self.matchings = matchings
