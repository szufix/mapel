#!/usr/bin/env python
import csv
import warnings
from abc import ABCMeta, abstractmethod
from multiprocessing import Process
from time import sleep
import ast
import time
from tqdm import tqdm

from mapel.elections.objects.ElectionFamily import ElectionFamily
from mapel.elections.objects.OrdinalElection import OrdinalElection
from mapel.elections.objects.ApprovalElection import ApprovalElection
import mapel.elections.distances_ as metr
import mapel.elections.other.rules as rules
import mapel.elections.features_ as features
from mapel.core.objects.Experiment import Experiment
import mapel.core.printing as pr
from mapel.core.utils import *
from mapel.core.glossary import *

import mapel.core.persistence.experiment_exports as exports

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


class ElectionExperiment(Experiment):
    __metaclass__ = ABCMeta
    """Abstract set of instances."""

    @abstractmethod
    def add_folders_to_experiment(self):
        pass

    @abstractmethod
    def prepare_instances(self):
        pass

    @abstractmethod
    def add_instance(self):
        pass

    @abstractmethod
    def add_feature(self, name, function):
        pass

    @abstractmethod
    def add_culture(self, name, function):
        pass

    def __init__(self, is_shifted=False, instance_type=None, **kwargs):
        self.instance_type = instance_type
        self.is_shifted = is_shifted
        self.default_num_candidates = 10
        self.default_num_voters = 100
        self.default_committee_size = 1
        self.all_winning_committees = {}
        super().__init__(**kwargs)

    def __getattr__(self, attr):
        if attr == 'elections':
            return self.instances
        elif attr == 'num_elections':
            return self.num_instances
        else:
            return self.__dict__[attr]

    def __setattr__(self, name, value):
        if name == "elections":
            self.instances = value
        elif name == "num_elections":
            self.num_instances = value
        else:
            self.__dict__[name] = value

    def prepare_matrices(self):
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "matrices")
        for file_name in os.listdir(path):
            os.remove(os.path.join(path, file_name))

        for election_id in self.elections:
            matrix = self.elections[election_id].votes_to_positionwise_matrix()
            file_name = election_id + ".csv"
            path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                "matrices", file_name)

            with open(path, 'w', newline='') as csv_file:

                writer = csv.writer(csv_file, delimiter=';')
                header = [str(i) for i in range(self.elections[election_id].num_candidates)]
                writer.writerow(header)
                for row in matrix:
                    writer.writerow(row)

    def add_instances_to_experiment(self):
        instances = {}

        for family_id in self.families:
            single = self.families[family_id].single
            ids = []
            for j in range(self.families[family_id].size):
                instance_id = get_instance_id(single, family_id, j)
                if self.instance_type == 'ordinal':
                    instance = OrdinalElection(self.experiment_id, instance_id,
                                               is_imported=True,
                                               fast_import=self.fast_import,
                                               with_matrix=self.with_matrix,
                                               label=self.families[family_id].label)
                elif self.instance_type == 'approval':
                    instance = ApprovalElection(self.experiment_id, instance_id,
                                                is_imported=True,
                                                fast_import=self.fast_import,
                                                label=self.families[family_id].label)
                else:
                    instance = None

                instances[instance_id] = instance
                ids.append(str(instance_id))

            self.families[family_id].election_ids = ids

        return instances

    def set_default_num_candidates(self, num_candidates: int) -> None:
        """ Set default number of candidates """
        self.default_num_candidates = num_candidates

    def set_default_num_voters(self, num_voters: int) -> None:
        """ Set default number of voters """
        self.default_num_voters = num_voters

    def set_default_committee_size(self, committee_size: int) -> None:
        """ Set default size of the committee """
        self.default_committee_size = committee_size

    def add_election(self,
                     culture_id="none",
                     params=None,
                     label=None,
                     color="black",
                     alpha=1.,
                     show=True,
                     marker='x',
                     starting_from=0,
                     size=1,
                     num_candidates=None,
                     num_voters=None,
                     election_id=None):
        """ Add election to the experiment """

        if num_candidates is None:
            num_candidates = self.default_num_candidates

        if num_voters is None:
            num_voters = self.default_num_voters

        return self.add_family(culture_id=culture_id,
                               params=params,
                               size=size,
                               label=label,
                               color=color,
                               alpha=alpha,
                               show=show,
                               marker=marker,
                               starting_from=starting_from,
                               family_id=election_id,
                               num_candidates=num_candidates,
                               num_voters=num_voters,
                               single=True)

    def add_family(self,
                   culture_id: str = "none",
                   params: dict = None,
                   size: int = 1,
                   label: str = None,
                   color: str = "black",
                   alpha: float = 1.,
                   show: bool = True,
                   marker: str = 'o',
                   starting_from: int = 0,
                   num_candidates: int = None,
                   num_voters: int = None,
                   family_id: str = None,
                   single: bool = False,
                   path: dict = None,
                   election_id: str = None,
                   **kwargs) -> list:
        """ Add family of elections to the experiment """

        if election_id is not None:
            family_id = election_id

        if num_candidates is None:
            num_candidates = self.default_num_candidates

        if num_voters is None:
            num_voters = self.default_num_voters

        if self.families is None:
            self.families = {}

        if family_id is None:
            family_id = culture_id + '_' + str(num_candidates) + '_' + str(num_voters)
            if culture_id in {'urn'} and params['alpha'] is not None:
                family_id += '_' + str(float(params['alpha']))
            elif culture_id in {'mallows'} and params['phi'] is not None:
                family_id += '_' + str(float(params['phi']))
            elif culture_id in {'norm-mallows', 'norm-mallows_matrix'} \
                    and params['normphi'] is not None:
                family_id += '_' + str(float(params['normphi']))

        elif label is None:
            label = family_id
        self.families[family_id] = ElectionFamily(culture_id=culture_id,
                                                  family_id=family_id,
                                                  params=params,
                                                  label=label,
                                                  color=color,
                                                  alpha=alpha,
                                                  show=show,
                                                  size=size,
                                                  marker=marker,
                                                  starting_from=starting_from,
                                                  num_candidates=num_candidates,
                                                  num_voters=num_voters,
                                                  path=path,
                                                  single=single,
                                                  instance_type=self.instance_type,
                                                  **kwargs)

        self.num_families = len(self.families)
        self.num_elections = sum([self.families[family_id].size for family_id in self.families])
        self.main_order = [i for i in range(self.num_elections)]

        new_instances = self.families[family_id].prepare_family(
            is_exported=self.is_exported,
            experiment_id=self.experiment_id,
            instance_type=self.instance_type)

        for instance_id in new_instances:
            self.instances[instance_id] = new_instances[instance_id]

        self.families[family_id].instance_ids = list(new_instances.keys())

        return list(new_instances.keys())

    def add_empty_family(self,
                         culture_id: str = "none",
                         label: str = None,
                         color: str = "black",
                         alpha: float = 1.,
                         show: bool = True,
                         marker: str = 'o',
                         num_candidates: int = None,
                         num_voters: int = None,
                         family_id: str = None):

        if label is None:
            label = family_id
        self.families[family_id] = ElectionFamily(culture_id=culture_id,
                                                  family_id=family_id,
                                                  label=label,
                                                  color=color,
                                                  alpha=alpha,
                                                  show=show,
                                                  size=0,
                                                  marker=marker,
                                                  num_candidates=num_candidates,
                                                  num_voters=num_voters,
                                                  instance_type=self.instance_type)

        self.families[family_id].prepare_family(
            is_exported=self.is_exported,
            experiment_id=self.experiment_id)

        return self.families[family_id]

    def prepare_elections(self, store_points=False, is_aggregated=True):
        """ Prepare elections for a given experiment """

        self.store_points = store_points
        self.is_aggregated = is_aggregated

        if self.instances is None:
            self.instances = {}

        for family_id in tqdm(self.families, desc="Preparing instances"):

            new_instances = self.families[family_id].prepare_family(
                is_exported=self.is_exported,
                experiment_id=self.experiment_id,
                store_points=store_points,
                is_aggregated=is_aggregated,
                instance_type=self.instance_type,
            )

            for instance_id in new_instances:
                self.instances[instance_id] = new_instances[instance_id]

    def compute_winners(self, method=None, num_winners=1):
        for election_id in self.elections:
            self.elections[election_id].compute_winners(method=method, num_winners=num_winners)

    def compute_alternative_winners(self, method=None, num_winners=None, num_parties=None):
        for election_id in self.elections:
            for party_id in range(num_parties):
                self.elections[election_id].compute_alternative_winners(
                    method=method, party_id=party_id, num_winners=num_winners)

    def compute_distances(self,
                          distance_id: str = None,
                          num_processes: int = 1,
                          self_distances: bool = False,
                          **kwargs) -> None:
        """ Compute distances between elections (using processes) """

        if distance_id is None:
            distance_id = self.distance_id

        self.distance_id = distance_id

        if '-approvalwise' in distance_id:
            for election in self.elections.values():
                election.votes_to_approvalwise_vector()
        elif '-coapproval_frequency' in distance_id:
            for election in self.elections.values():
                election.votes_to_coapproval_frequency_vectors(**kwargs)
        elif '-voterlikeness' in distance_id:
            for election in self.elections.values():
                election.votes_to_voterlikeness_matrix(**kwargs)
        elif '-candidatelikeness' in distance_id:
            for election in self.elections.values():
                election.votes_to_candidatelikeness_sorted_vectors()
        elif '-pairwise' in distance_id:
            for election in self.elections.values():
                election.votes_to_pairwise_matrix()

        matchings = {election_id: {} for election_id in self.elections}
        distances = {election_id: {} for election_id in self.elections}
        times = {election_id: {} for election_id in self.elections}

        ids = []
        for i, election_1 in enumerate(self.elections):
            for j, election_2 in enumerate(self.elections):
                if i < j or (i == j and self_distances):
                    ids.append((election_1, election_2))

        num_distances = len(ids)

        if self.experiment_id == 'virtual' or num_processes == 1:
            metr.run_single_process(self, ids, distances, times, matchings)

        else:
            processes = []
            for t in range(num_processes):
                print(f'Starting process: {t}')
                sleep(0.1)
                start = int(t * num_distances / num_processes)
                stop = int((t + 1) * num_distances / num_processes)
                instances_ids = ids[start:stop]

                process = Process(target=metr.run_multiple_processes, args=(self,
                                                                            instances_ids,
                                                                            distances,
                                                                            times,
                                                                            matchings,
                                                                            t))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            distances = {instance_id: {} for instance_id in self.instances}
            times = {instance_id: {} for instance_id in self.instances}

            for t in range(num_processes):

                file_name = f'{distance_id}_p{t}.csv'
                path = os.path.join(os.getcwd(),
                                    "experiments",
                                    self.experiment_id,
                                    "distances",
                                    file_name)

                with open(path, 'r', newline='') as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=';')

                    for row in reader:
                        distances[row['instance_id_1']][row['instance_id_2']] = \
                            float(row['distance'])
                        times[row['instance_id_1']][row['instance_id_2']] = \
                            float(row['time'])

        if self.is_exported:
            exports.export_distances_to_file(self, distance_id, distances, times, self_distances)

        self.distances = distances
        self.times = times
        self.matchings = matchings

    def get_election_id_from_model_name(self, culture_id: str) -> str:
        for family_id in self.families:
            if self.families[family_id].culture_id == culture_id:
                return family_id

    def print_matrix(self, **kwargs):
        pr.print_matrix(experiment=self, **kwargs)

    def import_controllers(self):
        """ Import controllers from a file """

        families = {}

        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'map.csv')
        with open(path, 'r') as file_:

            header = [h.strip() for h in file_.readline().split(';')]
            reader = csv.DictReader(file_, fieldnames=header, delimiter=';')

            all_num_candidates = []
            all_num_voters = []

            starting_from = 0
            for row in reader:

                culture_id = None
                color = None
                label = None
                params = None
                alpha = None
                size = None
                marker = None
                num_candidates = None
                num_voters = None
                family_id = None
                show = True

                if 'culture_id' in row.keys():
                    culture_id = str(row['culture_id']).strip()

                if 'color' in row.keys():
                    color = str(row['color']).strip()

                if 'label' in row.keys():
                    label = str(row['label'])

                if 'family_id' in row.keys():
                    family_id = str(row['family_id'])

                if 'params' in row.keys():
                    params = ast.literal_eval(str(row['params']))

                if 'alpha' in row.keys():
                    alpha = float(row['alpha'])

                if 'size' in row.keys():
                    size = int(row['size'])

                if 'marker' in row.keys():
                    marker = str(row['marker']).strip()

                if 'num_candidates' in row.keys():
                    num_candidates = int(row['num_candidates'])

                if 'num_voters' in row.keys():
                    num_voters = int(row['num_voters'])

                if 'path' in row.keys():
                    path = ast.literal_eval(str(row['path']))

                if 'show' in row.keys():
                    show = row['show'].strip() == 't'

                single = size == 1

                families[family_id] = ElectionFamily(culture_id=culture_id,
                                                     family_id=family_id,
                                                     params=params,
                                                     label=label,
                                                     color=color,
                                                     alpha=alpha,
                                                     show=show,
                                                     size=size,
                                                     marker=marker,
                                                     starting_from=starting_from,
                                                     num_candidates=num_candidates,
                                                     num_voters=num_voters,
                                                     path=path,
                                                     single=single)
                starting_from += size

                all_num_candidates.append(num_candidates)
                all_num_voters.append(num_voters)

            check_if_all_equal(all_num_candidates, 'num_candidates')
            check_if_all_equal(all_num_voters, 'num_voters')

            self.num_families = len(families)
            self.num_elections = sum([families[family_id].size for family_id in families])
            self.main_order = [i for i in range(self.num_elections)]

        return families

    def compute_feature(self,
                        feature_id: str = None,
                        feature_params=None,
                        overwrite=False,
                        saveas=None,
                        **kwargs) -> dict:

        if feature_params is None:
            feature_params = {}

        if feature_id in ['priceability', 'core', 'ejr']:
            feature_long_id = f'{feature_id}_{feature_params["rule"]}'
        elif feature_id in ['distortion', 'monotonicity']:
            feature_long_id = f'{feature_id}_{self.embedding_id}'
        else:
            feature_long_id = feature_id

        num_iterations = 1
        if 'num_interations' in feature_params:
            num_iterations = feature_params['num_interations']

        if feature_id == 'ejr':
            feature_dict = {'value': {}, 'time': {}, 'ejr': {}, 'pjr': {}, 'jr': {}, 'pareto': {}}
        else:
            feature_dict = {'value': {}, 'time': {}}

        if feature_id in MAIN_GLOBAL_FEATUERS or feature_id in ELECTION_GLOBAL_FEATURES:

            feature = features.get_global_feature(feature_id)

            values = feature(self, election_ids=list(self.instances), **kwargs)

            for instance_id in tqdm(self.instances, desc='Computing feature'):
                feature_dict['value'][instance_id] = values[instance_id]
                if values[instance_id] is None:
                    feature_dict['time'][instance_id] = None
                else:
                    feature_dict['time'][instance_id] = 0

        else:
            feature = features.get_local_feature(feature_id)

            for instance_id in tqdm(self.instances):
                instance = self.elections[instance_id]

                start = time.time()
                solution = None
                for _ in range(num_iterations):

                    if feature_id in ['monotonicity_1', 'monotonicity_triplets']:
                        value = feature(self, instance)

                    elif feature_id in {'avg_distortion_from_guardians',
                                        'worst_distortion_from_guardians',
                                        'distortion_from_all',
                                        'distortion_from_top_100'}:
                        value = feature(self, instance_id)
                    elif feature_id in ['ejr', 'core', 'pareto', 'priceability',
                                        'cohesiveness']:
                        value = instance.get_feature(feature_id, feature_long_id,
                                                     feature_params=feature_params)
                    else:
                        solution = instance.get_feature(feature_id, feature_long_id,
                                                        overwrite=overwrite, **kwargs)
                total_time = time.time() - start
                total_time /= num_iterations

                if solution is not None:
                    if type(solution) is dict:
                        for key in solution:
                            if key not in feature_dict:
                                feature_dict[key] = {}
                            feature_dict[key][instance_id] = solution[key]
                    else:
                        feature_dict['value'][instance_id] = solution
                    feature_dict['time'][instance_id] = total_time
                else:
                    feature_dict['value'][instance_id] = value
                    feature_dict['time'][instance_id] = total_time

        if saveas is None:
            saveas = feature_long_id

        if self.is_exported:
            exports.export_feature_to_file(self,
                                           feature_id,
                                           saveas,
                                           feature_dict)

        self.features[saveas] = feature_dict
        return feature_dict

    def compute_rules(self, list_of_rules,
                      committee_size: int = 10,
                      resolute: bool = False) -> None:
        for rule_name in list_of_rules:
            print('Computing', rule_name)
            if rule_name in NOT_ABCVOTING_RULES:
                rules.compute_not_abcvoting_rule(experiment=self,
                                                 rule_name=rule_name,
                                                 committee_size=committee_size,
                                                 resolute=resolute)
            else:
                rules.compute_abcvoting_rule(experiment=self, rule_name=rule_name,
                                             committee_size=committee_size,
                                             resolute=resolute)

    def import_committees(self, list_of_rules):
        for rule_name in list_of_rules:
            self.all_winning_committees[rule_name] = rules.import_committees_from_file(
                experiment_id=self.experiment_id, rule_name=rule_name)

    def add_election_to_family(self, election=None, family_id=None):
        self.instances[election.instance_id] = election
        self.families[family_id].add_election(election)


def check_if_all_equal(values, subject):
    if any(x != values[0] for x in values):
        text = f'Not all {subject} values are equal!'
        warnings.warn(text)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
