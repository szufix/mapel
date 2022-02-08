#!/usr/bin/env python
import copy
import csv
import itertools
import os
import warnings
from abc import abstractmethod
from threading import Thread
from time import sleep
import numpy as np
import ast

from mapel.elections.objects.ElectionFamily import ElectionFamily
import mapel.elections.metrics_main as metr
import mapel.elections.models_main as _elections
import mapel.elections.other.rules as rules
import mapel.elections.features_main as features
import mapel.elections.models.preflib as preflib
from mapel.elections._glossary import *
from mapel.main.objects.Experiment import Experiment
from mapel.main.objects.Family import Family
import mapel.elections._print as pr

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

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.default_num_candidates = 10
        self.default_num_voters = 100
        self.default_committee_size = 1

        self.all_winning_committees = {}

        # self.elections = None
        # self.num_elections = None

    def __getattr__(self, attr):
        if attr == 'elections':
            return self.instances
        elif attr == 'num_elections':
            return self.num_elections
        else:
            return getattr(self, attr)

    def add_instances_to_experiment(self):
        return self.add_elections_to_experiment()

    @abstractmethod
    def add_elections_to_experiment(self):
        pass

    def set_default_num_candidates(self, num_candidates: int) -> None:
        """ Set default number of candidates """
        self.default_num_candidates = num_candidates

    def set_default_num_voters(self, num_voters: int) -> None:
        """ Set default number of voters """
        self.default_num_voters = num_voters

    def set_default_committee_size(self, committee_size: int) -> None:
        """ Set default size of the committee """
        self.default_committee_size = committee_size

    def add_election(self, model_id="none", params=None, label=None,
                     color="black", alpha=1., show=True, marker='x', starting_from=0, size=1,
                     num_candidates=None, num_voters=None, election_id=None):
        """ Add election to the experiment """

        if num_candidates is None:
            num_candidates = self.default_num_candidates

        if num_voters is None:
            num_voters = self.default_num_voters

        return self.add_election_family(model_id=model_id, params=params, size=size, label=label,
                                        color=color, alpha=alpha, show=show, marker=marker,
                                        starting_from=starting_from, family_id=election_id,
                                        num_candidates=num_candidates, num_voters=num_voters,
                                         single_election=True)

    def add_election_family(self, model_id: str = "none", params: dict = None, size: int = 1,
                            label: str = None, color: str = "black", alpha: float = 1.,
                            show: bool = True, marker: str = 'o', starting_from: int = 0,
                            num_candidates: int = None, num_voters: int = None,
                            family_id: str = None, single_election: bool = False,
                             path: dict = None,
                            election_id: str = None) -> list:
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
            family_id = model_id + '_' + str(num_candidates) + '_' + str(num_voters)
            if model_id in {'urn_model'} and params['alpha'] is not None:
                family_id += '_' + str(float(params['alpha']))
            elif model_id in {'mallows'} and params['phi'] is not None:
                family_id += '_' + str(float(params['phi']))
            elif model_id in {'norm-mallows', 'norm-mallows_matrix'} \
                    and params['norm-phi'] is not None:
                family_id += '_' + str(float(params['norm-phi']))

        elif label is None:
            label = family_id

        self.families[family_id] = ElectionFamily(model_id=model_id, family_id=family_id,
                                                  params=params, label=label, color=color,
                                                  alpha=alpha,
                                                  show=show, size=size, marker=marker,
                                                  starting_from=starting_from,
                                                  num_candidates=num_candidates,
                                                  num_voters=num_voters, path=path,
                                                  single_election=single_election)

        self.num_families = len(self.families)
        self.num_elections = sum([self.families[family_id].size for family_id in self.families])
        self.main_order = [i for i in range(self.num_elections)]

        params = self.families[family_id].params
        model_id = self.families[family_id].model_id

        elections = _elections.prepare_statistical_culture_family(experiment=self,
                                                                  model_id=model_id,
                                                                  family_id=family_id,
                                                                  params=copy.deepcopy(params))

        self.families[family_id].instance_ids = list(elections.keys())

        return list(elections.keys())

    # def add_matrices_to_experiment(experiment):
    #     """ Import elections from a file """
    #
    #     matrices = {}
    #     vectors = {}
    #
    #     for family_id in experiment.families:
    #         for j in range(experiment.families[family_id].size):
    #             election_id = family_id + '_' + str(j)
    #             matrix = experiment.import_matrix(election_id)
    #             matrices[election_id] = matrix
    #             vectors[election_id] = matrix.transpose()
    #
    #     return matrices, vectors

    # def import_matrix(experiment, election_id):
    #
    #     file_name = election_id + '.csv'
    #     path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id,
    #     'matrices', file_name)
    #     num_candidates = experiment.elections[election_id].num_candidates
    #     matrix = np.zeros([num_candidates, num_candidates])
    #
    #     with open(path, 'r', newline='') as csv_file:
    #         reader = csv.DictReader(csv_file, delimiter=',')
    #         for i, row in enumerate(reader):
    #             for j, candidate_id in enumerate(row):
    #                 matrix[i][j] = row[candidate_id]
    #     return matrix

    def prepare_elections(self):
        """ Prepare elections for a given experiment """

        if self.elections is None:
            self.elections = {}

        elections = {}
        if self.store:

            for family_id in self.families:
                params = self.families[family_id].params
                model_id = self.families[family_id].model_id

                # if self.clean:
                #     path = os.path.join(os.getcwd(), "experiments", self.experiment_id, "elections")
                #     for file_name in os.listdir(path):
                #         os.remove(os.path.join(path, file_name))

                if model_id in LIST_OF_PREFLIB_MODELS:
                    ids = preflib.prepare_preflib_family(
                        experiment=self, model=model_id, family_id=family_id, params=params)

                    self.families[family_id].election_ids = ids

                elif model_id not in ['core', 'pabulib'] and (model_id in NICE_NAME or
                                                              model_id in LIST_OF_FAKE_MODELS or
                                                              model_id in APPROVAL_MODELS or
                                                              model_id in APPROVAL_FAKE_MODELS):

                    tmp_elections = _elections.prepare_statistical_culture_family(
                        experiment=self, model_id=model_id, family_id=family_id, params=params)

                    self.families[family_id].election_ids = tmp_elections.keys()

                    # for election_id in tmp_elections:
                    #     elections[election_id] = tmp_elections[election_id]

                else:
                    election_ids = {}
                    for j in range(self.families[family_id].size):
                        if self.families[family_id].single_election:
                            election_id = family_id
                        else:
                            election_id = family_id + '_' + str(j)
                        election_ids[election_id] = None
                    self.families[family_id].election_ids = election_ids.keys()


        # self.elections = elections

    def compute_winners(self, method=None, num_winners=1):
        for election_id in self.elections:
            self.elections[election_id].compute_winners(method=method, num_winners=num_winners)

    def compute_alternative_winners(self, method=None, num_winners=None, num_parties=None):
        for election_id in self.elections:
            for party_id in range(num_parties):
                self.elections[election_id].compute_alternative_winners(
                    method=method, party_id=party_id, num_winners=num_winners)

    def compute_distances(self, distance_id: str = 'emd-positionwise', num_threads: int = 1,
                          self_distances: bool = False, vector_type: str = 'A',
                          printing: bool = False) -> None:
        """ Compute distances between elections (using threads) """

        self.distance_id = distance_id

        # precompute vectors, matrices, etc...
        if '-approvalwise' in distance_id:
            for election in self.elections.values():
                election.votes_to_approvalwise_vector()
        elif '-coapproval_frequency' in distance_id or 'flow' in distance_id:
            for election in self.elections.values():
                election.votes_to_coapproval_frequency_vectors(vector_type=vector_type)
        elif '-voterlikeness' in distance_id:
            for election in self.elections.values():
                election.votes_to_voterlikeness_vectors(vector_type=vector_type)
        elif '-candidatelikeness' in distance_id:
            for election in self.elections.values():
                # print(election)
                election.votes_to_candidatelikeness_sorted_vectors()
        elif '-pairwise' in distance_id:
            for election in self.elections.values():
                election.votes_to_pairwise_matrix()
        # continue with normal code

        matchings = {election_id: {} for election_id in self.elections}
        distances = {election_id: {} for election_id in self.elections}
        times = {election_id: {} for election_id in self.elections}

        threads = [{} for _ in range(num_threads)]

        ids = []
        for i, election_1 in enumerate(self.elections):
            for j, election_2 in enumerate(self.elections):
                if i == j:
                    if self_distances:
                        ids.append((election_1, election_2))
                elif i < j:
                    ids.append((election_1, election_2))

        num_distances = len(ids)

        for t in range(num_threads):
            print(f'Starting thread: {t}')
            sleep(0.1)
            start = int(t * num_distances / num_threads)
            stop = int((t + 1) * num_distances / num_threads)
            thread_ids = ids[start:stop]

            threads[t] = Thread(target=metr.run_single_thread, args=(self, thread_ids,
                                                                     distances, times, matchings,
                                                                     printing, t))
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

    def get_election_id_from_model_name(self, model_id: str) -> str:
        for family_id in self.families:
            if self.families[family_id].model_id == model_id:
                return family_id

    def print_matrix(self, **kwargs):
        pr.print_matrix(experiment=self, **kwargs)

    def import_controllers(self):
        """ Import controllers from a file """

        families = {}

        path = os.path.join(os.getcwd(), 'experiments', self.experiment_id, 'map.csv')
        file_ = open(path, 'r')

        header = [h.strip() for h in file_.readline().split(';')]
        reader = csv.DictReader(file_, fieldnames=header, delimiter=';')

        all_num_candidates = []
        all_num_voters = []

        starting_from = 0
        for row in reader:

            model_id = None
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

            if 'model_id' in row.keys():
                model_id = str(row['model_id']).strip()

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
            #
            # if model_id in {'urn_model'} and 'alpha' in params:
            #     family_id += '_' + str(float(params['alpha']))
            # elif model_id in {'mallows'} and 'phi' in params:
            #     family_id += '_' + str(float(params['phi']))
            # elif model_id in {'norm-mallows', 'norm-mallows_matrix'} \
            #         and norm-phiparams['norm-phi'] is not None:
            #     family_id += '_' + str(float(params['norm-phi']))

            single_election = size == 1

            families[family_id] = ElectionFamily(model_id=model_id,
                                                 family_id=family_id,
                                                 params=params, label=label,
                                                 color=color, alpha=alpha, show=show,
                                                 size=size, marker=marker,
                                                 starting_from=starting_from,
                                                 num_candidates=num_candidates,
                                                 num_voters=num_voters, path=path,
                                                 single_election=single_election)
            starting_from += size

            all_num_candidates.append(num_candidates)
            all_num_voters.append(num_voters)

        check_if_all_equal(all_num_candidates, 'num_candidates')
        check_if_all_equal(all_num_voters, 'num_voters')

        self.num_families = len(families)
        self.num_elections = sum([families[family_id].size for family_id in families])
        self.main_order = [i for i in range(self.num_elections)]

        file_.close()
        return families

    def compute_feature(self, feature_id: str = None, feature_params=None) -> dict:

        if feature_params is None:
            feature_params = {}

        feature_dict = {}

        for election_id in self.elections:
            print(election_id)
            feature = features.get_feature(feature_id)
            election = self.elections[election_id]
            if feature_id in ['monotonicity_1', 'monotonicity_triplets']:
                value = feature(self, election)

            elif feature_id in ['largest_cohesive_group', 'number_of_cohesive_groups',
                                'number_of_cohesive_groups_brute',
                                'proportionality_degree_pav',
                                'proportionality_degree_av',
                                'proportionality_degree_cc',
                                'justified_ratio',
                                'cohesiveness',
                                'partylist',
                                'highest_cc_score',
                                'highest_hb_score']:
                value = feature(election, feature_params)

            elif feature_id in {'avg_distortion_from_guardians',
                                'worst_distortion_from_guardians',
                                'distortion_from_all',
                                'distortion_from_top_100'}:
                value = feature(self, election_id)
            else:
                value = feature(election)
            feature_dict[election_id] = value

        if self.store:
            if feature_id in EMBEDDING_RELATED_FEATURE:
                path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                    "features", f'{feature_id}__{self.distance_id}.csv')
            else:
                path = os.path.join(os.getcwd(), "experiments", self.experiment_id,
                                    "features", f'{feature_id}.csv')

            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                if feature_id in {'partylist'}:
                    writer.writerow(["election_id", "value", "bound", "num_large_parties"])
                    for key in feature_dict:
                        writer.writerow([key, feature_dict[key][0], feature_dict[key][1],
                                         feature_dict[key][2]])
                if feature_id in {'highest_cc_score', 'highest_hb_score'}:
                    writer.writerow(["election_id", "value", 'time'])
                    for key in feature_dict:
                        writer.writerow([key, feature_dict[key][0], round(feature_dict[key][1],3)])
                else:
                    writer.writerow(["election_id", "value"])
                    for key in feature_dict:
                        writer.writerow([key, str(feature_dict[key])])

        self.features[feature_id] = feature_dict
        return feature_dict

    @abstractmethod
    def create_structure(self):
        pass

    def compute_rules(self, list_of_rules, committee_size: int = 10, printing: bool = False,
                      resolute: bool = False) -> None:
        for rule_name in list_of_rules:
            print('Computing', rule_name)
            if rule_name in NOT_ABCVOTING_RULES:
                rules.compute_not_abcvoting_rule(experiment=self, rule_name=rule_name,
                                                 committee_size=committee_size,
                                                 printing=printing, resolute=resolute)
            else:
                rules.compute_abcvoting_rule(experiment=self, rule_name=rule_name,
                                             committee_size=committee_size,
                                             printing=printing, resolute=resolute)

    def import_committees(self, list_of_rules) -> None:
        for rule_name in list_of_rules:
            self.all_winning_committees[rule_name] = rules.import_committees_from_file(
                experiment_id=self.experiment_id, rule_name=rule_name)


def check_if_all_equal(values, subject):
    if any(x != values[0] for x in values):
        text = f'Not all {subject} values are equal!'
        warnings.warn(text)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
