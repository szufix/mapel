#!/usr/bin/env python
import copy
from abc import abstractmethod

import mapel.voting.elections_main as _elections
import mapel.voting.other.rules as rules
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


class ElectionExperiment(Experiment):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.default_num_candidates = 10
        self.default_num_voters = 100
        self.default_committee_size = 1

        self.all_winning_committees = {}

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
                     num_candidates=None, num_voters=None, election_id=None, num_nodes=None):
        """ Add election to the experiment """

        if num_candidates is None:
            num_candidates = self.default_num_candidates

        if num_voters is None:
            num_voters = self.default_num_voters

        return self.add_election_family(model_id=model_id, params=params, size=size, label=label,
                                        color=color, alpha=alpha, show=show, marker=marker,
                                        starting_from=starting_from, family_id=election_id,
                                        num_candidates=num_candidates, num_voters=num_voters,
                                        num_nodes=num_nodes, single_election=True)[0]

    def add_election_family(self, model_id: str = "none", params: dict = None, size: int = 1,
                            label: str = None, color: str = "black", alpha: float = 1.,
                            show: bool = True, marker: str = 'o', starting_from: int = 0,
                            num_candidates: int = None, num_voters: int = None,
                            family_id: str = None, single_election: bool = False,
                            num_nodes: int = None, path: dict = None,
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

        self.families[family_id] = Family(model_id=model_id, family_id=family_id,
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
        model_id = self.families[family_id].model_id

        ids = _elections.prepare_statistical_culture_family(experiment=self,
                                                            model_id=model_id,
                                                            family_id=family_id,
                                                            params=copy.deepcopy(params))

        self.families[family_id].election_ids = ids

        return ids

    @abstractmethod
    def create_structure(self):
        pass

    def compute_rules(self, list_of_rules, committee_size: int = 1, printing: bool = False) -> None:
        for rule_name in list_of_rules:
            print('Computing', rule_name)
            rules.compute_rule(experiment=self, rule_name=rule_name, committee_size=committee_size,
                               printing=printing)

    def import_committees(self, list_of_rules) -> None:
        for rule_name in list_of_rules:
            self.all_winning_committees[rule_name] = rules.import_committees_from_file(
                experiment_id=self.experiment_id, rule_name=rule_name)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
