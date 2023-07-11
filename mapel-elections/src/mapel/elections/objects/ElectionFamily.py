#!/usr/bin/env python

import copy
import logging

from mapel.core.objects.Family import Family
from mapel.elections.objects.OrdinalElection import OrdinalElection
from mapel.elections.objects.ApprovalElection import ApprovalElection
from mapel.core.utils import *
from mapel.elections.cultures.params import *


class ElectionFamily(Family):
    """ Family of elections: a set of elections from the same election culture_id """

    def __init__(self,
                 culture_id: str = None,
                 family_id: str = 'none',
                 params: dict = None,
                 size: int = 1,
                 label: str = None,
                 color: str = "black",
                 alpha: float = 1.,
                 ms: int = 20,
                 show: bool = True,
                 marker: str = 'o',
                 starting_from: int = 0,
                 path: dict = None,
                 single: bool = False,

                 instance_type = None,
                 num_candidates: int = None,
                 num_voters: int = None,
                 election_ids=None,
                 **kwargs):

        super().__init__(culture_id=culture_id,
                         family_id=family_id,
                         params=params,
                         size=size,
                         label=label,
                         color=color,
                         alpha=alpha,
                         ms=ms,
                         show=show,
                         marker=marker,
                         starting_from=starting_from,
                         path=path,
                         single=single,
                         instance_ids=election_ids,
                         **kwargs)

        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.instance_type = instance_type


    def __getattr__(self, attr):
        if attr == 'election_ids':
            return self.instance_ids
        else:
            return self.__dict__[attr]

    def __setattr__(self, name, value):
        if name == "election_ids":
            return setattr(self, 'instance_ids', value)
        else:
            self.__dict__[name] = value

    def prepare_family(self, experiment_id=None, is_exported=True,
                       store_points=False, is_aggregated=True,
                       instance_type=None):

        if instance_type is not None:
            self.instance_type = instance_type

        if self.instance_type == 'ordinal':

            elections = {}
            _keys = []
            for j in range(self.size):

                params = copy.deepcopy(self.params)

                variable = None
                path = self.path
                if path is not None and 'variable' in path:
                    new_params, variable = get_params_for_paths(self, j)
                    if params is None:
                        params = {}
                    params = {**params, **new_params}

                election_id = get_instance_id(self.single, self.family_id, j)

                election = OrdinalElection(experiment_id, election_id,
                                           culture_id=self.culture_id,
                                           num_voters=self.num_voters,
                                           label=self.label,
                                           num_candidates=self.num_candidates,
                                           params=copy.deepcopy(params),
                                           variable=variable,
                                           is_imported=False,
                                           )

                election.prepare_instance(is_exported=is_exported, is_aggregated=is_aggregated)

                if store_points:
                    try:
                        election.points['voters'] = election.import_ideal_points('voters')
                        election.points['candidates'] = election.import_ideal_points('candidates')
                    except:
                        pass

                election.compute_potes()

                elections[election_id] = election

                _keys.append(election_id)

            self.election_ids = _keys

        elif self.instance_type == 'approval':

            elections = {}
            _keys = []
            for j in range(self.size):

                params = copy.deepcopy(self.params)

                variable = None
                path = self.path
                if path is not None and 'variable' in path:
                    new_params, variable = get_params_for_paths(self, j)
                    params = {**params, **new_params}

                if self.culture_id in {'all_votes'}:
                    params['iter_id'] = j

                if self.culture_id in {'crate'}:
                    new_params = get_params_for_crate(j)
                    params = {**params, **new_params}

                election_id = get_instance_id(self.single, self.family_id, j)

                election = ApprovalElection(experiment_id,
                                            election_id,
                                            culture_id=self.culture_id,
                                            num_voters=self.num_voters,
                                            label=self.label,
                                            num_candidates=self.num_candidates,
                                            params=copy.deepcopy(params),
                                            ballot_type=self.instance_type,
                                            variable=variable,
                                            is_imported=False
                                            )
                election.prepare_instance(is_exported=is_exported, is_aggregated=is_aggregated)

                election.votes_to_approvalwise_vector()

                elections[election_id] = election

                _keys.append(election_id)

            self.election_ids = _keys

        else:
            logging.warning('No such instance type!')
            return None

        return elections

    def add_election(self, election):
        self.size += 1
        self.election_ids.append(election.instance_id)



# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
