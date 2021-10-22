#!/usr/bin/env python

class Family:
    """ Family of elections: a set of elections from the same election model_id """

    def __init__(self, model_id: str = None, family_id='none', params: dict = None,
                 size: int = 1, label: str = "none",
                 color: str = "black", alpha: float = 1.,
                 show=True, marker='o', starting_from: int = 0,
                 num_candidates=None, num_voters=None, single_election: bool = False,
                 election_ids=None, ballot: str = 'ordinal', path: dict = None,
                 num_nodes: int = None):

        self.family_id = family_id
        self.model = model_id
        self.params = params
        self.size = size
        self.label = label
        self.color = color
        self.alpha = alpha
        self.show = show
        self.marker = marker
        self.starting_from = starting_from
        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.single_election = single_election
        self.election_ids = election_ids
        self.ballot = ballot
        self.path = path
        self.num_nodes = num_nodes

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
