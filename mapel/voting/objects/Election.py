#!/usr/bin/env python

import copy
import csv
import os

import numpy as np

from mapel.voting._glossary import *
from mapel.voting.objects.Instance import Instance
from mapel.voting.other.winners import compute_sntv_winners, compute_borda_winners, \
    compute_stv_winners
from mapel.voting.other.winners2 import generate_winners


class Election(Instance):

    def __init__(self, experiment_id, election_id, votes=None, alpha=None, model_id=None,
                 ballot: str = 'ordinal', num_voters: int = None, num_candidates: int = None):

        super().__init__(experiment_id, election_id, model_id=model_id, alpha=alpha)

        self.election_id = election_id
        self.ballot = ballot

        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.winners = None
        self.alternative_winners = {}

        self.fake = model_id in LIST_OF_FAKE_MODELS

        self.votes = votes
        self.model_id = model_id

    def import_matrix(self) -> np.ndarray:

        file_name = f'{self.election_id}.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, 'matrices', file_name)
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for i, row in enumerate(reader):
                for j, candidate_id in enumerate(row):
                    matrix[i][j] = row[candidate_id]
        return matrix

    def votes_to_potes(self) -> np.ndarray:
        """ Convert votes to positional votes """
        return np.array([[vote.index(i) for i, _ in enumerate(vote)] for vote in self.votes])

    def vector_to_interval(self, vector, precision=None) -> list:
        # discreet version for now
        interval = []
        w = int(precision / self.num_candidates)
        for i in range(self.num_candidates):
            for j in range(w):
                interval.append(vector[i] / w)
        return interval

    def compute_alternative_winners(self, method=None, party_id=None, num_winners=None):

        election_without_party_id = remove_candidate_from_election(copy.deepcopy(self),
                                                                   party_id, num_winners)
        election_without_party_id = map_the_votes(election_without_party_id, party_id, num_winners)

        if method == 'sntv':
            winners_without_party_id = compute_sntv_winners(election=election_without_party_id,
                                                            num_winners=num_winners)
        elif method == 'borda':
            winners_without_party_id = compute_borda_winners(election=election_without_party_id,
                                                             num_winners=num_winners)
        elif method == 'stv':
            winners_without_party_id = compute_stv_winners(election=election_without_party_id,
                                                           num_winners=num_winners)
        elif method in {'approx_cc', 'approx_hb', 'approx_pav'}:
            winners_without_party_id = generate_winners(election=election_without_party_id,
                                                        num_winners=num_winners, method=method)
        else:
            winners_without_party_id = []

        winners_without_party_id = unmap_the_winners(winners_without_party_id, party_id, num_winners)

        self.alternative_winners[party_id] = winners_without_party_id


def map_the_votes(election, party_id, party_size) -> Election:
    new_votes = [[] for _ in range(election.num_voters)]
    for i in range(election.num_voters):
        for j in range(election.num_candidates):
            if election.votes[i][j] >= party_id * party_size:
                new_votes[i].append(election.votes[i][j]-party_size)
            else:
                new_votes[i].append(election.votes[i][j])
    election.votes = new_votes
    return election


def unmap_the_winners(winners, party_id, party_size):
    new_winners = []
    for j in range(len(winners)):
        if winners[j] >= party_id * party_size:
            new_winners.append(winners[j]+party_size)
        else:
            new_winners.append(winners[j])
    return new_winners


def remove_candidate_from_election(election, party_id, party_size) -> Election:
    for vote in election.votes:
        for i in range(party_size):
            _id = party_id*party_size + i
            vote.remove(_id)
    election.num_candidates -= party_size
    return election
