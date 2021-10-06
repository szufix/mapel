#!/usr/bin/env python

import csv
import os
import numpy as np
import copy


from mapel.voting.winners import compute_sntv_winners, compute_borda_winners, compute_stv_winners

from mapel.voting.glossary import LIST_OF_FAKE_MODELS, LIST_OF_PREFLIB_MODELS, PATHS

from mapel.voting.not_in_the_package.__winners import generate_winners

from mapel.voting.objects.Instance import Instance


class Election(Instance):

    def __init__(self, experiment_id, name, votes=None, with_matrix=False, alpha=None, model=None,
                 ballot='ordinal', num_voters=None, num_candidates=None):

        super().__init__(experiment_id, name, model=model, alpha=alpha)

        self.election_id = name
        self.ballot = ballot

        self.num_voters = num_voters
        self.num_candidates = num_candidates
        self.winners = None
        self.alternative_winners = {}

        if model in LIST_OF_FAKE_MODELS:
            self.fake = True
        else:
            self.fake = False

        if ballot == 'ordinal':
            if votes is not None:
                if str(votes[0]) in LIST_OF_FAKE_MODELS:
                    self.fake = True
                    self.votes = votes[0]
                    self.election_model = votes[0]
                    self.num_candidates = votes[1]
                    self.num_voters = votes[2]
                    self.fake_param = votes[3]
                else:
                    self.votes = votes
                    self.num_candidates = len(votes[0])
                    self.num_voters = len(votes)
                    self.election_model = model
                    self.potes = self.votes_to_potes()
            else:
                self.fake = check_if_fake(experiment_id, name)
                if self.fake:
                    self.election_model, self.fake_param, self.num_voters, \
                        self.num_candidates = import_fake_elections(experiment_id, name)
                else:
                    self.votes, self.num_voters, self.num_candidates, self.param, \
                        self.election_model = import_soc_election(experiment_id, name)

                    self.potes = self.votes_to_potes()

            if with_matrix:
                self.matrix = self.import_matrix()
                self.vectors = self.matrix.transpose()
            else:
                self.votes_to_positionwise_vectors()

        elif ballot in ['approval']:
            self.votes = votes
            self.election_model = model

    def import_matrix(self):

        file_name = self.election_id + '.csv'
        path = os.path.join(os.getcwd(), "experiments", self.experiment_id, 'matrices', file_name)
        matrix = np.zeros([self.num_candidates, self.num_candidates])

        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for i, row in enumerate(reader):
                for j, candidate_id in enumerate(row):
                    matrix[i][j] = row[candidate_id]
        return matrix

    def votes_to_potes(self):
        """ Prepare positional votes """
        potes = np.zeros([self.num_voters, self.num_candidates])
        for i in range(self.num_voters):
            for j in range(self.num_candidates):
                potes[i][self.votes[i][j]] = j
        return potes

    def vector_to_interval(self, vector, precision=None):
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
        # print(election_without_party_id.votes)
        election_without_party_id = map_the_votes(election_without_party_id, party_id, num_winners)
        # print(election_without_party_id.votes)

        if method == 'sntv':
            winners_without_party_id = compute_sntv_winners(election=election_without_party_id,
                                                            num_winners=num_winners)
        if method == 'borda':
            winners_without_party_id = compute_borda_winners(election=election_without_party_id,
                                                             num_winners=num_winners)
        if method == 'stv':
            winners_without_party_id = compute_stv_winners(election=election_without_party_id,
                                                           num_winners=num_winners)
        if method in {'approx_cc', 'approx_hb', 'approx_pav'}:
            winners_without_party_id = generate_winners(election=election_without_party_id,
                                                        num_winners=num_winners, method=method)

        # print(winners_without_party_id)
        winners_without_party_id = unmap_the_winners(winners_without_party_id, party_id, num_winners)
        # print(winners_without_party_id)

        self.alternative_winners[party_id] = winners_without_party_id


def map_the_votes(election, party_id, party_size):
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


def check_if_fake(experiment_id, election_id):
    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    line = my_file.readline().strip()
    if line[0] == '$':
        return True
    return False


def import_fake_elections(experiment_id, election_id):
    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", file_name)
    my_file = open(path, 'r')
    my_file.readline()  # line with $ fake

    num_voters = int(my_file.readline().strip())
    num_candidates = int(my_file.readline().strip())
    fake_model_name = str(my_file.readline().strip())
    params = {}
    if fake_model_name == 'crate':
        params = [float(my_file.readline().strip()), float(my_file.readline().strip()),
                  float(my_file.readline().strip()), float(my_file.readline().strip())]
    elif fake_model_name == 'norm-mallows_matrix':
        params['norm-phi'] = float(my_file.readline().strip())
        params['weight'] = float(my_file.readline().strip())
    elif fake_model_name in PATHS:
        params['alpha'] = float(my_file.readline().strip())
        if fake_model_name == 'mallows_matrix_path':
            params['norm-phi'] = params['alpha']
            params['weight'] = float(my_file.readline().strip())
    else:
        params = {}

    return fake_model_name, params, num_voters, num_candidates


def import_soc_election(experiment_id, election_id):
    file_name = str(election_id) + ".soc"
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "instances", file_name)
    my_file = open(path, 'r')

    param = 0
    first_line = my_file.readline()
    if first_line[0] != '#':
        model_name = 'empty'
        num_candidates = int(first_line)
    else:
        first_line = first_line.strip().split()
        model_name = first_line[1]
        if any(map(str.isdigit, first_line[len(first_line) - 1])):
            param = first_line[len(first_line) - 1]
        num_candidates = int(my_file.readline())

    for _ in range(num_candidates):
        my_file.readline()

    line = my_file.readline().rstrip("\n").split(',')
    num_voters = int(line[0])
    num_options = int(line[2])
    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    it = 0
    for j in range(num_options):
        line = my_file.readline().rstrip("\n").split(',')
        quantity = int(line[0])

        for k in range(quantity):
            for l in range(num_candidates):
                votes[it][l] = int(line[l + 1])
            it += 1

    # Shift by -1
    if model_name in LIST_OF_PREFLIB_MODELS:
        for i in range(num_voters):
            for j in range(num_candidates):
                votes[i][j] -= 1

    return votes, num_voters, num_candidates, param, model_name


def remove_candidate_from_election(election, party_id, party_size):
    for vote in election.votes:
        for i in range(party_size):
            _id = party_id*party_size + i
            vote.remove(_id)
    election.num_candidates -= party_size
    return election
