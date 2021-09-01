#!/usr/bin/env python

from mapel.voting.elections.group_separable import get_gs_caterpillar_vectors
from mapel.voting.elections.single_peaked import get_walsh_vectors, \
    get_conitzer_vectors
from mapel.voting.elections.single_crossing import get_single_crossing_vectors

import mapel.voting._elections as el

from mapel.voting.objects.Election import Election, get_fake_vectors_single, \
    get_fake_convex
from mapel.voting.objects.Experiment import Experiment

import os
import csv


def prepare_matrices(experiment_id):
    """ compute positionwise matrices and
    store them in the /matrices folder """
    experiment = Experiment(experiment_id, elections='import')

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "matrices")
    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))

    for election_id in experiment.elections:
        matrix = \
            experiment.elections[election_id].votes_to_positionwise_matrix()
        file_name = election_id + ".csv"
        path = os.path.join(os.getcwd(), "experiments", experiment_id,
                            "matrices", file_name)

        with open(path, 'w', newline='') as csv_file:

            writer = csv.writer(csv_file, delimiter=';')
            header = [str(i) for i in
                      range(experiment.elections[election_id].num_candidates)]
            writer.writerow(header)
            for row in matrix:
                writer.writerow(row)


def generate_positionwise_matrix(election_model=None, num_candidates=None,
                                 num_voters=100, params=None):
    # todo: there is a repetition of this code in Election file

    if election_model == 'conitzer_matrix':
        vectors = get_conitzer_vectors(num_candidates)
    elif election_model == 'walsh_matrix':
        vectors = get_walsh_vectors(num_candidates)
    elif election_model == 'single-crossing_matrix':
        vectors = get_single_crossing_vectors(num_candidates)
    elif election_model == 'gs_caterpillar_matrix':
        vectors = get_gs_caterpillar_vectors(num_candidates)
    elif election_model in {'identity', 'uniformity',
                            'antagonism', 'stratification'}:
        vectors = get_fake_vectors_single(election_model,
                                          num_candidates, num_voters)
    elif election_model in {'unid', 'anid', 'stid', 'anun', 'stun', 'stan'}:
        vectors = get_fake_convex(election_model, num_candidates,
                                  num_voters, params, get_fake_vectors_single)
    else:
        votes = el.generate_votes(election_model=election_model,
                                  num_candidates=num_candidates,
                                  num_voters=num_voters,
                                  params=params)
        return get_positionwise_matrix(votes)

    return vectors.transpose()


def get_positionwise_matrix(votes):
    election = Election("virtual", "virtual", votes=votes)
    return election.votes_to_positionwise_matrix()
