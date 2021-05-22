import math
import os
import time
#import concurrent.futures
import itertools

from threading import Thread
import numpy as np

from . import elections as el
from . import metrics as metr
from . import objects as obj


# TO BE UPDATED
def compute_highest_copeland_map(experiment_id):
    model = obj.Model(experiment_id)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                             "highest_copeland.txt")
    file_scores = open(file_name, 'w')

    for i in range(model.num_elections):
        election_id = "core_" + str(i)
        election = obj.Election(experiment_id, election_id)

        if not election.fake:
            score = metr.get_highest_copeland_score(election.potes, election.num_voters, election.num_candidates)
        else:
            score = -1

        file_scores.write(str(score) + "\n")

    file_scores.close()


def compute_highest_borda1m_map(experiment_id):
    model = obj.Model(experiment_id)
    scores = []

    for i in range(model.num_elections):
        election_id = 'core_' + str(i)
        election = obj.Election(experiment_id, election_id)

        score = metr.get_highest_borda12_score(election)
        scores.append(score)
        print(election_id, score)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                             "highest_borda1m.txt")
    file_scores = open(file_name, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_levels(experiment_id, election_model='identity'):
    model = obj.Model(experiment_id)

    # election_model = 'identity'
    num_voters = 100
    num_candidates = 10
    x = 'x'

    scores = []

    for i in range(model.num_elections):
        election_1_id = 'core_' + str(i)
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_candidates=num_candidates, num_voters=num_voters, special=0)

        elections_ids = [election_1_id, election_2_id]
        score = metr.get_distance(experiment_id, elections_ids)

        print(election_1_id, score)
        scores.append(score)

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", "levels.txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_distance_from_diameter(experiment_id, metric='positionwise'):
    model = obj.Model_xd(experiment_id, distance_name=metric)

    scores = []
    for i in range(model.num_elections):
        min_dist = math.inf
        for j in range(900, 1001):
            if model.distances[i][j] < min_dist:
                min_dist = model.distances[i][j]

        scores.append(min_dist)
        # print(min_dist)

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", "diameter.txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_distance_from_unid(experiment_id):
    model = obj.Model(experiment_id)

    num_elections = 1
    x = 'x'

    scores = []
    for i in range(model.num_elections):
        election_1_id = 'core_' + str(i)
        election_1 = obj.Election(experiment_id, election_1_id)

        election_model = 'identity'
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score_a = metr.get_distance_from_unid(election_1, election_2)

        election_model = 'equality'
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)

        election_2 = obj.Election(experiment_id, election_2_id)
        score_b = metr.get_distance_from_unid(election_1, election_2)

        score = score_a + score_b

        scores.append(score)

        print(election_1_id, round(score_a, 2), round(score_b, 2), round(score, 2))

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", "unid.txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_distance_from_dwa_sery(experiment_id):
    model = obj.Model(experiment_id)

    num_elections = 1
    x = 'x'

    scores = []
    for i in range(model.num_elections):
        score = 0

        election_1_id = 'core_' + str(i)
        election_1 = obj.Election(experiment_id, election_1_id)

        election_model = 'queue'
        x = 0
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score += metr.get_distance_from_unid(election_1, election_2)

        election_model = 'antagonism'
        x = 50
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score += metr.get_distance_from_unid(election_1, election_2)

        scores.append(score)

        print(election_1_id, round(score, 2))

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", "dwa_sery.txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_distance_from_cztery_sery(experiment_id):
    model = obj.Model(experiment_id)

    num_elections = 1
    x = 'x'

    scores = []
    for i in range(model.num_elections):
        score = 0

        election_1_id = 'core_' + str(i)
        election_1 = obj.Election(experiment_id, election_1_id)

        election_model = 'queue'
        x = 0
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score += metr.get_distance_from_unid(election_1, election_2)

        election_model = 'identity'
        x = 0
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score += metr.get_distance_from_unid(election_1, election_2)

        election_model = 'uniformity'
        x = 0
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score += metr.get_distance_from_unid(election_1, election_2)

        election_model = 'antagonism'
        x = 50
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score += metr.get_distance_from_unid(election_1, election_2)

        scores.append(score)

        print(election_1_id, round(score, 2))

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", "cztery_sery.txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_distance_from_chess(experiment_id):
    model = obj.Model(experiment_id)

    num_elections = 1
    x = 'x'

    scores = []
    for i in range(model.num_elections):
        election_1_id = 'core_' + str(i)
        election_1 = obj.Election(experiment_id, election_1_id)

        election_model = 'queue'
        x = 0
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score = metr.get_distance_from_unid(election_1, election_2)

        scores.append(score)

        print(election_1_id, round(score, 2))

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", "chess.txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_distance_from_random_point(experiment_id, rand_id=-1, nice_name=''):
    model = obj.Model_xd(experiment_id, distance_name='positionwise')

    scores = []
    for i in range(model.num_elections):
        score = model.distances[i][rand_id]
        scores.append(score)

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                        "rand_point_" + str(nice_name) + ".txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_distance_from_ant(experiment_id):
    model = obj.Model(experiment_id)

    num_elections = 1
    x = 'x'

    scores = []
    for i in range(model.num_elections):
        election_1_id = 'core_' + str(i)
        election_1 = obj.Election(experiment_id, election_1_id)

        election_model = 'antagonism'
        x = 50
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_elections=num_elections,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        election_2 = obj.Election(experiment_id, election_2_id)
        score = metr.get_distance_from_unid(election_1, election_2)

        scores.append(score)
        print(election_1_id, round(score, 2))

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", "ant.txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def triangle_test(experiment_id):
    num_voters = 100
    num_candidates = 10
    num_elections = 1
    elections_type = 'antagonism'
    # print(method, x)

    distance_name = "positionwise"
    metric_type = "emd"

    # COMPUTE ALL 'GUESS' ELECTIONS

    for T1 in range(100):
        x = T1
        elections_id_a = "guess_" + str(T1)
        el.generate_elections(experiment_id, election_model=elections_type, election_id=elections_id_a,
                              num_elections=num_elections,
                              num_candidates=num_candidates, num_voters=num_voters, special=x)

    # COMPUTE ALL DISTANCES

    for T1 in range(100):
        elections_id_a = "guess_" + str(T1)
        elections_id_b = "core_" + str(0)
        elections_ids = [elections_id_a, elections_id_b]
        distances_1 = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)[0]
        distances_1 = round(distances_1, 2)

        elections_id_a = "guess_" + str(T1)
        elections_id_b = "core_" + str(1)
        elections_ids = [elections_id_a, elections_id_b]
        distances_2 = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)[0]
        distances_2 = round(distances_2, 2)

        total = round(distances_1 + distances_2, 2)

        print(distances_1, distances_2, total)


# NEW 10.12.2020
def compute_distances_from_guardians(experiment_id):
    guardians = ['identity', 'uniformity', 'antagonism', 'stratification']
    # guardians = ['antagonism']

    for guardian in guardians:

        model = obj.Model(experiment_id)

        election_model = guardian
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_voters=100, num_candidates=10, special=0)
        # election_2 = obj.Election(experiment_id, election_2_id)

        scores = []
        for i in range(model.num_elections):
            election_1_id = 'core_' + str(i)
            election_1 = obj.Election(experiment_id, election_1_id)
            election_2 = obj.Election(experiment_id, election_2_id)

            score = metr.get_distance(election_1, election_2, distance_name='positionwise', metric_name='emd')

            scores.append(score)
            print(election_1_id, round(score, 4))

        path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", guardian + '.txt')
        file_scores = open(path, 'w')
        for i in range(model.num_elections):
            file_scores.write(str(scores[i]) + "\n")
        file_scores.close()


