
import os
import voting.metrics as metr
import voting.objects as obj
import voting.elections as el
import time
import math


# IN USE
def compute_highest_borda(experiment_id):

    model = obj.Model(experiment_id)
    scores = []

    for i in range(model.num_elections):
        election_id = 'core_' + str(i)
        election = obj.Election(experiment_id, election_id)

        score = metr.get_highest_borda_score(election)
        scores.append(score)
        print(election_id, score)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                             "highest_borda.txt")
    file_scores = open(file_name, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


def compute_highest_plurality(experiment_id):

    model = obj.Model(experiment_id)
    scores = []

    for i in range(model.num_elections):
        election_id = 'core_' + str(i)
        election = obj.Election(experiment_id, election_id)

        score = metr.get_highest_plurality_score(election)
        scores.append(score)
        print(election_id, score)

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                             "highest_plurality.txt")
    file_scores = open(file_name, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(scores[i]) + "\n")
    file_scores.close()


# TO BE UPDATED
def compute_highest_dodgson_map(experiment_id):

    model = obj.Model(experiment_id)

    for i in range(model.num_elections):
        election_id = 'core_' + str(i)
        election = obj.Election(experiment_id, election_id)

        start_time = time.time()

        condorcet_winner = metr.is_condorect_winner(election)
        if condorcet_winner:
            score = 0
        else:
            score = metr.get_dodgson_score(election)

        elapsed_time = time.time() - start_time
        elapsed_time = round(elapsed_time, 4)

        file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                                 "highest_dodgson.txt")
        file_scores = open(file_name, 'a')
        file_scores.write(str(score) + "\n")
        file_scores.close()

        file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                                 "highest_dodgson_time.txt")
        file_scores = open(file_name, 'a')
        file_scores.write(str(elapsed_time) + "\n")
        file_scores.close()


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

        file_scores.write(str(score)+"\n")

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

    #election_model = 'identity'
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
        #print(min_dist)

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

        print(election_1_id, round(score_a,2), round(score_b,2), round(score,2))

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


"""
def triangle_test(experiment_id):
    model = obj.Model(experiment_id)

    num_elections = 1
    x = 'x'

    import random as rand

    means = []

    #r1 = rand.randint(0, 1051)
    for r1 in range(900):
        election_1_id = 'core_' + str(r1)
        election_1 = obj.Election(experiment_id, election_1_id)

        import copy
        election_3 = copy.deepcopy(election_1)

        #r2 = rand.randint(0, 1051)
        #r2 = 64
        total = 0
        for r2 in range(30):
            election_2_id = 'core_' + str(r2)
            election_2 = obj.Election(experiment_id, election_2_id)

            for i in range(int(model.num_voters/2), model.num_voters):
                for j in range(model.num_candidates):
                    election_3.votes[i][j] = election_2.votes[i][j]

            distance_name = 'positionwise'
            metric_type = 'emd'
            distance_12 = [metr.get_basic_distance(election_1, election_2, distance_name, metric_type) for _ in range(num_elections)]
            distance_13 = [metr.get_basic_distance(election_1, election_3, distance_name, metric_type) for _ in
                           range(num_elections)]
            distance_23 = [metr.get_basic_distance(election_2, election_3, distance_name, metric_type) for _ in
                           range(num_elections)]

            #print(round(distance_12[0],2), round(distance_13[0],2), round(distance_23[0],2))
            try:
                total += (distance_13[0]+distance_23[0]) / distance_12[0]
            except:
                total += 0

        final_value = total / 30
        print(final_value)
        means.append(final_value)

    path = os.path.join(os.getcwd(), "experiments", experiment_id, "results", "scores", "avg_mix.txt")
    file_scores = open(path, 'w')
    for i in range(model.num_elections):
        file_scores.write(str(means[i]) + "\n")
    file_scores.close()
"""


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
        el.generate_elections(experiment_id, election_model=elections_type, election_id=elections_id_a, num_elections=num_elections,
                              num_candidates=num_candidates, num_voters=num_voters, special=x)

    # COMPUTE ALL DISTANCES
    import numpy as np

    for T1 in range(100):

        elections_id_a = "guess_" + str(T1)
        elections_id_b = "core_" + str(0)
        elections_ids = [elections_id_a, elections_id_b]
        distances_1 = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)[0]
        distances_1 = round(distances_1,2)


        elections_id_a = "guess_" + str(T1)
        elections_id_b = "core_" + str(1)
        elections_ids = [elections_id_a, elections_id_b]
        distances_2 = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)[0]
        distances_2 = round(distances_2, 2)

        total = round(distances_1+distances_2,2)

        print(distances_1, distances_2, total)


# NEW 10.12.2020
def compute_distances_from_guardians(experiment_id):

    guardians = ['identity', 'uniformity', 'antagonism', 'stratification']
    #guardians = ['antagonism']

    for guardian in guardians:

        model = obj.Model(experiment_id)

        election_model = guardian
        election_2_id = 'guess'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_voters=model.num_voters, num_candidates=model.num_candidates, special=0)
        #election_2 = obj.Election(experiment_id, election_2_id)

        scores = []
        for i in range(model.num_elections):
            election_1_id = 'core_' + str(i)
            election_1 = obj.Election(experiment_id, election_1_id)
            election_2 = obj.Election(experiment_id, election_2_id)

            score = metr.get_distance(election_1, election_2)


            scores.append(score)
            print(election_1_id, round(score, 4))

        path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances", guardian + '.txt')
        file_scores = open(path, 'w')
        for i in range(model.num_elections):
            file_scores.write(str(scores[i]) + "\n")
        file_scores.close()
