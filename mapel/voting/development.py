#!/usr/bin/env python

import os
import time

import matplotlib.pyplot as plt

import mapel.voting.features as features
import copy

from mapel.voting.objects.Experiment import Experiment
from mapel.voting.metrics.inner_distances import l1, l2

import mapel.voting._metrics as metr
import csv
import numpy as np


def potLadle(v, m):
    vv = sorted(v, reverse=True)
    cumv = np.cumsum(vv)
    n = 0
    while n < len(v) and vv[n] / cumv[n] >= 1 / (2 * m + n):
        n += 1
    v = np.array(v)
    vv = v / cumv[n - 1]
    s = vv * (1 + n / (2 * m)) - 1 / (2 * m)
    s[s < 0] = 0
    return list(s)


def compute_spoilers(election_model=None, method=None, num_winners=None, num_parties=None,
                     num_voters=100, num_districts=100, precision=100):
    party_size = num_winners
    num_candidates = num_parties * num_winners

    all_weight_spoilers = []
    all_weight_vectors = []
    all_shapley_spoilers = []
    all_shapley_vectors = []
    all_banzhaf_spoilers = []
    all_banzhaf_vectors = []
    all_borda = []

    all_alternative_weight_vectors = []
    all_alternative_shapley_vectors = []
    all_alternative_banzhaf_vectors = []

    all_separated_weight_vectors = []

    for _ in range(precision):
        experiment = prepare_experiment()
        experiment.set_default_num_candidates(num_candidates)
        experiment.set_default_num_voters(num_voters)

        ### MALLOWS ###
        if election_model == 'mallows_party_075':
            experiment.add_family(election_model='mallows_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'main-phi': 0.75, 'norm-phi': 0.75})

        elif election_model == 'mallows_party_05':
            experiment.add_family(election_model='mallows_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'main-phi': 0.5, 'norm-phi': 0.75})

        elif election_model == 'mallows_party_025':
            experiment.add_family(election_model='mallows_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'main-phi': 0.25, 'norm-phi': 0.75})

        ### 2D ###
        elif election_model == '2d_gaussian_party_005':
            experiment.add_family(election_model='2d_gaussian_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.05})

        elif election_model == '2d_gaussian_party_01':
            experiment.add_family(election_model='2d_gaussian_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.1})

        elif election_model == '2d_gaussian_party_02':
            experiment.add_family(election_model='2d_gaussian_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.2})

        ### 1D ###
        elif election_model == '1d_gaussian_party_005':
            experiment.add_family(election_model='1d_gaussian_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.05})

        elif election_model == '1d_gaussian_party_01':
            experiment.add_family(election_model='1d_gaussian_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.1})

        elif election_model == '1d_gaussian_party_02':
            experiment.add_family(election_model='1d_gaussian_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.2})

        ### SP by Walsh
        elif election_model == 'walsh_party_005':
            experiment.add_family(election_model='walsh_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.05})

        elif election_model == 'walsh_party_01':
            experiment.add_family(election_model='walsh_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.1})

        elif election_model == 'walsh_party_02':
            experiment.add_family(election_model='walsh_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.2})

        ### SP by Conitzer
        elif election_model == 'conitzer_party_005':
            experiment.add_family(election_model='conitzer_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.05})

        elif election_model == 'conitzer_party_01':
            experiment.add_family(election_model='conitzer_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.1})

        elif election_model == 'conitzer_party_02':
            experiment.add_family(election_model='conitzer_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties,
                                          'var': 0.2})

        ### IC
        elif election_model == 'ic_party':
            experiment.add_family(election_model='ic_party', size=num_districts,
                                  params={'num_winners': num_winners, 'num_parties': num_parties})

        experiment.compute_winners(method=method, num_winners=num_winners)

        if method == 'dhondt':
            # main election
            parties = [0 for _ in range(num_parties)]
            for election_id in experiment.instances:
                for vote in experiment.instances[election_id].votes:
                    parties[int(vote[0] / party_size)] += 1
            denominator = sum(parties)
            for i in range(num_parties):
                parties[i] /= denominator
            weight_vector = potLadle(parties, num_winners)

            # alternative instances
            alternative_weight_vectors = [[] for _ in range(num_parties)]
            for c in range(num_parties):
                parties = [0 for _ in range(num_parties)]
                for election_id in experiment.instances:
                    for vote in experiment.instances[election_id].votes:
                        ctr = 0
                        while int(vote[ctr] / party_size) == c:
                            ctr += 1
                        parties[int(vote[ctr] / party_size)] += 1

                denominator = sum(parties)
                for i in range(num_parties):
                    parties[i] /= denominator
                alternative_weight_vectors[c] = potLadle(parties, num_winners)

        else:

            experiment.compute_alternative_winners(method=method, num_winners=num_winners,
                                                   num_parties=num_parties)

            weight_vector = [0 for _ in range(num_parties)]
            separated_weight_vectors = []
            alternative_weight_vectors = {}
            for party_id in range(num_parties):
                alternative_weight_vectors[party_id] = [0 for i in range(num_parties)]

            for election_id in experiment.instances:
                single_vector = [0 for _ in range(num_parties)]
                for w in experiment.instances[election_id].winners:
                    weight_vector[int(w / party_size)] += 1
                    single_vector[int(w / party_size)] += 1
                separated_weight_vectors.append(single_vector)
                for party_id in range(num_parties):
                    for w in experiment.instances[election_id].alternative_winners[party_id]:
                        alternative_weight_vectors[party_id][int(w / party_size)] += 1

            all_separated_weight_vectors.append(separated_weight_vectors)

        borda = [0 for _ in range(num_parties)]
        for election_id in experiment.instances:
            for i in range(num_candidates):
                borda[int(i / party_size)] += experiment.instances[election_id].borda_points[i]

        denominator = sum(borda)
        borda = [elem / denominator for elem in borda]

        denominator = sum(weight_vector)
        weight_vector = [elem / denominator for elem in weight_vector]

        for spoiler_id in range(num_parties):
            denominator = sum(alternative_weight_vectors[spoiler_id])
            for party_id in range(num_parties):
                alternative_weight_vectors[spoiler_id][party_id] /= denominator

        # WEIGHT
        weight_spoilers = []
        for party_id in range(num_parties):
            value = l1(weight_vector, alternative_weight_vectors[party_id])
            value -= 2 * weight_vector[party_id]
            weight_spoilers.append(value / (value + borda[party_id]))
        all_weight_spoilers.append(weight_spoilers)
        all_weight_vectors.append(weight_vector)

        # SHAPLEY
        # shapley_spoilers = []
        # shapley_vector = features.shapley(weight_vector)
        # alternative_shapley_vectors = []
        # for party_id in range(num_parties):
        #     alternative_shapley_vector = features.shapley(
        #         alternative_weight_vectors[party_id])
        #     alternative_shapley_vectors.append(alternative_shapley_vector)
        #     value = l1(shapley_vector, alternative_shapley_vector)
        #     value -= 2 * shapley_vector[party_id]
        #     shapley_spoilers.append(value / (value + borda[party_id]))
        # all_shapley_spoilers.append(shapley_spoilers)
        # all_shapley_vectors.append(shapley_vector)

        # BANZHAF
        banzhaf_spoilers = []
        banzhaf_vector = features.banzhaf(weight_vector)
        alternative_banzhaf_vectors = []
        for party_id in range(num_parties):
            alternative_banzhaf_vector = features.banzhaf(
                alternative_weight_vectors[party_id])
            alternative_banzhaf_vectors.append(
                alternative_banzhaf_vector)
            value = l1(banzhaf_vector, alternative_banzhaf_vector)
            value -= 2 * banzhaf_vector[party_id]
            banzhaf_spoilers.append(value / (value + borda[party_id]))

        all_banzhaf_spoilers.append(banzhaf_spoilers)
        all_banzhaf_vectors.append(banzhaf_vector)

        # BORDA
        all_borda.append(borda)

        # BIG FILE ALL
        all_alternative_weight_vectors.append(alternative_weight_vectors)
        # all_alternative_shapley_vectors.append(alternative_shapley_vectors)
        all_alternative_banzhaf_vectors.append(alternative_banzhaf_vectors)

    # SAVE TO FILE -- SMALL
    file_name = election_model + '_p' + str(num_parties) + \
                '_k' + str(num_winners) + \
                '_' + str(method) + '_small'
    path = os.path.join(os.getcwd(), 'final_party', file_name + '.csv')
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["iteration", "party_id", "weight",
                         "sp_weight", "shapley", "sp_shapley",
                         "banzhaf", "sp_banzhaf", "borda"])

        for j in range(len(all_weight_vectors)):
            for p in range(num_parties):
                writer.writerow([j, p,
                                 all_weight_vectors[j][p],
                                 all_weight_spoilers[j][p],
                                 "NULL",  # all_shapley_vectors[j][p],
                                 "NULL",  # all_shapley_spoilers[j][p],
                                 all_banzhaf_vectors[j][p],
                                 all_banzhaf_spoilers[j][p],
                                 all_borda[j][p]])

    # SAVE TO FILE -- MEDIUM
    if num_winners == 15 and method != 'dhondt':
        file_name = election_model + '_p' + str(num_parties) + \
                    '_k' + str(num_winners) + \
                    '_' + str(method) + '_medium'
        path = os.path.join(os.getcwd(), 'final_party', file_name + '.csv')
        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(["iteration", "district", "vector"])

            for i in range(precision):
                for j in range(num_districts):
                    writer.writerow([i,j,all_separated_weight_vectors[i][j]])

    # SAVE TO FILE -- BIG
    file_name = election_model + '_p' + str(num_parties) + '_k' + str(
        num_winners) + '_' + str(method) + '_big'
    path = os.path.join(os.getcwd(), 'final_party', file_name + '.csv')

    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["iteration", "spoiler_id", 'party_id',
                         "weight", "shapley", "banzhaf"])

        for j in range(len(all_weight_vectors)):

            for party_id in range(num_parties):
                writer.writerow([j, "NULL", party_id,
                                 all_weight_vectors[j][party_id],
                                 "NULL", # all_shapley_vectors[j][party_id],
                                 all_banzhaf_vectors[j][party_id]])

            for spoiler_id in range(num_parties):
                for party_id in range(num_parties):
                    writer.writerow([j, spoiler_id, party_id,
                                     all_alternative_weight_vectors[j][spoiler_id][party_id],
                                     "NULL",  # all_alternative_shapley_vectors[j][spoiler_id][party_id],
                                     all_alternative_banzhaf_vectors[j][spoiler_id][party_id]])


def prepare_experiment(experiment_id=None, instances=None, distances=None,
                       coordinates=None, distance_name='emd-positionwise'):
    return Experiment("virtual", experiment_id=experiment_id, instances=instances,
                      distances=distances, coordinates=coordinates, distance_name=distance_name)


def generate_experiment(instances=None):
    return Experiment("virtual", instances=instances)


###############################################################################

## PART 0 ##

def compute_lowest_dodgson(experiment_id, clear=True):
    """ Compute lowest Dodgson score for all instances in a given experiment """

    experiment = Experiment(experiment_id)

    file_name_1 = os.path.join(os.getcwd(), "experiments", experiment_id,
                               "controllers", "advanced",
                               "lowest_dodgson.txt")

    file_name_2 = os.path.join(os.getcwd(), "experiments", experiment_id,
                               "controllers", "advanced",
                               "lowest_dodgson_time.txt")

    if clear:
        file_scores = open(file_name_1, 'w')
        file_scores.close()
        file_scores = open(file_name_2, 'w')
        file_scores.close()

    for election in experiment.instances:

        start_time = time.time()

        condorcet_winner = metr.is_condorect_winner(election)
        if condorcet_winner:
            score = 0
        else:
            score = metr.get_lowest_dodgson_score(election)

        elapsed_time = time.time() - start_time
        elapsed_time = round(elapsed_time, 4)

        print(elapsed_time)

        file_scores = open(file_name_1, 'a')
        file_scores.write(str(score) + "\n")
        file_scores.close()

        file_scores = open(file_name_2, 'a')
        file_scores.write(str(elapsed_time) + "\n")
        file_scores.close()


## PART 1 ##


def import_winners(experiment_id, method='hb', algorithm='greedy',
                   num_winners=10, num_instances=0):
    """ Import winners for all instances in a given experiment """

    winners = [[] for _ in range(num_instances)]

    path = "experiments/" + experiment_id + "/controllers/winners/" + method \
           + "_" + algorithm + ".txt"
    with open(path) as file_txt:
        for i in range(num_instances):
            for j in range(num_winners):
                value = int(float(file_txt.readline().strip()))
                winners[i].append(value)

    return winners


def compute_borda_winners(election, num_winners=10):
    """ Compute Borda winners for a given election """

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        for i in range(election.num_candidates):
            scores[vote[i]] += election.num_candidates - i - 1
    candidates = [i for i in range(election.num_candidates)]
    ranking = [x for _, x in sorted(zip(scores, candidates), reverse=True)]
    return ranking[0:num_winners]


def compute_plurality_winners(election, num_winners=10):
    """ Compute Plurality winners for a given election """

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        scores[vote[0]] += 1
    candidates = [i for i in range(election.num_candidates)]
    ranking = [x for _, x in sorted(zip(scores, candidates), reverse=True)]
    return ranking[0:num_winners]


def compute_effective_num_candidates(experiment_id, clear=True):
    """ Compute effective number of candidates """

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id,
                             "controllers", "advanced",
                             "effective_num_candidates.txt")

    if clear:
        file_scores = open(file_name, 'w')
        file_scores.close()

    experiment = Experiment(experiment_id=experiment_id, distance_name='swap')

    for election in experiment.instances:
        score = features.get_effective_num_candidates(election)

        file_scores = open(file_name, 'a')
        file_scores.write(str(score) + "\n")
        file_scores.close()


def compute_condorcet_existence(experiment_id):
    experiment = Experiment(experiment_id)
    path = "experiments/" + experiment_id + "/controllers/winners/condorcet_existence.txt"
    with open(path, 'w') as file_txt:
        for election in experiment.instances:
            exists = is_condorect_winner(election)
            file_txt.write(str(exists) + "\n")


def print_chart_overlapping_of_winners(experiment_id, method='hb',
                                       algorithm='greedy', num_winners=10):
    experiment = Experiment(experiment_id, raw=True)

    mallows = []
    norm_mallows = []
    impartial_culture = []
    path = "experiments/" + experiment_id + "/controllers/approx/" + 'overlap' + ".txt"
    with open(path) as file_txt:
        spam_line = file_txt.readline()
        for i in range(experiment.num_families):
            local_sum = 0.
            for j in range(experiment.families[i].size):
                value = float(file_txt.readline().strip())
                local_sum += value
            avg = local_sum / float(experiment.families[i].size)

            # print(avg)

            if i % 3 == 0:
                mallows.append(avg)
            elif i % 3 == 1:
                norm_mallows.append(avg)
            else:
                impartial_culture.append(avg)

    # X = [20,30,40,50,60,70,80,90,100]
    # plt.plot(X, mallows, label='Mallows')
    # plt.plot(X, norm_mallows, label='Norm-Mallows')

    # X = [x for x in range(5,105,5)]
    X = [10, 20, 30, 40]
    plt.plot(X, mallows, label='Mallows')
    plt.plot(X, norm_mallows, label='Norm-Mallows')
    plt.plot(X, impartial_culture, label='IC')

    plt.xticks(size=14)
    plt.yticks(size=14)

    plt.legend(prop={'size': 18})
    plt.ylabel('overlap of winners', size=18)
    plt.xlabel('number of candidates', size=18)
    path = 'images/' + experiment_id + '_overlap.png'
    plt.savefig(path, bbox_inches='tight')
    plt.show()


def print_chart_condorcet_existence(experiment_id):
    experiment = Experiment(experiment_id, raw=True)

    mallows = []
    norm_mallows = []
    impartial_culture = []
    path = "experiments/" + experiment_id + "/controllers/winners/condorcet_existence.txt"

    with open(path, 'r') as file_txt:
        for i in range(experiment.num_families):
            local_sum = 0.
            for j in range(experiment.families[i].size):
                if file_txt.readline().strip() == "True":
                    local_sum += 1
            avg = local_sum / float(experiment.families[i].size)

            print(avg)

            if i % 3 == 0:
                mallows.append(avg)
            elif i % 3 == 1:
                norm_mallows.append(avg)
            else:
                impartial_culture.append(avg)

    # X = [20,30,40,50,60,70,80,90,100]
    # plt.plot(X, mallows, label='Mallows')
    # plt.plot(X, norm_mallows, label='Norm-Mallows')

    X = [x for x in range(10, 150 + 10, 10)]
    # X = [x for x in range(5, 105, 5)]
    # X = [10,20,30,40]
    plt.plot(X, mallows,
             label='Mallows ' + str(experiment.families[0].param_1))
    # plt.plot(X, norm_mallows, label='Norm-Mallows')
    # plt.plot(X, impartial_culture, label='IC')

    plt.xticks(size=14)
    plt.yticks(size=14)

    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.legend(prop={'size': 18})
    plt.ylabel('condorcet winner existence', size=18)
    plt.xlabel('number of candidates', size=18)
    plt.title(str(experiment.families[0].num_voters) + ' voters', size=14)
    path = 'images/' + experiment_id + '_condorcet.png'
    plt.savefig(path, bbox_inches='tight')
    plt.show()


## PART 3 ## SUBinstances


### PART X ###


import math


def is_condorect_winner(election):
    """ Check if election witness Condorect winner"""
    for i in range(election.num_candidates):

        condocret_winner = True
        for j in range(election.num_candidates):

            diff = 0
            for k in range(election.num_voters):

                if election.potes[k][i] <= election.potes[k][j]:
                    diff += 1

            if diff < math.ceil((election.num_voters + 1) / 2.):
                condocret_winner = False
                break

        if condocret_winner:
            return True

    return False


def map_diameter(c):
    """ Compute the diameter """
    return 1 / 3 * (c + 1) * (c - 1)


def result_backup(experiment_id, result, i, j):
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "distances",
                        "backup.txt")
    with open(path, 'a') as txt_file:
        txt_file.write(str(i) + ' ' + str(j) + ' ' + str(result) + '\n')


def get_borda_ranking(votes, num_voters, num_candidates):
    points = [0 for _ in range(num_candidates)]
    scoring = [1. for _ in range(num_candidates)]

    for i in range(len(scoring)):
        scoring[i] = (len(scoring) - float(i) - 1.) / (len(scoring) - 1.)

    for i in range(num_voters):
        for j in range(num_candidates):
            points[int(votes[i][j])] += scoring[j]

    candidates = [x for x in range(num_candidates)]
    ordered_candidates = [x for _, x in
                          sorted(zip(points, candidates), reverse=True)]

    return ordered_candidates


### DISTORTION SECTION ###
def compute_distortion(experiment, attraction_factor=1, saveas='tmp'):
    # experiment = Experiment_2d(experiment_id, attraction_factor=attraction_factor)
    X = []
    A = []
    B = []

    for i, election_id_1 in enumerate(experiment.instances):
        for j, election_id_2 in enumerate(experiment.instances):
            if i < j:
                m = experiment.instances[election_id_1].num_candidates
                true_distance = experiment.distances[election_id_1][
                    election_id_2]
                true_distance /= map_diameter(m)

                embedded_distance = l2(experiment.coordinates[election_id_1],
                                       experiment.coordinates[election_id_2],
                                       1)
                embedded_distance /= l2(
                    experiment.coordinates['identity_10_100_0'],
                    experiment.coordinates['uniformity_10_100_0'], 1)
                proportion = float(true_distance) / float(embedded_distance)
                X.append(proportion)
                A.append(true_distance)
                B.append(embedded_distance)

    plt.scatter(A, B)
    plt.xlabel('true_distance/diameter')
    plt.ylabel('embedded_distance/diameter')
    name = 'images/' + saveas + '.png'
    plt.savefig(name)
    plt.show()

    # for i in [1, 0.99, 0.95, 0.9]:
    #     X = sorted(X)
    #     X = X[0:int(i*len(X))]
    #     X = [X[i] for i in range(len(X))]  # normalize by median
    #     #X = [math.log(X[i]) for i in range(len(X))] # use log scale
    #     plt.plot(X)
    #     plt.title(i)
    #     plt.ylabel('true_distance/diameter / embedded_distance/diameter')
    #     plt.xlabel('meaningless')
    #     name = 'images/' + str(i)+'_' + str(attraction_factor) + '.png'
    #     plt.savefig(name)
    #     plt.show()

# def compute_distortion_paths(experiment_id, attraction_factor=1, saveas='tmp'):
#     experiment = Experiment_2d(experiment_id, attraction_factor=attraction_factor)
#     X = []
#     A = []
#     B = []
#
#     id_x = []
#     id_y = []
#     un_x = []
#     un_y = []
#     an_x = []
#     an_y = []
#     st_x = []
#     st_y = []
#
#     for i, election_id_1 in enumerate(experiment.instances):
#         for j, election_id_2 in enumerate(experiment.instances):
#             if i < j:
#                 m = experiment.instances[election_id_1].num_candidates
#                 true_distance = experiment.distances[election_id_1][election_id_2]
#                 true_distance /= map_diameter(m)
#
#                 embedded_distance = l2(experiment.points[election_id_1], experiment.points[election_id_2], 1)
#                 embedded_distance /= l2(experiment.points['identity_10_100_0'], experiment.points['uniformity_10_100_0'], 1)
#                 # print(true_distance, embedded_distance)
#                 proportion = float(true_distance) / float(embedded_distance)
#                 X.append(proportion)
#                 A.append(true_distance)
#                 B.append(embedded_distance)
#
#                 if election_id_1 == 'identity_10_100_0' or election_id_2 == 'identity_10_100_0':
#                     id_x.append(true_distance)
#                     id_y.append(embedded_distance)
#
#                 if election_id_1 == 'uniformity_10_100_0' or election_id_2 == 'uniformity_10_100_0':
#                     un_x.append(true_distance)
#                     un_y.append(embedded_distance)
#
#                 if election_id_1 == 'stratification_10_100_0' or election_id_2 == 'stratification_10_100_0':
#                     st_x.append(true_distance)
#                     st_y.append(embedded_distance)
#
#                 if election_id_1 == 'antagonism_10_100_0' or election_id_2 == 'antagonism_10_100_0':
#                     an_x.append(true_distance)
#                     an_y.append(embedded_distance)
#
#
#     plt.scatter(id_x, id_y, color='blue', label='ID')
#     plt.scatter(un_x, un_y, color='black', label='UN')
#     plt.scatter(st_x, st_y, color='green', label='ST')
#     plt.scatter(an_x, an_y, color='red', label='AN')
#     plt.legend()
#     plt.xlabel('true_distance/diameter')
#     plt.ylabel('embedded_distance/diameter')
#     name = 'images/' + saveas + '.png'
#     plt.savefig(name)
#     plt.show()
#
#     # for i in [1, 0.99, 0.95, 0.9]:
#     #     X = sorted(X)
#     #     X = X[0:int(i*len(X))]
#     #     X = [X[i] for i in range(len(X))]  # normalize by median
#     #     #X = [math.log(X[i]) for i in range(len(X))] # use log scale
#     #     plt.plot(X)
#     #     plt.title(i)
#     #     plt.ylabel('true_distance/diameter / embedded_distance/diameter')
#     #     plt.xlabel('meaningless')
#     #     name = 'images/' + str(i)+'_' + str(attraction_factor) + '.png'
#     #     plt.savefig(name)
#     #     plt.show()
