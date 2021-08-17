#!/usr/bin/env python

import os
import time
import csv
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np

import mapel.voting._elections as el

import mapel.voting.winners as win
import mapel.voting.features as features
import copy

import itertools
from mapel.voting.objects.Experiment import Experiment
from mapel.voting.objects.Election import Election
from mapel.voting.metrics.inner_distances import l2

import mapel.voting._metrics as metr



def prepare_experiment(experiment_id=None, elections=None, distances=None,
                       coordinates=None, distance_name='emd-positionwise'):
    return Experiment("virtual",
                      experiment_id=experiment_id,
                      elections=elections, distances=distances,
                      coordinates=coordinates,
                      distance_name=distance_name)


def generate_experiment(elections=None):

    experiment = Experiment("virtual", elections=elections,
                            )

    return experiment


###############################################################################

## PART 0 ##

def compute_lowest_dodgson(experiment_id, clear=True):
    """ Compute lowest Dodgson score for all elections in a given experiment """

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

    for election in experiment.elections:

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

def compute_winners(experiment_id, method='hb', algorithm='exact',
                    num_winners=10):
    """ Compute winners for all elections in a given experiment """

    experiment = Experiment(experiment_id)

    file_name = "experiments/" + experiment_id + "/controllers/winners/" + \
                method + "_" + algorithm + ".txt"
    file_output = open(file_name, 'w')

    Z = 0
    for fam in range(experiment.num_families):

        for _ in range(experiment.families[fam].size):

            print(Z)

            params = {}
            params['orders'] = num_winners
            params['voters'] = experiment.families[fam].num_voters
            params['candidates'] = experiment.families[fam].num_candidates
            params['elections'] = 1

            election = experiment.elections[Z]

            winners = []
            if method in {'pav', 'hb'}:
                if algorithm == 'exact':
                    rule = {'name': method, 'length': num_winners,
                            'type': 'borda_owa'}
                    winners = win.get_winners(params, copy.deepcopy(experiment.elections[Z].votes), rule)

                elif algorithm == 'greedy':
                    if method == "pav":
                        winners = win.get_winners_approx_pav(copy.deepcopy(experiment.elections[Z].votes), params, algorithm)
                    elif method == "hb":
                        winners = win.get_winners_approx_hb(copy.deepcopy(experiment.elections[Z].votes), params, algorithm)

            elif method == 'borda':
                winners = compute_borda_winners(election,
                                                num_winners=num_winners)

            elif method == 'plurality':
                winners = compute_plurality_winners(election,
                                                    num_winners=num_winners)


            for winner in winners:
                file_output.write(str(winner) + "\n")

            Z = Z+1

    file_output.close()

    print("\nDone.")


def import_winners(experiment_id, method='hb', algorithm='greedy',
                   num_winners=10, num_elections=0):
    """ Import winners for all elections in a given experiment """

    winners = [[] for _ in range(num_elections)]

    path = "experiments/" + experiment_id + "/controllers/winners/" + method \
           + "_" + algorithm + ".txt"
    with open(path) as file_txt:
        for i in range(num_elections):
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

    for election in experiment.elections:

        score = features.get_effective_num_candidates(election)

        file_scores = open(file_name, 'a')
        file_scores.write(str(score) + "\n")
        file_scores.close()


def compute_condorcet_existence(experiment_id):
    experiment = Experiment(experiment_id)
    path = "experiments/" + experiment_id + "/controllers/winners/condorcet_existence.txt"
    with open(path, 'w') as file_txt:
        for election in experiment.elections:
            exists = is_condorect_winner(election)
            file_txt.write(str(exists) + "\n")


## PART 2 ##


def compute_statistics_tmp(experiment_id, method='hb', algorithm='greedy', num_winners=10):

    experiment = Experiment(experiment_id)


    file_name = "experiments/" + experiment_id + "/controllers/approx/" + method + "_" + algorithm + ".txt"
    file_output = open(file_name, 'w')
    num_lines = experiment.num_elections
    file_output.write(str(num_lines) + "\n")

    winners_1 = import_winners(experiment_id, method='hb', algorithm=algorithm, num_winners=num_winners, num_elections=experiment.num_elections)
    winners_2 = import_winners(experiment_id, method='hb', algorithm='exact', num_winners=num_winners, num_elections=experiment.num_elections)

    Z = 0
    for fam in range(experiment.num_families):

        for _ in range(experiment.families[fam].size):

            print(Z)

            params = {}
            params['orders'] = num_winners
            params['voters'] = experiment.num_voters
            params['candidates'] = experiment.families[fam].num_candidates
            params['elections'] = experiment.num_elections

            if method == "pav":
                score_1 = win.check_pav_dissat(copy.deepcopy(experiment.elections[Z].votes), params, winners_1[Z])
            elif method == "hb":
                score_1 = win.check_hb_dissat(copy.deepcopy(experiment.elections[Z].votes), params, winners_1[Z])

            if method == "pav":
                score_2 = win.check_pav_dissat(copy.deepcopy(experiment.elections[Z].votes), params, winners_2[Z])
            elif method == "hb":
                score_2 = win.check_hb_dissat(copy.deepcopy(experiment.elections[Z].votes), params, winners_2[Z])

            output = score_1 / score_2

            file_output.write(str(output) + "\n")

            Z = Z+1

    file_output.close()

    print("\nDone.")


# def compute_overlapping_of_winners(experiment_id, num_winners=10):
#
#     model = obj.Model(experiment_id, raw=True)
#
#     file_name = "experiments/" + experiment_id + "/controllers/approx/" + 'overlap' + ".txt"
#     file_output = open(file_name, 'w')
#     num_lines = model.num_elections
#     file_output.write(str(num_lines) + "\n")
#
#     winners_1 = import_winners(experiment_id, method='plurality', algorithm='exact',
#                                num_winners=num_winners, num_elections=model.num_elections)
#     winners_2 = app.import_winners(experiment_id, method='borda', algorithm='exact',
#                                num_winners=num_winners, num_elections=model.num_elections)
#
#     Z = 0
#     for fam in range(model.num_families):
#
#         for _ in range(model.families[fam].size):
#
#             print(Z)
#
#             winners = (set(winners_1[Z])).intersection(set(winners_2[Z]))
#
#             output = len(winners)
#
#             file_output.write(str(output) + "\n")
#
#             Z = Z+1
#
#     file_output.close()
#
#     print("\nDone.")
#
#
# def print_statistics(experiment_id, method='hb', algorithm='greedy', num_winners=10):
#
#     model = Experiment(experiment_id)
#
#     mallows = []
#     norm_mallows = []
#     path = "experiments/" + experiment_id + "/controllers/approx/" + method + "_" + algorithm + ".txt"
#     with open(path) as file_txt:
#         spam_line = file_txt.readline()
#         for i in range(model.num_families):
#             local_sum = 0.
#             for j in range(model.families[i].size):
#                 value = float(file_txt.readline().strip())
#                 local_sum += value
#             avg = local_sum / float(model.families[i].size)
#             if i%2==0:
#                 mallows.append(avg)
#             else:
#                 norm_mallows.append(avg)
#
#     X = [20,30,40,50,60,70,80,90,100]
#     plt.plot(X, mallows, label='Mallows')
#     plt.plot(X, norm_mallows, label='Norm-Mallows')
#     plt.legend()
#     plt.ylabel('dissatisfaction ratio')
#     plt.xlabel('number of candidates')
#     plt.savefig('mallows_vs_norm-mallows.png')
#     plt.show()


def print_chart_overlapping_of_winners(experiment_id, method='hb', algorithm='greedy', num_winners=10):

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

            if i%3 == 0:
                mallows.append(avg)
            elif i%3 == 1:
                norm_mallows.append(avg)
            else:
                impartial_culture.append(avg)

    # X = [20,30,40,50,60,70,80,90,100]
    # plt.plot(X, mallows, label='Mallows')
    # plt.plot(X, norm_mallows, label='Norm-Mallows')

    # X = [x for x in range(5,105,5)]
    X = [10,20,30,40]
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

    X = [x for x in range(10, 150+10, 10)]
    # X = [x for x in range(5, 105, 5)]
    # X = [10,20,30,40]
    plt.plot(X, mallows, label='Mallows '+str(experiment.families[0].param_1))
    #plt.plot(X, norm_mallows, label='Norm-Mallows')
    #plt.plot(X, impartial_culture, label='IC')

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


## PART 3 ## SUBELECTIONS
 






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
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "distances", "backup.txt")
    with open(path, 'a') as txt_file:
        txt_file.write(str(i) + ' ' + str(j) + ' ' + str(result) + '\n')


def get_borda_ranking(votes, num_voters, num_candidates):

    # todo: compute ranking based on positionwise vectors

    points = [0 for _ in range(num_candidates)]
    scoring = [1. for _ in range(num_candidates)]

    for i in range(len(scoring)):
        scoring[i] = (len(scoring) - float(i) - 1.) / (len(scoring) - 1.)

    for i in range(num_voters):
        for j in range(num_candidates):
            points[int(votes[i][j])] += scoring[j]

    candidates = [x for x in range(num_candidates)]
    ordered_candidates = [x for _, x in sorted(zip(points, candidates), reverse=True)]

    return ordered_candidates


### DISTORTION SECTION ###
def compute_distortion(experiment, attraction_factor=1, saveas='tmp'):
    # experiment = Experiment_2d(experiment_id, attraction_factor=attraction_factor)
    X = []
    A = []
    B = []

    for i, election_id_1 in enumerate(experiment.elections):
        for j, election_id_2 in enumerate(experiment.elections):
            if i < j:
                m = experiment.elections[election_id_1].num_candidates
                true_distance = experiment.distances[election_id_1][election_id_2]
                true_distance /= map_diameter(m)

                embedded_distance = l2(experiment.coordinates[election_id_1], experiment.coordinates[election_id_2], 1)
                embedded_distance /= l2(experiment.coordinates['identity_10_100_0'], experiment.coordinates['uniformity_10_100_0'], 1)
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
#     for i, election_id_1 in enumerate(experiment.elections):
#         for j, election_id_2 in enumerate(experiment.elections):
#             if i < j:
#                 m = experiment.elections[election_id_1].num_candidates
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


