#!/usr/bin/env python

import os
import time
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np

from . import _elections as el
from . import metrics as metr
from . import winners as win
from . import features
import copy

import itertools
from .objects.Experiment import Experiment, Experiment_xD, Experiment_2D, Experiment_3D
from .objects.Election import Election
from .metrics.inner_distances import l2




def create_structure(experiment_id):

    # PREPARE STRUCTURE

    if not os.path.isdir("experiments/"):
        os.mkdir(os.path.join(os.getcwd(), "experiments"))

    if not os.path.isdir("images/"):
        os.mkdir(os.path.join(os.getcwd(), "images"))

    if not os.path.isdir("trash/"):
        os.mkdir(os.path.join(os.getcwd(), "trash"))

    try:
        os.mkdir(os.path.join(os.getcwd(), "experiments", experiment_id))

        os.mkdir(os.path.join(os.getcwd(), "experiments", experiment_id, "distances"))
        os.mkdir(os.path.join(os.getcwd(), "experiments", experiment_id, "features"))
        os.mkdir(os.path.join(os.getcwd(), "experiments", experiment_id, "coordinates"))
        os.mkdir(os.path.join(os.getcwd(), "experiments", experiment_id, "elections"))
        os.mkdir(os.path.join(os.getcwd(), "experiments", experiment_id, "matrices"))

        # PREPARE MAP.CSV FILE

        path = os.path.join(os.getcwd(), "experiments", experiment_id, "map.csv")
        with open(path, 'w') as file_csv:
            file_csv.write("family_size,num_candidates,num_voters,election_model,param_1,param_2,color,alpha,label,marker,show\n")
            file_csv.write("3,10,100,impartial_culture,0,0,black,1,Impartial Culture,o,t\n")
            file_csv.write("3,10,100,iac,0,0,black,0.7,IAC,o,t\n")
            file_csv.write("3,10,100,conitzer,0,0,limegreen,1,SP by Conitzer,o,t\n")
            file_csv.write("3,10,100,walsh,0,0,olivedrab,1,SP by Walsh,o,t\n")
            file_csv.write("3,10,100,spoc_conitzer,0,0,DarkRed,0.7,SPOC,o,t\n")
            file_csv.write("3,10,100,group-separable,0,0,blue,1,Group-Separable,o,t\n")
            file_csv.write("3,10,100,single-crossing,0,0,purple,0.6,Single-Crossing,o,t\n")
            file_csv.write("3,10,100,1d_interval,0,0,DarkGreen,1,1D Interval,o,t\n")
            file_csv.write("3,10,100,2d_disc,0,0,Green,1,2D Disc,o,t\n")
            file_csv.write("3,10,100,3d_cube,0,0,ForestGreen,0.7,3D Cube ,o,t\n")
            file_csv.write("3,10,100,2d_sphere,0,0,black,0.2,2D Sphere,o,t\n")
            file_csv.write("3,10,100,3d_sphere,0,0,black,0.4,3D Sphere,o,t\n")
            file_csv.write("3,10,100,urn_model,0.1,0,yellow,1,Urn model 0.1,o,t\n")
            file_csv.write("3,10,100,norm-mallows,0.5,0,blue,1,Norm-Mallows 0.5,o,t\n")
            file_csv.write("3,10,100,urn_model,0,0,orange,1,Urn model (gamma),o,t\n")
            file_csv.write("3,10,100,norm-mallows,0,0,cyan,1,Norm-Mallows (uniform),o,t\n")
            file_csv.write("1,10,100,identity,0,0,blue,1,Identity,x,t\n")
            file_csv.write("1,10,100,uniformity,0,0,black,1,Uniformity,x,t\n")
            file_csv.write("1,10,100,antagonism,0,0,red,1,Antagonism,x,t\n")
            file_csv.write("1,10,100,stratification,0,0,green,1,Stratification,x,t\n")
            file_csv.write("1,10,100,walsh_matrix,0,0,olivedrab,1,Walsh Matrix,x,t\n")
            file_csv.write("1,10,100,conitzer_matrix,0,0,limegreen,1,Conitzer Matrix,x,t\n")
            file_csv.write("1,10,100,single-crossing_matrix,0,0,purple,0.6,Single-Crossing Matrix,x,t\n")
            file_csv.write("1,10,100,gs_caterpillar_matrix,0,0,green,1,GS Caterpillar Matrix,x,t")
            file_csv.write("3,10,100,unid,4,0,blue,1,UNID,3,f\n")
            file_csv.write("3,10,100,anid,4,0,black,1,ANID,3,f\n")
            file_csv.write("3,10,100,stid,4,0,black,1,STID,3,f\n")
            file_csv.write("3,10,100,stan,4,0,black,1,STAN,3,f\n")
            file_csv.write("3,10,100,stun,4,0,black,1,stun,3,f\n")
            file_csv.write("3,10,100,stan,4,0,red,1,stan,3,f\n")
    except:
        print("Experiment already exists!")

###############################################################################

## PART 0 ##

def compute_lowest_dodgson(experiment_id, clear=True):
    """ Compute lowest Dodgson score for all elections in a given experiment """

    experiment = Experiment(experiment_id)

    file_name_1 = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                               "lowest_dodgson.txt")

    file_name_2 = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
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

def compute_winners(experiment_id, method='hb', algorithm='exact', num_winners=10):
    """ Compute winners for all elections in a given experiment """

    experiment = Experiment(experiment_id)

    file_name = "experiments/" + experiment_id + "/controllers/winners/" + method + "_" + algorithm + ".txt"
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
                    rule = {}
                    rule['name'] = method
                    rule['length'] = num_winners
                    rule['type'] = 'borda_owa'
                    winners = win.get_winners(params, copy.deepcopy(experiment.elections[Z].votes), rule)

                elif algorithm == 'greedy':
                    if method == "pav":
                        winners = win.get_winners_approx_pav(copy.deepcopy(experiment.elections[Z].votes), params, algorithm)
                    elif method == "hb":
                        winners = win.get_winners_approx_hb(copy.deepcopy(experiment.elections[Z].votes), params, algorithm)

            elif method == 'borda':
                winners = compute_borda_winners(election, num_winners=num_winners)

            elif method == 'plurality':
                winners = compute_plurality_winners(election, num_winners=num_winners)


            for winner in winners:
                file_output.write(str(winner) + "\n")

            Z = Z+1

    file_output.close()

    print("\nDone.")


def import_winners(experiment_id, method='hb', algorithm='greedy', num_winners=10, num_elections=0):
    """ Import winners for all elections in a given experiment """

    winners = [[] for _ in range(num_elections)]

    path = "experiments/" + experiment_id + "/controllers/winners/" + method + "_" + algorithm + ".txt"
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

    file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                             "effective_num_candidates.txt")

    if clear:
        file_scores = open(file_name, 'w')
        file_scores.close()

    experiment = Experiment_xD(experiment_id, distance_name='swap')

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
 

def thread_function(experiment_id, distance_name, all_pairs, model, election_models, specials, thread_ids, results, t, precision, metric_name):

    for i, j in thread_ids:

        result = 0
        local_ctr = 0
        for p in range(precision):
            if t == 1:
                print("ctr: ", local_ctr)
                local_ctr += 1
            election_id_1 = "tmp_" + str(i) + '_' + str(j) + '_1'
            election_id_2 = "tmp_" + str(i) + '_' + str(j) + '_2'

            el.generate_elections(experiment_id, election_model=election_models[i], election_id=election_id_1,
                                  num_candidates=model.num_candidates, num_voters=model.num_voters, special=specials[i])
            el.generate_elections(experiment_id, election_model=election_models[j], election_id=election_id_2,
                                  num_candidates=model.num_candidates, num_voters=model.num_voters, special=specials[j])

            election_1 = Election(experiment_id, election_id_1)
            election_2 = Election(experiment_id, election_id_2)

            distance = metr.get_distance(election_1, election_2, distance_name=distance_name, metric_name=metric_name)

            # delete tmp files

            file_name_1 = str(election_id_1) + ".soc"
            path_1 = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", "soc_original", file_name_1)
            os.remove(path_1)

            file_name_2 = str(election_id_2) + ".soc"
            path_2 = os.path.join(os.getcwd(), "experiments", experiment_id, "elections", "soc_original", file_name_2)
            os.remove(path_2)

            all_pairs[i][j][p] = round(distance, 5)
            result += distance

        results[i][j] = result / precision
    print("thread " + str(t) + " is ready :)")


def compute_subelection_weird(experiment_id, distance_name='voter_subelection', metric_name=0, num_threads=1, precision=10):

    time_start = time.time()
    print("hello")

    experiment = Experiment(experiment_id, raw=True)

    num_distances = experiment.num_families * (experiment.num_families+1) / 2

    results = np.zeros([experiment.num_elections, experiment.num_elections])
    threads = [None for _ in range(num_threads)]

    election_models = []
    specials = []

    for i in range(experiment.num_families):
        for j in range(experiment.families[i].size):
            election_models.append(experiment.families[i].election_model)
            specials.append(experiment.families[i].param_1)

    ids = [[(i, j) for j in range(i, experiment.num_families)] for i in range(experiment.num_families)]
    ids = list(itertools.chain(*ids))

    all_pairs = np.zeros([experiment.num_elections, experiment.num_elections, precision])

    for t in range(num_threads):


        start = int(t * num_distances / num_threads)
        stop = int((t+1) * num_distances / num_threads)
        thread_ids = ids[start:stop]
        print('t: ', t)
        #print(thread_ids)


        threads[t] = Thread(target=thread_function, args=(experiment_id, distance_name, all_pairs,
                                                          experiment, election_models, specials,
                                                          thread_ids, results, t, precision, metric_name))
        threads[t].start()

    #"""
    for t in range(num_threads):
        threads[t].join()
    #"""
    #print(results)

    # compute STD
    std = np.zeros([experiment.num_elections, experiment.num_elections])
    for i in range(experiment.num_elections):
        for j in range(experiment.num_elections):
                std[i][j] = round(np.std(all_pairs[i][j]), 5)

    ctr = 0
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances",
                        str(metric_name) + '-' + str(distance_name) + ".txt")
    with open(path, 'w') as txtfile:
        txtfile.write(str(experiment.num_elections) + '\n')
        txtfile.write(str(experiment.num_families) + '\n')
        txtfile.write(str(int(num_distances)) + '\n')
        for i in range(experiment.num_elections):
            for j in range(i, experiment.num_elections):
                txtfile.write(str(i) + ' ' + str(j) + ' ' + str(results[i][j]) + ' ' + str(std[i][j]) + '\n')
                ctr += 1

    ctr = 0
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "distances",
                        str(metric_name) + '-' + str(distance_name) + "_all_pairs.txt")
    with open(path, 'w') as txtfile:
        for i in range(experiment.num_elections):
            for j in range(i, experiment.num_elections):
                for p in range(precision):
                    txtfile.write(str(i) + ' ' + str(j) + ' ' + str(p) + ' ' + str(all_pairs[i][j][p]) + '\n')
                    ctr += 1



    time_stop = time.time()
    print(time_stop - time_start)


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
def compute_distortion(experiment_id, attraction_factor=1):
    experiment = Experiment_2D(experiment_id, attraction_factor=attraction_factor)
    X = []
    A = []
    B = []

    for i, election_id_1 in enumerate(experiment.elections):
        for j, election_id_2 in enumerate(experiment.elections):
            if i < j:
                true_distance = experiment.distances[election_id_1][election_id_2]
                embedded_distance = l2(experiment.points[election_id_1], experiment.points[election_id_2], 1)
                proportion = float(true_distance) / float(embedded_distance)
                X.append(proportion)
                A.append(true_distance)
                B.append(embedded_distance)

    plt.scatter(A,B)
    plt.show()

    for i in [1, 0.99, 0.95, 0.9]:
        X = sorted(X)
        X = X[0:int(i*len(X))]
        X = [X[i]/X[int(len(X)/2)] for i in range(len(X))]  # normalize by median
        #X = [math.log(X[i]) for i in range(len(X))] # use log scale
        plt.plot(X)
        plt.title(i)
        plt.ylabel('(true_distance / embedded_distance) / median')
        plt.xlabel('meaningless')
        name = 'images/' + str(i)+'_' + str(attraction_factor) + '.png'
        plt.savefig(name)
        plt.show()

