#!/usr/bin/env python
# UNDER CONSTRUCTION #

import os
from collections import Counter
import random as rand

import numpy as np


# GENERATE
def generate_preflib_election(experiment=None, model=None, name=None,
                              num_voters=None, num_candidates=None, folder=None,
                              selection_method='random'):
    """ main function: generate elections"""

    votes = generate_votes_preflib(model, selection_method=selection_method,
                                   num_voters=num_voters, num_candidates=num_candidates,
                                   folder=folder)

    path = os.path.join("experiments", experiment.experiment_id, "elections", name + ".soc")
    file_ = open(path, 'w')

    file_.write(str(num_candidates) + "\n")

    for i in range(num_candidates):
        file_.write(str(i) + ', c' + str(i) + "\n")

    c = Counter(map(tuple, votes))
    counted_votes = [[count, list(row)] for row, count in c.items()]
    counted_votes = sorted(counted_votes, reverse=True)

    file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' + str(len(counted_votes)) + "\n")

    for i in range(len(counted_votes)):
        file_.write(str(counted_votes[i][0]) + ', ')
        for j in range(num_candidates):
            file_.write(str(counted_votes[i][1][j]))
            if j < num_candidates - 1:
                file_.write(", ")
            else:
                file_.write("\n")

    file_.close()


# REAL
def generate_votes_preflib(model, num_candidates=None, num_voters=None,  folder=None,
                           selection_method='borda'):
    """ Generate votes based on elections from Preflib """

    long_name = str(model)
    file_name = '_real_data/' + folder + '/' + long_name + '.txt'
    file_votes = open(file_name, 'r')
    original_num_voters = int(file_votes.readline())
    if original_num_voters == 0:
        return [0, 0, 0]
    original_num_candidates = int(file_votes.readline())
    choice = [x for x in range(original_num_voters)]
    np.random.shuffle(choice)

    votes = np.zeros([num_voters, original_num_candidates], dtype=int)
    original_votes = np.zeros([original_num_voters, original_num_candidates], dtype=int)

    for j in range(original_num_voters):
        value = file_votes.readline().strip().split(',')
        for k in range(original_num_candidates):
            original_votes[j][k] = int(value[k])

    file_votes.close()
    print(model, len(original_votes), len(np.unique(original_votes, axis=0)))

    for j in range(num_voters):
        r = np.random.randint(0, original_num_voters - 1)
        for k in range(original_num_candidates):
            votes[j][k] = original_votes[r][k]

    for i in range(num_voters):
        if len(votes[i]) != len(set(votes[i])):
            print('wrong data')

    # REMOVE SURPLUS CANDIDATES
    if num_candidates < original_num_candidates:
        new_votes = []

        # NEW 17.12.2020
        if selection_method == 'random':
            selected_candidates = [j for j in range(original_num_candidates)]
            np.random.shuffle(selected_candidates)
            selected_candidates = selected_candidates[0:num_candidates]
        elif selection_method == 'borda':
            scores = get_borda_scores(original_votes, original_num_voters, original_num_candidates)
            order_by_score = [x for _, x in
                              sorted(zip(scores, [i for i in range(original_num_candidates)]),
                                     reverse=True)]
            selected_candidates = order_by_score[0:num_candidates]
        elif selection_method == 'freq':
            freq = import_freq(model)
            # print(freq)
            selected_candidates = freq[0:num_candidates]
        else:
            raise NameError('No such selection method!')

        mapping = {}
        for j in range(num_candidates):
            mapping[str(selected_candidates[j])] = j
        for j in range(num_voters):
            vote = []
            for k in range(original_num_candidates):
                cand = votes[j][k]
                if cand in selected_candidates:
                    vote.append(mapping[str(cand)])
            if len(vote) != len(set(vote)):
                print(vote)
            new_votes.append(vote)
        return np.array(new_votes)
    else:
        return np.array(votes)


def import_freq(elections_model):
    path = 'real_data/freq/' + elections_model + '.txt'
    with open(path, 'r', newline='') as txt_file:
        line = txt_file.readline().strip().split(',')
        line = line[0:len(line) - 1]
        for i in range(len(line)):
            line[i] = int(line[i])
    return line


def get_borda_scores(votes, num_voters, num_candidates):
    scores = [0 for _ in range(num_candidates)]
    for i in range(num_voters):
        for j in range(num_candidates):
            scores[votes[i][j]] += num_candidates - j - 1

    return scores


def prepare_preflib_family(experiment=None, model=None, family_id=None,
                           params=None) -> list:
    # NEEDS UPDATE #

    selection_method = 'random'
    keys = []

    # list of IDs larger than 10
    if model == 'irish':
        folder = 'irish_s1'
        # folder = 'irish_f'
        ids = [1, 3]
    elif model == 'glasgow':
        folder = 'glasgow_s1'
        # folder = 'glasgow_f'
        ids = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 21]
    elif model == 'formula':
        folder = 'formula_s1'
        # 17 races or more
        ids = [17, 35, 37, 40, 41, 42, 44, 45, 46, 47, 48]
    elif model == 'skate':
        folder = 'skate_ic'
        # 9 judges
        ids = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48]
    elif model == 'sushi':
        folder = 'sushi_ff'
        ids = [1]
    elif model == 'grenoble':
        folder = 'grenoble_ff'
        ids = [1]
    elif model == 'tshirt':
        folder = 'tshirt_ff'
        ids = [1]
    elif model == 'cities_survey':
        folder = 'cities_survey_s1'
        ids = [1, 2]
    elif model == 'aspen':
        folder = 'aspen_s1'
        ids = [1]
    elif model == 'marble':
        folder = 'marble_ff'
        ids = [1, 2, 3, 4, 5]
    elif model == 'cycling_tdf':
        folder = 'cycling_tdf_s1'
        # ids = [e for e in range(1, 69+1)]
        selection_method = 'random'
        ids = [14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]
    elif model == 'cycling_gdi':
        folder = 'cycling_gdi_s1'
        ids = [i for i in range(2, 23 + 1)]
    elif model == 'ers':
        folder = 'ers_s1'
        # folder = 'ers_f'
        # 500 voters or more
        ids = [3, 9, 23, 31, 32, 33, 36, 38, 40, 68, 77, 79, 80]
    elif model == 'ice_races':
        folder = 'ice_races_s1'
        # 80 voters or more
        ids = [4, 5, 8, 9, 15, 20, 23, 24, 31, 34, 35, 37, 43, 44, 49]
    else:
        ids = []

    ctr = 0
    # print(experiment.families)
    # print(ids, experiment.families[family_id].size)
    rand_ids = rand.choices(ids, k=experiment.families[family_id].size)
    for ri in rand_ids:
        name = family_id + '_' + str(ctr)
        tmp_election_type = model + '_' + str(ri)

        generate_preflib_election(
            experiment=experiment, model=tmp_election_type,
            name=name,
            num_voters=experiment.families[family_id].num_voters,
            num_candidates=experiment.families[family_id].num_candidates,
            folder=folder, selection_method=selection_method)
        ctr += 1

        keys.append(name)
    return keys


# MATRICES
def get_sushi_vectors():
    return np.array(get_sushi_matrix()).transpose()


def get_sushi_matrix():
    return [[0.11, 0.0808, 0.0456, 0.1494, 0.109, 0.0412, 0.3426, 0.0226, 0.0072, 0.0916],
            [0.1058, 0.1546, 0.0644, 0.1302, 0.1304, 0.0358, 0.2056, 0.051, 0.0132, 0.109],
            [0.1138, 0.1594, 0.0884, 0.096, 0.1266, 0.0548, 0.1276, 0.0874, 0.0246, 0.1214],
            [0.109, 0.1474, 0.1048, 0.0754, 0.1182, 0.068, 0.0862, 0.1168, 0.034, 0.1402],
            [0.1004, 0.1376, 0.129, 0.061, 0.0924, 0.0888, 0.0666, 0.1394, 0.047, 0.1378],
            [0.1042, 0.116, 0.1348, 0.0632, 0.0846, 0.0954, 0.048, 0.1528, 0.0804, 0.1206],
            [0.097, 0.0832, 0.1324, 0.0568, 0.079, 0.1306, 0.0422, 0.1682, 0.1094, 0.1012],
            [0.0982, 0.0636, 0.1214, 0.0638, 0.0728, 0.1546, 0.0376, 0.1396, 0.166, 0.0824],
            [0.0836, 0.0352, 0.1048, 0.1084, 0.101, 0.1692, 0.0236, 0.0958, 0.2222, 0.0562],
            [0.078, 0.0222, 0.0744, 0.1958, 0.086, 0.1616, 0.02, 0.0264, 0.296, 0.0396]]


def generate_preflib_votes(culture_id=None, num_candidates=None, num_voters=None, params=None):
    model = f'{culture_id}_{params["id"]}'
    votes = generate_votes_preflib(model, num_candidates, num_voters, params['folder'])
    return np.array(votes)