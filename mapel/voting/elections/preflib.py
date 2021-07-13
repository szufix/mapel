#!/usr/bin/env python
# UNDER CONSTRUCTION #

import os
import random as rand
from collections import Counter

import numpy as np
from scipy.stats import gamma


LIST_OF_PREFLIB_ELECTIONS = {'sushi', 'irish', 'glasgow', 'skate', 'formula',
                             'tshirt', 'cities_survey', 'aspen', 'ers',
                             'marble', 'cycling_tdf', 'cycling_gdi', 'ice_races',
                             'grenoble'}

# GENERATE
def generate_elections_preflib(experiment_id, election_model=None, elections_id=None,
                       num_voters=None, num_candidates=None, special=None, folder=None, selection_method='random'):
    """ main function: generate elections"""

    votes = generate_votes_preflib(election_model, selection_method=selection_method,
                          num_voters=num_voters, num_candidates=num_candidates, folder=folder)

    path = os.path.join("experiments", experiment_id, "elections", elections_id + ".soc")
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
def generate_votes_preflib(elections_model, num_voters=None, num_candidates=None, folder=None,
                           selection_method=None):
    """ Generate votes based on elections from Preflib """

    long_name = str(elections_model)
    file_name = 'real_data/' + folder + '/' + long_name + '.txt'
    file_votes = open(file_name, 'r')
    original_num_voters = int(file_votes.readline())
    if original_num_voters == 0:
        return [0, 0, 0]
    original_num_candidates = int(file_votes.readline())
    choice = [x for x in range(original_num_voters)]
    rand.shuffle(choice)

    votes = np.zeros([num_voters, original_num_candidates], dtype=int)
    original_votes = np.zeros([original_num_voters, original_num_candidates], dtype=int)

    for j in range(original_num_voters):
        value = file_votes.readline().strip().split(',')
        for k in range(original_num_candidates):
            original_votes[j][k] = int(value[k])

    file_votes.close()

    for j in range(num_voters):
        r = rand.randint(0, original_num_voters - 1)
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
            selected_candidates = rand.sample([j for j in range(original_num_candidates)], num_candidates)
        elif selection_method == 'borda':
            scores = get_borda_scores(original_votes, original_num_voters, original_num_candidates)
            order_by_score = [x for _, x in
                              sorted(zip(scores, [i for i in range(original_num_candidates)]), reverse=True)]
            selected_candidates = order_by_score[0:num_candidates]
        elif selection_method == 'freq':
            freq = import_freq(elections_model)
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
        return new_votes
    else:
        return votes



def import_freq(elections_model):
    path = 'real_data/freq/' + elections_model + '.txt'
    with open(path, 'r', newline='') as txt_file:
        line = txt_file.readline().strip().split(',')
        line = line[0:len(line)-1]
        for i in range(len(line)):
            line[i] = int(line[i])
    return line


def get_borda_scores(votes, num_voters, num_candidates):

    scores = [0 for _ in range(num_candidates)]
    for i in range(num_voters):
        for j in range(num_candidates):
            scores[votes[i][j]] += num_candidates - j - 1

    return scores


