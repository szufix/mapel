#!/usr/bin/env python
""" this module is used to generate and import elections"""

import math
import os
import random as rand
import time

import numpy as np


# NOWE
def get_preflib_meta(data_name):

    return {
        'irish': (3, '01'),
        'netflix': (200, '04'),
        'burlington': (2, '05'),
        'skate': (48, '06'),
        'ers': (87, '07'),
        'glasgow': (21, '08'),
        'formula': (48, '10'),
        'ski': (2, '10'),
        'tshirt': (1, '12'),
        'asen': (19, '13'),
        'sushi': (1, '14'),
        'aspen': (2, '16'),
        'sf': (14, '21'),
        'poster': (2, '33'),
        'oakland': (7, '19'),
        'vermont': (15, '31'),
        'cities_survey': (2, '34'),
        'marble': (5, 'A'),
        'cycling_gdi': (23, 'A'),
        'cycling_tdf': (69, 'A'),
        'ice_races': (50, 'A'),
    }.get(data_name)



# STARE - needs update
def compute_preflib_elections(elections_name):
    if elections_name == "netflix3":
        return compute_netflix(elections_name, 3)
    elif elections_name == "netflix4":
        return compute_netflix(elections_name, 4)
    elif elections_name == "formula1":
        return compute_formula1(elections_name)
    elif elections_name == "sushi10":
        return compute_sushi(elections_name, 10)
    elif elections_name == "sushi100":
        return compute_sushi(elections_name, 100)
    elif elections_name == "skijumping2013":
        return compute_skijumping2013(elections_name)


# STARE - needs update
def compute_skijumping2013(elections_name):
    params = {'elections': 1, 'voters': 0,
              'candidates': 0, 'options': 0}

    total_max_cand_pos_value = [0] * 120

    year = "2013"
    limit = 94

    players = {}

    number_of_players = 0
    number_of_competitions = 0
    dirr = 'preflib/skijumping/' + year + '/'
    for filename in os.listdir(dirr):

        path = 'preflib/skijumping/' + year + '/' + str(filename)
        my_file = open(path, 'r')

        for dirty_line in my_file:
            line = dirty_line.rstrip("\n").split(',')
            key = line[1]
            if key not in players:
                players[key] = [number_of_players, 1]
                number_of_players += 1
            else:
                players[key][1] += 1

        number_of_competitions += 1

    votes = [[0 for _ in range(number_of_players)] for _ in range(number_of_competitions)]

    # print(players)

    num = 0
    for item in players:
        if players[item][1] > limit:
            num += 1
    print(num)

    magic_number = num

    ctr = 0
    for filename in os.listdir(dirr):

        path = 'preflib/skijumping/' + year + '/' + str(filename)
        my_file = open(path, 'r')

        pos = 0
        for dirty_line in my_file:
            line = dirty_line.rstrip("\n").split(',')
            key = line[1]

            if players[key][1] > limit:
                votes[ctr][pos] = players[key][0]
                pos += 1

        ctr += 1

    # print(votes)

    params['voters'] = number_of_competitions
    params['candidates'] = number_of_players

    for j in range(magic_number):

        cand_pos_value = [0] * params['candidates']
        max_cand_pos_value = 0
        for k in range(params['voters']):
            vote = votes[k][j]
            cand_pos_value[vote] += 1
            if cand_pos_value[vote] > max_cand_pos_value:
                max_cand_pos_value = cand_pos_value[vote]

        total_max_cand_pos_value[j] += max_cand_pos_value / float(params['voters'])

    total_max_cand_pos_value[:] = [x / float(params['elections']) for x in total_max_cand_pos_value]

    file_name = "elections/mfcr/" + str(elections_name) + ".txt"
    file_mfcr = open(file_name, 'w')
    # for i in range(params["candidates"]):
    for i in range(magic_number):
        file_mfcr.write(str(total_max_cand_pos_value[i] * 100) + "\n")
    file_mfcr.close()


# STARE - needs update
def compute_sushi(elections_name, size):
    params = {'elections': 1, 'voters': 0,
              'candidates': 0, 'options': 0}

    max_cand = size
    magic_number = 10

    total_max_cand_pos_value = [0] * max_cand

    for i in range(params['elections']):

        file_type = ""
        if elections_name == "sushi10":
            file_type = "soc"
            id = i + 1
        elif elections_name == "sushi100":
            file_type = "soi"
            id = i + 2

        file_name = ""
        if 0 <= id <= 9:
            file_name = "ED-00014-0000000" + str(id)
        elif 10 <= id <= 99:
            file_name = "ED-00014-000000" + str(id)

        path = "preflib/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
        my_file = open(path, 'r')

        params['candidates'] = int(my_file.readline())
        for _ in range(params['candidates']):
            my_file.readline()

        line = my_file.readline().rstrip("\n").split(',')
        params['voters'] = int(line[0])
        params['options'] = int(line[2])

        votes = [[0 for _ in range(params['candidates'])] for _ in range(params['voters'])]
        lines = [[0 for _ in range(params['candidates'] + 1)] for _ in range(params['voters'])]

        it = 0
        for j in range(params['options']):
            lines[j] = my_file.readline().rstrip("\n").split(',')
            quantity = int(lines[j][0])

            for k in range(quantity):
                for l in range(magic_number):
                    votes[it][l] = int(lines[j][l + 1]) - 1
                it += 1

        my_file.close()

        # for j in range(params['candidates']):
        for j in range(magic_number):

            cand_pos_value = [0] * params['candidates']
            max_cand_pos_value = 0
            for k in range(params['voters']):
                vote = votes[k][j]
                cand_pos_value[vote] += 1
                if cand_pos_value[vote] > max_cand_pos_value:
                    max_cand_pos_value = cand_pos_value[vote]

            total_max_cand_pos_value[j] += max_cand_pos_value / float(params['voters'])

    total_max_cand_pos_value[:] = [x / float(params['elections']) for x in total_max_cand_pos_value]

    file_name = "elections/mfcr/" + str(elections_name) + ".txt"
    file_mfcr = open(file_name, 'w')
    # for i in range(params["candidates"]):
    for i in range(magic_number):
        file_mfcr.write(str(total_max_cand_pos_value[i] * 100) + "\n")
    file_mfcr.close()


# STARE - needs update
def compute_netflix(elections_name, size):
    params = {'elections': 100, 'voters': 0,
              'candidates': size, 'options': math.factorial(size)}

    total_max_cand_pos_value = [0] * params['candidates']

    elections_id = elections_name

    file_name = "real_data/" + str(elections_id) + ".txt"
    file_votes = open(file_name, 'w')
    file_votes.write(str(params['elections']) + "\n")

    for i in range(params['elections']):

        if size == 3:
            id = i + 1
        elif size == 4:
            id = i + 101

        file_name = ""
        if 0 <= id <= 9:
            file_name = "ED-00004-0000000" + str(id)
        elif 10 <= id <= 99:
            file_name = "ED-00004-000000" + str(id)
        elif 100 <= id <= 999:
            file_name = "ED-00004-00000" + str(id)

        file_type = "soc"
        path = "preflib/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
        my_file = open(path, 'r')

        for _ in range(params['candidates'] + 1):
            my_file.readline()

        line = my_file.readline().rstrip("\n").split(',')
        params['voters'] = int(line[0])
        votes = [[0 for _ in range(params['candidates'])] for _ in range(params['voters'])]
        lines = [[0 for _ in range(params['candidates'] + 1)] for _ in range(params['options'])]

        it = 0
        for j in range(params['options']):
            lines[j] = my_file.readline().rstrip("\n").split(',')
            quantity = int(lines[j][0])

            for k in range(quantity):
                for l in range(params['candidates']):
                    votes[it][l] = int(lines[j][l + 1]) - 1
                it += 1

        my_file.close()

        # print(votes)
        print(params['voters'])

        file_votes.write(str(params['voters']) + "\n")

        for j in range(params['voters']):
            for k in range(params['candidates']):
                file_votes.write(str(votes[j][k]) + "\n")

    file_votes.close()

    """
        for j in range(params['candidates']):

            cand_pos_value = [0] * params['candidates']
            max_cand_pos_value = 0
            for k in range(params['voters']):
                vote = votes[k][j]
                cand_pos_value[vote] += 1
                if cand_pos_value[vote] > max_cand_pos_value:
                    max_cand_pos_value = cand_pos_value[vote]

            total_max_cand_pos_value[j] += max_cand_pos_value/ float(params['voters'])

    total_max_cand_pos_value[:] = [x / float(params['elections']) for x in total_max_cand_pos_value]

    file_name = "elections/mfcr/" + str(elections_name) + ".txt"
    file_mfcr = open(file_name, 'w')
    for i in range(params["candidates"]):
        file_mfcr.write(str(total_max_cand_pos_value[i]) + "\n")
    file_mfcr.close()

    """


# STARE - needs update
def save_sushi():
    params = {'elections': 1, 'voters': 0,
              'candidates': 0, 'options': 0}

    elections_name = "sushi10"
    max_cand = 10
    magic_number = 10

    total_max_cand_pos_value = [0] * max_cand

    for i in range(params['elections']):

        file_type = ""
        if elections_name == "sushi10":
            file_type = "soc"
            id = i + 1
        elif elections_name == "sushi100":
            file_type = "soi"
            id = i + 2

        file_name = ""
        if 0 <= id <= 9:
            file_name = "ED-00014-0000000" + str(id)
        elif 10 <= id <= 99:
            file_name = "ED-00014-000000" + str(id)

        path = "preflib/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
        my_file = open(path, 'r')

        params['candidates'] = int(my_file.readline())
        for _ in range(params['candidates']):
            my_file.readline()

        line = my_file.readline().rstrip("\n").split(',')
        params['voters'] = int(line[0])
        params['options'] = int(line[2])

        votes = [[0 for _ in range(params['candidates'])] for _ in range(params['voters'])]
        lines = [[0 for _ in range(params['candidates'] + 1)] for _ in range(params['voters'])]

        it = 0
        for j in range(params['options']):
            lines[j] = my_file.readline().rstrip("\n").split(',')
            quantity = int(lines[j][0])

            for k in range(quantity):
                for l in range(magic_number):
                    votes[it][l] = int(lines[j][l + 1]) - 1
                it += 1

        my_file.close()

        file_name = "elections/info/" + str(elections_name) + ".txt"
        file_info = open(file_name, 'w')
        file_info.write(str(0) + "\n")
        file_info.write(str(params["elections"]) + "\n")
        file_info.write(str(params["voters"]) + "\n")
        file_info.write(str(params["candidates"]) + "\n")
        file_info.close()

        file_name = "elections/votes/" + str(elections_name) + ".txt"
        file_votes = open(file_name, 'w')
        for j in range(params["voters"]):
            for k in range(params["candidates"]):
                file_votes.write(str(votes[j][k]) + "\n")
        file_votes.close()


# STARE - needs update
def get_axis_from_sp(elections_id):
    elections, params = import_elections(elections_id)

    number_of_elections = params['elections']
    number_of_voters = params['voters']
    number_of_candidates = params['candidates']

    for e in range(number_of_elections):
        # axis = np.zeros(number_of_candidates)
        left = []
        right = []

        # pierwszy element
        left.append(elections['votes'][e][0][number_of_candidates - 1])

        # pierwsza petla 2
        ctr = 1
        cand = elections['votes'][e][ctr][number_of_candidates - 1]
        ctr += 1
        while ctr < number_of_voters and cand in left:
            cand = elections['votes'][e][ctr][number_of_candidates - 1]
            ctr += 1

        if cand not in left:
            right.append(cand)

        for X in range(2, number_of_candidates + 1):
            # druga peentla 1
            ctr = 0
            cand = elections['votes'][e][ctr][number_of_candidates - X]
            while ctr < number_of_voters - 1 and (cand in left or cand in right):
                ctr += 1
                cand = elections['votes'][e][ctr][number_of_candidates - X]

            if cand not in left and cand not in right:
                if elections['votes'][e][ctr][number_of_candidates - 1] in left:
                    left.append(cand)
                else:
                    right.append(cand)

            # druga peentla 2
            while ctr < number_of_voters - 1 and (cand in left or cand in right):
                ctr += 1
                cand = elections['votes'][e][ctr][number_of_candidates - X]

            if cand not in left and cand not in right:
                if elections['votes'][e][ctr][number_of_candidates + 1 - X] in left:
                    left.append(cand)
                else:
                    right.append(cand)

            if (len(left) + len(right)) == number_of_candidates:
                break

        for r in range(len(right)):
            left.append(right[len(right) - r - 1])

        # print(left)

    # print(elections['votes'][0])
    return left


# STARE - needs update
def generate_irish_version(elections_name, elections_id):
    elections_irish, _ = import_elections(elections_name)

    elections, params = import_elections(elections_id)

    file_name = "elections/info/" + str(elections_id) + "_" + str(elections_name) + ".txt"
    file_info = open(file_name, 'w')
    file_info.write(str("0") + "\n")
    file_info.write(str(params["elections"]) + "\n")
    file_info.write(str(params["voters"]) + "\n")
    file_info.write(str(params["candidates"]) + "\n")
    file_info.close()

    file_name = "elections/votes/" + str(elections_id) + "_" + str(elections_name) + ".txt"
    file_votes = open(file_name, 'w')
    for i in range(params["elections"]):
        for j in range(params["voters"]):
            for k in range(params["candidates"]):
                if elections_irish['votes'][i][j][k] != -1:
                    file_votes.write(str(elections['votes'][i][j][k]) + "\n")
                else:
                    file_votes.write(str(-1) + "\n")
    file_votes.close()


# STARE - needs update
def snoch_irish_elections(elections_name):
    if elections_name == "dublin_north":
        file_name = "ED-00001-00000001"
    elif elections_name == "dublin_west":
        file_name = "ED-00001-00000002"
    elif elections_name == "meath":
        file_name = "ED-00001-00000003"

    file_type = "soi"
    path = "preflib/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
    my_file = open(path, 'r')

    num_elections = 1
    num_candidates = int(my_file.readline())
    for _ in range(num_candidates):
        my_file.readline()

    line = my_file.readline().rstrip("\n").split(',')
    num_voters = int(line[0])
    num_options = int(line[2])

    votes = [[-1 for _ in range(num_candidates)] for _ in range(num_voters)]
    lines = [[0 for _ in range(num_candidates + 1)] for _ in range(num_voters)]
    shape = [0 for _ in range(num_candidates)]

    it = 0
    for j in range(num_options):
        lines[j] = my_file.readline().rstrip("\n").split(',')
        quantity = int(lines[j][0])
        # shape[len(lines[j])-2] += quantity

        for k in range(quantity):
            for l in range(len(lines[j]) - 1):
                votes[it][l] = int(lines[j][l + 1]) - 1
                if l == 0:
                    shape[votes[it][l]] += 1
            it += 1

    my_file.close()

    for i in shape:
        print(i)


# STARE - needs update
def compute_formula1(elections_name):
    params = {'elections': 48, 'voters': 0,
              'candidates': 0, 'options': 0}

    max_cand = 100
    magic_number = 10

    total_max_cand_pos_value = [0] * max_cand

    for i in range(params['elections']):

        id = i + 1

        file_name = ""
        if 0 <= id <= 9:
            file_name = "ED-00010-0000000" + str(id)
        elif 10 <= id <= 99:
            file_name = "ED-00010-000000" + str(id)

        file_type = "soi"
        path = "preflib/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
        my_file = open(path, 'r')

        params['candidates'] = int(my_file.readline())
        for _ in range(params['candidates']):
            my_file.readline()

        line = my_file.readline().rstrip("\n").split(',')
        params['voters'] = int(line[0])
        params['options'] = int(line[2])

        votes = [[0 for _ in range(params['candidates'])] for _ in range(params['voters'])]
        lines = [[0 for _ in range(params['candidates'] + 1)] for _ in range(params['voters'])]

        it = 0
        for j in range(params['options']):
            lines[j] = my_file.readline().rstrip("\n").split(',')
            quantity = int(lines[j][0])

            for k in range(quantity):
                for l in range(magic_number):
                    votes[it][l] = int(lines[j][l + 1]) - 1
                it += 1

        my_file.close()

        # for j in range(params['candidates']):
        for j in range(magic_number):

            cand_pos_value = [0] * params['candidates']
            max_cand_pos_value = 0
            for k in range(params['voters']):
                vote = votes[k][j]
                cand_pos_value[vote] += 1
                if cand_pos_value[vote] > max_cand_pos_value:
                    max_cand_pos_value = cand_pos_value[vote]

            total_max_cand_pos_value[j] += max_cand_pos_value / float(params['voters'])

    total_max_cand_pos_value[:] = [x / float(params['elections']) for x in total_max_cand_pos_value]

    file_name = "elections/mfcr/" + str(elections_name) + ".txt"
    file_mfcr = open(file_name, 'w')
    # for i in range(params["candidates"]):
    for i in range(magic_number):
        file_mfcr.write(str(total_max_cand_pos_value[i] * 100) + "\n")
    file_mfcr.close()


# Aktualne
def save_formula(top):
    params = {'elections': 1, 'voters': 0,
              'candidates': 0, 'options': 0}

    elections_name = "formula"

    params['elections'] = 48

    top_drivers = []
    mapping = []

    # GET TOP 20
    for i in range(params['elections'] ):#params['elections']):

        file_type = "soi"
        id = i + 1
        print(id)

        file_name = ""
        if 0 <= id <= 9:
            file_name = "ED-00010-0000000" + str(id)
        elif 10 <= id <= 99:
            file_name = "ED-00010-000000" + str(id)

        path = "raw_data/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
        my_file = open(path, 'r')

        params['candidates'] = int(my_file.readline())
        for _ in range(params['candidates']):
            my_file.readline()

        line = my_file.readline().rstrip("\n").split(',')
        params['voters'] = int(line[0])
        params['options'] = int(line[2])

        votes = [[0 for _ in range(params['candidates'])] for _ in range(params['voters'])]
        lines = [[0 for _ in range(params['candidates'] + 1)] for _ in range(params['voters'])]

        it = 0
        for j in range(params['options']):
            lines[j] = my_file.readline().rstrip("\n").\
                replace("{", "").replace("}", "").split(',')
            quantity = int(lines[j][0])

            for k in range(quantity):
                for l in range(len(lines[j])-1):
                    votes[it][l] = int(lines[j][l + 1]) - 1
                it += 1

        #print(votes)

        num_candidates = params['candidates']
        num_voters = params['voters']

        order = [x for x in range(num_candidates)]
        score = np.zeros(num_candidates)

        for j in range(num_candidates):
            for k in range(num_voters):
                if j in votes[k]:
                    score[j] += 1

        #for x in range(num_candidates):
            #print(x, score[x])

        order = [x for _, x in sorted(zip(score, order))]
        order = [x for x in sorted(order, reverse=True)]

        top_drivers = order[0:top]

        #print(top_drivers)

        mapping = [-1 for _ in range(num_candidates)]
        for x in range(top):
            mapping[order[x]] = x

        #for x in range(num_candidates):
        #    print(x, mapping[x])
        my_file.close()

        # PREPARE ELECTIONS

        file_type = "toc"
        id = i + 1

        file_name = ""
        if 0 <= id <= 9:
            file_name = "ED-00010-0000000" + str(id)
        elif 10 <= id <= 99:
            file_name = "ED-00010-000000" + str(id)

        path = "raw_data/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
        my_file = open(path, 'r')

        params['candidates'] = int(my_file.readline())
        for _ in range(params['candidates']):
            my_file.readline()

        line = my_file.readline().rstrip("\n").split(',')
        params['voters'] = int(line[0])
        params['options'] = int(line[2])

        votes = [[0 for _ in range(top)] for _ in range(params['voters'])]
        lines = [[0 for _ in range(params['candidates'] + 1)] for _ in range(params['voters'])]

        it = 0
        for j in range(params['options']):
            lines[j] = my_file.readline().rstrip("\n").\
                replace("{", "").replace("}", "").split(',')
            quantity = int(lines[j][0])

            for k in range(quantity):
                pos = 0
                for l in range(len(lines[j])-1):
                    driver = int(lines[j][l + 1]) - 1
                    #print(driver)
                    if driver in top_drivers:
                        votes[it][pos] = mapping[driver]
                        pos += 1
                it += 1

        #print(votes)

        file_name = "real_data/" + str(elections_name) + "_" + str(top) + "/" + \
                    str(elections_name) + "_" + str(top) + "_" + str(i) + ".txt"

        file_votes = open(file_name, 'w')
        file_votes.write(str(num_voters) + "\n")

        for j in range(num_voters):
            for k in range(top):
                file_votes.write(str(votes[j][k]) + "\n")
        file_votes.close()


        print("YOU ARE A PRO!")


# Aktualne
def transform_irish_elections(elections_name, length_of_vote):

    if elections_name == "dublin_north":
        file_name = "ED-00001-00000001"
    elif elections_name == "dublin_west":
        file_name = "ED-00001-00000002"
    elif elections_name == "meath":
        file_name = "ED-00001-00000003"

    file_type = "soi"
    path = "raw_data/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
    my_file = open(path, 'r')

    num_elections = 1
    num_candidates = int(my_file.readline())
    for _ in range(num_candidates):
        my_file.readline()

    line = my_file.readline().rstrip("\n").split(',')
    num_voters = int(line[0])
    num_options = int(line[2])

    votes = []
    lines = [[0 for _ in range(num_candidates + 1)] for _ in range(num_voters)]
    # shape = [0 for _ in range(num_candidates)]

    for j in range(num_options):
        lines[j] = my_file.readline().rstrip("\n").split(',')
        quantity = int(lines[j][0])
        for k in range(quantity):
            size = len(lines[j]) - 1
            if size >= length_of_vote:
                vote = []
                for l in range(size):
                    vote.append(int(lines[j][l + 1]) - 1)
                votes.append(vote)

    #print(votes)
    num_voters = len(votes)
    print(num_voters)

    my_file.close()

    long_name = str(elections_name) + "_" + str(length_of_vote)
    file_name = "real_data/" + long_name + "/" + long_name + ".txt"
    file_votes = open(file_name, 'w')
    file_votes.write(str(num_voters) + "\n")
    for j in range(num_voters):
        for k in range(num_candidates):
            file_votes.write(str(votes[j][k]) + "\n")
    file_votes.close()


# Aktualne
def transform_ers_elections(elections_name='ers'):

    total_voters = 0

    for i in range(1, 88):
        if i < 10:
            file_name = "ED-00007-0000000" + str(i)
        else:
            file_name = "ED-00007-000000" + str(i)

        file_type = "soi"
        path = "raw_data/" + str(elections_name) + "/" + str(file_name) + "." + str(file_type)
        my_file = open(path, 'r')

        num_elections = 1
        num_candidates = int(my_file.readline())
        for _ in range(num_candidates):
            my_file.readline()
        length_of_vote = num_candidates

        line = my_file.readline().rstrip("\n").split(',')
        num_voters = int(line[0])
        num_options = int(line[2])

        votes = []
        lines = [[0 for _ in range(num_candidates + 1)] for _ in range(num_voters)]
        # shape = [0 for _ in range(num_candidates)]

        for j in range(num_options):
            lines[j] = my_file.readline().rstrip("\n").split(',')
            quantity = int(lines[j][0])
            for k in range(quantity):
                size = len(lines[j]) - 1
                if size >= length_of_vote:
                    vote = []
                    for l in range(size):
                        vote.append(int(lines[j][l + 1]) - 1)
                    votes.append(vote)

        my_file.close()

        num_voters = len(votes)
        total_voters += num_voters
        print(num_voters)

        if num_voters >= 50 and num_candidates >= 3:
            print(i, num_candidates, num_voters)

            long_name = str(i) + "_" + str(elections_name) + "_" + str(length_of_vote)
            path = "real_data/ers/" + long_name
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)

                file_name = "real_data/ers/" + long_name + "/" + long_name + ".txt"
                file_votes = open(file_name, 'w')
                file_votes.write(str(num_voters) + "\n")
                for j in range(num_voters):
                    for k in range(num_candidates):
                        file_votes.write(str(votes[j][k]) + "\n")
                file_votes.close()


    print('SUMA: ', total_voters)


# NOWE
def transform_elections_modern(data_name=None, fill=None):

    size, data_id = get_preflib_meta(data_name)

    tmp_list = []

    for i in range(1, size+1):

        if data_name == 'marble':
            years = [str(year) for year in range(2016,2020+1)]
            path = "raw_data/marble/marble" + years[i-1] + "_filtered"

        elif data_name == 'cycling_tdf':
            years = [str(year) for year in range(1910, 1914+1)] + \
                    [str(year) for year in range(1919, 1934 + 1)] + \
                    [str(1936)] + \
                    [str(year) for year in range(1947, 1953 + 1)] + \
                    [str(1958), str(1959), str(1979)] + \
                    [str(year) for year in range(1984, 2020 + 1)]
            path = "raw_data/cycling_tdf/tdf" + years[i-1] + "_filtered"

        elif data_name == 'cycling_gdi':
            years = [str(year) for year in range(1997,2019+1)]
            path = "raw_data/cycling_gdi/gdi" + years[i-1] + "_filtered"

        elif data_name == 'ice_races':
            path = "raw_data/ice_races/" + str(i-1)

        else:   # real preflib
            if i < 10:
                file_name = 'ED-000' + data_id + '-0000000' + str(i)
            elif i < 100:
                file_name = 'ED-000' + data_id + '-000000' + str(i)
            else:
                file_name = 'ED-000' + data_id + '-00000' + str(i)
            path = "raw_data/" + data_name + "/" + str(file_name)

        all_votes = 0
        # IMPORT
        try:
            my_file = open(path + '.soc', 'r')
            num_candidates = int(my_file.readline())

            for _ in range(num_candidates):
                my_file.readline()
            length_of_vote = num_candidates

            line = my_file.readline().rstrip("\n").split(',')
            num_voters = int(line[0])
            num_options = int(line[2])

            votes = []
            lines = [[0 for _ in range(num_candidates + 1)] for _ in range(num_voters)]

            for j in range(num_options):
                lines[j] = my_file.readline().rstrip("\n").split(',')
                quantity = int(lines[j][0])
                for _ in range(quantity):
                    vote = []
                    for k in range(num_candidates):
                        vote.append(int(lines[j][k + 1]) - 1)
                    votes.append(vote)

            my_file.close()
            num_voters = len(votes)

        except:

            votes = []
            if fill in ['id', 'ic', 'f']:

                my_file = open(path + '.toc', 'r')
                num_candidates = int(my_file.readline())
                for _ in range(num_candidates):
                    my_file.readline()
                length_of_vote = num_candidates

                line = my_file.readline().rstrip("\n").split(',')
                num_voters = int(line[0])
                num_options = int(line[2])

                for j in range(num_options):
                    if fill == 'id':
                        line = my_file.readline().strip()
                        line = line.replace(',{}', '')
                        line = line.replace('{', '').replace('}', '')
                        line = line.split(',')
                        quantity = int(line[0])
                        for _ in range(quantity):
                            vote = []
                            for k in range(num_candidates):
                                vote.append(int(line[k + 1]) - 1)
                            votes.append(vote)

                    elif fill == 'ic':
                        line = my_file.readline().strip()
                        line = line.replace(',{}', '')
                        line = line.split(',')
                        quantity = int(line[0])
                        for _ in range(quantity):
                            my_line = line[1:(num_candidates+1)]
                            beginning = 0
                            for k in range(len(my_line)):
                                if '{' in my_line[k]:
                                    beginning = k
                                    my_line[k] = my_line[k].replace('{', '')
                                if '}' in my_line[k]:
                                    my_line[k] = my_line[k].replace('}', '')
                                    if k > beginning:
                                        ties = my_line[beginning:(k+1)]
                                        rand.shuffle(ties)
                                        my_line[beginning:(k+1)] = ties
                            vote = []
                            for k in range(num_candidates):
                                vote.append(int(my_line[k]) - 1)
                            votes.append(vote)

                    elif fill == 'f':
                        line = my_file.readline().rstrip("\n")
                        if '{' not in line:
                            line = line.split(',')
                            quantity = int(line[0])
                            for _ in range(quantity):
                                vote = []
                                for k in range(num_candidates):
                                    vote.append(int(line[k + 1]) - 1)
                                votes.append(vote)

            elif fill in ['s1']:

                try:
                    print(path)
                    my_file = open(path + '.soi', 'r')

                    num_candidates = int(my_file.readline())
                    if num_candidates >= 10:
                        print(i)
                        tmp_list.append(i)
                    for _ in range(num_candidates):
                        my_file.readline()
                    length_of_vote = num_candidates

                    line = my_file.readline().rstrip("\n").split(',')
                    num_voters = int(line[0])
                    num_options = int(line[2])

                    original_votes = []
                    original_options = []
                    original_options_quantity = []

                    for j in range(num_options):
                        line = my_file.readline().strip()
                        line = line.split(',')
                        quantity = int(line[0])
                        vote = []
                        for k in range(len(line) - 1):
                            vote.append(int(line[k + 1]) - 1)
                        original_options.append(vote)
                        original_options_quantity.append(quantity)
                        for _ in range(quantity):
                            original_votes.append(vote)

                except:
                    my_file = open(path + '.toi', 'r')
                    num_candidates = int(my_file.readline())
                    for _ in range(num_candidates):
                        my_file.readline()
                    length_of_vote = num_candidates

                    line = my_file.readline().rstrip("\n")
                    line = line.split(',')
                    num_voters = int(line[0])
                    num_options = int(line[2])

                    original_votes = []
                    original_options = []
                    original_options_quantity = []

                    # glosy z remisami usuwam
                    for j in range(num_options):
                        line = my_file.readline().strip()
                        if '{' not in line:
                            line = line.split(',')
                            quantity = int(line[0])
                            all_votes += quantity
                            vote = []
                            for k in range(len(line) - 1):
                                vote.append(int(line[k + 1]) - 1)
                            if len(vote) == len(set(vote)):
                                original_options.append(vote)
                                original_options_quantity.append(quantity)
                                for _ in range(quantity):
                                    original_votes.append(vote)

                # GET FREQ
                num_voters = len(original_votes)
                cand_freq = [0 for _ in range(num_candidates)]
                for v in range(num_voters):
                    for c in range(len(original_votes[v])):
                        cand_freq[original_votes[v][c]] += 1
                ordered_cands = [x for _, x in sorted(zip(cand_freq, [i for i in range(num_candidates)]), reverse=True)]
                # SAVE FREQ TO FILE
                long_name = str(data_name) + "_" + str(i) + '.txt'
                path = 'real_data/freq/' + long_name
                file_votes = open(path, 'w')
                for c in range(num_candidates):
                    file_votes.write(str(ordered_cands[c]) + ',')
                file_votes.close()

                start_time = time.time()
                for j in range(len(original_votes)):

                    #if j%100 == 0:
                    #    print(j)

                    vote = []
                    vote[0:len(original_votes[j])] = original_votes[j][0:len(original_votes[j])]

                    while len(vote) < num_candidates:

                        q = [0 for _ in range(num_candidates)]
                        for k in range(len(original_options)):
                            if original_options[k][0:len(vote)] == vote and len(original_options[k]) > len(vote):
                                q[original_options[k][len(vote)]] += original_options_quantity[k]
                        it = 0
                        if sum(q) > 0:
                            r = rand.randint(0, sum(q)-1)
                            while sum(q[0:(it+1)]) <= r:
                                it += 1
                            vote.append(it)
                        else:
                            ending = []
                            for c in range(num_candidates):
                                if c not in vote:
                                    ending.append(c)
                            rand.shuffle(ending)
                            for c in range(len(ending)):
                                vote.append(ending[c])
                    votes.append(vote)

                elapsed_time = time.time() - start_time
                #print(round(elapsed_time, 2))

            my_file.close()
            num_voters = len(votes)


        # SAVE TO FILE
        long_name = str(data_name) + "_" + str(i) + '.txt'
        path = 'real_data/' + data_name + '_' + fill + '/' + long_name
        file_votes = open(path, 'w')
        file_votes.write(str(num_voters) + "\n")
        file_votes.write(str(num_candidates) + "\n")
        for j in range(num_voters):
            for k in range(num_candidates):
                if k < num_candidates - 1:
                    file_votes.write(str(votes[j][k]) + ',')
                else:
                    file_votes.write(str(votes[j][k]) + '\n')
        file_votes.close()

    print(tmp_list)


def print_meta_preflib_params(data_name=None, fill=None):
    size, data_id = get_preflib_meta(data_name)
    values = []
    cands = []

    for i in range(1, size+1):
        path = 'real_data/' + data_name + '_' + fill + '/' + data_name + '_' + str(i) + '.txt'
        with open(path, 'r') as txt_file:
            num_voters = int(txt_file.readline())
            num_candidates = int(txt_file.readline())
            values.append(num_voters)
            cands.append(num_candidates)

    ids = []
    avg_candidates = 0
    avg_voters = 0

    if data_name == 'skate':
        for i in range(1, size+1):
            if values[i-1] >= 9 and cands[i-1] >= 10:
                ids.append(i)
                avg_voters += values[i-1]
                avg_candidates += cands[i-1]

    elif data_name in {'irish', 'glasgow', 'aspen', 'cycling_gdi', 'tshirt',
                       'cities_survey'}:
        for i in range(1, size+1):
            if cands[i-1] >= 10:
                ids.append(i)
                avg_voters += values[i-1]
                avg_candidates += cands[i-1]

    elif data_name == 'formula':
        for i in range(1, size+1):
            if values[i-1] >= 17:
                ids.append(i)

    elif data_name == 'ers':
        for i in range(1, size+1):
            if values[i-1] >= 500 and cands[i-1] >= 10:
                ids.append(i)
                avg_voters += values[i-1]
                avg_candidates += cands[i-1]

    elif data_name in {'cycling_tdf'}:
        for i in range(1, size + 1):
            if values[i - 1] >= 20 and cands[i - 1] <= 75:
                ids.append(i)
                avg_voters += values[i-1]
                avg_candidates += cands[i-1]

    elif data_name in {'ice_races'}:
        for i in range(1, size + 1):
            if values[i - 1] >= 80 and cands[i - 1] >= 10:
                ids.append(i)
                avg_voters += values[i-1]
                avg_candidates += cands[i-1]

    print(ids)
    print(len(ids))
    print(avg_candidates/len(ids))
    print(avg_voters/len(ids))


def print_num_voters_and_candidates(data_name=None, fill=None):
    size, data_id = get_preflib_meta(data_name)
    V = []
    C = []


    for i in range(1, size+1):
        path = 'real_data/' + data_name + '_' + fill + '/' + data_name + '_' + str(i) + '.txt'
        with open(path, 'r') as txt_file:
            num_voters = int(txt_file.readline())
            num_candidates = int(txt_file.readline())
            V.append(num_voters)
            C.append(num_candidates)
            print(num_voters, num_candidates)

    print('V', min(V), max(V))
    print('C', min(C), max(C))
