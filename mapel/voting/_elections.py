#!/usr/bin/env python
""" this module is used to generate and import elections"""

from mapel.voting.elections.group_separable import generate_group_separable_election
from mapel.voting.elections.mallows import generate_mallows_election, \
    phi_mallows_helper
from mapel.voting.elections.euclidean import generate_elections_1d_simple, \
    generate_elections_2d_simple, generate_elections_nd_simple
from mapel.voting.elections.single_peaked import generate_conitzer_election, \
    generate_walsh_election, generate_spoc_conitzer_election
from mapel.voting.elections.single_crossing import generate_single_crossing_election
from mapel.voting.elections.impartial import generate_impartial_culture_election, \
    generate_impartial_anonymous_culture_election
from mapel.voting.elections.guardians import generate_real_antagonism_election, \
    generate_real_identity_election, generate_real_stratification_election, generate_real_uniformity_election

import os
import random as rand
from collections import Counter

import numpy as np
from scipy.stats import gamma

from . import objects as obj


LIST_OF_PREFLIB_ELECTIONS = {'sushi', 'irish', 'glasgow', 'skate', 'formula',
                             'tshirt', 'cities_survey', 'aspen', 'ers',
                             'marble', 'cycling_tdf', 'cycling_gdi', 'ice_races',
                             'grenoble'}


def nice_name(name):

    return {
        '1d_interval': '1D_Interval',
        '1d_gaussian': '1D_Gaussian',
        '2d_disc': '2D_Disc',
        '2d_square': '2D_Square',
        '2d_gaussian': '2D_Gaussian',
        '3d_cube': '3D_Cube',
        '4d_cube': '4D_Cube',
        '5d_cube': '5D_Cube',
        '10d_cube': '10D_Cube',
        '20d_cube': '20D_Cube',
        '2d_sphere': '2D_Sphere',
        '3d_sphere': '3D_Sphere',
        '4d_sphere': '4D_Sphere',
        '5d_sphere': '5D_Sphere',
        'impartial_culture': 'Impartial_Culture',
        'iac': 'Impartial_Anonymous_Culture',
        'urn_model': 'Urn_Model',
        'conitzer': 'Single-Peaked_(by_Conitzer)',
        'spoc_conitzer': 'SPOC',
        'walsh': 'Single-Peaked_(by_Walsh)',
        'mallows': 'Mallows',
        'norm_mallows': 'Norm_Mallows',
        'single-crossing': 'Single-Crossing',
        'didi': 'DiDi',
        'pl': 'Plackett-Luce',
        'netflix': 'Netflix',
        'sushi': 'Sushi',
        'formula': 'Formula',
        'meath': 'Meath',
        'dublin_north': 'North_Dublin',
        'dublin_west': 'West_Dublin',
        'identity': 'Identity',
        'antagonism': 'Antagonism',
        'uniformity': 'Uniformity',
        'stratification': 'Stratification',
        'unid': "UNID",
        'real_identity': 'Real_Identity',
        'real_uniformity':  'Real_Uniformity',
        'real_antagonism': 'Real_Antagonism',
        'real_stratification': 'Real_Stratification',
        'group-separable': 'Group-Separable',
    }.get(name)


LOOKUP_TABLE = {(5, 0.5): 0.50417,
                (10, 0.5): 0.65345,
                (15, 0.5): 0.73378,
                (20, 0.5): 0.78389,
                (25, 0.5): 0.81814,
                (30, 0.5): 0.84302,
                (35, 0.5): 0.86191,
                (40, 0.5): 0.87674,
                (45, 0.5): 0.8887,
                (50, 0.5): 0.89854,
                (55, 0.5): 0.90679,
                (60, 0.5): 0.91379,
                (65, 0.5): 0.91982,
                (70, 0.5): 0.92505,
                (75, 0.5): 0.92965,
                (80, 0.5): 0.93371,
                (85, 0.5): 0.93733,
                (90, 0.5): 0.94058,
                (95, 0.5): 0.94351,
                (100, 0.5): 0.94616,

                (5, 0.75): 0.73407,
                (10, 0.75): 0.82952,
                (15, 0.75): 0.87458,
                (20, 0.75): 0.9008,
                (25, 0.75): 0.91795,
                (30, 0.75): 0.93005,
                (35, 0.75): 0.93904,
                (40, 0.75): 0.94598,
                (45, 0.75): 0.9515,
                (50, 0.75): 0.956,
                (55, 0.75): 0.95973,
                (60, 0.75): 0.96288,
                (65, 0.75): 0.96558,
                (70, 0.75): 0.96791,
                (75, 0.75): 0.96994,
                (80, 0.75): 0.97173,
                (85, 0.75): 0.97332,
                (90, 0.75): 0.97474,
                (95, 0.75): 0.97602,
                (100, 0.75): 0.97717,

                # NEW
                (10, 0.9): 0.9301369856,
                (20, 0.9): 0.960527289,
                (30, 0.9): 0.972493169,
                (40, 0.9): 0.9788919858,
                (50, 0.9): 0.9828755937,
                (60, 0.9): 0.985594312,
                (70, 0.9): 0.9875680461,
                (80, 0.9): 0.9890661082,
                (90, 0.9): 0.9902419615,
                (100, 0.9): 0.9911894651,
                (110, 0.9): 0.9919692496,
                (120, 0.9): 0.9926222263,
                (130, 0.9): 0.9931770014,
                (140, 0.9): 0.9936541785,
                (150, 0.9): 0.9940689741,

                (10, 0.95): 0.964589542,
                (20, 0.95): 0.9801654616,
                (30, 0.95): 0.9862247902,
                (40, 0.95): 0.989448283,
                (50, 0.95): 0.9914492248,
                (60, 0.95): 0.9928122517,
                (70, 0.95): 0.9938004798,
                (80, 0.95): 0.9945498142,
                (90, 0.95): 0.9951375388,
                (100, 0.95): 0.9956108463,
                (110, 0.95): 0.9960001847,
                (120, 0.95): 0.9963260786,
                (130, 0.95): 0.9966028676,
                (140, 0.95): 0.9968408726,
                (150, 0.95): 0.9970477116,

                (10, 0.99): 0.9928254408,
                (20, 0.99): 0.9960077199,
                (30, 0.99): 0.9972344025,
                (40, 0.99): 0.9978844379,
                (50, 0.99): 0.9982870544,
                (60, 0.99): 0.9985609271,
                (70, 0.99): 0.9987592959,
                (80, 0.99): 0.9989096017,
                (90, 0.99): 0.9990274248,
                (100, 0.99): 0.9991222683,
                (110, 0.99): 0.9992002574,
                (120, 0.99): 0.9992655184,
                (130, 0.99): 0.999320932,
                (140, 0.99): 0.9993685707,
                (150, 0.99): 0.9994099636,}



# PREPARE
def prepare_elections(experiment_id, folder=None, starting_from=0, ending_at=1000000):
    """ Prepare elections for a given experiment """

    model = obj.Model(experiment_id, raw=True)
    id_ = 0

    for i in range(model.num_families):

        print('family:', model.families[i].label)

        election_model = model.families[i].election_model
        param_1 = model.families[i].param_1
        param_2 = model.families[i].param_2
        copy_param_1 = param_1

        #print(election_model)
        if starting_from <= id_ < ending_at:

            # PREFLIB
            if election_model in LIST_OF_PREFLIB_ELECTIONS:

                selection_method = 'random'

                # list of IDs larger than 10
                if election_model == 'irish':
                    folder = 'irish_s1'
                    #folder = 'irish_f'
                    ids = [1, 3]
                elif election_model == 'glasgow':
                    folder = 'glasgow_s1'
                    #folder = 'glasgow_f'
                    ids = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 21]
                elif election_model == 'formula':
                    folder = 'formula_s1'
                    # 17 races or more
                    ids = [17, 35, 37, 40, 41, 42, 44, 45, 46, 47, 48]
                elif election_model == 'skate':
                    folder = 'skate_ic'
                    # 9 judges
                    ids = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48]
                elif election_model == 'sushi':
                    folder = 'sushi_ff'
                    ids = [1]
                elif election_model == 'grenoble':
                    folder = 'grenoble_ff'
                    ids = [1]
                elif election_model == 'tshirt':
                    folder = 'tshirt_ff'
                    ids = [1]
                elif election_model == 'cities_survey':
                    folder = 'cities_survey_s1'
                    ids = [1, 2]
                elif election_model == 'aspen':
                    folder = 'aspen_s1'
                    ids = [1]
                elif election_model == 'marble':
                    folder = 'marble_ff'
                    ids = [1, 2, 3, 4, 5]
                elif election_model == 'cycling_tdf':
                    folder = 'cycling_tdf_s1'
                    #ids = [e for e in range(1, 69+1)]
                    selection_method = 'random'
                    ids = [14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]
                elif election_model == 'cycling_gdi':
                    folder = 'cycling_gdi_s1'
                    ids = [i for i in range(2, 23+1)]
                elif election_model == 'ers':
                    folder = 'ers_s1'
                    #folder = 'ers_f'
                    # 500 voters or more
                    ids = [3, 9, 23, 31, 32, 33, 36, 38, 40, 68, 77, 79, 80]
                elif election_model == 'ice_races':
                    folder = 'ice_races_s1'
                    # 80 voters or more
                    ids = [4, 5, 8, 9, 15, 20, 23, 24, 31, 34, 35, 37, 43, 44, 49]
                else:
                    ids = []

                #print(model.families[i].size)
                rand_ids = rand.choices(ids, k=model.families[i].size)
                for ri in rand_ids:
                    elections_id = "core_" + str(id_)
                    tmp_elections_type = election_model + '_' + str(ri)
                    #print(tmp_elections_type)
                    generate_elections_preflib(experiment_id, election_model=tmp_elections_type, elections_id=elections_id,
                                                  num_voters=model.families[i].num_voters,
                                                  num_candidates=model.families[i].num_candidates,
                                                  special=param_1,
                                                  folder=folder, selection_method=selection_method)
                    id_ += 1

            # STATISTICAL CULTURE
            else:

                base = []
                if election_model == 'crate':
                    my_size = 9
                    # with_edge
                    for p in range(my_size):
                        for q in range(my_size):
                            for r in range(my_size):
                                a = p / (my_size - 1)
                                b = q / (my_size - 1)
                                c = r / (my_size - 1)
                                d = 1 - a - b - c
                                tmp = [a, b, c, d]
                                if d >= 0 and sum(tmp) == 1:
                                    base.append(tmp)
                    # without_edge
                    """
                    for p in range(1, my_size-1):
                        for q in range(1, my_size-1):
                            for r in range(1, my_size-1):
                                a = p / (my_size - 1)
                                b = q / (my_size - 1)
                                c = r / (my_size - 1)
                                d = 1 - a - b - c
                                tmp = [a, b, c, d]
                                if d >= 0 and sum(tmp) == 1:
                                    #print(tmp)
                                    base.append(tmp)
                    """
                #print(len(base))

                for j in range(model.families[i].size):

                    if election_model in {'unid', 'stan', 'anid', 'stid', 'anun', 'stun'}:
                        if copy_param_1 == 0:    # with both
                            param_1 = j / (model.families[i].size - 1)
                        elif copy_param_1 == 1:   # without first (which is last)
                            param_1 = j / model.families[i].size
                        elif copy_param_1 == 2:   # without second (which is first)
                            param_1 = (j+1) / model.families[i].size
                        elif copy_param_1 == 4:   # without both
                            param_1 = (j+1) / (model.families[i].size + 1)
                        #print(param_1)
                    elif election_model in {'crate'}:
                        param_1 = base[j]
                    elif election_model == 'urn_model' and copy_param_1 == -2.:
                        param_1 = round(j/10000., 2)

                    elections_id = "core_" + str(id_)
                    generate_elections(experiment_id, election_model=election_model, election_id=elections_id,
                                          num_voters=model.families[i].num_voters,
                                          num_candidates=model.families[i].num_candidates,
                                          special=param_1, second_param=param_2)
                    id_ += 1

        else:
            id_ += model.families[i].size


# GENERATE
def generate_elections(experiment_id, election_model=None, election_id=None,
                       num_candidates=None, num_voters=None, special=None, second_param=None):
    """ main function: generate elections """


    if election_model in  {'identity', 'uniformity', 'antagonism', 'stratification',
                           'unid', 'anid', 'stid', 'anun', 'stun', 'stan',
                           'crate', 'walsh_fake', 'conitzer_fake'}:
        path = os.path.join("experiments", str(experiment_id), "elections", "soc_original",
                            (str(election_id) + ".soc"))
        file_ = open(path, 'w')
        file_.write('$ fake' + '\n')
        file_.write(str(num_voters) + '\n')
        file_.write(str(num_candidates) + '\n')
        file_.write(str(election_model) + '\n')
        if election_model == 'crate':
            file_.write(str(round(special[0], 5)) + '\n')
            file_.write(str(round(special[1], 5)) + '\n')
            file_.write(str(round(special[2], 5)) + '\n')
            file_.write(str(round(special[3], 5)) + '\n')
        else:
            file_.write(str(round(special, 5)) + '\n')
        file_.close()

    else:

        if election_model == 'mallows' and special == 0:
            special = rand.random()
        elif election_model == 'norm_mallows' and special == 0:
            special = phi_mallows_helper(num_candidates)
        elif election_model == 'norm_mallows' and special >= 0:
            if (num_candidates, special) in LOOKUP_TABLE:
                special = LOOKUP_TABLE[(num_candidates, special)]
            else:
                special = phi_mallows_helper(num_candidates, rdis=special)
        elif election_model == 'urn_model' and special == 0:
            special = gamma.rvs(0.8)

        naked_models = {'impartial_culture': generate_impartial_culture_election,
                        'iac': generate_impartial_anonymous_culture_election,
                        'conitzer': generate_conitzer_election,
                        'spoc_conitzer': generate_spoc_conitzer_election,
                        'walsh': generate_walsh_election,
                        'single-crossing': generate_single_crossing_election,
                        'real_identity': generate_real_identity_election,
                        'real_uniformity': generate_real_uniformity_election,
                        'real_antagonism': generate_real_antagonism_election,
                        'real_stratification': generate_real_stratification_election}

        euclidean_models = {'1d_interval': generate_elections_1d_simple,
                            '1d_gaussian': generate_elections_1d_simple,
                            '1d_one_sided_triangle': generate_elections_1d_simple,
                            '1d_full_triangle': generate_elections_1d_simple,
                            '1d_two_party': generate_elections_1d_simple,
                            '2d_disc': generate_elections_2d_simple,
                            '2d_square': generate_elections_2d_simple,
                            '2d_gaussian': generate_elections_2d_simple,
                            '3d_cube': generate_elections_nd_simple,
                            '4d_cube': generate_elections_nd_simple,
                            '5d_cube': generate_elections_nd_simple,
                            '10d_cube': generate_elections_nd_simple,
                            '15d_cube': generate_elections_nd_simple,
                            '20d_cube': generate_elections_nd_simple,
                            '40d_cube': generate_elections_nd_simple,
                            '2d_sphere': generate_elections_2d_simple,
                            '3d_sphere': generate_elections_nd_simple,
                            '4d_sphere': generate_elections_nd_simple,
                            '5d_sphere': generate_elections_nd_simple,
                            '4d_ball': generate_elections_nd_simple,
                            '5d_ball': generate_elections_nd_simple,
                            '40d_ball': generate_elections_nd_simple,}

        single_param_models = {'urn_model': generate_elections_urn_model,
                               'group-separable': generate_group_separable_election,}

        double_param_models = {'mallows': generate_mallows_election,
                               'norm_mallows': generate_mallows_election,}


        if election_model in naked_models:
            votes = naked_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates)

        elif election_model in euclidean_models:
            votes = euclidean_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                                            election_model=election_model)

        elif election_model in single_param_models:
            votes = single_param_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                                            param=special)

        elif election_model in double_param_models:
            votes = double_param_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                                            param=special, second_param=second_param)


        path = os.path.join("experiments", str(experiment_id), "elections", "soc_original",
                            (str(election_id) + ".soc"))
        file_ = open(path, 'w')

        if election_model in ["norm_mallows", "mallows", "urn_model"]:
            file_.write("# " + nice_name(election_model) + " " + str(round(special, 5)) + "\n")
        else:
            file_.write("# " + nice_name(election_model) + "\n")

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
                file_.write(str(int(counted_votes[i][1][j])))
                if j < num_candidates - 1:
                    file_.write(", ")
                else:
                    file_.write("\n")

        file_.close()

# GENERATE
def generate_elections_preflib(experiment_id, election_model=None, elections_id=None,
                       num_voters=None, num_candidates=None, special=None, folder=None, selection_method='random'):
    """ main function: generate elections"""

    votes = generate_votes_preflib(election_model, selection_method=selection_method,
                          num_voters=num_voters, num_candidates=num_candidates, folder=folder)

    path = os.path.join("experiments", experiment_id, "elections", "soc_original", elections_id + ".soc")
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


########################################################################################################################


########################################################################################################################


def generate_elections_urn_model(num_voters=None, num_candidates=None, param=None):
    """ helper function: generate polya-eggenberger urn model elections"""

    alpha = param

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    urn_size = 1.
    for j in range(num_voters):
        rho = rand.uniform(0, urn_size)
        if rho <= 1.:
            votes[j] = np.random.permutation(num_candidates)
        else:
            votes[j] = votes[rand.randint(0, j - 1)]
        urn_size += alpha

    return votes






# REAL
def generate_votes_preflib(elections_model, num_voters=None, num_candidates=None, folder=None,
                           selection_method=None):
    """ Generate votes based on elections from Preflib """

    #print(elections_model)
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
        r = rand.randint(0, original_num_voters-1)
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
            order_by_score = [x for _, x in sorted(zip(scores, [i for i in range(original_num_candidates)]), reverse=True)]
            selected_candidates = order_by_score[0:num_candidates]
        elif selection_method == 'freq':
            freq = import_freq(elections_model)
            #print(freq)
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


# IMPORT
def import_election(experiment_id, election_id):
    """ main function: import single election """
    return obj.Election(experiment_id, election_id)


def import_soc_elections(experiment_id, election_id):
    """ main function: import elections """

    path = "experiments/" + str(experiment_id) + "/elections/soc_original/" + str(election_id) + ".soc"
    my_file = open(path, 'r')

    first_line = my_file.readline()
    if first_line[0] != '#':
        num_candidates = int(first_line)
    else:
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

    return votes, num_voters, num_candidates


def import_approval_elections(experiment_id, elections_id, params):
    file_votes = open("experiments/" + experiment_id + "/elections/votes/" + str(elections_id) + ".txt", 'r')
    votes = [[[] for _ in range(params['voters'])] for _ in
             range(params['elections'])]
    for i in range(params['elections']):
        for j in range(params['voters']):
            line = file_votes.readline().rstrip().split(" ")
            if line == ['']:
                votes[i][j] = []
            else:
                for k in range(len(line)):
                    votes[i][j].append(int(line[k]))
    file_votes.close()

    # empty candidates
    candidates = [[0 for _ in range(params['candidates'])] for _ in range(params['elections'])]
    for i in range(params['elections']):
        for j in range(params['candidates']):
            candidates[i][j] = 0

    elections = {"votes": votes, "candidates": candidates}

    return elections, params




# PROBABLY NOT USED AT ALL

# def some_f(ops, i):
#     if i == 1.:
#         return 1.
#     mapping = [1., i - 1.]
#     return mapping[ops]
#

#
# def get_vector(type, num_candidates):
#     if type == "uniform":
#         return [1.] * num_candidates
#     elif type == "linear":
#         return [(num_candidates - x) for x in range(num_candidates)]
#     elif type == "linear_low":
#         return [(float(num_candidates) - float(x)) / float(num_candidates) for x in range(num_candidates)]
#     elif type == "square":
#         return [(float(num_candidates) - float(x)) ** 2 / float(num_candidates) ** 2 for x in
#                 range(num_candidates)]
#     elif type == "square_low":
#         return [(num_candidates - x) ** 2 for x in range(num_candidates)]
#     elif type == "cube":
#         return [(float(num_candidates) - float(x)) ** 3 / float(num_candidates) ** 3 for x in
#                 range(num_candidates)]
#     elif type == "cube_low":
#         return [(num_candidates - x) ** 3 for x in range(num_candidates)]
#     elif type == "split_2":
#         values = [1.] * num_candidates
#         for i in range(num_candidates / 2):
#             values[i] = 10.
#         return values
#     elif type == "split_4":
#         size = num_candidates / 4
#         values = [1.] * num_candidates
#         for i in range(size):
#             values[i] = 1000.
#         for i in range(size, 2 * size):
#             values[i] = 100.
#         for i in range(2 * size, 3 * size):
#             values[i] = 10.
#         return values
#     else:
#         return type
#
# def get_num_changes(num_candidates, PHI):
#     power = [pow(PHI, n) for n in range(0, num_candidates + 1)]
#     intervals = [sum(power[0:n]) for n in range(num_candidates + 2)]
#     answer = np.searchsorted(intervals, (rand.uniform(0, sum(power))), side='left')
#     return answer - 1
#
#
# def mallowsAlternativeVote(base, num_candidates, PHI):
#     # num_changes = get_num_changes(num_candidates, PHI)
#     num_changes = int(PHI * num_candidates)
#     places = rand.sample([c for c in range(num_candidates)], num_changes)
#     vote = list(base)
#     for p in places:
#         r = rand.randint(0, 1)
#         if r == 1:  # ma byc
#             if p not in vote:
#                 vote.append(p)
#         else:  # ma nie byc
#             if p in vote:
#                 vote.remove(p)
#     return vote