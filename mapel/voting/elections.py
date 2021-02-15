#!/usr/bin/env python
""" this module is used to generate and import elections"""


### COMMENT ON SERVER ###
# import matplotlib.pyplot as plt
#########################

import random as rand
import numpy as np
import math
import os
from collections import Counter
from scipy.stats import gamma
from . import objects as obj


LIST_OF_PREFLIB_ELECTIONS = {'sushi', 'irish', 'glasgow', 'skate', 'formula',
                             'tshirt', 'cities_survey', 'aspen', 'ers',
                             'marble', 'cycling_tdf', 'cycling_gdi', 'ice_races',
                             'grenoble'}


def nice_name(name):

    return {
        '1d_interval': '1D Interval',
        '1d_gaussian': '1D Gaussian',
        '2d_disc': '2D Disc',
        '2d_square': '2D Square',
        '2d_gaussian': '2D Gaussian',
        '3d_cube': '3D Cube',
        '4d_cube': '4D Cube',
        '5d_cube': '5D Cube',
        '10d_cube': '10D Cube',
        '20d_cube': '20D Cube',
        '2d_sphere': '2D Sphere',
        '3d_sphere': '3D Sphere',
        '4d_sphere': '4D Sphere',
        '5d_sphere': '5D Sphere',
        'impartial_culture': 'Impartial Culture',
        'iac': 'Impartial Anonymous Culture',
        'urn_model': 'Urn Model',
        'conitzer': 'Single-Peaked (by Conitzer)',
        'spoc_conitzer': 'SPOC',
        'walsh': 'Single-Peaked (by Walsh)',
        'mallows': 'Mallows',
        'phi_mallows': 'Phi-Mallows',
        'single_crossing': 'Single Crossing',
        'didi': 'DiDi',
        'pl': 'Plackett Luce',
        'netflix': 'Netflix',
        'sushi': 'Sushi',
        'formula': 'Formula',
        'meath': 'Meath',
        'dublin_north': 'North Dublin',
        'dublin_west': 'West Dublin',
        'identity': 'Identity',
        'antagonism': 'Antagonism',
        'uniformity': 'Uniformity',
        'stratification': 'Stratification',
        'mallows05': 'Mallows 0.5',
        'unid': "UNID"
    }.get(name)


# PREPARE
def prepare_elections(experiment_id, folder=None, starting_from=0, ending_at=10000):

    model = obj.Model(experiment_id)
    id_ = 0

    for i in range(model.num_families):

        election_model = model.families[i].election_model
        param_1 = model.families[i].param_1
        param_2 = model.families[i].param_2
        copy_param_1 = param_1

        if id_ >= starting_from and id_ < ending_at:

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

                print(model.families[i].size)
                rand_ids = rand.choices(ids, k=model.families[i].size)
                for ri in rand_ids:
                    elections_id = "core_" + str(id_)
                    tmp_elections_type = election_model + '_' + str(ri)
                    print(tmp_elections_type)
                    el.generate_elections_preflib(experiment_id, election_model=tmp_elections_type, elections_id=elections_id,
                                                  num_voters=model.num_voters, num_candidates=model.num_candidates,
                                                  special=param_1, folder=folder, selection_method=selection_method)
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

                    if election_model in {'unid', 'stan'}:
                        param_1 = j / (model.families[i].size - 1)
                    elif election_model in {'anid', 'stid', 'anun', 'stun'}:
                        param_1 = (j+1) / (model.families[i].size + 1)
                        #print(param_1)
                    elif election_model in {'crate'}:
                        param_1 = base[j]
                    elif election_model == 'urn_model' and copy_param_1 == -2.:
                        param_1 = round(j/10000., 2)

                    elections_id = "core_" + str(id_)
                    generate_elections(experiment_id, election_model=election_model, election_id=elections_id,
                                          num_voters=model.num_voters,
                                          #num_candidates=model.families[i].num_candidates,
                                          num_candidates=model.num_candidates,
                                          special=param_1)
                    id_ += 1

        else:
            id_ += model.families[i].size


# GENERATE
def generate_elections(experiment_id, election_model=None, election_id=None,
                       num_candidates=None, num_voters=None, special=None):
    """ main function: generate elections"""


    if election_model in  {'identity', 'uniformity', 'antagonism', 'stratification',
                           'unid', 'anid', 'stid', 'anun', 'stun', 'stan',
                           'crate'}:
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

        #print(election_model)

        if election_model == 'mallows' and special == -1:
            special = rand.random()
        elif election_model == 'phi_mallows' and special == -1:
            special = phi_mallows_helper(num_candidates)
        elif election_model == 'phi_mallows' and special >= 0:
            special = phi_mallows_helper(num_candidates, rdis=special)
        elif election_model == 'urn_model' and special == -1:
            special = gamma.rvs(0.8)
        #print(special)

        naked_models = {'impartial_culture': generate_elections_impartial_culture,
                        'impartial_anonymous_culture': generate_elections_impartial_anonymous_culture,
                        'conitzer': generate_elections_conitzer,
                        'spoc_conitzer': generate_elections_spoc_conitzer,
                        'walsh': generate_elections_walsh,
                        'single_crossing': generate_elections_single_crossing,}

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

        single_param_models = {'mallows': generate_elections_mallows,
                               'phi_mallows': generate_elections_mallows,
                               'urn_model': generate_elections_urn_model,}


        if election_model in naked_models:
            votes = naked_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates)

        elif election_model in euclidean_models:
            votes = euclidean_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                                            election_model=election_model)

        elif election_model in single_param_models:
            votes = single_param_models.get(election_model)(num_voters=num_voters, num_candidates=num_candidates,
                                                            param=special)


        path = os.path.join("experiments", str(experiment_id), "elections", "soc_original",
                            (str(election_id) + ".soc"))
        file_ = open(path, 'w')

        if election_model in ["mallows", "urn_model"]:
            file_.write("# " + nice_name(election_model) + " " + str(round(special,5)) + "\n")
        else:
            file_.write("# " + nice_name(election_model) + "\n")

        file_.write(str(num_candidates) + "\n")

        for i in range(num_candidates):
            #file_.write(str(i + 1) + ', ' + chr(97 + i) + "\n")
            file_.write(str(i + 1) + ', c' + str(i) + "\n")


        #print(votes[0])
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
    #print(votes)

    # NEW PART -- I assume there is always single election
    path = os.path.join("experiments", experiment_id, "elections", "soc_original", elections_id + ".soc")
    file_ = open(path, 'w')

    file_.write(str(num_candidates) + "\n")

    for i in range(num_candidates):
        #file_.write(str(i + 1) + ', ' + chr(97 + i) + "\n")
        file_.write(str(i + 1) + ', c' + str(i) + "\n")

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


def generate_elections_impartial_anonymous_culture(num_voters=None, num_candidates=None):
    alpha = 1. / math.factorial(num_candidates)

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    urn_size = 1.
    for j in range(num_voters):
        rho = rand.random()
        if rho <= urn_size:
            votes[j] = np.random.permutation(num_candidates)
        else:
            votes[j] = votes[rand.randint(0, j - 1)]
        urn_size = 1. / (1. + alpha * (j + 1.))

    return votes


def generate_elections_impartial_culture(num_voters=None, num_candidates=None):
    """ helper function: generate impartial culture elections"""

    votes = np.zeros([num_voters, num_candidates], dtype=int)

    for j in range(num_voters):
        votes[j] = np.random.permutation(num_candidates)

    return votes


def generate_elections_conitzer(num_voters=None, num_candidates=None):
    """ helper function: generate conitzer single-peaked elections"""

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    for j in range(num_voters):
        votes[j][0] = rand.randint(0, num_candidates - 1)
        left = votes[j][0] - 1
        right = votes[j][0] + 1
        for k in range(1, num_candidates):
            side = rand.choice([0, 1])
            if side == 0:
                if left >= 0:
                    votes[j][k] = left
                    left -= 1
                else:
                    votes[j][k] = right
                    right += 1
            else:
                if right < num_candidates:
                    votes[j][k] = right
                    right += 1
                else:
                    votes[j][k] = left
                    left -= 1

    return votes


def generate_elections_spoc_conitzer(num_voters=None, num_candidates=None):
    """ helper function: generate spoc_conitzer single-peaked elections"""

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    for j in range(num_voters):
        votes[j][0] = rand.randint(0, num_candidates - 1)
        left = votes[j][0] - 1
        left %= num_candidates
        right = votes[j][0] + 1
        right %= num_candidates
        for k in range(1, num_candidates):
            side = rand.choice([0, 1])
            if side == 0:
                votes[j][k] = left
                left -= 1
                left %= num_candidates
            else:
                votes[j][k] = right
                right += 1
                right %= num_candidates

    return votes


def generate_elections_walsh(num_voters=None, num_candidates=None):
    """ helper function: generate walsh single-peaked elections"""

    votes = np.zeros([num_voters, num_candidates])

    for j in range(num_voters):
        votes[j] = walsh_sp(0, num_candidates - 1)

    return votes


def get_num_changes(num_candidates, PHI):
    power = [pow(PHI, n) for n in range(0, num_candidates + 1)]
    intervals = [sum(power[0:n]) for n in range(num_candidates + 2)]
    answer = np.searchsorted(intervals, (rand.uniform(0, sum(power))), side='left')
    return answer - 1


def mallowsAlternativeVote(base, num_candidates, PHI):
    # num_changes = get_num_changes(num_candidates, PHI)
    num_changes = int(PHI * num_candidates)
    places = rand.sample([c for c in range(num_candidates)], num_changes)
    vote = list(base)
    for p in places:
        r = rand.randint(0, 1)
        if r == 1:  # ma byc
            if p not in vote:
                vote.append(p)
        else:  # ma nie byc
            if p in vote:
                vote.remove(p)
    return vote


def hamming_distance(set_1, set_2):
    return len(set_1) + len(set_2) - 2 * len(set_1.intersection(set_2))


def phi_mallows_helper(num_candidates, rdis=None):

    def nswap(m):
        return int((m * (m - 1)) / 2)

    def generateVoteSwapArray(m):
        S = [[0 for _ in range(int(m * (m + 1) / 2) + 1)] for _ in range(m + 1)]

        for n in range(m + 1):
            for d in range(0, int((n + 1) * n / 2 + 1)):
                if d == 0:
                    S[n][d] = 1
                else:
                    S[n][d] = S[n][d - 1] + S[n - 1][d] - S[n - 1][d - n]
        return S

    def calculateZpoly(m):
        res = [1]
        for i in range(1, m + 1):
            mult = [1] * i
            res2 = [0] * (len(res) + len(mult) - 1)
            for o1, i1 in enumerate(res):
                for o2, i2 in enumerate(mult):
                    res2[o1 + o2] += i1 * i2
            res = res2
        return res

    def calcPolyExp(table, m):
        coeff = (nswap(m) + 1) * [0]
        for i in range(nswap(m) + 1):
            coeff[i] = i * table[i]
        return coeff

    # Function that for some given number of candidates m and some number rDisID\in[0,1] returns the dispersion parameter \phi
    # such that the relative expected distance of the resulting election sampled from Mallows model from identity is rDisID.
    # Table[i] is the number of votes at swap distance i from some m-candidate vote.
    def findPhiFromDistanceID(m, rDisID, table):
        coeffZ = calculateZpoly(m)
        for i in range(len(coeffZ)):
            coeffZ[i] = (rDisID / 2) * nswap(m) * coeffZ[i]
        coeffExp = calcPolyExp(table, m)
        finalcoeff = (nswap(m) + 1) * [0]
        for i in range(nswap(m) + 1):
            finalcoeff[i] = coeffExp[i] - coeffZ[i]
        r = np.roots(list(reversed(finalcoeff)))
        real_valued = r.real[abs(r.imag) < 1e-5]
        res = real_valued[real_valued >= 0]
        return res[0]

    m = num_candidates
    if rdis is None:
        rdis = rand.random()
    S = generateVoteSwapArray(m)
    new_param = findPhiFromDistanceID(m, rdis, S[m])
    #print(new_param)
    return new_param


def generate_elections_mallows(num_voters=None, num_candidates=None, param=None):
    """ helper function: generate mallows elections"""

    PHI = param
    votes = mallowsProfile(num_candidates, num_voters, PHI)

    return votes


def generate_elections_single_crossing(num_voters=None, num_candidates=None):
    """ helper function: generate simple single-crossing elections"""

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    # GENERATE DOMAIN

    domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)

    domain = [[i for i in range(num_candidates)] for _ in range(domain_size)]

    for line in range(1, domain_size):

        poss = []
        for i in range(num_candidates - 1):
            if domain[line - 1][i] < domain[line - 1][i + 1]:
                poss.append([domain[line - 1][i], domain[line - 1][i + 1]])

        r = rand.randint(0, len(poss) - 1)

        for i in range(num_candidates):

            domain[line][i] = domain[line - 1][i]

            if domain[line][i] == poss[r][0]:
                domain[line][i] = poss[r][1]

            elif domain[line][i] == poss[r][1]:
                domain[line][i] = poss[r][0]

    # GENERATE VOTES

    for j in range(num_voters):
        r = rand.randint(0, domain_size - 1)
        votes[j] = list(domain[r])

    return votes


def generate_elections_1d_simple(election_model=None, num_voters=None, num_candidates=None):
    """ helper function: generate simple 1d elections"""

    voters = [0 for _ in range(num_voters)]
    candidates = [0 for _ in range(num_candidates)]
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(num_voters):
        voters[j] = get_rand(election_model)
    voters = sorted(voters)

    for j in range(num_candidates):
        candidates[j] = get_rand(election_model)
    candidates = sorted(candidates)

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(1, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


def generate_elections_2d_simple(election_model=None, num_voters=None, num_candidates=None):
    """ helper function: generate simple 2d elections"""

    voters = [[0, 0] for _ in range(num_voters)]
    candidates = [[0, 0] for _ in range(num_candidates)]

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(num_voters):
        voters[j] = get_rand(election_model)
    voters= sorted(voters)

    for j in range(num_candidates):
        candidates[j] = get_rand(election_model)
    candidates = sorted(candidates)

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(2, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


def generate_elections_nd_simple(election_model=None, num_voters=None, num_candidates=None):
    """ helper function: generate simple nd elections"""

    n_dim = 0

    if election_model == "3d_sphere" or election_model == "3d_cube" or election_model == "3d_ball":
        n_dim = 3
    elif election_model == "4d_sphere" or election_model == "4d_cube" or election_model == "4d_ball":
        n_dim = 4
    elif election_model == "5d_sphere" or election_model == "5d_cube" or election_model == "5d_ball":
        n_dim = 5
    elif election_model == "10d_cube":
        n_dim = 10
    elif election_model == "15d_cube":
        n_dim = 15
    elif election_model == "20d_cube":
        n_dim = 20
    elif election_model == "40d_cube" or election_model == "40d_ball":
        n_dim = 40

    voters = [[0 for _ in range(n_dim)] for _ in range(num_voters)]
    candidates = [[0 for _ in range(n_dim)] for _ in range(num_candidates)]
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    distances = np.zeros([num_voters, num_candidates], dtype=float)

    for j in range(num_voters):
        voters[j] = get_rand(election_model)
    voters = sorted(voters)

    for j in range(num_candidates):
        candidates[j] = get_rand(election_model)
    candidates = sorted(candidates)

    for j in range(num_voters):
        for k in range(num_candidates):
            votes[j][k] = k
            distances[j][k] = distance(n_dim, voters[j], candidates[k])

        votes[j] = [x for _, x in sorted(zip(distances[j], votes[j]))]

    return votes


# REAL
def generate_votes_preflib(elections_model, num_voters=None, num_candidates=None, folder=None,
                           selection_method=None):

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
            print(freq)
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


# AUXILIARY
def random_ball(dimension, num_points=1, radius=1):
    from numpy import random, linalg
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random.random(num_points) ** (1 / dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


def get_rand(elections_model, cat="voters"):
    """ generate random values"""
    if elections_model in {"1d_interval", "1d_interval_bis"}:
        point = rand.random()
    elif elections_model in {"1d_gaussian", "1d_gaussian_bis"}:
        point = rand.gauss(0.5, 0.15)
        while point > 1 or point < 0:
            point = rand.gauss(0.5, 0.15)
    elif elections_model == "1d_one_sided_triangle":
        point = rand.uniform(0, 1) ** (0.5)
    elif elections_model == "1d_full_triangle":
        point = rand.choice([rand.uniform(0, 1) ** (0.5), 2 - rand.uniform(0, 1) ** (0.5)])
    elif elections_model == "1d_two_party":
        point = rand.choice([rand.uniform(0, 1), rand.uniform(2, 3)])
    elif elections_model in {"2d_disc", "2d_range_disc"}:
        phi = 2.0 * 180.0 * rand.random()
        radius = math.sqrt(rand.random()) * 0.5
        point = [0.5 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
    elif elections_model == "2d_range_overlapping":
        phi = 2.0 * 180.0 * rand.random()
        radius = math.sqrt(rand.random()) * 0.5
        if cat == "voters":
            point = [0.25 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]
        elif cat == "candidates":
            point = [0.75 + radius * math.cos(phi), 0.5 + radius * math.sin(phi)]

    elif elections_model in {"2d_square"}:
        point = [rand.random(), rand.random()]

    elif elections_model == "2d_sphere":
        alpha = 2 * math.pi * rand.random()
        x = 1. * math.cos(alpha)
        y = 1. * math.sin(alpha)
        point = [x, y]

    elif elections_model in ["2d_gaussian", "2d_range_gaussian"]:
        point = [rand.gauss(0.5, 0.15), rand.gauss(0.5, 0.15)]
        while distance(2, point, [0.5, 0.5]) > 0.5:
            point = [rand.gauss(0.5, 0.15), rand.gauss(0.5, 0.15)]

    elif elections_model in ["2d_range_fourgau"]:
        r = rand.randint(1, 4)
        size = 0.06
        if r == 1:
            point = [rand.gauss(0.25, size), rand.gauss(0.5, size)]
        if r == 2:
            point = [rand.gauss(0.5, size), rand.gauss(0.75, size)]
        if r == 3:
            point = [rand.gauss(0.75, size), rand.gauss(0.5, size)]
        if r == 4:
            point = [rand.gauss(0.5, size), rand.gauss(0.25, size)]

    elif elections_model == "3d_interval_bis" or elections_model == "3d_cube":
        point = [rand.random(), rand.random(), rand.random()]
    elif elections_model == "3d_gaussian_bis":
        point = [rand.gauss(0.5, 0.15), rand.gauss(0.5, 0.15), rand.gauss(0.5, 0.15)]
        while distance(3, point, [0.5, 0.5, 0.5]) > 0.5:
            point = [rand.gauss(0.5, 0.15), rand.gauss(0.5, 0.15), rand.gauss(0.5, 0.15)]
    elif elections_model == "4d_cube":
        dim = 4
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "5d_cube":
        dim = 5
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "10d_cube":
        dim = 10
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "20d_cube":
        dim = 20
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "40d_cube":
        dim = 40
        point = [rand.random() for _ in range(dim)]
    elif elections_model == "3d_sphere":
        dim = 3
        point = list(random_ball(dim)[0])
    elif elections_model == "4d_sphere":
        dim = 4
        point = list(random_ball(dim)[0])
    elif elections_model == "5d_sphere":
        dim = 5
        point = list(random_ball(dim)[0])
    elif elections_model == "40d_ball":
        dim = 40
        point = list(random_ball(dim)[0])
    else:
        point = [0, 0]
    return point


def walsh_sp(a, b):
    if a == b:
        return [a]
    elif rand.choice([0, 1]) == 0:
        return walsh_sp(a + 1, b) + [a]
    else:
        return walsh_sp(a, b - 1) + [b]


def some_f(ops, i):
    if i == 1.:
        return 1.
    mapping = [1., i - 1.]
    return mapping[ops]


def phiVector(phi, i):
    phi = float(phi)
    den = sum([phi ** j for j in range(i)])
    vec = [(phi ** (i - (j + 1))) / den for j in range(i)]
    return vec


def iDraw(vector):
    x = rand.uniform(0, 1)
    for i in range(len(vector)):
        if (x < sum(vector[0:i + 1])):
            return i


def mallows(center, PHI):
    m = len(center)

    PHI_VEC = [phiVector(PHI, qs + 1) for qs in range(m)]

    r = []
    for i in range(m):
        v = PHI_VEC[i]
        j = iDraw(v)
        r = r[0:j] + [center[i]] + r[j:]
    return r


def mallowsProfile(m, n, PHI):
    V = []
    for i in range(n):
        # print "Mallows %d/%d" % (i, n)
        V += [mallows(range(m), PHI)]
    # return (range(m), [0] * n, V)
    return V


def distance(dim, x_1, x_2):
    """ compute distance between two points """

    if dim == 1:
        return abs(x_1 - x_2)

    output = 0.
    for i in range(dim):
        output += (x_1[i] - x_2[i]) ** 2

    return output ** 0.5


# AUXILIARY (alphas)

def get_vector(type, num_candidates):
    if type == "uniform":
        return [1.] * num_candidates
    elif type == "linear":
        return [(num_candidates - x) for x in range(num_candidates)]
    elif type == "linear_low":
        return [(float(num_candidates) - float(x)) / float(num_candidates) for x in range(num_candidates)]
    elif type == "square":
        return [(float(num_candidates) - float(x)) ** 2 / float(num_candidates) ** 2 for x in
                range(num_candidates)]
    elif type == "square_low":
        return [(num_candidates - x) ** 2 for x in range(num_candidates)]
    elif type == "cube":
        return [(float(num_candidates) - float(x)) ** 3 / float(num_candidates) ** 3 for x in
                range(num_candidates)]
    elif type == "cube_low":
        return [(num_candidates - x) ** 3 for x in range(num_candidates)]
    elif type == "split_2":
        values = [1.] * num_candidates
        for i in range(num_candidates / 2):
            values[i] = 10.
        return values
    elif type == "split_4":
        size = num_candidates / 4
        values = [1.] * num_candidates
        for i in range(size):
            values[i] = 1000.
        for i in range(size, 2 * size):
            values[i] = 100.
        for i in range(2 * size, 3 * size):
            values[i] = 10.
        return values
    else:
        return type
