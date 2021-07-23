import math
import os
import time
#import concurrent.futures
import itertools
import matplotlib.pyplot as plt

from threading import Thread
import numpy as np

from mapel.voting import elections as el
from mapel.voting import metrics as metr
from mapel.voting import objects as obj

# KIND OF A BACKUP
def print_map(self, mask=False, mixed=False, fuzzy_paths=True, xlabel=None,
              angle=0, reverse=False, update=False, feature=None, attraction_factor=1, axis=False,
              distance_name="emd-positionwise", guardians=False, tmp2=[1, 1, 1], zorder=[1, 1, 1], ticks=None,
              title=None,
              saveas="map_2d", show=True, ms=20, normalizing_func=None, xticklabels=None, cmap=None,
              ignore=None, marker_func=None, tex=False, black=False, legend=True, levels=False, tmp=False):
    """ Print the two-dimensional embedding of multi-dimensional map of the elections """

    self.compute_points_by_families()

    if angle != 0:
        self.rotate(angle)

    if reverse:
        self.reverse()

    if update:
        self.update()

    if cmap is None:
        cmap = pr.custom_div_cmap()

    if feature is not None:
        fig = plt.figure(figsize=(6.4, 4.8 + 0.48))
    else:
        fig = plt.figure()
    ax = fig.add_subplot()

    if not axis:
        plt.axis('off')

    # COLORING
    if feature is not None:
        pr.color_map_by_feature(fig=fig, ax=ax, experiment=self, experiment_id=self.experiment_id,
                                feature=feature, normalizing_func=normalizing_func, marker_func=marker_func,
                                xticklabels=xticklabels, ms=ms, cmap=cmap, ticks=ticks)
    elif tmp:
        pr.ijcai_2021_coloring_1(ax=ax, experiment=self, ms=ms, tmp=tmp, tmp2=tmp2, zorder=zorder)
    elif mixed:
        pr.ijcai_2021_coloring_2(ax=ax, experiment=self, ms=ms, tmp=tmp, tmp2=tmp2, zorder=zorder,
                                 fuzzy_paths=fuzzy_paths)
    else:
        pr.basic_coloring(ax=ax, experiment=self, ms=ms)

    # BACKGROUND
    if mask:
        pr.mask_background(fig=fig, ax=ax, black=black, saveas=saveas, tex=tex)
    elif levels:
        pr.level_background(fig=fig, ax=ax, saveas=saveas, tex=tex)
    else:
        pr.basic_background(ax=ax, values=feature, legend=legend, saveas=saveas, xlabel=xlabel,
                            title=title)

    if guardians:
        pr.add_guardians_to_picture(self.experiment_id, ax=ax, values=feature, legend=legend, saveas=saveas)

    if tex:
        pr.saveas_tex(saveas=saveas)

    if show:
        plt.show()


# TO BE UPDATED
def compute_highest_copeland_map(experiment_id):
    model = obj.Experiment(experiment_id)

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
    model = obj.Experiment(experiment_id)
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
    model = obj.Experiment(experiment_id)

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
    model = obj.Experiment(experiment_id)

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
    model = obj.Experiment(experiment_id)

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
    model = obj.Experiment(experiment_id)

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
    model = obj.Experiment(experiment_id)

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
    model = obj.Experiment(experiment_id)

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

        model = obj.Experiment(experiment_id)

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



def paths_for_emd_positionwise(m=10, scale=1.):

    unid = 1/3 * (m*m -1)
    anid = m*m/4
    stid = 2/3 * (m*m/4 - 1)
    anun = 2/3 * (m*m/4 - 1)
    stun = m*m/4
    stan = 13/48 * m*m - 1/3

    unid = round(unid*scale, 0)
    anid = round(anid*scale, 0)-2
    stid = round(stid*scale, 0)-2
    anun = round(anun*scale, 0)-2
    stun = round(stun*scale, 0)-2
    stan = round(stan*scale, 0)

    total = unid+anid+stid+anun+stun+stan

    print('\nPaths for emd-Positionwise:')

    print('unid: ', unid)
    print('anid: ', anid)
    print('stid: ', stid)
    print('anun: ', anun)
    print('stun: ', stun)
    print('stan: ', stan)

    print('total:', total)


def paths_for_l1_positionwise(m=10, scale=1.):

    unid = 2*(m-1)
    anid = m/2
    stid = 2*(m-2)
    anun = 2*(m-2)
    stun = m/2
    stan = 2*(m-2)

    unid = round(unid*scale, 0)
    anid = round(anid*scale, 0)-2
    stid = round(stid*scale, 0)-2
    anun = round(anun*scale, 0)-2
    stun = round(stun*scale, 0)-2
    stan = round(stan*scale, 0)

    total = unid+anid+stid+anun+stun+stan

    print('\nPATHS for emd-Positionwise:')

    print('unid: ', unid)
    print('anid: ', anid)
    print('stid: ', stid)
    print('anun: ', anun)
    print('stun: ', stun)
    print('stan: ', stan)

    print('total:', total)


def paths_for_emd_bordawise(m=10, scale=1.):

    idun = 1/12 * m * (m*m - 1)
    stid = 1/48 * m * (m*m + 3*m - 4)
    stun = 1/16 * m*m * (m - 1)

    idun = round(idun*scale, 0)
    stid = round(stid*scale, 0)-1
    stun = round(stun*scale, 0)-2

    total = idun+stid+stun

    print('\nPATHS for emd-Bordawise:')

    print('idun: ', idun)
    print('stid: ', stid)
    print('stun: ', stun)

    print('total:', total)


def paths_for_l1_bordawise(m=10, scale=1.):

    idun = 1/4 * m * m
    stid = 1/8 * m * (m - 1)
    stun = 1/4 * m * (m - 1)

    idun = round(idun*scale, 0)
    stid = round(stid*scale, 0)-1
    stun = round(stun*scale, 0)-2

    total = idun+stid+stun

    print('\nPATHS for l1-Bordawise:')

    print('idun: ', idun)
    print('stid: ', stid)
    print('stun: ', stun)

    print('total:', total)


def paths_for_pairwise(m=10, scale=1.):

    idun = 1/2 * m * (m-1)
    stid = 1/4 * m * (m-2)
    stun = 1/4 * m*m

    idun = round(idun*scale, 0)
    stid = round(stid*scale, 0)-1
    stun = round(stun*scale, 0)-2

    total = idun+stid+stun

    print('\nPATHS for Pairwise:')

    print('idun: ', idun)
    print('stid: ', stid)
    print('stun: ', stun)

    print('total:', total)


def precompute_phi_params():

    for special in {0.2}:

        for num_candidates in range(5,105,5):
            norm_phi = el.phi_mallows_helper(num_candidates, rdis=special)

            with open('mallows_helper.txt', 'a') as file_txt:
                line = '(' + str(num_candidates) + ', ' + str(special) + '): ' + str(round(norm_phi, 10)) + ',\n'
                file_txt.write(line)


### AFTER 01.06.2021 ###

def single_peaked_special(experiment_id):

    num_sets = 15

    model = obj.Model_xd(experiment_id, distance_name='positionwise', metric_name='emd')

    order = {'identity': 0, 'uniformity': 1, 'antagonism': 2, 'stratification': 3,
             'walsh_fake': 4, 'conitzer_fake': 5}

    x_values = [10*i for i in range(1, num_sets+1)]

    for name_1 in order:
        for name_2 in order:
            if order[name_1] >= order[name_2]:
                continue

            y_values = []
            for i in range(num_sets):
                a = num_sets*order[name_1]+i
                b = num_sets*order[name_2]+i
                value = model.distances[a][b] / metr.map_diameter(x_values[i])
                y_values.append(value)

            plt.plot(x_values, y_values)
            title = str(name_1) + '-' + str(name_2)
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            plt.title(title)

            file_name = title + ".png"
            path = os.path.join(os.getcwd(), "images", "single-peaked", file_name)
            plt.savefig(path, bbox_inches='tight')

            plt.show()


def single_peaked_special_double(experiment_id):

    num_sets = 15

    model = obj.Model_xd(experiment_id, distance_name='positionwise', metric_name='emd')

    order = {'identity': 0, 'uniformity': 1, 'antagonism': 2, 'stratification': 3,
             'walsh_fake': 4, 'conitzer_fake': 5}

    x_values = [10*i for i in range(1, num_sets+1)]

    name_1 = 'walsh_fake'
    name_2b = 'identity'
    name_2c = 'uniformity'

    y_values = []
    for i in range(num_sets):
        a = num_sets*order[name_1]+i
        b = num_sets*order[name_2b]+i
        c = num_sets*order[name_2c]+i
        value = model.distances[a][b] / metr.map_diameter(x_values[i])
        value += model.distances[a][c] / metr.map_diameter(x_values[i])
        y_values.append(value)

    plt.plot(x_values, y_values, label='IDUN')

    name_1 = 'walsh_fake'
    name_2b = 'antagonism'
    name_2c = 'stratification'

    y_values = []
    for i in range(num_sets):
        a = num_sets*order[name_1]+i
        b = num_sets*order[name_2b]+i
        c = num_sets*order[name_2c]+i
        value = model.distances[a][b] / metr.map_diameter(x_values[i])
        value += model.distances[a][c] / metr.map_diameter(x_values[i])
        y_values.append(value)

    plt.plot(x_values, y_values, label='STAN')


    plt.yticks([0, 0.25, 0.5, 0.75, 1, 1.25])
    plt.legend()

    title = 'walsh_vs_all'
    plt.title(title)

    file_name = title + ".png"
    path = os.path.join(os.getcwd(), "images", "single-peaked", file_name)
    plt.savefig(path, bbox_inches='tight')

    plt.show()


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

#
# if election_model == 'crate':
                #     my_size = 9
                #     # with_edge
                #     for p in range(my_size):
                #         for q in range(my_size):
                #             for r in range(my_size):
                #                 a = p / (my_size - 1)
                #                 b = q / (my_size - 1)
                #                 c = r / (my_size - 1)
                #                 d = 1 - a - b - c
                #                 tmp = [a, b, c, d]
                #                 if d >= 0 and sum(tmp) == 1:
                #                     base.append(tmp)
                #     # without_edge
                #     """
                #     for p in range(1, my_size-1):
                #         for q in range(1, my_size-1):
                #             for r in range(1, my_size-1):
                #                 a = p / (my_size - 1)
                #                 b = q / (my_size - 1)
                #                 c = r / (my_size - 1)
                #                 d = 1 - a - b - c
                #                 tmp = [a, b, c, d]
                #                 if d >= 0 and sum(tmp) == 1:
                #                     #print(tmp)
                #                     base.append(tmp)
                #     """
                # #print(len(base))