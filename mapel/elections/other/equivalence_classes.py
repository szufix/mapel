#!/usr/bin/env python

import itertools
from matplotlib import pyplot as plt
from matplotlib.text import TextPath
import math
import mapel
import random
import os
import csv

from PIL import Image


from scipy.stats import stats

base = {
    '3x3': 10,
    '3x4': 24,
    '3x5': 42,
    '3x6': 83,
    '3x7': 132,
    '4x3': 111,
    '4x4': 762,
    '4x5': 4095,
    # '7x2': 2636
    # '5x3': 2467,
}

# num_elections = 10

def get_id(name):
    return name.split('_')[-1]


def text_path(name):
    return TextPath((0, 0), get_id(name))

def count_number_of_eq_classes():

    for exp_id in base:
        print("\n\n\n")
        num_elections = base[exp_id]
        # swap_exp = mapel.prepare_experiment(f'abstract_classes/{exp_id}', election_type='ordinal',
        #                                     distance_id='swap_bf')

        # groups_pos = {}
        # for name_1, name_2 in itertools.combinations(swap_exp.elections, 2):
        #     if swap_exp.distances[name_1][name_2] == 0:
        #         name_1 = get_id(name_1)
        #         name_2 = get_id(name_2)
        #         if name_1 in groups_pos:
        #             groups_pos[name_1].append(name_2)
        #         else:
        #             if not any(name_1 in group for group in groups_pos.values()):
        #                 groups_pos[name_1] = [name_1, name_2]
        # groups_pos = [[name.split('_')[-1] for name in group] for group in groups_pos.values()]
        #

        distance_ids = [
            # 'l1-bordawise',
            # 'emd-bordawise',
            # 'l1-pairwise',
            # 'l1-positionwise',
            # 'emd-positionwise',
            # 'swap_bf'
            'discrete'
        ]

        for distance_id in distance_ids:
            target_exp = mapel.prepare_experiment(f'abstract_classes/{exp_id}', instance_type='ordinal',
                                                  distance_id=distance_id)

            groups_pos = {}
            for name_1, name_2 in itertools.combinations(target_exp.elections, 2):
                # print(name_1, name_2)

                if target_exp.distances[name_1][name_2] < 0.0001:
                    # print('tmp')
                    name_1 = get_id(name_1)
                    name_2 = get_id(name_2)
                    if name_1 in groups_pos:
                        groups_pos[name_1].append(name_2)
                    else:
                        if not any(name_1 in group for group in groups_pos.values()):
                            groups_pos[name_1] = [name_1, name_2]

            groups_pos = [[int(name.split('_')[-1]) for name in group] for group in groups_pos.values()]
            flat_groups = [item for sublist in groups_pos for item in sublist]
            # print(flat_groups)
            for i in range(num_elections):
                if i not in flat_groups:
                    groups_pos.append([i])
            # print(groups_pos)

            # for group in groups_pos:
            #     print(group)

            print(distance_id, len(groups_pos))
            #
            # individual = {}
            # individual['color'] = {election_id: 'green' for election_id in target_exp.elections}
            # individual['alpha'] = {election_id: 0.9 for election_id in target_exp.elections}
            # individual['marker'] = {election_id: 'o' for election_id in target_exp.elections}
            # individual['ms'] = {election_id: 30 for election_id in target_exp.elections}
            #
            #
            # individual['color']['all_0'] = 'red'
            # # individual['alpha']['all_0'] = 0.9
            # # individual['marker']['all_0'] = 'o'
            # individual['ms']['all_0'] = 60


            # print(groups_pos)
            # print(groups_swap)

            # print("num of classes", len(groups/_pos), len(groups_pos))
            #
            # ctr = 0
            # for i, group in enumerate(groups_pos):
            #     # print(group)
            #     if group not in groups_pos:
            #         if ctr < 10:
            #             # print(group)
            #             color = COLORS[ctr]
            #             ctr += 1
            #             for element in group:
            #                 election_id = f'all_{int(element)}'
            #                 individual['color'][election_id] = color
            #                 individual['alpha'][election_id] = 0.8
            #                 individual['marker'][election_id] = 'o'
            #                 individual['ms'][election_id] = 30
            #         else:
            #             for element in group:
            #                 election_id = f'all_{int(element)}'
            #                 individual['marker'][election_id] = 'o'


            #
            # target_exp.embed(algorithm='mds')
            # target_exp.print_map(title=f'{exp_id} {distance_id}', ms=50,
            #                      saveas=f'equivalence/{exp_id} {distance_id}',
            #                      individual=individual)


def func_5():
    for exp_id in base:
        list_of_names = []
        for distance_id in distance_ids:
            list_of_names.append(f'images/equivalence/{exp_id} {distance_id}.png')


        images = [Image.open(x) for x in list_of_names]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(f'images/equivalence/{exp_id}.jpg')
    #
    # new_im.save('test.jpg')

    #
    # def pil_grid(images, max_horiz=np.iinfo(int).max):
    #     n_images = len(images)
    #     n_horiz = min(n_images, max_horiz)
    #     h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    #     for i, im in enumerate(images):
    #         h, v = i % n_horiz, i // n_horiz
    #         h_sizes[h] = max(h_sizes[h], im.size[0])
    #         v_sizes[v] = max(v_sizes[v], im.size[1])
    #     h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    #     im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    #     for i, im in enumerate(images):
    #         im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    #     return im_grid
    #
    #
    # imag = pil_grid(dummy_images, 4)

def func_7():

    mses = {
        '3x3': 100,
        '3x4': 80,
        '3x5': 60,
        '4x3': 40,
        '4x4': 20,
    }

    for exp_id in base:
        list_of_names = []
        for distance_id in distance_ids:
            experiment = mapel.prepare_experiment(experiment_id=f'abstract_classes/{exp_id}',
                                                  instance_type='ordinal',
                                                  distance_id=distance_id)

            feature_id = 'swap_distance_from_id'
            title = 'swap_distance_from_id'

            # experiment.compute_feature(feature_id=feature_id)
            # experiment.embed(algorithm='mds')
            experiment.print_map(feature_id=feature_id,
                                 title=f'{exp_id} {distance_id}', ms=mses[exp_id],
                                 adjust_single=True, rounding=0,
                                 xticklabels=['0', ' ', 'max'],
                                 saveas=f'equivalence/{exp_id} {distance_id}', )

            # experiment.embed(algorithm='mds', dim=3)
            # experiment.print_map(dim=3, feature_id=feature_id)
            # experiment.print_map(feature_id=feature_id, adjust=True, rounding=0,
            #                      saveas=f'full_egg/features/{name}_{distance_id}_{feature_id}',
            #                      title=title, show=False, legend=False,
            #                      # normalizing_func=normalizing_function,
            #                      bbox_inches=Bbox([[0 + 1.2, 0.5], [6.4 - 1.2, 6.4]]),
            #                      textual=['ID 0.5', 'IC 0.5', 'empty', 'full']
            #                      # textual=['IC 0.25'])
            #                      # xticklabels=['too slow', '0.0', '17.2', '34.6', '52.0', '69.3'],
            #                      # textual=['IC 0.25', 'ID 0.25', 'empty']
            #                      )



def func_swap():

    for exp_id in base:
        num_elections = base[exp_id]
        swap_exp = mapel.prepare_experiment(f'abstract_classes/{exp_id}', instance_type='ordinal',
                                            distance_id='swap_bf')

        feature_dict = swap_exp.distances['all_0']
        print(feature_dict)

        path = os.path.join(os.getcwd(), "experiments", f'abstract_classes/{exp_id}',
                            "features", f'swap_distance_from_id.csv')
        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(["election_id", "value"])
            writer.writerow(["all_0", "0"])
            ctr = 0
            for key in feature_dict:
                ctr += 1
                writer.writerow([key, feature_dict[key]])

def func_correlation():

    for exp_id in base:
        list_of_names = []
        output = []
        for distance_id in distance_ids:
            experiment_id = f'abstract_classes/{exp_id}'
            names = ['swap_bf', distance_id]
            all_distances = {}

            for name in names:
                experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                                      instance_type='ordinal', distance_id=name)
                all_distances[name] = experiment.distances

            for name_1, name_2 in itertools.combinations(names, 2):
                values_x = []
                values_y = []
                empty_x = []
                empty_y = []
                for e1, e2 in itertools.combinations(all_distances[name_1], 2):

                    values_x.append(all_distances[name_1][e1][e2])
                    values_y.append(all_distances[name_2][e1][e2])

                pear = round(stats.pearsonr(values_x, values_y)[0], 4)
                output.append(f'{exp_id} {distance_id} {pear}')

        print(output)

def func_2():

    for distance_id in distance_ids:
        target_exp = mapel.prepare_experiment(f'abstract_classes/{exp_id}', instance_type='ordinal',
                                              distance_id=distance_id)
        groups_pos = {}
        for name_1, name_2 in itertools.combinations(target_exp.elections, 2):
            if target_exp.distances[name_1][name_2] < 0.000001:
                name_1 = get_id(name_1)
                name_2 = get_id(name_2)
                if name_1 in groups_pos:
                    groups_pos[name_1].append(name_2)
                else:
                    if not any(name_1 in group for group in groups_pos.values()):
                        groups_pos[name_1] = [name_1, name_2]

        groups_pos = [[int(name.split('_')[-1]) for name in group] for group in groups_pos.values()]
        flat_groups = [item for sublist in groups_pos for item in sublist]
        for i in range(num_elections):
            if i not in flat_groups:
                groups_pos.append([i])

        for group in groups_pos:
            # for exp_id in group:
            print(group[0])
            print(target_exp.elections[f'all_{group[0]}'].votes_to_bordawise_vector())


        print(distance_id, len(groups_pos))


def eq_cl():



    # distance_id = 'l1-positionwise'
    exp_id = '5x3'
    exp = mapel.prepare_experiment(f'abstract_classes/{exp_id}', instance_type='ordinal')
    # exp.prepare_elections()

    distance_ids = [
        # 'l1-bordawise',
        'emd-bordawise',
        'emd-positionwise',
        'l1-positionwise',
        'l1-pairwise',
        # 'swap_bf'
        # 'spearman'
    ]

    for distance_id in distance_ids:
        print(distance_id)
        exp.compute_distances(distance_id=distance_id, printing=True)
