
import numpy as np

from mapel.core.inner_distances import l2

from itertools import combinations

def map_diameter(c: int) -> float:
    """ Compute the diameter """
    return 1. / 3. * (c + 1) * (c - 1)

def distortion_from_guardians(experiment, election_id) -> np.ndarray:
    values = np.array([])
    election_id_1 = election_id

    for election_id_2 in experiment.elections:
        if election_id_2 in {'identity_10_100_0', 'uniformity_10_100_0',
                             'antagonism_10_100_0', 'stratification_10_100_0'}:
            if election_id_1 != election_id_2:
                m = experiment.elections[election_id_1].num_candidates
                true_distance = experiment.distances[election_id_1][election_id_2]
                true_distance /= map_diameter(m)
                embedded_distance = l2(experiment.coordinates[election_id_1],
                                       experiment.coordinates[election_id_2])

                embedded_distance /= \
                    l2(experiment.coordinates['identity_10_100_0'],
                       experiment.coordinates['uniformity_10_100_0'])
                ratio = float(true_distance) / float(embedded_distance)
                values = np.append(values, ratio)

    return values


def distortion_from_all(experiment, election_ids):

    all_values = {}

    for election_id in election_ids:

        values = np.array([])
        one_side_values = np.array([])
        election_id_1 = election_id

        for election_id_2 in experiment.instances:
            # if election_id_2 in {'identity_10_100_0', 'uniformity_10_100_0',
            #                      'antagonism_10_100_0', 'stratification_10_100_0'}:
            if election_id_1 != election_id_2:
                # m = experiment.instances[election_id_1].num_candidates
                # print(election_id_1, election_id_2)
                true_distance = experiment.distances[election_id_1][election_id_2]
                true_distance /= experiment.distances['ID']['UN']
                embedded_distance = l2(np.array(experiment.coordinates[election_id_1]),
                                       np.array(experiment.coordinates[election_id_2]))

                embedded_distance /= \
                    l2(np.array(experiment.coordinates['ID']),
                       np.array(experiment.coordinates['UN']))
                # try:
                #     ratio = float(embedded_distance) / float(true_distance)
                # except:
                #     ratio = 1.
                one_side_ratio = embedded_distance / true_distance
                one_side_values = np.append(one_side_values, one_side_ratio)

                ratio = max(embedded_distance, true_distance) / min(embedded_distance, true_distance)

                values = np.append(values, ratio)

        # print(min(one_side_values), max(one_side_values))
        # if election_id_1 == 'IC_0':
        #     print(values)

        all_values[election_id_1] = np.mean(values)

    return all_values


# def distortion_from_all(experiment, election_id) -> np.ndarray:
#     values = np.array([])
#     election_id_1 = election_id
#
#     for election_id_2 in experiment.elections:
#         if election_id_1 != election_id_2:
#             m = experiment.elections[election_id_1].num_candidates
#             true_distance = experiment.distances[election_id_1][election_id_2]
#             true_distance /= map_diameter(m)
#             embedded_distance = l2(np.array(experiment.coordinates[election_id_1]),
#                                    np.array(experiment.coordinates[election_id_2]))
#
#             embedded_distance /= \
#                 l2(np.array(experiment.coordinates['core_800']),
#                    np.array(experiment.coordinates['core_849']))
#             try:
#                 ratio = float(embedded_distance) / float(true_distance)
#             except:
#                 ratio = 1.
#             values = np.append(values, ratio)
#
#     return np.mean(abs(1.-values))


# def distortion_from_top_100(experiment, election_id) -> np.ndarray:
#     values = np.array([])
#     election_id_1 = election_id
#
#     euc_dist = {}
#     for election_id_2 in experiment.elections:
#         if election_id_1 != election_id_2:
#             euc_dist[election_id_2] = l2(np.array(experiment.coordinates[election_id_1]),
#                                            np.array(experiment.coordinates[election_id_2]))
#
#     all = (sorted(euc_dist.items(), key=lambda item: item[1]))
#     top_100 = [x for x,_ in all[0:100]]
#
#
#     # all = (sorted(experiment.distances[election_id_1].items(), key=lambda item: item[1]))
#     # top_100 = [x for x,_ in all[0:100]]
#
#     for election_id_2 in experiment.elections:
#         if election_id_1 != election_id_2:
#             if election_id_2 in top_100:
#                 m = experiment.elections[election_id_1].num_candidates
#                 true_distance = experiment.distances[election_id_1][election_id_2]
#                 true_distance /= map_diameter(m)
#                 embedded_distance = l2(np.array(experiment.coordinates[election_id_1]),
#                                        np.array(experiment.coordinates[election_id_2]))
#
#                 embedded_distance /= \
#                     l2(np.array(experiment.coordinates['core_800']),
#                        np.array(experiment.coordinates['core_849']))
#                 try:
#                     ratio = float(embedded_distance) / float(true_distance)
#                 except:
#                     ratio = 1.
#                 values = np.append(values, ratio)
#
#     return np.mean(abs(1.-values))


def avg_distortion_from_guardians(experiment, election_id):
    values = distortion_from_guardians(experiment, election_id)
    return np.mean(values)


def worst_distortion_from_guardians(experiment, election_id):
    values = distortion_from_guardians(experiment, election_id)
    return np.max(values)



def distortion(experiment, election) -> float:
    e0 = election.election_id
    c0 = np.array(experiment.coordinates[e0])
    distortion = 0
    for e1, e2 in combinations(experiment.elections, 2):
        if e1 != e0 and e2 != e0:
            original_d1 = experiment.distances[e0][e1]
            original_d2 = experiment.distances[e0][e2]
            original_proportion = original_d1 / original_d2
            embedded_d1 = np.linalg.norm(c0 - experiment.coordinates[e1], ord=2)
            embedded_d2 = np.linalg.norm(c0 - experiment.coordinates[e2], ord=2)
            embedded_proportion = embedded_d1 / embedded_d2
            _max = max(original_proportion, embedded_proportion)
            _min = min(original_proportion, embedded_proportion)
            distortion += _max / _min
    return distortion


def monotonicity_triplets(experiment, election) -> float:
    epsilon = 0.1
    e0 = election.election_id
    c0 = np.array(experiment.coordinates[e0])
    distortion = 0.
    ctr = 0.
    for e1, e2 in combinations(experiment.elections, 2):
        if e1 != e0 and e2 != e0:
            original_d1 = experiment.distances[e0][e1]
            original_d2 = experiment.distances[e0][e2]
            embedded_d1 = np.linalg.norm(c0 - experiment.coordinates[e1], ord=2)
            embedded_d2 = np.linalg.norm(c0 - experiment.coordinates[e2], ord=2)
            if (original_d1 < original_d2 and embedded_d1 > embedded_d2 * (1. + epsilon)) or \
                    (original_d2 < original_d1 and embedded_d2 > embedded_d1 * (1. + epsilon)):
                distortion += 1.
            ctr += 1.
    distortion /= ctr
    print(distortion)
    return distortion
