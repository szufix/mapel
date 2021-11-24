import matplotlib.pyplot as plt

import mapel
import math
import os
import numpy as np
import itertools
import struct
import csv


def l2(vector_1, vector_2):
    """ compute L2 metric """
    return math.pow(sum([math.pow((vector_1[i] - vector_2[i]), 2)
                         for i in range(len(vector_1))]), 0.5)

def l1(vector_1, vector_2):
    """ compute L2 metric """
    return sum([abs((vector_1[i] - vector_2[i])) for i in range(len(vector_1))])


def banzhaf(rgWeights, fpThold=0.5, normalize=True):
    n = len(rgWeights)
    wSum = sum(rgWeights)
    wThold = fpThold * wSum
    rgWeights = np.array(rgWeights)
    cDecisive = np.zeros(n, dtype=np.uint64)
    for bitmask in range(0, 2**n-1):
        w = rgWeights * np.unpackbits(np.uint8(struct.unpack('B' * (8), np.uint64(bitmask))), bitorder='little')[:n]
        wSum = sum(w)
        if (wSum >= wThold):
            cDecisive = np.add(cDecisive, (w > (wSum - wThold)))
    phi = cDecisive / (2**n) * 2
    if (normalize):
        return phi / sum(phi)
    else:
        return phi


def shapley(rgWeights, fpThold=0.5):
    n = len(rgWeights)
    wSum = sum(rgWeights)
    wThold = fpThold * wSum
    rgWeights = np.array(rgWeights)
    cDecisive = np.zeros(n, dtype=np.uint64)
    for perm in itertools.permutations(range(n)):
        w = rgWeights[list(perm)]
        dec = 0
        wSum = w[dec]
        while wSum < wThold:
            dec += 1
            wSum += w[dec]
        cDecisive[perm[dec]] += 1
    return cDecisive / sum(cDecisive)


def potLadle(v, m):
    vv = sorted(v, reverse=True)
    cumv = np.cumsum(vv)
    n = 0
    while n < len(v) and vv[n] / cumv[n] >= 1 / (2 * m + n):
        n += 1
    v = np.array(v)
    vv = v / cumv[n-1]
    s = vv * (1 + n/(2*m)) - 1/(2*m)
    s[s < 0] = 0
    return list(s)


if __name__ == "__main__":

    election_model = '2d_gaussian_party'

    methods = ['plurality', 'borda', 'stv', 'approx_cc', 'approx_hb', 'approx_pav', 'dhondt']
    # methods = ['dhondt']

    num_winners = 3  # equivalent to party size
    party_size = num_winners

    num_voters = 100
    size = 100
    precision = 10

    for method in methods:

        result_max = []
        result_avg = []
        for num_parties in [3, 4, 5, 6, 7]:
            print(method, num_parties)
            num_candidates = num_parties * num_winners

            all_x = []
            all_y = []
            max_es = []
            avg_es = []
            all_weight_spoilers = []
            all_weight_vectors = []
            all_shapley_spoilers = []
            all_shapley_vectors = []
            all_banzhaf_spoilers = []
            all_banzhaf_vectors = []
            all_borda = []

            all_alternative_weight_vectors = []
            all_alternative_shapley_vectors = []
            all_alternative_banzhaf_vectors = []

            for _ in range(precision):
                experiment = mapel.prepare_experiment()
                experiment.set_default_num_candidates(num_candidates)
                experiment.set_default_num_voters(num_voters)

                experiment.add_family(election_model=election_model, size=size,
                                      params={'num_winners': num_winners,
                                              'num_parties': num_parties,
                                              'main-phi': 0.5,
                                              'norm-phi': 0.5})

                experiment.compute_winners(method=method,
                                           num_winners=num_winners)

                # do poprawy
                if method == 'dhondt':
                    # main election
                    parties = [0 for _ in range(num_parties)]
                    for election_id in experiment.elections:
                        for vote in experiment.elections[election_id].votes:
                            parties[int(vote[0]/party_size)] += 1
                    denominator = sum(parties)
                    for i in range(num_parties):
                        parties[i] /= denominator
                    # print(parties, num_winners)
                    weight_vector = potLadle(parties, num_winners)

                    # alternative elections
                    alternative_weight_vectors = [[] for _ in range(num_parties)]
                    for c in range(num_parties):
                        parties = [0 for _ in range(num_parties)]
                        for election_id in experiment.elections:
                            for vote in experiment.elections[election_id].votes:
                                ctr = 0
                                while int(vote[ctr]/party_size) == c:
                                    ctr += 1
                                parties[int(vote[ctr]/party_size)] += 1

                        denominator = sum(parties)
                        for i in range(num_parties):
                            parties[i] /= denominator
                        alternative_weight_vectors[c] = potLadle(parties, num_winners)
                    # print(alternative_weight_vector)

                else:

                    experiment.compute_alternative_winners(method=method,
                                                           num_winners=num_winners,
                                                           num_parties=num_parties)

                    weight_vector = [0 for _ in range(num_parties)]
                    alternative_weight_vectors = {}
                    for party_id in range(num_parties):
                        alternative_weight_vectors[party_id] = [0 for i in range(num_parties)]

                    for election_id in experiment.elections:
                        for w in experiment.elections[election_id].winners:
                            weight_vector[int(w / party_size)] += 1
                        for party_id in range(num_parties):
                            for w in experiment.elections[election_id].alternative_winners[party_id]:
                                alternative_weight_vectors[party_id][int(w / party_size)] += 1

                borda = [0 for _ in range(num_parties)]
                for election_id in experiment.elections:
                    for i in range(num_candidates):
                        borda[int(i / party_size)] += experiment.elections[election_id].borda_points[i]

                denominator = sum(borda)
                for party_id in range(num_parties):
                    borda[party_id] /= denominator

                denominator = sum(weight_vector)
                for i in range(num_parties):
                    weight_vector[i] /= denominator

                for spoiler_id in range(num_parties):
                    denominator = sum(alternative_weight_vectors[spoiler_id])
                    for party_id in range(num_parties):
                        alternative_weight_vectors[spoiler_id][party_id] /= denominator

                #
                # print(weight_vector)
                # print(alternative_weight_vector)

                # weight_vector = shapley(weight_vector)

                # WEIGHT
                weight_spoilers = []
                for party_id in range(num_parties):
                    value = l1(weight_vector, alternative_weight_vectors[party_id])
                    value -= 2*weight_vector[party_id]
                    weight_spoilers.append(value / (value + borda[party_id]))
                all_weight_spoilers.append(weight_spoilers)
                all_weight_vectors.append(weight_vector)

                # print(max(spo/ilers_weight))

                # SHAPLEY
                shapley_spoilers = []
                shapley_vector = shapley(weight_vector)
                alternative_shapley_vectors = []
                for party_id in range(num_parties):
                    alternative_shapley_vector = shapley(alternative_weight_vectors[party_id])
                    alternative_shapley_vectors.append(alternative_shapley_vector)
                    value = l1(shapley_vector, alternative_shapley_vector)
                    value -= 2 * shapley_vector[party_id]
                    shapley_spoilers.append(value / (value + borda[party_id]))
                all_shapley_spoilers.append(shapley_spoilers)
                all_shapley_vectors.append(shapley_vector)

                # BANZHAF
                banzhaf_spoilers = []
                banzhaf_vector = banzhaf(weight_vector)
                alternative_banzhaf_vectors = []
                for party_id in range(num_parties):
                    alternative_banzhaf_vector = banzhaf(
                        alternative_weight_vectors[party_id])
                    alternative_banzhaf_vectors.append(
                        alternative_banzhaf_vector)
                    value = l1(banzhaf_vector, alternative_banzhaf_vector)
                    value -= 2 * banzhaf_vector[party_id]
                    banzhaf_spoilers.append(value / (value + borda[party_id]))

                all_banzhaf_spoilers.append(banzhaf_spoilers)
                all_banzhaf_vectors.append(banzhaf_vector)


                # BORDA
                all_borda.append(borda)

                # BIF FILE ALL
                all_alternative_weight_vectors.append(alternative_weight_vectors)
                all_alternative_shapley_vectors.append(alternative_shapley_vectors)
                all_alternative_banzhaf_vectors.append(alternative_banzhaf_vectors)

                # print(all_spoilers_weight)

                # for i in range(len(spoilers_weight)):
                #     all_y.append(spoilers_weight[i])
                #     all_x.append(borda[i])

            #     max_spoiler = max(spoilers)
            #     avg_spoiler = sum(spoilers)/num_parties
            #     max_es.append(max_spoiler)
            #     avg_es.append(avg_spoiler)
            # result_max.append(sum(max_es)/float(precision))
            # result_avg.append(sum(avg_es)/float(precision))

            # plt.scatter(all_x, all_y)
            # plt.show()


            # SMALL
            file_name = election_model + '_p' + str(num_parties) + '_k' + str(num_winners) + '_' + str(method) + '_small'
            path = os.path.join(os.getcwd(), 'party', file_name)
            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(["iteration", "party_id",
                                 "weight",  "sp_weight",
                                 "shapley", "sp_shapley",
                                 "banzhaf", "sp_banzhaf",
                                 "borda"])

                for j in range(len(all_weight_vectors)):
                    for p in range(num_parties):
                        writer.writerow([j, p,
                                         all_weight_vectors[j][p],
                                         all_weight_spoilers[j][p],
                                         all_shapley_vectors[j][p],
                                         all_shapley_spoilers[j][p],
                                         all_banzhaf_vectors[j][p],
                                         all_banzhaf_spoilers[j][p],
                                         all_borda[j][p]])

            # BIG
            file_name = election_model + '_p' + str(num_parties) + '_k' + str(num_winners) + '_' + str(method) + '_big'
            path = os.path.join(os.getcwd(), 'party', file_name)
            with open(path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(["iteration", "spoiler_id", 'party_id',
                                 "weight", "shapley", "banzhaf"])

                for j in range(len(all_weight_vectors)):

                    for party_id in range(num_parties):
                        writer.writerow([j, "NULL", party_id,
                                         all_weight_vectors[j][party_id],
                                         all_shapley_vectors[j][party_id],
                                         all_banzhaf_vectors[j][party_id]])

                    for spoiler_id in range(num_parties):
                        for party_id in range(num_parties):
                            writer.writerow([j, spoiler_id, party_id,
                                all_alternative_weight_vectors[j][spoiler_id][party_id],
                                all_alternative_shapley_vectors[j][spoiler_id][party_id],
                                all_alternative_banzhaf_vectors[j][spoiler_id][party_id]])

