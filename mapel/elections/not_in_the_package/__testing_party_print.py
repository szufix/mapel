import matplotlib.pyplot as plt

import mapel
import math
import os
import csv
import numpy as np

COLORS = {
'plurality': 'black',
             'borda': 'green',
             'stv': 'blue',
             'approx_cc': 'grey',
             'approx_hb': 'orange',
            'approx_pav': 'purple',
            'dhondt': 'red',
}


if __name__ == "__main__":



    methods = ['plurality', 'borda', 'stv', 'approx_cc', 'approx_hb', 'approx_pav', 'dhondt']
    # methods = ['plurality', 'borda']

    x_values = [3, 4, 5, 6, 7]

    election_model = '2d_gaussian_party'

    num_winners = 3  # equivalent to party size
    party_size = num_winners

    num_voters = 100
    size = 20
    precision = 10
    for method in methods:

        y_values = []
        for num_parties in x_values:

            results = np.zeros([precision, num_parties])
            borda = np.zeros([precision, num_parties])

            # READ RESULTS FROM FILE
            file_name = election_model + '_p' + str(num_parties) + '_k' + str(
                num_winners) + '_' + str(method) + '_small'
            path = os.path.join(os.getcwd(), 'party',file_name)
            with open(path, 'r', newline='') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=';')
                for row in reader:
                    results[int(row['iteration'])][int(row['party_id'])] = float(row['sp_weight'])
                    # borda[int(row['iteration'])][int(row['party_id'])] = float(row['borda'])

            # AGREGATE RESULTS
            avg_max = []
            for row in results:
                avg_max.append(max(row))
            avg_max = sum(avg_max) / len(results)
            y_values.append(avg_max)

            # plt.scatter(borda, results)
            # plt.show()

        plt.plot(x_values, y_values, label=method, color=COLORS[method])

    plt.legend()
    plt.xlabel("number of parties")
    plt.ylabel("spoilerity")
    plt.title("weight, c=30, k=3, 2D")

    plt.savefig('spoiler', bbox_inches='tight')
    plt.show()