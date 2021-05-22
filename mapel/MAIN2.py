#!/usr/bin/env python

import mapel
from voting import _tmp


if __name__ == "__main__":

    experiment_id = "comparison/true_swap_10x50"

    # mapel.compute_lowest_dodgson(experiment_id)

    mapel.print_2d(experiment_id, distance_name='swap', metric_name='', attraction_factor=2,
                   values='lowest_dodgson_time')



    #
    # experiment_id = "ijcai/paths+urn+phi_mallows+preflib"
    #
    # mapel.compute_effective_num_candidates(experiment_id)
    #
    # mapel.print_2d(experiment_id, saveas="distances/effective_num_candidates_2", ms=16, values='effective_num_candidates',
    #             attraction_factor=2, distance_name='positionwise', metric_name='emd')