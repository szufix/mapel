#!/usr/bin/env python

import mapel

if __name__ == "__main__":

    experiment_id = 'tmp_ordinal'
    distance_name = 'emd-positionwise'

    experiment = mapel.prepare_experiment(experiment_id=experiment_id, shift=False,
                                          election_type='ordinal',
                                          elections='import',
                                          distances='import')

    # experiment.prepare_elections()

    # experiment.compute_distances(distance_name=distance_name)

    experiment.embed(algorithm='spring')

    experiment.print_map(saveas='saveas', title='title', shift_legend=0.8, adjust=True)


