#!/usr/bin/env python3

import mapel.elections as mapel

from mapel.elections import prepare_offline_ordinal_experiment
from mapel.main.features.distortion import calculate_distortion
from mapel.main.features.monotonicity import calculate_monotonicity
from mapel.main.features.stability import calculate_stability


def import_experiment():
    experiment_id = 'kamada-kawai_example'
    distance_id = 'emd-positionwise'

    experiment = prepare_offline_ordinal_experiment(experiment_id=experiment_id,
                                          distance_id=distance_id)

    experiment.prepare_elections()
    experiment.compute_distances()
    experiment.embed(algorithm='kamada-kawai')
    experiment.print_map()
    
    # To be fixed: the following function does not work at the moment as we do
    # not have more lists of coordinates
    #print(calculate_stability(experiment))

if __name__ == '__main__':
    import_experiment()
