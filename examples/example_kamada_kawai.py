import mapel
from mapel.main.features.distortion import calculate_distortion
from mapel.main.features.monotonicity import calculate_monotonicity


def import_experiment():
    experiment_id = 'emd-positionwise'
    instance_type = 'ordinal'
    distance_id = 'emd-positionwise'

    experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                          instance_type=instance_type,
                                          distance_id=distance_id)

    # experiment.embed(algorithm='kamada-kawai')
    # experiment.print_map()
    calculate_monotonicity(experiment)#, ['Impartial Culture_0', 'Impartial Culture_1', 'Impartial Culture_4', 'Impartial Culture_5'])


if __name__ == '__main__':
    import_experiment()
