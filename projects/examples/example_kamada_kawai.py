import mapel
from mapel.main.features.distortion import calculate_distortion
from mapel.main.features.monotonicity import calculate_monotonicity
from mapel.main.features.stability import calculate_stability


def import_experiment():
    experiment_id = 'emd-positionwise'
    instance_type = 'ordinal'
    distance_id = 'emd-positionwise'

    experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                          instance_type=instance_type,
                                          distance_id=distance_id,
                                          coordinates_names=[
                                              'emd-positionwise-paths-big-fixed-4_circle_kamada_bb_2_steps.csv',
                                              'emd-positionwise-paths-big-fixed-4_square_kamada_bb_2_steps.csv'
                                          ])

    experiment.embed(algorithm='kamada-kawai')
    experiment.print_map()
    print(calculate_stability(experiment))


if __name__ == '__main__':
    import_experiment()
