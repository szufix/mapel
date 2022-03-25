import mapel


def import_experiment():
    experiment_id = 'emd-positionwise'
    instance_type = 'ordinal'
    distance_id = 'emd-positionwise'

    experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                          instance_type=instance_type,
                                          distance_id=distance_id)

    experiment.embed(algorithm='kamada-kawai')
    experiment.print_map()


if __name__ == '__main__':
    import_experiment()
