import mapel


if __name__ == "__main__":

    experiment_id = 'mallows'
    distance_id = 'emd-positionwise'
    instance_type = 'ordinal'

    experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                          distance_id=distance_id,
                                          instance_type=instance_type)

    # generate elections according to map.csv file
    experiment.prepare_elections()

    # compute distances between each pair of elections
    experiment.compute_distances(distance_id=distance_id)





