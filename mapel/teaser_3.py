import mapel

if __name__ == "__main__":

    # VARIANT 1
    experiment = mapel.prepare_experiment(experiment_id='kangaroo')
    experiment.prepare_elections()
    experiment.compute_distances()
    experiment.embed()
    experiment.print_map()

    # VARIANT 2
    # experiment = mapel.prepare_experiment(experiment_id='kangaroo', elections='import')
    # experiment.compute_distances()
    # experiment.embed()
    # experiment.print_map()

    # VARIANT 3
    # experiment = mapel.prepare_experiment(experiment_id='kangaroo', elections='import', distances='import')
    # experiment.embed()
    # experiment.print_map()

    # VARIANT 4
    # experiment = mapel.prepare_experiment(experiment_id='kangaroo', elections='import', distances='import',
    #                                       coordinates='import')
    # experiment.print_map()



