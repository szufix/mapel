import mapel

if __name__ == "__main__":

    # VARIANT 1
    experiment = mapel.prepare_experiment(experiment_id='tmp_example')
    experiment.prepare_elections()
    experiment.compute_distances()
    experiment.embed()
    experiment.compute_feature('highest_borda_score')
    cmap = mapel.custom_div_cmap(colors=["orange", "red", "purple", "black"], num_colors=11)
    experiment.print_map(feature='highest_borda_score', cmap=cmap)

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



