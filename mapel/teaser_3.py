import mapel

if __name__ == "__main__":

    experiment_id = 'test'

    mapel.create_structure(experiment_id)
    mapel.prepare_elections(experiment_id)
    # mapel.prepare_matrices(experiment_id)
    mapel.compute_distances(experiment_id)
    mapel.convert_xd_to_2d(experiment_id)
    mapel.print_2d(experiment_id)
