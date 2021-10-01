import mapel


if __name__ == "__main__":

    distance_name = 'emd-positionwise'

    experiment = mapel.prepare_experiment()

    experiment.set_default_num_candidates(20)
    experiment.set_default_num_voters(100)

    experiment.add_family(model='impartial_culture', color='grey', name='ic', size=10)

    experiment.add_family(model='urn_model', color='black', name='urn',
                          params={'alpha': 0.2}, size=10)

    experiment.add_family(model='norm-mallows', color='blue', name='mallows',
                          params={'norm-phi': 0.5}, size=10)

    experiment.compute_distances(distance_name=distance_name)

    experiment.embed()

    experiment.print_map(legend=True)

