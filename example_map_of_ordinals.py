import mapel


if __name__ == "__main__":

    distance_id = 'emd-positionwise'

    experiment = mapel.prepare_experiment()

    experiment.set_default_num_candidates(20)
    experiment.set_default_num_voters(100)

    experiment.add_election_family(model_id='impartial_culture', color='grey', election_id='ic',
                                   size=10)

    experiment.add_election_family(model_id='urn_model', color='black', election_id='urn',
                                   params={'alpha': 0.2}, size=10)

    experiment.add_election_family(model_id='norm-mallows', color='blue', election_id='mallows',
                                   params={'norm-phi': 0.5}, size=10)

    experiment.compute_distances(distance_id=distance_id)

    experiment.embed()

    experiment.print_map(legend=True)

