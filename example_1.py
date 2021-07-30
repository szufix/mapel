import mapel

if __name__ == "__main__":
    experiment = mapel.prepare_experiment()

    experiment.set_default_num_candidates(8)
    experiment.set_default_num_voters(50)

    experiment.add_election(election_model='identity', election_id='id')

    experiment.add_family(election_model='impartial_culture', size=20, color='blue')

    experiment.add_family(election_model='1d_interval', size=20, color='green')

    experiment.compute_distances()

    experiment.embed()

    experiment.print_map(title='My First Map', ms=30, legend=True)
