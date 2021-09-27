import mapel


if __name__ == "__main__":

    experiment = mapel.prepare_experiment()

    experiment.set_default_num_candidates(20)
    experiment.set_default_num_voters(100)

    experiment.add_family(election_model='approval_ic', color='grey', family_id='ic0.5',
                          params={'p': 0.5}, size=10)

    experiment.add_family(election_model='approval_ic', color='black', family_id='ic0.25',
                          params={'p': 0.25}, size=10)

    experiment.add_family(election_model='approval_ic', color='blue', family_id='ic0.75',
                          params={'p': 0.75}, size=10)

    experiment.add_family(election_model='approval_2d_disc', color='green', family_id='2d',
                          size=10)

    experiment.compute_distances(distance_name='l1-approval_frequency')

    experiment.embed()

    experiment.print_map(legend=True)

