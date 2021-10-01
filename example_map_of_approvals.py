import mapel


if __name__ == "__main__":

    distance_name = 'l1-approval_frequency'

    experiment = mapel.prepare_experiment()

    experiment.set_default_num_candidates(20)
    experiment.set_default_num_voters(100)

    experiment.add_family(model='approval_ic', color='grey', name='ic0.5',
                          params={'p': 0.5}, size=10)

    experiment.add_family(model='approval_ic', color='black', name='ic0.25',
                          params={'p': 0.25}, size=10)

    experiment.add_family(model='approval_ic', color='blue', name='ic0.75',
                          params={'p': 0.75}, size=10)

    experiment.compute_distances(distance_name=distance_name)

    experiment.embed()

    experiment.print_map(legend=True)

