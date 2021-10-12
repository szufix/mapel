import mapel


if __name__ == "__main__":

    distance_name = 'l1-approvalwise'

    experiment = mapel.prepare_experiment(election_type='approval')

    experiment.set_default_num_candidates(50)
    experiment.set_default_num_voters(100)

    experiment.add_election(model='approval_empty', name='Empty', label='_nolegend_')
    experiment.add_election(model='approval_full', name='Full', label='_nolegend_')
    experiment.add_election(model='approval_ic_0.5', name='IC 0.5', label='_nolegend_')
    experiment.add_election(model='approval_id_0.5', name='ID 0.5', label='_nolegend_')

    experiment.add_election_family(model='approval_ic', color='black', name='IC_path',
                                   params={}, size=10, path={'variable': 'p'})

    experiment.add_election_family(model='approval_ic', color='black', name='IC_path',
                                   params={}, size=10, path={'variable': 'p'})

    experiment.add_election_family(model='approval_id', color='brown', name='ID_path',
                                   params={}, size=10, path={'variable': 'p'})

    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        name = f'Mal_{p}_path'
        experiment.add_election_family(model='approval_mallows', color='blue', name=name,
                                       params={'p': p}, size=10, path={'variable': 'phi'})

    experiment.compute_distances(distance_name=distance_name)

    experiment.embed(algorithm='spring', attraction_factor=2)

    experiment.print_map(shading=True, adjust=True, legend=True,
                         skeleton=['Empty', 'Full', 'IC 0.5', 'ID 0.5'])

