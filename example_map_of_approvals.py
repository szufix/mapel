import mapel


if __name__ == "__main__":

    distance_id = 'l1-approvalwise'

    experiment = mapel.prepare_experiment(election_type='approval')

    experiment.set_default_num_candidates(50)
    experiment.set_default_num_voters(100)

    experiment.add_election(model_id='approval_empty', election_id='Empty', label='_nolegend_')
    experiment.add_election(model_id='approval_full', election_id='Full', label='_nolegend_')
    experiment.add_election(model_id='approval_ic', election_id='IC 0.5', label='_nolegend_',
                            params={'p': .5})
    experiment.add_election(model_id='approval_id', election_id='ID 0.5', label='_nolegend_',
                            params={'p': .5})

    experiment.add_election_family(model_id='approval_ic', color='black', election_id='IC_path',
                                   params={}, size=10, path={'variable': 'p'})

    experiment.add_election_family(model_id='approval_ic', color='black', election_id='IC_path',
                                   params={}, size=10, path={'variable': 'p'})

    experiment.add_election_family(model_id='approval_id', color='brown', election_id='ID_path',
                                   params={}, size=10, path={'variable': 'p'})

    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        name = f'Mal_{p}_path'
        experiment.add_election_family(model_id='approval_shumallows', color='blue', election_id=name,
                                       params={'p': p}, size=10, path={'variable': 'phi'})

    experiment.compute_distances(distance_id=distance_id)

    experiment.embed(algorithm='spring')

    experiment.print_map(shading=True, adjust=True, legend=True, legend_pos=[1., .75],
                         skeleton=['Empty', 'Full', 'IC 0.5', 'ID 0.5'])

