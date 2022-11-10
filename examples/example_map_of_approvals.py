#!/usr/bin/env python3

import mapel.elections as mapel


if __name__ == "__main__":

    distance_id = 'l1-approvalwise'

    experiment = mapel.prepare_experiment(instance_type='approval', distance_id\
    = distance_id)

    experiment.set_default_num_candidates(50)
    experiment.set_default_num_voters(100)

    experiment.add_election(culture_id='approval_empty', election_id='Empty',
    label='Empty')
    experiment.add_election(culture_id='approval_full', election_id='Full',
    label='Full')
    experiment.add_election(culture_id='ic', election_id='IC 0.5', label='IC 0.5',
                            params={'p': .5})
    experiment.add_election(culture_id='id', election_id='ID 0.5', label='ID 0.5',
                            params={'p': .5})

    experiment.add_family(culture_id='ic', color='black', family_id='IC_path',
                                   params={}, size=10, path={'variable': 'p'})

    experiment.add_family(culture_id='ic', color='black', family_id='IC_path',
                                   params={}, size=10, path={'variable': 'p'})

    experiment.add_family(culture_id='id', color='brown', family_id='ID_path',
                                   params={}, size=10, path={'variable': 'p'})

    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        name = f'Mal_{p}_path'
        experiment.add_family(culture_id='resampling', color='blue', election_id=name,
                                       params={'p': p}, size=10, path={'variable': 'phi'})

    experiment.compute_distances()

    experiment.embed(algorithm='spring')

    experiment.print_map(legend=True, legend_pos=[1., .75],
                         textual=['Empty', 'Full', 'IC 0.5', 'ID 0.5'])

