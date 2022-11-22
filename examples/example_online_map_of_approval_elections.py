#!/usr/bin/env python3

import mapel.elections as mapel

if __name__ == "__main__":

    distance_id = 'l1-approvalwise'

    experiment = mapel.prepare_online_approval_experiment(distance_id =
    distance_id)

    experiment.set_default_num_candidates(20)
    experiment.set_default_num_voters(50)

    experiment.add_family(culture_id='impartial_culture', color='black', family_id='ic', size=8)

    experiment.add_family(culture_id='urn', color='orange', family_id='urn_1',
                                   params={'alpha': 0.1}, size=8)

    experiment.add_family(culture_id='urn', color='red', family_id='urn_2',
                                   params={'alpha': 0.5}, size=8)

    experiment.compute_distances()

    experiment.embed()
    
    experiment.print_map()
    
