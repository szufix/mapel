#!/usr/bin/env python3

import mapel.elections as mapel

if __name__ == "__main__":

    distance_id = 'emd-positionwise'

    experiment = mapel.prepare_online_ordinal_experiment()

    experiment.set_default_num_candidates(20)
    experiment.set_default_num_voters(50)

    experiment.add_family(culture_id='impartial_culture', color='black', family_id='ic', size=8)

    experiment.add_family(culture_id='urn', color='orange', family_id='urn_1',
                                   params={'alpha': 0.1}, size=8)

    experiment.add_family(culture_id='urn', color='red', family_id='urn_2',
                                   params={'alpha': 0.5}, size=8)

    experiment.add_family(culture_id='norm-mallows', color='cyan', family_id='mallows_1',
                                   params={'norm-phi': 0.1}, size=8)

    experiment.add_family(culture_id='norm-mallows', color='blue', family_id='mallows_2',
                                   params={'norm-phi': 0.5}, size=8)

    experiment.compute_distances(distance_id=distance_id)

    experiment.embed()

    experiment.print_map()

