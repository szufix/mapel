#!/usr/bin/env python3

import mapel.elections as mapel


if __name__ == "__main__":


    election_1 = mapel.generate_approval_election(culture_id='impartial_culture',
                                                  num_voters=5, num_candidates=3)
    election_2 = mapel.generate_approval_election(culture_id='impartial_culture',
                                                  num_voters=5, num_candidates=3)
    election_3 = mapel.generate_approval_election(culture_id='resampling',
                                                  params={'p': 0.1, 'phi': 0.4},
                                                  num_voters=5, num_candidates=3)


    print(f'votes_1 {election_1.votes}')
    print(f'votes_2 {election_2.votes}')

    distances = mapel.compute_distance(election_1, election_2, distance_id='l1-approvalwise')

    print(f'distances {distances}')
