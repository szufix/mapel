import numpy as np

import pytest

import mapel.elections as mapel

registered_approval_cultures_to_test = {
    'ic',
    'id',
    'resampling',
    'disjoint_resampling',
    'moving_resampling',
    'noise',
    'euclidean',
    'truncated_urn',
    'urn_partylist',
    'full',
    'empty',
}


class TestCultures:

    @pytest.mark.parametrize("culture_id", registered_approval_cultures_to_test)
    def test_approval_cultures(self, culture_id):

        num_voters = np.random.randint(10, 100)
        num_candidates = np.random.randint(10, 100)

        if culture_id in ['resampling',
                          'disjoint_resampling',
                          'moving_resampling',
                          'noise']:
            election = mapel.generate_approval_election(culture_id=culture_id,
                                                        num_voters=num_voters,
                                                        num_candidates=num_candidates,
                                                        phi=0.4,
                                                        p=0.4,)
        elif culture_id in ['truncated_urn']:
            election = mapel.generate_approval_election(culture_id=culture_id,
                                                    num_voters=num_voters,
                                                    num_candidates=num_candidates,
                                                        p=0.4,
                                                        alpha=0.1)

        elif culture_id in ['urn_partylist']:
            election = mapel.generate_approval_election(culture_id=culture_id,
                                                    num_voters=num_voters,
                                                    num_candidates=num_candidates,
                                                        p=0.4,
                                                        alpha=0.1,
                                                        parties=2)
        elif culture_id in ['ic']:
            election = mapel.generate_approval_election(culture_id=culture_id,
                                                    num_voters=num_voters,
                                                    num_candidates=num_candidates,
                                                        p=0.4)
        else:
            election = mapel.generate_approval_election(culture_id=culture_id,
                                                    num_voters=num_voters,
                                                    num_candidates=num_candidates)



        # assert election.num_candidates == num_candidates
        # assert election.num_voters == num_voters
