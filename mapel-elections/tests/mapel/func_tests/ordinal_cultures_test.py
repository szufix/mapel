
import pytest
import numpy as np

import mapel.elections as mapel


registered_ordinal_cultures_to_test = {
    'ic',
    'iac',
    'euclidean',
    'urn',
    'group-separable',
    'single-crossing',
    'weighted_stratification',
    'mallows',
    'norm-mallows',
    'conitzer',
    'spoc_conitzer',
    'walsh',
    'real_identity',
}


class TestCultures:

    @pytest.mark.parametrize("culture_id", registered_ordinal_cultures_to_test)
    def test_ordinal_cultures(self, culture_id):

        num_voters = np.random.randint(10, 100)
        num_candidates = np.random.randint(10, 100)

        election = mapel.generate_ordinal_election(culture_id=culture_id,
                                                   num_voters=num_voters,
                                                   num_candidates=num_candidates)

        assert election.num_candidates == num_candidates
        assert election.num_voters == num_voters

