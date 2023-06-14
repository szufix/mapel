import pytest
import numpy as np

import mapel.elections as mapel

registered_approval_distances_to_test = {
    'l1-approvalwise',
    # 'hamming',  # tested independently
}


class TestOrdinalDistances:

    @pytest.mark.parametrize("distance_id", registered_approval_distances_to_test)
    def test_approval_distances(self, distance_id):

        num_voters = np.random.randint(10, 20)
        num_candidates = np.random.randint(5, 10)

        ele_1 = mapel.generate_approval_election(culture_id='ic', p=0.5,
                                                 num_voters=num_voters,
                                                 num_candidates=num_candidates)

        ele_2 = mapel.generate_approval_election(culture_id='ic', p=0.5,
                                                 num_voters=num_voters,
                                                 num_candidates=num_candidates)

        distance, mapping = mapel.compute_distance(ele_1, ele_2, distance_id=distance_id)

        assert type(float(distance)) is float
