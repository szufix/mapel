import pytest
import numpy as np

import mapel.elections as mapel


registered_approval_features_to_test = {
    'max_approval_score',
    'number_of_cohesive_groups',
    'number_of_cohesive_groups_brute',
    'proportionality_degree_av',
    'proportionality_degree_pav',
    'proportionality_degree_cc',
}


class TestFeatures:

    @pytest.mark.parametrize("feature_id", registered_approval_features_to_test)
    def test_approval_features(self, feature_id):

        num_voters = np.random.randint(10, 20)
        num_candidates = np.random.randint(5, 10)

        election = mapel.generate_approval_election(culture_id='ic',
                                                    num_voters=num_voters,
                                                    num_candidates=num_candidates,
                                                    p=0.5,)

        election.compute_feature(feature_id)
