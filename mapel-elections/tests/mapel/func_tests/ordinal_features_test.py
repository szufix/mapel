
import pytest
import numpy as np

import mapel.elections as mapel

registered_ordinal_features_to_test = {
    'highest_borda_score',
    'highest_plurality_score',
    'highest_copeland_score',
    'lowest_dodgson_score',
    'highest_cc_score',
    'highest_hb_score',
    'highest_pav_score',
    'greedy_approx_cc_score',
    'removal_approx_cc_score',
    'greedy_approx_hb_score',
    'removal_approx_hb_score',
    'greedy_approx_pav_score',
    'removal_approx_pav_score',
    'banzhaf_cc_score',
    'ranging_cc_score',
    'num_of_diff_votes',
    'voterlikeness_sqrt',
    'voterlikeness_harmonic',
    'borda_diversity',
    'borda_std',
    'borda_range',
    'borda_gini',
    'borda_meandev',
    'cand_dom_dist_mean',
    'cand_dom_dist_std',
    'cand_pos_dist_std',
    'cand_pos_dist_gini',
    'cand_pos_dist_meandev',
    'med_cands_summed',
    'vote_dist_mean',
    'vote_dist_max',
    'vote_dist_med',
    'vote_dist_gini',
    'vote_sqr_dist_mean',
    'vote_sqr_dist_med',
    'vote_diversity_Karpov',
    'greedy_kKemenys_summed',
    'greedy_2kKemenys_summed',
    'greedy_kKemenys_divk_summed',
    'polarization_1by2Kemenys',
    'greedy_kmeans_summed',
    'support_pairs',
    'support_triplets',
    'support_votes',
    'support_diversity_summed',
    'support_diversity_normed_summed',
    'support_diversity_normed2_summed',
    'support_diversity_normed3_summed',
    'dist_to_Borda_mean',
    'dist_to_Kemeny_mean',
    'borda_spread',
    'Agreement',
    'Diversity',
    'Polarization',
}


class TestFeatures:

    @pytest.mark.parametrize("feature_id", registered_ordinal_features_to_test)
    def test_ordinal_features(self, feature_id):

        num_voters = np.random.randint(10, 20)
        num_candidates = np.random.randint(5, 10)

        election = mapel.generate_ordinal_election(culture_id='ic',
                                                   num_voters=num_voters,
                                                   num_candidates=num_candidates)
        election.compute_feature(feature_id)
