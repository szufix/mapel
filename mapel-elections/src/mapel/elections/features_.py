#!/usr/bin/env python

import mapel.elections.features.partylist as partylist
# import mapel.elections.features.cohesive as cohesive
# import mapel.elections.features.proportionality_degree as prop_deg
import mapel.elections.features.scores as scores
import mapel.elections.features.approx as approx
import mapel.elections.features.banzhaf_cc as banzhaf_cc
import mapel.elections.features.ranging_cc as ranging_cc
import mapel.elections.features.vc_diversity as vcd
import mapel.elections.features.diversity as diversity
import mapel.elections.features.clustering as clustering
import mapel.elections.features.distortion as distortion
import mapel.elections.features.other as other
import mapel.elections.features.justified_representation as jr
import mapel.elections.features.dimensionality as dimensionality


from mapel.core.glossary import MAIN_LOCAL_FEATUERS, MAIN_GLOBAL_FEATUERS
from mapel.core.features_main import get_main_local_feature, get_main_global_feature


def get_global_feature(feature_id):
    """ Global feature depends on all instances """
    if feature_id in MAIN_GLOBAL_FEATUERS:
        return get_main_global_feature(feature_id)

    return {'clustering': clustering.clustering_v1,
            'clustering_kmeans': clustering.clustering_kmeans,
            'distortion_from_all': distortion.distortion_from_all,
            'id_vs_un': clustering.id_vs_un,
            'an_vs_st': clustering.an_vs_st,
            }.get(feature_id)


LIST_OF_LOCAL_FEATURES = {'highest_borda_score': scores.highest_borda_score,
            'highest_plurality_score': scores.highest_plurality_score,
            'highest_copeland_score': scores.highest_copeland_score,
            'lowest_dodgson_score': scores.lowest_dodgson_score,
            'avg_distortion_from_guardians': distortion.avg_distortion_from_guardians,
            'worst_distortion_from_guardians': distortion.worst_distortion_from_guardians,
            'max_approval_score': other.max_approval_score,
            # 'cohesiveness': cohesive.count_largest_cohesiveness_level_l_of_cohesive_group,
            # 'number_of_cohesive_groups': cohesive.count_number_of_cohesive_groups,
            # 'number_of_cohesive_groups_brute': cohesive.count_number_of_cohesive_groups_brute,
            # 'proportionality_degree_av': prop_deg.proportionality_degree_av,
            # 'proportionality_degree_pav': prop_deg.proportionality_degree_pav,
            # 'proportionality_degree_cc': prop_deg.proportionality_degree_cc,
            'abstract': other.abstract,
            'distortion': distortion,
            'monotonicity_triplets': distortion.monotonicity_triplets,
            'partylist': partylist.partylistdistance,
            'pav_time': partylist.pav_time,
            'justified_ratio': other.justified_ratio,
            'highest_cc_score': scores.highest_cc_score,
            'highest_hb_score': scores.highest_hb_score,
            'highest_pav_score': scores.highest_pav_score,
            'greedy_approx_cc_score': approx.get_greedy_approx_cc_score,
            'removal_approx_cc_score': approx.get_removal_approx_cc_score,
            'greedy_approx_hb_score': approx.get_greedy_approx_hb_score,
            'removal_approx_hb_score': approx.get_removal_approx_hb_score,
            'greedy_approx_pav_score': approx.get_greedy_approx_pav_score,
            'removal_approx_pav_score': approx.get_removal_approx_pav_score,
            'rand_approx_pav_score': approx.get_rand_approx_pav_score,
            'banzhaf_cc_score': banzhaf_cc.get_banzhaf_cc_score,
            'ranging_cc_score': ranging_cc.get_ranging_cc_score,
            'num_of_diff_votes': vcd.num_of_diff_votes,
            'voterlikeness_sqrt': vcd.voterlikeness_sqrt,
            'voterlikeness_harmonic': vcd.voterlikeness_harmonic,
            'borda_diversity': vcd.borda_diversity,
            'borda_std': diversity.borda_std,
            'borda_range': diversity.borda_range,
            'borda_gini': diversity.borda_gini,
            'borda_meandev': diversity.borda_meandev,
            'cand_dom_dist_mean': diversity.cand_dom_dist_mean,
            'cand_dom_dist_std': diversity.cand_dom_dist_std,
            'cand_pos_dist_std': diversity.cand_pos_dist_std,
            'cand_pos_dist_gini': diversity.cand_pos_dist_gini,
            'cand_pos_dist_meandev': diversity.cand_pos_dist_meandev,
            'med_cands_summed': diversity.med_cands_summed,
            'vote_dist_mean': diversity.vote_dist_mean,
            'vote_dist_max': diversity.vote_dist_max,
            'vote_dist_med': diversity.vote_dist_med,
            'vote_dist_gini': diversity.vote_dist_gini,
            'vote_sqr_dist_mean': diversity.vote_sqr_dist_mean,
            'vote_sqr_dist_med': diversity.vote_sqr_dist_med,
            'vote_diversity_Karpov': diversity.vote_diversity_Karpov,
            'greedy_kKemenys_summed': diversity.greedy_kKemenys_summed,
            'greedy_2kKemenys_summed': diversity.greedy_2kKemenys_summed,
            'greedy_kKemenys_divk_summed': diversity.greedy_kKemenys_divk_summed,
            'polarization_1by2Kemenys': diversity.polarization_1by2Kemenys,
            'greedy_kmeans_summed': diversity.greedy_kmeans_summed,
            'support_pairs': diversity.support_pairs,
            'support_triplets': diversity.support_triplets,
            'support_votes': diversity.support_votes,
            'support_diversity_summed': diversity.support_diversity_summed,
            'support_diversity_normed_summed': diversity.support_diversity_normed_summed,
            'support_diversity_normed2_summed': diversity.support_diversity_normed2_summed,
            'support_diversity_normed3_summed': diversity.support_diversity_normed3_summed,
            'dist_to_Borda_mean': diversity.dist_to_Borda_mean,
            'dist_to_Kemeny_mean': diversity.dist_to_Kemeny_mean,
            'ejr': jr.test_ejr,
            'min_dim': dimensionality.min_dim,
            'borda_spread': scores.borda_spread,
            }


def get_local_feature(feature_id):
    """ Local feature depends only on a single instance """

    if feature_id in MAIN_LOCAL_FEATUERS:
        return get_main_local_feature(feature_id)
    elif feature_id in LIST_OF_LOCAL_FEATURES:
        return LIST_OF_LOCAL_FEATURES.get(feature_id)
    else:
        raise ValueError(f'Incorrect feature_id: {feature_id}')


def add_local_feature(name, function):
    LIST_OF_LOCAL_FEATURES[name] = function


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 10.05.2022 #
# # # # # # # # # # # # # # # #
