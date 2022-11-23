
import mapel.elections as mapel


class TestSwapDistance:

    def test_bf_vs_ilp_swap_distance(self):
        for _ in range(20):
            election_1 = mapel.generate_ordinal_election(culture_id='impartial_culture',
                                                         num_voters=5, num_candidates=3)
            election_2 = mapel.generate_ordinal_election(culture_id='impartial_culture',
                                                         num_voters=5, num_candidates=3)

            distances_1 = mapel.compute_distance(election_1, election_2,
                                                 distance_id='swap')

            distances_2 = mapel.compute_distance(election_1, election_2,
                                                 distance_id='ilp_swap')

            assert distances_1 == distances_2, "BF swap distance differs from ILP swap distance"
