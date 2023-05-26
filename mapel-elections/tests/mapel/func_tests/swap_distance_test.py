import mapel.elections as mapel


class TestSwapDistance:

    def test_bf_vs_ilp_swap_distance(self):
        for _ in range(20):
            election_1 = mapel.generate_ordinal_election(culture_id='ic',
                                                         num_voters=5,
                                                         num_candidates=3)
            election_2 = mapel.generate_ordinal_election(culture_id='ic',
                                                         num_voters=5,
                                                         num_candidates=3)

            distance_1, _ = mapel.compute_distance(election_1, election_2,
                                                   distance_id='swap')

            distance_2, _ = mapel.compute_distance(election_1, election_2,
                                                   distance_id='ilp_swap')

            assert distance_1 == distance_2, "BF swap distance differs from ILP swap distance"
