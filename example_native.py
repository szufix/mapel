import mapel


if __name__ == "__main__":

    ele_1 = mapel.generate_election(model='impartial_culture', num_voters=5, num_candidates=3)
    ele_2 = mapel.generate_election(model='impartial_culture', num_voters=5, num_candidates=3)

    distance, mapping = mapel.compute_distance(ele_1, ele_2, distance_name='emd-positionwise')

    print('votes_1',  ele_1.votes)
    print('votes_2', ele_2.votes)
    print('distance', distance)
    print('mapping', mapping)
