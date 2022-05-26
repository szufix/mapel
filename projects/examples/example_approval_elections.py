import mapel


if __name__ == "__main__":

    election_1 = mapel.generate_approval_election(model_id='impartial_culture',
                                                  num_voters=5, num_candidates=3)
    election_2 = mapel.generate_approval_election(model_id='impartial_culture',
                                                  num_voters=5, num_candidates=3)

    print(f'votes_1 {election_1.votes}')
    print(f'votes_2 {election_2.votes}')

    distances = mapel.compute_distance(election_1, election_2, distance_id='l1-approvalwise')

    print(f'distances {distances}')