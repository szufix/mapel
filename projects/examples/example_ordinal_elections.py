import mapel


def compute_borda_scores(election) -> list:
    """ Return: List with all Borda scores """
    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        for c in range(election.num_candidates):
            scores[vote[c]] += c
    return scores


if __name__ == "__main__":

    election_1 = mapel.generate_ordinal_election(model_id='impartial_culture',
                                                 num_voters=5, num_candidates=3)
    election_2 = mapel.generate_ordinal_election(model_id='impartial_culture',
                                                 num_voters=5, num_candidates=3)

    print(f'votes_1 {election_1.votes}')
    print(f'votes_2 {election_2.votes}')

    distances, mapping = mapel.compute_distance(election_1, election_2,
                                                distance_id='emd-positionwise')
    print(f'distances {distances}')
    print(f'mapping {mapping}')

    print(f'Borda scores {compute_borda_scores(election_1)}')
