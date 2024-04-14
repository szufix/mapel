from collections import defaultdict
import math


def _calculate_maximum_entropy(m):
    num_candidates = m

    # Initialize a dictionary to count rank frequencies for each candidate
    rank_frequencies = defaultdict(lambda: [0] * num_candidates)

    # Count rank frequencies for each candidate
    for i in range(num_candidates):
        for j in range(num_candidates):
            rank_frequencies[i][j] = 1

    # Count rank frequencies for each candidate
    # Calculate the probability for each rank of each candidate
    probabilities = {candidate: [1. / num_candidates for _ in range(num_candidates)] for candidate, ranks in
                     rank_frequencies.items()}
    # print(probabilities)
    # Calculate the entropy for each candidate
    candidate_entropies = [-sum(p * math.log2(p) for p in prob if p > 0) for prob in
                           probabilities.values()]

    # Calculate the average entropy across all candidates
    average_entropy = sum(candidate_entropies)

    return average_entropy


def _calculate_entropy_ordinal(votes):
    num_voters = len(votes)
    num_candidates = len(votes[0])

    # Initialize a dictionary to count rank frequencies for each candidate
    rank_frequencies = defaultdict(lambda: [0] * num_candidates)

    # Count rank frequencies for each candidate
    for vote in votes:
        for rank, candidate in enumerate(vote):
            rank_frequencies[candidate][rank] += 1

    # Calculate the probability for each rank of each candidate
    probabilities = {candidate: [freq / num_voters for freq in ranks] for candidate, ranks in rank_frequencies.items()}

    # Calculate the entropy for each candidate
    candidate_entropies = [-sum(p * math.log2(p) for p in prob if p > 0) for prob in probabilities.values()]

    # Calculate the average entropy across all candidates
    average_entropy = sum(candidate_entropies)

    return average_entropy / _calculate_maximum_entropy(num_candidates)


def entropy(election) -> dict:
    if election.fake:
        return {'value': None}

    score = _calculate_entropy_ordinal(election.votes)

    return score