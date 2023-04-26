import numpy as np


def convert_ordinal_to_approval(votes):
    approval_votes = [{} for _ in range(len(votes))]
    for i, vote in enumerate(votes):
        k = int(np.random.beta(2, 6) * len(votes[0])) + 1
        approval_votes[i] = set(vote[0:k])

    return approval_votes