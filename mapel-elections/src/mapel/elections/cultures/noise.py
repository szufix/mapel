import numpy as np
import scipy.special


def generate_approval_noise_model_votes(num_voters=None,
                                        num_candidates=None,
                                        p=None, phi=0.5, type_id='hamming'):

    model_type = type_id

    k = int(p * num_candidates)

    A = {i for i in range(k)}
    B = set(range(num_candidates)) - A

    choices = []
    probabilites = []

    # PREPARE BUCKETS
    for x in range(len(A) + 1):
        num_options_in = scipy.special.binom(len(A), x)
        for y in range(len(B) + 1):
            num_options_out = scipy.special.binom(len(B), y)

            if model_type == 'hamming':
                factor = phi ** (len(A) - x + y)  # Hamming
            elif model_type == 'jaccard':
                factor = phi ** ((len(A) - x + y) / (len(A) + y))  # Jaccard
            elif model_type == 'zelinka':
                factor = phi ** max(len(A) - x, y)  # Zelinka
            elif model_type == 'bunke-shearer':
                factor = phi ** (max(len(A) - x, y) / max(len(A), x + y))  # Bunke-Shearer
            else:
                "No such model type_id!!!"

            num_options = num_options_in * num_options_out * factor

            choices.append((x, y))
            probabilites.append(num_options)

    denominator = sum(probabilites)
    probabilites = [p / denominator for p in probabilites]

    # SAMPLE VOTES
    votes = []
    for _ in range(num_voters):
        _id = np.random.choice(range(len(choices)), 1, p=probabilites)[0]
        x, y = choices[_id]
        vote = set(np.random.choice(list(A), x, replace=False))
        vote = vote.union(set(np.random.choice(list(B), y, replace=False)))
        votes.append(vote)

    return votes
