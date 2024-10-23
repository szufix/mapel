import numpy as np
from mapel.roommates.cultures._utils import convert_votes_to_sr


def generate_roommates_urn_votes(num_agents: int = None,
                                 alpha: int = 0.1,
                                 **kwargs):

    votes = np.zeros([num_agents, num_agents], dtype=int)
    urn_size = 1.
    for j in range(num_agents):
        rho = np.random.uniform(0, urn_size)
        if rho <= 1.:
            votes[j] = np.random.permutation(num_agents)
        else:
            votes[j] = votes[np.random.randint(0, j)]
        urn_size += alpha

    return convert_votes_to_sr(votes)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON:  9.06.2023 #
# # # # # # # # # # # # # # # #
