import numpy as np


def generate_roommates_ic_votes(num_agents: int = None, params=None):

        votes = np.zeros([num_agents, num_agents-1], dtype=int)
        for j in range(num_agents):
            base = np.random.permutation(num_agents)
            base = base[base != j]
            votes[j] = base
        return votes

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 14.10.2021 #
# # # # # # # # # # # # # # # #
