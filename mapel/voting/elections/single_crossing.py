import random as rand


def generate_single_crossing_election(num_voters=None, num_candidates=None):
    """ helper function: generate simple single-crossing elections"""

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    # GENERATE DOMAIN

    domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)

    domain = [[i for i in range(num_candidates)] for _ in range(domain_size)]

    for line in range(1, domain_size):

        poss = []
        for i in range(num_candidates - 1):
            if domain[line - 1][i] < domain[line - 1][i + 1]:
                poss.append([domain[line - 1][i], domain[line - 1][i + 1]])

        r = rand.randint(0, len(poss) - 1)

        for i in range(num_candidates):

            domain[line][i] = domain[line - 1][i]

            if domain[line][i] == poss[r][0]:
                domain[line][i] = poss[r][1]

            elif domain[line][i] == poss[r][1]:
                domain[line][i] = poss[r][0]

    # GENERATE VOTES

    for j in range(num_voters):
        r = rand.randint(0, domain_size - 1)
        votes[j] = list(domain[r])

    return votes

