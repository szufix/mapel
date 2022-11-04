import numpy as np
from mapel.elections.metrics.lp import solve_rand_approx_pav
from mapel.elections.features.scores import get_score, get_dissat


# NEW LP
def get_rand_approx_pav_score(election, committee_size=1):
    if election.fake:
        return 'None'
    W = [1 / (i + 1) for i in range(election.num_candidates)]

    C = np.zeros([election.num_voters, election.num_candidates])
    for i, vote in enumerate(election.votes):
        for j, c in enumerate(vote):
            C[i][c] = j
    # is C equivalent to potes?

    return solve_rand_approx_pav(election, committee_size, W, C)


# GREEDY
def get_greedy_approx_cc_score(election, committee_size=1):
    return get_greedy_approx_score(election, 'cc', committee_size=committee_size)


def get_greedy_approx_hb_score(election, committee_size=1):
    return get_greedy_approx_score(election, 'hb', committee_size=committee_size)


def get_greedy_approx_pav_score(election, committee_size=1):
    return get_greedy_approx_score(election, 'pav', committee_size=committee_size)


def get_greedy_approx_score(election, rule, committee_size=1):
    if election.fake:
        return 'None', 'None'
    winners = get_winners_approx_greedy(election, committee_size, rule)
    print(winners)
    return get_score(election, winners, rule), get_dissat(election, winners, rule)


def get_winners_approx_greedy(election, committee_size, rule):
    """ universal function """

    owa_vector, scoring_vector = get_vectors(election, rule, committee_size)

    if rule == 'hb':
        owa_vector = [1/(i+1) for i in range(election.num_candidates)]
        scoring_vector = [election.num_candidates - i - 1 for i in range(election.num_candidates)]
    elif rule == 'pav':
        owa_vector = [1/(i+1) for i in range(election.num_candidates)]
        scoring_vector = [0] * election.num_candidates
        for i in range(committee_size):
            scoring_vector[i] = 1
    elif rule == 'cc':
        owa_vector = [0] * election.num_candidates
        owa_vector[0] = 1
        scoring_vector = [election.num_candidates - i - 1 for i in range(election.num_candidates)]

    def simple(active, vote, target, owa_vector, scoring_vector):
        income = 0
        ctr = 0
        for x in range(len(vote)):
            if not active[vote[x]] or target == vote[x]:
                income += owa_vector[ctr] * scoring_vector[x]
                ctr += 1
        return income


    winners = []
    voter_sat = [0 for _ in range(election.num_voters)]
    active = [True for _ in range(election.num_candidates)]

    for Z in range(committee_size):

        points = [0 for _ in range(election.num_candidates)]
        income = np.zeros([election.num_voters, election.num_candidates], dtype=float)

        # Compute points
        for i in range(election.num_voters):
            # ctr = 1.
            for j in range(election.num_candidates):
                if active[election.votes[i][j]]:
                    value = simple(active, election.votes[i], election.votes[i][j],
                                   owa_vector, scoring_vector)
                    points[election.votes[i][j]] += value - voter_sat[i]
                    income[i][election.votes[i][j]] = value
                # else:
                    # ctr += 1.

        winner_id = -1
        winner_value = -1

        # Select winner
        # random_order = np.random.permutation(election.num_candidates)
        random_order = [i for i in range(election.num_candidates)]
        for i in random_order:
            if active[i] and points[i] > winner_value:
                winner_value = points[i]
                winner_id = i

        # Update voter satisfaction
        for i in range(election.num_voters):
            voter_sat[i] = income[i][winner_id]

        winners.append(winner_id)
        active[winner_id] = False

    return winners


# REMOVAL
def get_removal_approx_cc_score(election, committee_size=1):
    return get_removal_approx_score(election, "cc", committee_size=committee_size)


def get_removal_approx_hb_score(election, committee_size=1):
    return get_removal_approx_score(election, "hb", committee_size=committee_size)


def get_removal_approx_pav_score(election, committee_size=1):
    return get_removal_approx_score(election, "pav", committee_size=committee_size)


def get_removal_approx_score(election, rule, committee_size=1):
    if election.fake:
        return 'None', 'None'
    winners = get_winners_approx_removal(election, committee_size, rule)

    return get_score(election, winners, rule), get_dissat(election, winners, rule)


def get_winners_approx_removal(election, committee_size, rule):

    owa_vector, scoring_vector = get_vectors(election, rule, committee_size)

    def simple(active, vote, target, owa_vector, scoring_vector):
        income = 0
        ctr = 0
        for x in range(len(vote)):
            if active[vote[x]] and target != vote[x]:
                income += owa_vector[ctr] * scoring_vector[x]
                ctr += 1
        return income

    num_voters = election.num_voters
    num_candidates = election.num_candidates
    votes = election.votes
    removed = 0
    active = [True for _ in range(num_candidates)]

    # PRECOMPUTE
    starting_voter_sat = 0
    for i, x in enumerate(owa_vector):
        starting_voter_sat += scoring_vector[i] * x

    voter_sat = [starting_voter_sat for _ in range(election.num_voters)]

    # STANDARD
    for Z in range(num_candidates - committee_size):

        points = [0. for _ in range(num_candidates)]
        income = np.zeros([election.num_voters, election.num_candidates], dtype=float)

        for i in range(num_voters):
            for j in range(num_candidates):
                if active[votes[i][j]]:

                    value = simple(active, election.votes[i], election.votes[i][j],
                                   owa_vector, scoring_vector)
                    points[election.votes[i][j]] += voter_sat[i] - value
                    income[i][election.votes[i][j]] = value

        loser_id = -1
        loser_value = 9999999
        for i in range(num_candidates):
            if (active[i]) and (0 <= points[i] <= loser_value):
                loser_value = points[i]
                loser_id = i

        active[loser_id] = False
        removed += 1

        # Update voter satisfaction
        for i in range(election.num_voters):
            voter_sat[i] = income[i][loser_id]

    winners = []
    for i in range(num_candidates):
        if active[i]:
            winners.append(i)

    return winners



def get_vectors(election, rule, committee_size):

    if rule == 'hb':
        owa_vector = [1/(i+1) for i in range(election.num_candidates)]
        scoring_vector = [election.num_candidates - i - 1 for i in range(election.num_candidates)]
    elif rule == 'pav':
        owa_vector = [1/(i+1) for i in range(election.num_candidates)]
        scoring_vector = [0] * election.num_candidates
        for i in range(committee_size):
            scoring_vector[i] = 1
    elif rule == 'cc':
        owa_vector = [0] * election.num_candidates
        owa_vector[0] = 1
        scoring_vector = [election.num_candidates - i - 1 for i in range(election.num_candidates)]

    return owa_vector, scoring_vector

