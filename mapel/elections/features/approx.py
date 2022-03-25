import numpy as np
from mapel.elections.metrics.lp import solve_rand_approx_pav


# NEW LP
def get_rand_approx_pav_score(election, feature_params):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)

    W = [1 / (i + 1) for i in range(election.num_candidates)]

    C = np.zeros([election.num_voters, election.num_candidates])
    for i, vote in enumerate(election.votes):
        for j, c in enumerate(vote):
            C[i][c] = j
    # is C equivalent to potes?

    return solve_rand_approx_pav(election, committee_size, W, C)


# GREEDY
def get_greedy_approx_cc_score(election, feature_params):
    return get_greedy_approx_score(election, feature_params, "cc")


def get_greedy_approx_hb_score(election, feature_params):
    return get_greedy_approx_score(election, feature_params, "hb")


def get_greedy_approx_pav_score(election, feature_params):
    return get_greedy_approx_score(election, feature_params, "pav")


def get_greedy_approx_score(election, feature_params, rule):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)
    winners = get_winners_approx_greedy(election, committee_size, rule)
    return get_score(election, winners, rule)


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
        random_order = np.random.permutation(election.num_candidates)
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
def get_removal_approx_cc_score(election, feature_params):
    return get_removal_approx_score(election, feature_params, "cc")


def get_removal_approx_hb_score(election, feature_params):
    return get_removal_approx_score(election, feature_params, "hb")


def get_removal_approx_pav_score(election, feature_params):
    return get_removal_approx_score(election, feature_params, "pav")


def get_removal_approx_score(election, feature_params, rule):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)
    winners = get_winners_approx_removal(election, committee_size, rule)

    return get_score(election, winners, rule)


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


# GET SCORE
def get_score(election, winners, rule) -> float:
    if rule == 'cc':
        return get_cc_score(election, winners)
    elif rule == 'hb':
        return get_hb_score(election, winners)
    elif rule == 'pav':
        return get_pav_score(election, winners)


def get_cc_score(election, winners) -> float:
    num_voters = election.num_voters
    num_candidates = election.num_candidates
    votes = election.votes

    score = 0

    for i in range(num_voters):
        for j in range(num_candidates):
            if votes[i][j] in winners:
                score += 1.
                break

    return score

def get_hb_score(election, winners) -> float:
    num_voters = election.num_voters
    num_candidates = election.num_candidates
    votes = election.votes

    score = 0

    for i in range(num_voters):
        ctr = 1.
        for j in range(num_candidates):
            if votes[i][j] in winners:
                score += (1. / ctr) * (num_candidates - j - 1)
                ctr += 1

    return score


def get_pav_score(election, winners) -> float:
    num_voters = election.num_voters
    num_candidates = election.num_candidates
    votes = election.votes

    score = 0

    vector = [0.] * num_candidates
    for i in range(len(winners)):
        vector[i] = 1.

    for i in range(num_voters):
        ctr = 1.
        for j in range(num_candidates):
            if votes[i][j] in winners:
                score += (1. / ctr) * vector[j]
                ctr += 1

    return score


# OTHER
def unpack_committee_size(feature_params):
    if 'committee_size' in feature_params:
        return feature_params['committee_size']
    return 10


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

