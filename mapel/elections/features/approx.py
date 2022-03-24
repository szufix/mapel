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


# MAIN FUNCTIONS
def get_greedy_approx_cc_score(election, feature_params):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)
    winners = get_winners_approx_cc_greedy(election, committee_size)
    return get_cc_score(election, winners)


def get_removal_approx_cc_score(election, feature_params):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)
    winners = get_winners_approx_cc_removal(election, committee_size)
    return get_cc_score(election, winners)


def get_greedy_approx_hb_score(election, feature_params):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)
    winners = get_winners_approx_hb_greedy(election, committee_size)
    return get_hb_score(election, winners)


def get_removal_approx_hb_score(election, feature_params):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)
    winners = get_winners_approx_hb_removal(election, committee_size)
    return get_hb_score(election, winners)


def get_greedy_approx_pav_score(election, feature_params):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)
    winners = get_winners_approx_pav_greedy(election, committee_size)
    return get_pav_score(election, winners)


def get_removal_approx_pav_score(election, feature_params):
    if election.fake:
        return 'None'
    committee_size = unpack_committee_size(feature_params)
    winners = get_winners_approx_pav_removal(election, committee_size)
    return get_pav_score(election, winners)


# HELPER FUNCTIONS
def get_winners_approx_cc_greedy(votes, num_voters, num_candidates, num_winners):

    winners = []
    voter_sat = [0 for _ in range(num_voters)]
    active = [True for _ in range(num_candidates)]

    for Z in range(num_winners):

        points = [0 for _ in range(num_candidates)]

        for i in range(num_voters):
            for j in range(num_candidates):
                if voter_sat[i] == (num_candidates - j - 1):
                    break
                else:
                    points[votes[i][j]] += (num_candidates - j - 1) - voter_sat[i]

        winner_id = -1
        winner_value = -1

        random_order = np.random.permutation(num_candidates)
        for i in random_order:

            if active[i] and points[i] > winner_value:
                winner_value = points[i]
                winner_id = i

        for i in range(num_voters):
            for j in range(num_candidates):
                if votes[i][j] == winner_id:
                    if num_candidates - j - 1 > voter_sat[i]:
                        voter_sat[i] = num_candidates - j - 1
                    break

        winners.append(winner_id)
        active[winner_id] = False

    return winners


def get_winners_approx_cc_removal(votes, num_voters, num_candidates, num_winners):

    winners = []

    return winners


# def get_winners_approx_hb(election, committee_size, algorithm='greedy'):
#
#     if algorithm == "removal":
#         return get_winners_approx_hb_removal(election, committee_size)
#     elif algorithm == "greedy":
#         return get_winners_approx_hb_greedy(election, committee_size)


def get_winners_approx_hb_greedy(election, committee_size):
    num_voters = election.num_voters
    num_candidates = election.num_candidates
    votes = election.votes

    winners = []
    active = [True for _ in range(num_candidates)]

    for Z in range(committee_size):

        points = [0 for _ in range(num_candidates)]

        for i in range(num_voters):
            ctr = 1.
            for j in range(num_candidates):
                if active[votes[i][j]]:
                    points[votes[i][j]] += (1. / ctr) * (num_candidates - j - 1)
                else:
                    ctr += 1.

        winner_id = 0
        winner_value = -1
        for i in range(num_candidates):
            if active[i] and points[i] > winner_value:
                winner_value = points[i]
                winner_id = i

        active[winner_id] = False
        winners.append(winner_id)

    return winners


def get_winners_approx_hb_removal(election, committee_size):
    num_voters = election.num_voters
    num_candidates = election.num_candidates
    votes = election.votes
    removed = 0
    active = [True for _ in range(num_candidates)]

    for Z in range(num_candidates - committee_size):

        points = [0. for _ in range(num_candidates)]

        for i in range(num_voters):
            ctr = 1.
            for j in range(num_candidates):
                if active[votes[i][j]]:
                    points[votes[i][j]] += (1. / ctr) * (num_candidates - j - 1)
                    ctr += 1

        loser_id = -1
        loser_value = 9999999
        for i in range(num_candidates):
            if (active[i]) and (0 <= points[i] <= loser_value):
                loser_value = points[i]
                loser_id = i

        active[loser_id] = False
        removed += 1

    winners = []
    for i in range(num_candidates):
        if active[i]:
            winners.append(i)

    return winners


def get_winners_approx_pav_greedy(election, committee_size):
    print(election.votes)

    num_voters = election.num_voters
    num_candidates = election.num_candidates
    votes = election.votes

    winners = []
    active = [True for _ in range(num_candidates)]

    for Z in range(committee_size):

        points = [0 for _ in range(num_candidates)]

        for i in range(num_voters):
            ctr = 1.
            for j in range(committee_size):
                if active[votes[i][j]]:
                    points[votes[i][j]] += (1. / ctr)
                else:
                    ctr += 1.

        winner_id = 0
        winner_value = -1
        for i in range(num_candidates):
            if active[i] and points[i] > winner_value:
                winner_value = points[i]
                winner_id = i

        active[winner_id] = False
        winners.append(winner_id)

    return winners


def get_winners_approx_pav_removal(election, committee_size):
    num_voters = election.num_voters
    num_candidates = election.num_candidates
    votes = election.votes

    removed = 0
    active = [True for _ in range(num_candidates)]

    for Z in range(num_candidates - committee_size):

        points = [0 for _ in range(num_candidates)]

        for i in range(num_voters):
            ctr = 1.
            for j in range(committee_size):
                if active[votes[i][j]]:
                    points[votes[i][j]] += (1. / ctr)
                    ctr += 1

        loser_id = 0
        loser_value = 9999999
        for i in range(num_candidates):
            if active[i] and 0 <= points[i] < loser_value:
                loser_value = points[i]
                loser_id = i

        active[loser_id] = False
        removed += 1

    winners = []
    for i in range(num_candidates):
        if active[i]:
            winners.append(i)

    return winners


def get_cc_score(election, winners) -> float:
    score = 0

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


def unpack_committee_size(feature_params):
    if 'committee_size' in feature_params:
        return feature_params['committee_size']
    return 10
