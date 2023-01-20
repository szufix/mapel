#!/usr/bin/env python
import math
import numpy as np


def randomize(vector, num_winners):
    scores = [x for x, _ in vector]
    ranking = [x for _, x in vector]
    last_value = scores[num_winners-1]
    # LEFT
    left = num_winners - 2
    while left >= 0 and scores[left] == last_value:
        left -= 1
    left += 1
    # RIGHT
    right = num_winners
    while right < len(scores) and scores[right] == last_value:
        right += 1
    # print(left, right)
    if left < right:
        ranking[left:right] = np.random.choice(ranking[left:right], right - left,
                                               replace=False)
    # print(ranking)
    return ranking


def compute_sntv_winners(election=None, num_winners=1):
    """ Compute SNTV winners for a given election """
    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        scores[vote[0]] += 1
    candidates = [i for i in range(election.num_candidates)]
    results = sorted(zip(scores, candidates), reverse=True)
    ranking = randomize(results, num_winners)
    return ranking[0:num_winners]


def compute_borda_winners(election=None, num_winners=1):
    """ Compute Borda winners for a given election """

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        for i in range(election.num_candidates):
            scores[vote[i]] += election.num_candidates - i - 1
    candidates = [i for i in range(election.num_candidates)]
    results = sorted(zip(scores, candidates), reverse=True)
    ranking = randomize(results, num_winners)
    return ranking[0:num_winners]


def compute_stv_winners(election=None, num_winners=1):
    """ Compute STV winners for a given election """

    winners = [] #[0] * params['orders']
    active = [True] * election.num_candidates

    droop_quota = math.floor(election.num_voters / (num_winners + 1.) ) + 1

    votes_on_1 = [0.] * election.num_candidates
    for i in range(election.num_voters):
        votes_on_1[election.votes[i][0]] += 1

    v_power = [1.] * election.num_voters


    while len(winners) + sum(active) > num_winners:

        ctr = election.num_candidates
        winner_id = 0
        while ctr > 0:
            if active[winner_id] and votes_on_1[winner_id] >= droop_quota:

                winners += [winner_id]

                total = 0
                for i in range(election.num_voters):
                    for j in range(election.num_candidates):
                        if active[election.votes[i][j]]:
                            if election.votes[i][j] == winner_id:
                                for k in range(j + 1, election.num_candidates):
                                    if active[election.votes[i][k]]:
                                        v_power[i] *= float(votes_on_1[winner_id] - droop_quota) / float(votes_on_1[winner_id])
                                        votes_on_1[election.votes[i][k]] += 1. * v_power[i]
                                        total += 1. * v_power[i]
                                        ctr = election.num_candidates
                                        break
                            break
                votes_on_1[winner_id] = 0
                active[winner_id] = False

            ctr -= 1
            winner_id += 1
            winner_id %= election.num_candidates

        loser_votes = droop_quota
        loser_id = 0
        for i in range(election.num_candidates):
            if active[i] and votes_on_1[i] < loser_votes:
                loser_votes = votes_on_1[i]
                loser_id = i

        votes_on_1[loser_id] = 0
        for i in range(election.num_voters):
            for j in range(election.num_candidates):
                if active[election.votes[i][j]]:
                    if election.votes[i][j] == loser_id:
                        for k in range(j+1, election.num_candidates):
                            if active[election.votes[i][k]]:
                                votes_on_1[election.votes[i][k]] += 1. * v_power[i]
                                break
                    break
        active[loser_id] = False


    for i in range(election.num_candidates):
        if active[i]:
            winners += [i]

    winners = sorted(winners)

    return winners