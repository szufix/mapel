#!/usr/bin/env python

import math
import os
import random as rand

import numpy as np

from mapel.voting.metrics import lp as lp


def generate_winners(election=None, num_winners=1, ballot="ordinal", method=None):
    votes, num_voters, num_candidates = election.votes, election.num_voters, election.num_candidates
    params = {}
    params["orders"] = num_winners
    params["pure"] = True
    params["elections"] = 1
    params['candidates'] = num_candidates
    params['voters'] = num_voters
    rule = {}
    rule['type'] = method
    winners = get_winners(params, votes, rule, ballot)
    return winners


def get_winners(params, votes, rule, ballot='ordinal'):
    if ballot == "ordinal":
        return get_ordinal_winners(params, votes, rule)
    elif ballot == "approval":
        return get_approval_winners(params, votes, rule)


def get_approval_winners(params, elections, rule):
    if rule['type'] == 'app_cc':
        all_winners = []
        for i in range(params['elections']):
            winners = get_winners_app_cc(params, elections['votes'][i], elections['candidates'][i])
            all_winners += winners
        return all_winners


# Need update
def get_ordinal_winners(params, votes, rule):
    if rule['type'] == 'scoring':
        scoring = get_rule(rule['name'], rule['length'])
        all_winners = []
        for i in range(params['elections']):
            winners = get_winners_scoring(params, votes[i], params['candidates'], scoring)
            all_winners += winners
        return all_winners

    elif rule['type'] == 'borda_owa':
        owa = get_rule(rule['name'], rule['length'])
        all_winners = []
        for i in range(params['elections']):
            winners = get_winners_borda_owa(params, votes, params['candidates'], owa)
            all_winners += winners
        return all_winners

    elif rule['type'] == 'bloc_owa':
        owa = get_rule(rule['name'], rule['length'])
        t_bloc = rule['special']
        all_winners = []
        for i in range(params['elections']):
            winners = get_winners_bloc_owa(params, votes[i], params['candidates'], owa, t_bloc)
            all_winners += winners
        return all_winners

    elif rule['type'] == 'stv':
        all_winners = []
        for i in range(params['elections']):
            winners = get_winners_stv(params, votes[i], params['candidates'])
            all_winners += winners
        return all_winners

    elif rule['type'] == 'self':
        all_winners = []
        for i in range(params['elections']):
            winners = []
            for j in range(params['voters']):
                winners += [votes[i][j]]

            all_winners += winners
        return all_winners

    elif rule['type'] == "approx_cc":
        all_winners = []
        winners = get_winners_approx_cc(votes, params)
        all_winners += winners
        return all_winners

    elif rule['type'] == "approx_hb":
        all_winners = []
        winners = get_winners_approx_hb(votes, params, 'greedy')
        all_winners += winners
        return all_winners

    elif rule['type'] == "approx_pav":
        all_winners = []
        winners = get_winners_approx_pav(votes, params, 'greedy')
        all_winners += winners
        return all_winners


def get_rule(name, length):

    rule = [0.] * length
    if name == "borda":
        for i in range(length):
            rule[i] = (length - float(i) - 1.) / (length - 1.)
    elif name == 'sntv':
        rule[1] = 1.
    elif name == 'cc':
        rule[1] = 1.
    elif name == 'hb':
        for i in range(length):
            rule[i] = 1. / (i + 1.)
    else:
        return name
    return rule


def get_winners_app_cc(params, votes, candidates):

    points = [0 for _ in range(params["candidates"])]

    for i in range(params['voters']):
        for j in range(len(votes[i])):
            points[int(votes[i][j])] += 1.

    tmp_candidates = [x for _, x in sorted(zip(points, candidates))]
    winners = tmp_candidates[params['candidates'] - params['orders']: params['candidates']]

    winners = sorted(winners)
    return winners


def get_winners_scoring(params, votes, candidates, scoring):

    points = [0 for _ in range(params["candidates"])]

    for i in range(params['voters']):
        for j in range(params['candidates']):
            points[int(votes[i][j])] += scoring[j]

    tmp_candidates = [x for _, x in sorted(zip(points, candidates))]
    winners = tmp_candidates[params['candidates'] - params['orders']: params['candidates']]

    points = sorted(points)
    ### DRAW ###
    if params['orders'] > 1:
        breaking_point = params['candidates'] - params['orders']
        if points[breaking_point-1] == points[breaking_point]:
            left = -1
            while points[breaking_point-1 + left] == points[breaking_point + left]:
                left -= 1
            right = 0

            while right < params['orders']-2 and points[breaking_point+right] == points[breaking_point+right+1]:
                right += 1
            while right >= 0:
                r = rand.randint(left,right)
                tmp_candidate = winners[right]
                winners[right] = tmp_candidates[breaking_point+r]
                tmp_candidates[breaking_point + r] = tmp_candidate
                right -= 1
    winners = sorted(winners)
    return winners


def get_winners_borda_owa(params, votes, candidates, owa):
    rand_name = str(rand.random())
    lp_file_name = str(rand_name + ".lp")
    lp.generate_lp_file_borda_owa(owa, lp_file_name, params, votes)
    winners = lp.get_winners_from_lp(lp_file_name, params, candidates)
    os.remove(lp_file_name)
    winners = sorted(winners)
    return winners


def get_winners_bloc_owa(params, votes, candidates, owa, t_bloc):

    rand_name = str(rand.random())
    lp_file_name = str(rand_name + ".lp")
    lp.generate_lp_file_bloc_owa(owa, lp_file_name, params, votes, t_bloc)
    winners = lp.get_winners_from_lp(lp_file_name, params, candidates)
    os.remove(lp_file_name)
    winners = sorted(winners)
    return winners


def get_winners_stv(params, votes, candidates):

    winners = [] #[0] * params['orders']
    active = [True] * params['candidates']

    droop_quota = math.floor(params['voters'] / (params['orders'] + 1.) ) + 1

    votes_on_1 = [0.] * params['candidates']
    for i in range(params['voters']):
        votes_on_1[votes[i][0]] += 1

    v_power = [1.] * params['voters']

    while len(winners) + sum(active) > params['orders']:

        ctr = params['candidates']
        winner_id = 0
        while ctr > 0:

            if active[winner_id] and votes_on_1[winner_id] >= droop_quota:

                if params['pure']:
                    winners += [winner_id]
                else:
                    winners += [candidates[winner_id]]


                total = 0
                for i in range(params['voters']):
                    for j in range(params['candidates']):
                        if active[votes[i][j]]:
                            if votes[i][j] == winner_id:
                                for k in range(j + 1, params['candidates']):
                                    if active[votes[i][k]]:
                                        v_power[i] *= float(votes_on_1[winner_id] - droop_quota) / float(votes_on_1[winner_id])
                                        votes_on_1[votes[i][k]] += 1. * v_power[i]
                                        total += 1. * v_power[i]
                                        ctr = params['candidates']
                                        break
                            break

                votes_on_1[winner_id] = 0
                active[winner_id] = False

            ctr -= 1
            winner_id += 1
            winner_id %= params['candidates']


        loser_votes = droop_quota
        loser_id = 0
        for i in range(params['candidates']):
            if active[i] and votes_on_1[i] < loser_votes:
                loser_votes = votes_on_1[i]
                loser_id = i

        votes_on_1[loser_id] = 0
        for i in range(params['voters']):
            for j in range(params['candidates']):
                if active[votes[i][j]]:
                    if votes[i][j] == loser_id:
                        for k in range(j+1, params['candidates']):
                            if active[votes[i][k]]:
                                votes_on_1[votes[i][k]] += 1. * v_power[i]
                                break
                    break
        active[loser_id] = False

        #print(votes_on_1)

        #print("suma po: ", sum(votes_on_1))

    for i in range(params['candidates']):
        if active[i]:
            if params['pure']:
                winners += [i]
            else:
                winners += [candidates[i]]

    winners = sorted(winners)

    return winners


def get_winners_approx_pav(votes, params, algorithm="greedy"):

    if algorithm == "removal":
        return get_winners_approx_pav_removal(votes, params)
    elif algorithm == "greedy":
        return get_winners_approx_pav_greedy(votes, params)


def get_winners_approx_hb(votes, params, algorithm):

    if algorithm == "removal":
        return get_winners_approx_hb_removal(votes, params)
    elif algorithm == "greedy":
        return get_winners_approx_hb_greedy(votes, params)


def get_winners_approx_cc(votes, params):

    return get_winners_approx_cc_greedy(votes, params)


def get_winners_approx_pav_greedy(votes, params):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

    winners = []
    active = [True for _ in range(num_candidates)]

    for Z in range(num_winners):

        points = [0 for _ in range(num_candidates)]

        for i in range(num_voters):
            ctr = 1.
            for j in range(num_winners):
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


def get_winners_approx_pav_removal(votes, params):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

    removed = 0
    active = [True for _ in range(num_candidates)]

    for Z in range(num_candidates - num_winners):

        points = [0 for _ in range(num_candidates)]

        for i in range(num_voters):
            ctr = 1.
            for j in range(num_winners):
                if active[votes[i][j]]:
                    points[votes[i][j]] += (1./ctr)
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


def check_pav_score(votes, params, winners):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

    #print(votes)

    score = 0

    vector = [0.] * 100
    for i in range(10):
        vector[i] = 1.


    mask = num_winners
    for i in range(num_voters):
        ctr = 1.
        for j in range(num_candidates):
            if votes[i][j] in winners:
                score += (1. / ctr) * vector[i]
                ctr += 1

    return score


def check_hb_score(votes, params, winners):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

    #print(votes)

    score = 0

    for i in range(num_voters):
        ctr = 1.
        for j in range(num_candidates):
            if votes[i][j] in winners:
                score += (1. / ctr) * (num_candidates - j - 1)
                ctr += 1

    return score


def get_winners_approx_hb_greedy(votes, params):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

    winners = []
    active = [True for _ in range(num_candidates)]

    for Z in range(num_winners):

        points = [0 for _ in range(num_candidates)]

        for i in range(num_voters):
            ctr = 1.
            for j in range(num_candidates):
                if active[votes[i][j]]:
                    points[votes[i][j]] += (1./ctr) * (num_candidates - j - 1)
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


def get_winners_approx_hb_removal(votes, params):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

    removed = 0
    active = [True for _ in range(num_candidates)]

    for Z in range(num_candidates - num_winners):

        points = [0. for _ in range(num_candidates)]

        for i in range(num_voters):
            ctr = 1.
            for j in range(num_candidates):
                if active[votes[i][j]]:
                    points[votes[i][j]] += (1./ctr) * (num_candidates - j - 1)
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


def check_hb_dissat(votes, params, winners):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

    dissat = 0

    for i in range(num_voters):
        ctr = 1.
        for j in range(num_candidates):
            if votes[i][j] in winners:
                dissat += (1. / ctr) * (j)
                ctr += 1

    return dissat


def check_pav_dissat(votes, params, winners):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

    #print(votes)

    dissat = 0

    vector = [0. for _ in range(100)]
    for i in range(10,100):
        vector[i] = 1.

    for i in range(num_voters):
        ctr = 1.
        for j in range(num_candidates):
            if votes[i][j] in winners:
                dissat += ((1. / ctr) * vector[j])
                ctr += 1

    return dissat


def get_winners_approx_cc_greedy(votes, params):

    num_voters = params['voters']
    num_candidates = params['candidates']
    num_winners = params['orders']

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

