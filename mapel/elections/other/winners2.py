#!/usr/bin/env python

import math
import os

import numpy as np

from mapel.elections.metrics import lp as lp


def generate_winners(election=None, num_winners=1, ballot="ordinal", type=None, name=None):
    votes, num_voters, num_candidates = election.votes, election.num_voters, election.num_candidates
    params = {"orders": num_winners,
              "pure": True,
              "elections": 1,
              'candidates': num_candidates,
              'voters': num_voters}
    rule = {'type_id': type,
            'name': name,
            'length': num_candidates}
    winners, total_time = get_winners(params, votes, rule, ballot)
    return winners, total_time


def get_winners(params, votes, rule, ballot='ordinal'):
    if ballot == "ordinal":
        return get_ordinal_winners(params, votes, rule)
    elif ballot == "approval":
        return get_approval_winners(params, votes, rule)


def get_approval_winners(params, elections, rule):
    if rule['type_id'] == 'app_cc':
        all_winners = []
        for i in range(params['elections']):
            winners = get_winners_app_cc(params, elections['votes'][i], elections['candidates'][i])
            all_winners += winners
        return all_winners


# Need update
def get_ordinal_winners(params, votes, rule):
    if rule['type_id'] == 'scoring':
        scoring = get_rule(rule['name'], rule['length'])
        all_winners = []
        for i in range(params['elections']):
            winners = get_winners_scoring(params, votes[i], params['candidates'], scoring)
            all_winners += winners
        return all_winners

    elif rule['type_id'] == 'borda_owa':
        owa = get_rule(rule['name'], rule['length'])
        all_winners = []
        for i in range(params['elections']):
            winners, total_time = get_winners_borda_owa(params, votes, owa)
            all_winners += winners
        # print(obj_vaue)
        return all_winners, total_time

    elif rule['type_id'] == 'bloc_owa':
        owa = get_rule(rule['name'], rule['length'])
        # t_bloc = rule['special']
        t_bloc = params["orders"]
        all_winners = []
        for i in range(params['elections']):
            winners, total_time = get_winners_bloc_owa(params, votes, owa, t_bloc)
            all_winners += winners
        # print(winners)
        return all_winners, total_time

    elif rule['type_id'] == 'stv':
        all_winners = []
        for i in range(params['elections']):
            winners = get_winners_stv(params, votes[i], params['candidates'])
            all_winners += winners
        return all_winners

    elif rule['type_id'] == 'experiment':
        all_winners = []
        for i in range(params['elections']):
            winners = []
            for j in range(params['voters']):
                winners += [votes[i][j]]

            all_winners += winners
        return all_winners

    elif rule['type_id'] == "approx_cc":
        all_winners = []
        winners = get_winners_approx_cc(votes, params)
        all_winners += winners
        return all_winners

    elif rule['type_id'] == "approx_hb":
        all_winners = []
        winners = get_winners_approx_hb(votes, params, 'greedy')
        all_winners += winners
        return all_winners

    elif rule['type_id'] == "approx_pav":
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
        rule[0] = 1.
    elif name == 'cc':
        rule[0] = 1.
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
                r = np.random.randint(left,right)
                tmp_candidate = winners[right]
                winners[right] = tmp_candidates[breaking_point+r]
                tmp_candidates[breaking_point + r] = tmp_candidate
                right -= 1
    winners = sorted(winners)
    return winners


def get_winners_borda_owa(params, votes, owa):
    return lp.solve_lp_borda_owa(params, votes, owa)

# def get_winners_borda_owa(params, votes, candidates, owa):
#     rand_name = str(np.random.random())
#     lp_file_name = str(rand_name + ".lp")
#     lp.generate_lp_file_borda_owa(owa, lp_file_name, params, votes)
#     winners, obj_value, total_time = lp.get_winners_from_lp(lp_file_name, params, candidates)
#     os.remove(lp_file_name)
#     winners = sorted(winners)
#     return winners, obj_value, total_time


def get_winners_bloc_owa(params, votes, owa, t_bloc):
    return lp.solve_lp_bloc_owa(params, votes, owa, t_bloc)



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


# def get_winners_approx_pav(votes, params, algorithm='greedy'):
#
#     if algorithm == "removal":
#         return get_winners_approx_pav_removal(votes, params)
#     elif algorithm == "greedy":
#         return get_winners_approx_pav_greedy(votes, params)




