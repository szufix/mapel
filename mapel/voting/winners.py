#!/usr/bin/env python
import math


def compute_plurality_winners(election=None, num_winners=1):
    """ Compute Plurality winners for a given election """

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        scores[vote[0]] += 1
    candidates = [i for i in range(election.num_candidates)]
    ranking = [x for _, x in sorted(zip(scores, candidates), reverse=True)]
    return ranking[0:num_winners]


def compute_borda_winners(election=None, num_winners=1):
    """ Compute Borda winners for a given election """

    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        for i in range(election.num_candidates):
            scores[vote[i]] += election.num_candidates - i - 1
    candidates = [i for i in range(election.num_candidates)]
    ranking = [x for _, x in sorted(zip(scores, candidates), reverse=True)]
    return ranking[0:num_winners]



def compute_stv_winners(election=None, num_winners=1):

    #print(votes)
    # print(election.votes)

    winners = [] #[0] * params['orders']
    active = [True] * election.num_candidates

    droop_quota = math.floor(election.num_voters / (num_winners + 1.) ) + 1
    #print(droop_quota)

    votes_on_1 = [0.] * election.num_candidates
    for i in range(election.num_voters):
        votes_on_1[election.votes[i][0]] += 1

    v_power = [1.] * election.num_voters

    #print(votes_on_1)

    while len(winners) + sum(active) > num_winners:

        #print("---")
        #print(votes_on_1)

        #print("part 1")

        ctr = election.num_candidates
        winner_id = 0
        while ctr > 0:
            #print(winner_id)
            if active[winner_id] and votes_on_1[winner_id] >= droop_quota:
                #orders += [candidates[winner_id]]

                winners += [winner_id]
                # else:
                #     winners += [candidates[winner_id]]

                #print("kandydat: ", votes_on_1[winner_id])
                #print("suma przed: ",sum(votes_on_1))
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
                #print("przechodzi: ",total)
                votes_on_1[winner_id] = 0
                active[winner_id] = False
                #print("suma po: ",sum(votes_on_1))

                #print(votes_on_1)
            ctr -= 1
            winner_id += 1
            winner_id %= election.num_candidates


        #print("part 2")

        #print(votes_on_1)

        #print("suma przed: ", sum(votes_on_1))

        loser_votes = droop_quota
        loser_id = 0
        for i in range(election.num_candidates):
            if active[i] and votes_on_1[i] < loser_votes:
                loser_votes = votes_on_1[i]
                loser_id = i

        #print("lv: ",loser_votes)
        #print(votes_on_1[loser_id])
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

        #print(votes_on_1)

        #print("suma po: ", sum(votes_on_1))

    for i in range(election.num_candidates):
        if active[i]:
            # if params['pure']:
            winners += [i]
            # else:
            #     winners += [candidates[i]]

    winners = sorted(winners)

    """
    #hill = [0] * 10
    for h in range(10):
        hill = 0
        player = orders[h]
        for i in range(params['voters']):
            for j in range(params['candidates']):
                if candidates[votes[i][j]] == player:
                    hill += j

        print hill/float(params['voters'])
    """

    return winners