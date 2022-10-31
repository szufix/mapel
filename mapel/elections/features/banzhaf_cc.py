
import scipy.special

from mapel.elections.features.scores import get_cc_score
import time


def get_banzhaf_cc_score(election, committee_size=1):
    if election.fake:
        return 'None'

    winners = set()
    BASE = {}
    BINOM = {}

    for _ in range(committee_size):
        highest_c = 0
        highest_score = 0
        for c in range(election.num_candidates):
            if c in winners:
                continue
            candidate_score = sum([voter_score(BASE, BINOM, election, v, c, committee_size, winners)
                                   for v in range(election.num_voters)])
            if candidate_score > highest_score:
                highest_score = candidate_score
                highest_c = c
        winners.add(highest_c)

    return get_cc_score(election, winners)


def voter_score(BASE, BINOM, election, v, c, committee_size, winners):

    potes = election.get_potes()
    m = election.num_candidates

    Wa = 0
    for i in range(potes[v][c]):
        if election.votes[v][i] in winners:
            Wa += 1
    Wb = len(winners) - Wa

    c_score = delta_c(BASE, BINOM, potes[v][c], Wa, Wb, m, committee_size)

    r = []
    for i in range(m):
        if potes[v][i] < potes[v][c]:
            r.append(0)
        else:
            r.append(1)

    d_score = sum(delta_d(BASE, BINOM, potes[v][d], Wa, Wb, m, committee_size, r[d])
                  for d in range(election.num_candidates) if d != c)

    return c_score + d_score


def delta_c(BASE, BINOM, pos, Wa, Wb, m, committee_size):
    name = f'c_{pos}_{Wa}_{Wb}'
    if name not in BASE:
        score = big_C(BINOM, pos, Wa, Wb, m, committee_size) * (m-pos-1)
        BASE[name] = score
    return BASE[name]


def delta_d(BASE, BINOM, pos, Wa, Wb, m, committee_size, r_d):
    name = f'd_{pos}_{Wa}_{Wb}_{r_d}'
    if name not in BASE:
        score = big_C(BINOM, pos, Wa, Wb, m, committee_size) * ( (m-pos-1) - (m-pos-1)*(1-r_d) )
        BASE[name] = score
    return BASE[name]


def big_C(BINOM, pos, Wa, Wb, m, committee_size):
    if Wa == 0:
        name = f'{pos}_{Wb}'
        if name not in BINOM:

            if m - pos - Wb <= 0 or 1 - committee_size - Wb <= 0:
                b_2 = 1
            else:
                b_2 = scipy.special.binom(m - pos - Wb, 1 - committee_size - Wb)

            BINOM[name] = b_2

        return BINOM[name]
    return 0
