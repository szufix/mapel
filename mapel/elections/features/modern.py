
import scipy.special

from mapel.elections.features.approx import get_cc_score


def get_banzhaf_cc_score(election, feature_params):

    committee_size = feature_params['committee_size']

    m = election.num_candidates
    k = committee_size
    W = set()

    for _ in range(k):
        highest_c = 0
        highest_score = 0
        for c in range(m):
            if c in W:
                continue
            candidate_score = sum([voter_score(election, v, c, k, W) for v in range(election.num_voters)])
            if candidate_score > highest_score:
                highest_score = candidate_score
                highest_c = c
        W.add(highest_c)

    return get_cc_score(election, W)


def voter_score(election, v, c, k, W):
    potes = election.votes_to_potes()

    m = election.num_candidates

    Wa = 0
    for i in range(potes[v][c]):
        if election.votes[v][i] in W:
            Wa += 1
    Wb = len(W) - Wa - 1

    c_score = delta_c(potes[v][c], Wa, Wb, m, k)

    r = []
    for i in range(m):
        if potes[v][i] < potes[v][c]:
            r.append(0)
        else:
            r.append(1)

    d_score = sum(delta_d(potes[v][d], Wa, Wb, m, k, r[d]) for d in range(election.num_candidates) if d != c)

    return c_score + d_score


def delta_c(pos, Wa, Wb, m, k):
    score = 0
    for t in range(1, k+1):
        score += big_C(pos, t, Wa, Wb, m, k) * gamma(pos, t, m)
    return score


def delta_d(pos, Wa, Wb, m, k, r_d):
    score = 0
    for t in range(1, k+1):
        score += big_C(pos, t, Wa, Wb, m, k) * (gamma(pos, t, m) - gamma(pos, t+r_d, m))
    return score


def big_C(pos, t, Wa, Wb, m, k):
    if t > Wa:
        if pos - 1 - Wa <= 0 or t - 1 - Wa <= 0:
            b_1 = 1
        else:
            b_1 = scipy.special.binom(pos - 1 - Wa, t - 1 - Wa)
        if m - pos - Wb <= 0 or t - k - Wb <= 0:
            b_2 = 1
        else:
            b_2 = scipy.special.binom(m - pos - Wb, t - k - Wb)
        return b_1 * b_2
    else:
        return 0


def gamma(pos, t, m):
    if t == 1:
        return m-pos-1
    else:
        return 0
