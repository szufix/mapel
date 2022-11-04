
import scipy.special
import numpy as np

from mapel.elections.features.scores import get_cc_score


def get_ranging_cc_score(election, committee_size=1):
    if election.fake:
        return 'None'

    x = election.num_candidates * scipy.special.lambertw(committee_size).real / committee_size

    scores = []
    for threshold in range(1, int(x)):
        scores.append(get_algorithm_p_committee(election, committee_size, x))

    return max(scores)


def get_algorithm_p_committee(election, committee_size, x):

    winners = set()

    active = [True for _ in range(election.num_voters)]

    for i in range(committee_size):
        tops = np.zeros([election.num_candidates])
        for v, vote in enumerate(election.votes):
            if active[v]:
                for c in range(int(x)):
                    tops[vote[c]] += 1

        winner_id = np.argmax(tops)
        winners.add(winner_id)
        for v, vote in enumerate(election.votes):
            if active[v]:
                for c in range(int(x)):
                    if winner_id == vote[c]:
                        active[v] = False
                        break

    return get_cc_score(election, winners)
