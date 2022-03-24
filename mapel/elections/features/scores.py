#!/usr/bin/env python

import math
import os
import sys
from typing import Union
from mapel.elections.features.approx import get_hb_score

import numpy as np
try:
    import pulp
except Exception:
    pulp = None


try:
    sys.path.append('/Users/szufa/PycharmProjects/abcvoting/')
    from abcvoting import abcrules, preferences
except ImportError:
    abcrules = None
    preferences = None

from mapel.elections._glossary import *
from mapel.elections.metrics import lp
from mapel.elections.other import winners2 as win


# MAIN FUNCTIONS
def highest_borda_score(election) -> int:
    """ Compute highest BORDA score of a given election """
    c = election.num_candidates
    vectors = election.get_vectors()
    borda = [sum([vectors[i][pos] * (c - pos - 1) for pos in range(c)])
             for i in range(c)]
    return max(borda) * election.num_voters


def highest_plurality_score(election) -> int:
    """ compute highest PLURALITY score of a given election"""
    first_pos = election.get_matrix()[0]
    return max(first_pos) * election.num_voters


def highest_copeland_score(election) -> Union[int, str]:
    """ compute highest COPELAND score of a given election """
    if election.model_id in LIST_OF_FAKE_MODELS:
        return 'None'

    election.votes_to_potes()

    scores = np.zeros([election.num_candidates])

    for i in range(election.num_candidates):
        for j in range(i + 1, election.num_candidates):
            result = 0
            for k in range(election.num_voters):
                if election.potes[k][i] < election.potes[k][j]:
                    result += 1
            if result > election.num_voters / 2:
                scores[i] += 1
                scores[j] -= 1
            elif result < election.num_voters / 2:
                scores[i] -= 1
                scores[j] += 1

    return max(scores)


def lowest_dodgson_score(election):
    """ compute lowest DODGSON score of a given election """
    if election.model_id in LIST_OF_FAKE_MODELS:
        return 'None', 'None'

    min_score = math.inf

    for target_id in range(election.num_candidates):

        # PREPARE N
        unique_potes, N = _potes_to_unique_potes(election.potes)

        e = np.zeros([len(N), election.num_candidates,
                      election.num_candidates])

        # PREPARE e
        for i, p in enumerate(unique_potes):
            for j in range(election.num_candidates):
                for k in range(election.num_candidates):
                    if p[target_id] <= p[k] + j:
                        e[i][j][k] = 1

        # PREPARE D
        D = [0 for _ in range(election.num_candidates)]
        threshold = math.ceil((election.num_voters + 1) / 2.)
        for k in range(election.num_candidates):
            diff = 0
            for i, p in enumerate(unique_potes):
                if p[target_id] < p[k]:
                    diff += N[i]
                if diff >= threshold:
                    D[k] = 0
                else:
                    D[k] = threshold - diff
        D[target_id] = 0  # always winning

        # file_name = f'{np.random.random()}.lp'
        # file_name = 'tmp_old.lp'
        # path = os.path.join(os.getcwd(), "trash", file_name)
        # lp.generate_lp_file_dodgson_score_old(path, N=N, e=e, D=D)
        # score, total_time = lp.solve_lp_dodgson_score(path)
        # lp.remove_lp_file(path)

        score = lp.solve_lp_file_dodgson_score(election, N=N, e=e, D=D)

        # if election.election_id == 'Impartial Culture_2' and target_id == 0:
        #     exit()

        if score < min_score:
            min_score = score

    print(min_score)
    return min_score, 0


def highest_cc_score(election, feature_params):
    if election.model_id in LIST_OF_FAKE_MODELS:
        return 'None', 'None'
    winners, obj_value, total_time = win.generate_winners(election=election,
                                             num_winners=feature_params['committee_size'],
                                             ballot="ordinal",
                                             type='borda_owa', name='cc')
    return obj_value, total_time


def highest_hb_score(election, feature_params):
    if election.model_id in LIST_OF_FAKE_MODELS:
        return 'None', 'None'
    winners, total_time = win.generate_winners(election=election,
                                             num_winners=feature_params['committee_size'],
                                             ballot="ordinal",
                                             type='borda_owa', name='hb')
    return get_hb_score(election, winners), total_time


def highest_pav_score(election, feature_params):
    if election.model_id in LIST_OF_FAKE_MODELS:
        return 'None', 'None'
    winners, obj_value, total_time = win.generate_winners(election=election,
                                             num_winners=feature_params['committee_size'],
                                             ballot="ordinal",
                                             type='bloc_owa', name='hb')
    return obj_value, total_time


# HELPER FUNCTIONS
def _potes_to_unique_potes(potes):
    """ Remove repetitions from potes (positional votes) """
    unique_potes = []
    n = []
    for pote in potes:
        flag_new = True
        for i, p in enumerate(unique_potes):
            if list(pote) == list(p):
                n[i] += 1
                flag_new = False
        if flag_new:
            unique_potes.append(pote)
            n.append(1)
    return unique_potes, n


