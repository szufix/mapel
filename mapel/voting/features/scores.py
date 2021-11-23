#!/usr/bin/env python

import math
from typing import Union

import numpy as np
from pulp import *

from mapel.voting._glossary import *
from mapel.voting.metrics import lp


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


def highest_copeland_score(election) -> Union[int, None]:
    """ compute highest COPELAND score of a given election """
    if election.model_id in LIST_OF_FAKE_MODELS:
        return None

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

def lowest_dodgson_score(election) -> Union[int, None]:
    """ compute lowest DODGSON score of a given election """
    if election.model_id in LIST_OF_FAKE_MODELS:
        return None

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

        file_name = f'{np.random.random()}.lp'
        path = os.path.join(os.getcwd(), "trash", file_name)
        lp.generate_lp_file_dodgson_score(path, N=N, e=e, D=D)
        score = lp.solve_lp_dodgson_score(path)

        lp.remove_lp_file(path)

        if score < min_score:
            min_score = score

    return min_score


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


