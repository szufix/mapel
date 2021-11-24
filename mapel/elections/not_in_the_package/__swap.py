#!/usr/bin/env python

import itertools


def vote_to_pote(vote: list) -> list:
    """ Return: Positional vote """
    return [vote.index(i) for i, _ in enumerate(vote)]


def swap_distance(vote_1: list, vote_2: list) -> int:
    """ Return: Swap distance between two votes """

    pote_1 = vote_to_pote(vote_1)
    pote_2 = vote_to_pote(vote_2)

    swap_distance = 0
    for i, j in itertools.combinations(pote_1, 2):
        if (pote_1[i] > pote_1[j] and pote_2[i] < pote_2[j]) or \
                (pote_1[i] < pote_1[j] and pote_2[i] > pote_2[j]):
            swap_distance += 1
    return swap_distance


if __name__ == "__main__":

    v1 = [1, 2, 3, 4, 5, 0]
    v2 = [3, 4, 2, 1, 0, 5]
    print(swap_distance(v1, v2))

