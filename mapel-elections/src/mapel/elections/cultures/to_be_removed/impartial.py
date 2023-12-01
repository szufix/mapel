#!/usr/bin/env python


def generate_approval_full_votes(num_voters: int = None,
                                 num_candidates: int = None,
                                 **kwargs) -> list:
    """ Return: approval votes where each voter approves all the candidates """
    vote = {i for i in range(num_candidates)}
    return [vote for _ in range(num_voters)]


def generate_approval_empty_votes(num_voters: int = None,
                                  **kwargs) -> list:
    """ Return: approval votes where each vote is empty """
    return [set() for _ in range(num_voters)]
