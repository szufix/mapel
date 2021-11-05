import math
import random as rand

from itertools import product
import numpy as np

from mapel.voting.elections_main import store_ordinal_election
from mapel.voting.metrics.main_ordinal_distances import compute_swap_bf_distance
from mapel.voting.objects.OrdinalElection import OrdinalElection

try:
    from sympy.utilities.iterables import multiset_permutations
except:
    pass


def generate_all_ordinal_elections(experiment, num_candidates, num_voters):
    """ At the same time generate elections and compute distances """

    id_ctr = 0

    experiment.elections = {}

    a = [i for i in range(num_candidates)]
    A = list(multiset_permutations(a))
    if num_voters == 3:
        X = [p for p in product([a], A, A)]
    elif num_voters == 4:
        X = [tuple(p) for p in product([a], A, A, A)]
    elif num_voters == 5:
        X = [tuple(p) for p in product([a], A, A, A, A)]

    Y = []
    for votes in X:
        ordered_votes = sorted(votes)
        Y.append(ordered_votes)

    Z = []
    tmp_ctr = 0
    for ordered_votes in Y:
        if ordered_votes not in Z:

            model_id = 'all'
            election_id = f'{model_id}_{id_ctr}'
            params = {'id_ctr': id_ctr}
            ballot = 'ordinal'

            new_election = OrdinalElection(experiment.experiment_id, election_id, votes=ordered_votes,
                                           num_voters=num_voters, num_candidates=num_candidates)

            for target_election in experiment.elections.values():
                if target_election.election_id != new_election.election_id:

                    obj_value, _ = compute_swap_bf_distance(target_election, new_election)

                    if obj_value == 0:
                        print('dist == 0')
                        break
            else:
                print(id_ctr, tmp_ctr)

                store_ordinal_election(experiment, model_id, election_id, num_candidates, num_voters, params,
                                       ballot, votes=ordered_votes)

                id_ctr += 1
                experiment.elections[election_id] = new_election
                Z.append(ordered_votes)

            tmp_ctr += 1

    print(len(X), len(Y), len(Z))

    # Compute distances between current election and all previous elections

    # for i in range(id_ctr):
    #     # experiment.elections[election_id]
    #
    #

    # if a dist=0 break
    # for
    # else:
    #    store the election
    #     model_id =''
    #     election_id = ''
    #     store_ordinal_election(experiment, model_id, election_id, num_candidates, num_voters,
    #                            params, ballot)



    # Store the distances
