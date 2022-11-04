import numpy as np
import mapel.elections.cultures.mallows as mallows
import itertools as it
import copy
import random
import mapel.elections.cultures.sampling.samplemat as smpl


# Auxiliary functions

def distribute_in_matrix(n,m):
    if m == 0:
        return []
    k = n // m
    r = n - k * m
    matrix = []
    for i in range(m):
        row = [k for _ in range(m)]
        for j in range(i,i+r):
            if j >= m:
                j = j - m
            row[j] = row[j] + 1
        matrix.append(row)
    return matrix

def distribute_in_block_matrix(n,blocks):
    before = 0
    after = sum(blocks)
    matrix = []
    for b in blocks:
        after = after - b
        block = distribute_in_matrix(n,b)
        for row in block:
            matrix.append([0 for _ in range(before)] + row + [0 for _ in range(after)])
        before = before + b
    return(matrix)


def draw_election(matrix):
    return smpl.sample_election_using_permanent(matrix)


def build_perms(matrix):
    if len(matrix[0]) == 0:
        return [[]]
    perms = []
    last_col = []
    for j in range(len(matrix)):
        last_col.append(matrix[j].pop())
    if sum(last_col) > 0:
        for last_cand, elig in enumerate(last_col):
            if elig <= 0:
                continue
            zeroed_vector = matrix[last_cand][:]
            matrix[last_cand][:] = len(matrix[last_cand]) * [0]
            for perm in build_perms(matrix):
                perms.append(perm + [last_cand])
            matrix[last_cand][:] = zeroed_vector[:]
    for j in range(len(matrix)):
        matrix[j].append(last_col[j])
    return perms


# Models

def generate_un_from_matrix_votes(num_voters=None, num_candidates=None):
    """ Generate real elections that have UN positionwise matrix """
    matrix = distribute_in_matrix(num_voters, num_candidates)
    return draw_election(matrix)


def generate_idan_part_votes(num_voters=None, num_candidates=None, params=None):
    """ Generate real elections between (ID) and (AN) """
    if params is None or not ('part_share' in params):
        print("IDAN_part generation : params None : random param generated")
        part_size = np.random.choice(range(num_voters))
    else:
        part_size = params['part_share'] * (num_voters)
    part_size = int(round(part_size))
    id_share = num_voters - (part_size // 2)
    op_share = part_size // 2
    votes = [[j for j in range(num_candidates)] for _ in range(id_share)]
    votes = votes + [[(num_candidates - j - 1) for j in range(num_candidates)] for _ in range(op_share)]
    return votes

def generate_idun_part_votes(num_voters=None, num_candidates=None, params=None):
    """ Generate elections realizing linear combinations of pos-matrices between (ID) and (UN) """
    if params is None or not ('part_share' in params):
        print("IDUN_part generation : params None : random param generated")
        part_size = np.random.choice(range(num_voters))
    else:
        part_size = params['part_share'] * (num_voters)
    part_size = int(round(part_size))
    id_share = num_voters - part_size
    un_share = part_size
    votes = [[j for j in range(num_candidates)] for _ in range(id_share)]
    votes = votes + draw_election(distribute_in_matrix(un_share,num_candidates))
    return votes

def generate_idst_part_votes(num_voters=None, num_candidates=None, params=None):
    """ Generate elections realizing linear combinations of pos-matrices between (ID) and (ST) """
    if params is None or not ('part_share' in params):
        print("IDST_part generation : params None : random param generated")
        part_size = np.random.choice(range(num_voters))
    else:
        part_size = params['part_share'] * (num_voters)
    part_size = int(round(part_size))
    id_share = num_voters - part_size
    st_share = part_size
    topsize = num_candidates // 2
    bottomsize = num_candidates - topsize
    votes_id = [[j for j in range(num_candidates)] for _ in range(id_share)]
    votes_st = draw_election(distribute_in_block_matrix(st_share,[topsize,bottomsize]))
    return votes_id + votes_st

def generate_anun_part_votes(num_voters=None, num_candidates=None, params=None):
    """ Generate elections realizing linear combinations of pos-matrices between (AN) and (UN) """
    if params is None or not ('part_share' in params):
        print("ANUN_part generation : params None : random param generated")
        part_size = np.random.choice(range(num_voters))
    else:
        part_size = params['part_share'] * (num_voters)
    part_size = int(round(part_size))
    id_share = (num_voters - part_size) // 2
    op_share = num_voters - part_size - id_share
    un_share = num_voters - id_share - op_share
    votes = [[j for j in range(num_candidates)] for _ in range(id_share)]
    votes = votes + [[(num_candidates - j - 1) for j in range(num_candidates)] for _ in range(op_share)]
    votes = votes + draw_election(distribute_in_matrix(un_share,num_candidates))
    return votes

def generate_anst_part_votes(num_voters=None, num_candidates=None, params=None):
    """ Generate elections realizing linear combinations of pos-matrices between (AN) and (ST) """
    if params is None or not ('part_share' in params):
        print("ANST_part generation : params None : random param generated")
        part_size = np.random.choice(range(num_voters))
    else:
        part_size = params['part_share'] * (num_voters)
    part_size = int(round(part_size))
    id_share = (num_voters - part_size) // 2
    op_share = num_voters - part_size - id_share
    st_share = num_voters - id_share - op_share
    topsize = num_candidates // 2
    bottomsize = num_candidates - topsize
    votes = [[j for j in range(num_candidates)] for _ in range(id_share)]
    votes = votes + [[(num_candidates - j - 1) for j in range(num_candidates)] for _ in range(op_share)]
    votes = votes + draw_election(distribute_in_block_matrix(st_share,[topsize,bottomsize]))
    return votes

def generate_unst_part_votes(num_voters=None, num_candidates=None, params=None):
    """ Generate elections realizing linear combinations of pos-matrices between (UN) and (ST) """
    if params is None or not ('part_share' in params):
        print("UNST_part generation : params None : random param generated")
        part_size = np.random.choice(range(num_voters))
    else:
        part_size = params['part_share'] * (num_voters)
    part_size = int(round(part_size))
    un_share = num_voters - part_size
    st_share = part_size
    topsize = num_candidates // 2
    bottomsize = num_candidates - topsize
    votes = draw_election(distribute_in_matrix(un_share,num_candidates))
    votes = votes + draw_election(distribute_in_block_matrix(st_share,[topsize,bottomsize]))
    return votes

def generate_idan_mallows_votes(num_voters=None, num_candidates=None, params=None):
    if params is None or not ('scaled-phi' in params):
        print("IDAN_mallows generation : params None : random param generated")
        is_reversed = np.random.choice([True,False])
        phi = mallows.phi_from_relphi(num_candidates, relphi=np.random.uniform())
    else:
        if params['scaled-phi'] > 0.5:
            is_reversed = True
            phi = mallows.phi_from_relphi(num_candidates, (1-params['scaled-phi'])*2)
        else:
            is_reversed = False
            phi = mallows.phi_from_relphi(num_candidates, params['scaled-phi']*2)
    id_share = num_voters // 2
    op_share = num_voters - id_share
    votes_id = [[j for j in range(num_candidates)] for _ in range(id_share)]
    votes_op = mallows.generate_mallows_votes(op_share,num_candidates,{'phi' : phi})
    if is_reversed:
        for v in votes_op:
            v.reverse()
    return votes_id + votes_op

def generate_idst_mallows_votes(num_voters=None, num_candidates=None, params=None):
    if params is None or not ('phi' in params):
        print("IDST_mallows generation : params None : random param generated")
        phi = mallows.phi_from_relphi(num_candidates, relphi=np.random.uniform())
    else:
        phi = params['phi']
    better = num_candidates // 2
    worse = num_candidates - better
    votes_better = mallows.generate_mallows_votes(num_voters,better,{'phi' : phi})
    votes_worse = mallows.generate_mallows_votes(num_voters,worse,{'phi' : phi})
    votes = []
    for b, w in zip(votes_better,votes_worse):
        w_ = [c + better for c in w]
        v = b + w_
        votes.append(v)
    return votes

def generate_anun_mallows_votes(num_voters=None, num_candidates=None, params=None):
    mallows_params = {}
    if params is None or not ('phi' in params):
        print("IDST_mallows generation : params None : random param generated")
        mallows_params['phi'] = mallows.phi_from_relphi(num_candidates, relphi=np.random.uniform())
    else:
        mallows_params['phi'] = params['phi']
    mallows_params['weight'] = 0.5
    return mallows.generate_mallows_votes(num_voters,num_candidates,mallows_params)

def generate_unst_mallows_votes(num_voters=None, num_candidates=None, params=None):
    if params is None or not ('phi' in params):
        print("IDST_mallows generation : params None : random param generated")
        phi = mallows.phi_from_relphi(num_candidates, relphi=np.random.uniform()) / 2
    else:
        phi = params['phi'] / 2
    better = num_candidates // 2
    worse = num_candidates - better
    votes = draw_election(distribute_in_block_matrix(num_voters,[better,worse]))
    # The next part works poorly for odd number of candidates (the last one is always from worse part)
    for v in votes:
        for i in range(better):
            if np.random.random() < phi:
                c = v[i]
                v[i] = v[i + better]
                v[i + better] = c
    return votes

def generate_unst_topsize_votes(num_voters=None, num_candidates=None, params=None):
    """ Generate kind of real elections between (UN) and (ST) """
    if params is None or not ('top_share' in params):
        print("UNST_topsize generation : params None : random param generated")
        top_share = np.random.random()
    else:
        top_share = params['top_share']
    top_size = int(round(top_share * num_candidates))
    better = top_size
    worse = num_candidates - top_size
    matrix = distribute_in_block_matrix(num_voters,[better,worse])
    return draw_election(matrix)

def generate_idst_blocks_votes(num_voters=None, num_candidates=None, params=None):
    """ Generate kind of real elections between (ID) and (UN) """
    if params is None or not ('no_blocks' in params):
        print("IDST_blocks generation : params None : random param generated")
        no_blocks = np.random.choice(range(num_candidates + 1))
    else:
        no_blocks = params['no_blocks']
    no_blocks = int(round(no_blocks))
    k = num_candidates // no_blocks
    r = num_candidates - k * no_blocks
    blocks = [k for _ in range(no_blocks)]
    with_one_more = list(np.random.choice(range(no_blocks),r,replace=False))
    for i in with_one_more:
        blocks[i] = blocks[i] + 1
    print(blocks)
    matrix = distribute_in_block_matrix(num_voters,blocks)
    return draw_election(matrix)
