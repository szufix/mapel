import random as rand
import numpy as np
import os
import pickle
#Given the number m of candidates and a phi\in [0,1] function computes the expected number of swaps in a vote sampled from Mallows model
def calculateExpectedNumberSwaps(num_candidates,phi):
    res= phi*num_candidates/(1-phi)
    for j in range(1,num_candidates+1):
        res = res + (j*(phi**j))/((phi**j)-1)
    return res

#Given the number m of candidates and a absolute number of expected swaps exp_abs, this function returns a value of phi such that in a vote sampled from Mallows model with this parameter the expected number of swaps is exp_abs
def phi_from_relphi(num_candidates,relphi=None):
    if relphi is None:
        relphi = rand.random()
    if relphi==1:
        return 1
    exp_abs=relphi*(num_candidates*(num_candidates-1))/4
    low=0
    high=1
    while low <= high:
        mid = (high + low) / 2
        cur=calculateExpectedNumberSwaps(num_candidates, mid)
        if abs(cur-exp_abs)<1e-5:
            return mid
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1

def computeInsertionProbas(i,phi):
    probas = (i+1)*[0]
    for j in range(i+1):
        probas[j]=pow(phi,(i+1)-(j+1))
    return probas

def weighted_choice(choices):
    total=0
    for w in choices:
        total=total + w
    r = rand.uniform(0, total)
    upto = 0.0
    for i,w in enumerate(choices):
        if upto + w >= r:
            return i
        upto =upto + w
    assert False, "Shouldn't get here"

def mallowsVote(m,insertion_probabilites_list):
    vote=[0]
    for i in range(1,m):
        index = weighted_choice(insertion_probabilites_list[i-1])
        vote.insert(index,i)
    return vote


def generate_mallows_election(num_voters, num_candidates,params):
    insertion_probabilites_list = []
    for i in range(1, num_candidates):
        insertion_probabilites_list.append(computeInsertionProbas(i, params['phi']))
    V = []
    for i in range(num_voters):
        vote = mallowsVote(num_candidates, insertion_probabilites_list)
        if params['weight']>0:
            probability = rand.random()
            if probability >= params['weight']:
                vote.reverse()
        V += [vote]
    return V

def calculateZpoly(m):
    res = [1]
    for i in range(1, m + 1):
        mult = [1] * i
        res2 = [0] * (len(res) + len(mult) - 1)
        for o1, i1 in enumerate(res):
            for o2, i2 in enumerate(mult):
                res2[o1 + o2] += i1 * i2
        res = res2
    return res


def evaluatePolynomial(coeff, x):
    res = 0
    for i, c in enumerate(coeff):
        res += c * (x ** i)
    return res


def calculateZ(m, phi):
    coeff = calculateZpoly(m)
    # print(coeff)
    return evaluatePolynomial(coeff, phi)


#mat[i][j] is the probability with which candidate i ends up in position j
def mallowsMatrix(num_candidates,lphi,pos,normalize=True):
    mat = np.zeros([num_candidates,num_candidates])
    if normalize:
        phi = phi_from_relphi(num_candidates,lphi)
    else:
        phi=lphi
    Z = calculateZ(num_candidates, phi)
    for i in range(num_candidates):
        for j in range(num_candidates):
            freqs=[pos[k][i][j] for k in range(1+int(num_candidates*(num_candidates-1)/2))]
            unnormal_prob=evaluatePolynomial(freqs,phi)
            mat[i][j]=unnormal_prob/Z
    return mat


def get_mallows_matrix(num_candidates, params,normalize=True):
    lphi=params['norm-phi']
    weight=params['weight']
    # print(lphi, weight)
    try:
        path = os.path.join(os.getcwd(), 'mapel','voting', 'elections', 'mallows_positionmatrices',str(num_candidates) + "_matrix.txt")
        with open(path, "rb") as file:
            #print(path)
            pos = pickle.load(file)
    except FileNotFoundError:
        print("Mallows matrix only supported for up to 30 candidates")
    #print(pos)
    mat1 = mallowsMatrix(num_candidates, lphi, pos,normalize)
    res = np.zeros([num_candidates,num_candidates])
    for i in range(num_candidates):
        for j in range(num_candidates):
            res[i][j] = weight * mat1[i][j] + (1 - weight) * mat1[i][num_candidates - 1 - j]
    return res


def get_mallows_vectors(num_candidates, fake_param):
    return get_mallows_matrix(num_candidates, fake_param).transpose()


def generate_mallows_party(num_voters=None, num_candidates=None,
                           election_model=None, params=None):
    num_parties = params['num_parties']
    num_winners = params['num_winners']
    party_size = num_winners

    params['phi'] = phi_from_relphi(num_parties, relphi=params['main-phi'])
    mapping = generate_mallows_election(num_voters, num_parties, params)[0]

    params['phi'] = phi_from_relphi(num_parties, relphi=params['norm-phi'])
    votes = generate_mallows_election(num_voters, num_parties, params)

    for i in range(num_voters):
        for j in range(num_parties):
            votes[i][j] = mapping[votes[i][j]]

    new_votes = [[] for _ in range(num_voters)]

    for i in range(num_voters):
        for j in range(num_parties):
            for w in range(party_size):
                _id = votes[i][j] * party_size + w
                new_votes[i].append(_id)

    return new_votes


def generate_approval_mallows_election(num_voters=None, num_candidates=None, params=None):
    # central_vote = set()
    # for c in range(num_candidates):
    #     if rand.random() <= params['p']:
    #         central_vote.add(c)
    k = int(params['p']*num_candidates)
    central_vote = {i for i in range(k)}

    votes = [0 for _ in range(num_voters)]
    for v in range(num_voters):
        vote = set()
        for c in range(num_candidates):
            if rand.random() <= params['phi']:
                if rand.random() <= params['p']:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote

    return votes


def generate_approval_raw_mallows_election(num_voters=None, num_candidates=None, params=None):

    k = int(params['p']*num_candidates)
    central_vote = {i for i in range(k)}

    votes = [0 for _ in range(num_voters)]
    for v in range(num_voters):
        vote = set()
        for c in range(num_candidates):
            if c in central_vote:
                if rand.random() <= params['phi']:
                    vote.add(c)
            else:
                if rand.random() < 1 - params['phi']:
                    vote.add(c)
        votes[v] = vote

    return votes


def generate_approval_disjoint_mallows_election(num_voters=None, num_candidates=None, params=None):

    k = int(params['p']*num_candidates)
    votes = [0 for _ in range(num_voters)]


    central_vote_1 = {i for i in range(k)}

    for v in range(int(num_voters/2)):
        vote = set()
        for c in range(num_candidates):
            if c in central_vote_1:
                if rand.random() <= params['phi']:
                    vote.add(c)
            else:
                if rand.random() < 1 - params['phi']:
                    vote.add(c)
        votes[v] = vote


    central_vote_2 = {i+k for i in range(k)}

    for v in range(int(num_voters/2), num_voters):
        vote = set()
        for c in range(num_candidates):
            if c in central_vote_2:
                if rand.random() <= params['phi']:
                    vote.add(c)
            else:
                if rand.random() < 1 - params['phi']:
                    vote.add(c)
        votes[v] = vote

    return votes
