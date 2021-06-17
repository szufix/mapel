import random as rand
import numpy as np


def generate_mallows_election(num_voters=None, num_candidates=None, param=None, second_param=None):
    """ helper function: generate mallows elections"""

    PHI = param
    votes = mallowsProfile(num_candidates, num_voters, PHI, reverse=second_param)

    return votes



def phi_mallows_helper(num_candidates, rdis=None):

    def nswap(m):
        return int((m * (m - 1)) / 2)

    def generateVoteSwapArray(m):
        S = [[0 for _ in range(int(m * (m + 1) / 2) + 1)] for _ in range(m + 1)]

        for n in range(m + 1):
            for d in range(0, int((n + 1) * n / 2 + 1)):
                if d == 0:
                    S[n][d] = 1
                else:
                    S[n][d] = S[n][d - 1] + S[n - 1][d] - S[n - 1][d - n]
        return S

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

    def calcPolyExp(table, m):
        coeff = (nswap(m) + 1) * [0]
        for i in range(nswap(m) + 1):
            coeff[i] = i * table[i]
        return coeff

    # Function that for some given number of candidates m and some number rDisID\in[0,1] returns the dispersion parameter \phi
    # such that the relative expected distance of the resulting election sampled from Mallows model from identity is rDisID.
    # Table[i] is the number of votes at swap distance i from some m-candidate vote.
    def findPhiFromDistanceID(m, rDisID, table):
        coeffZ = calculateZpoly(m)
        for i in range(len(coeffZ)):
            coeffZ[i] = (rDisID / 2) * nswap(m) * coeffZ[i]
        coeffExp = calcPolyExp(table, m)
        finalcoeff = (nswap(m) + 1) * [0]
        for i in range(nswap(m) + 1):
            finalcoeff[i] = coeffExp[i] - coeffZ[i]
        r = np.roots(list(reversed(finalcoeff)))
        real_valued = r.real[abs(r.imag) < 1e-5]
        res = real_valued[real_valued >= 0]
        return res[0]

    m = num_candidates
    if rdis is None:
        rdis = rand.random()
    S = generateVoteSwapArray(m)
    new_param = findPhiFromDistanceID(m, rdis, S[m])
    #print(new_param)
    return new_param





def phiVector(phi, i):
    phi = float(phi)
    den = sum([phi ** j for j in range(i)])
    vec = [(phi ** (i - (j + 1))) / den for j in range(i)]
    return vec


def iDraw(vector):
    x = rand.uniform(0, 1)
    for i in range(len(vector)):
        if (x < sum(vector[0:i + 1])):
            return i


def mallows(center, PHI):
    m = len(center)

    PHI_VEC = [phiVector(PHI, qs + 1) for qs in range(m)]

    r = []
    for i in range(m):
        v = PHI_VEC[i]
        j = iDraw(v)
        r = r[0:j] + [center[i]] + r[j:]
    return r


def mallowsProfile(m, n, PHI, reverse=0):

    reversed_range = [i for i in range(m)]
    reversed_range.reverse()

    V = []
    for i in range(n):
        probability = rand.random()

        if probability >= reverse:
            V += [mallows(range(m), PHI)]
        else:
            V += [mallows(reversed_range, PHI)]
    return V