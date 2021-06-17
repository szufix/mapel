from math import *
from random import *
from sys import argv

from scipy.special import binom

# binom = comb



# WALSH

def f(i, j):
    if i < 0: return 0
    if j < 0: return 0
    return (1.0 / (2 ** (i + j))) * binom(i + j, i)


def probW(m, i, t):
    # probability that c_i is ranked t among m candidates
    return 0.5 * f(i - 1, m - t - (i - 1)) + 0.5 * f(i - t, m - i)


# RANDOM CONITZER

def random_conitzer(C):
    # generate a random vote from the Conitzer model for axis
    # C[0], ..., C[m-1]
    m = len(C)
    center = randint(0, m - 1)
    left = center
    right = center
    vote = [C[center]]
    for i in range(m - 1):
        L = False
        R = False

        if left > 0 and right < m - 1:
            if random() < 0.5:
                L = True
            else:
                R = True
        elif left > 0:
            L = True
        else:
            R = True

        if L:
            left -= 1
            vote.append(C[left])
        else:
            right += 1
            vote.append(C[right])

    return vote


# CONITZER

def g(m, i, j):
    if i > j: return 0
    if i == j: return 1.0 / m
    if i == 1 and j < m: return g(m, 1, j - 1) + 0.5 * g(m, 2, j)
    if j == m and i > 1: return g(m, i + 1, m) + 0.5 * g(m, i, m - 1)
    if i == 1 and j == m: return 1.0
    return 1.0 / m


#  return 0.5*g(m,i+1,j) + 0.5*g(m,i,j-1)


def probC(m, i, t):
    # probability that c_i is ranked t among m candidates
    p = 0.0
    if t == 1: return 1.0 / m

    if i - (t - 1) > 1:
        p += 0.5 * g(m, i - (t - 1), i - 1)
    elif i - (t - 1) == 1:
        p += g(m, i - (t - 1), i - 1)

    if i + (t - 1) < m:
        p += 0.5 * g(m, i + 1, i + (t - 1))
    elif i + (t - 1) == m:
        p += g(m, i + 1, i + (t - 1))

    return p


PRECISION = 1000
DIGITS = 4


def conitzer(m):
    P = [[0] * m for i in range(m)]
    for i in range(m):
        for j in range(m):
            P[i][j] = probC(m, i + 1, j + 1)
    return P


def simconitzer(m):
    P = [[0] * m for i in range(m)]
    T = 100000

    C = list(range(m))
    for t in range(T):
        if t % 10000 == 0: print(t)
        v = random_conitzer(C)
        for i in range(m):
            P[v[i]][i] += 1

    for j in range(m):
        for i in range(m):
            P[i][j] = str(int(PRECISION * (P[i][j] / T))).rjust(DIGITS)
    return P


def walsh(m):
    P = [[0] * m for i in range(m)]
    for i in range(m):
        for t in range(m):
            P[i][t] = probW(m, i + 1, t + 1)
    return P


# size = 20
# wal = walsh(size)
# for i in range(size):
#     for j in range(size):
#         print(round(wal[j][i],3), end=' ')
#     print()

# if len(argv) < 2:
#     print("Invocation:")
#     print("   python walsh.py conitzer|simconitzer|walsh m")
#     exit()
#
# m = int(argv[2])
# if argv[1] == "conitzer":
#     ff = conitzer
# elif argv[1] == "simconitzer":
#     ff = simconitzer
# elif argv[1] == "walsh":
#     ff = walsh
# else:
#     print("unknown distribution")
#     exit()

# P = ff(m)
#
# B = [0] * m
# for j in range(m):
#     for i in range(m):
#         print(str(int(PRECISION * (P[i][j]))).rjust(DIGITS), end="")
#         B[i] += P[i][j] * (m - j - 1)
#     print()
#
# S = 0
# for i in range(m):
#     print(B[i], end=" ")
#     S += B[i]
# print()
# print(S)
# print(" ")
# for i in range(m - 1):
#     print(B[i + 1] - B[i], end=" ")
# print()
#
# exit()