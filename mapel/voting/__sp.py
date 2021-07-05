from math import *
from random import *
from sys import argv

from scipy.special import binom

# binom = comb





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