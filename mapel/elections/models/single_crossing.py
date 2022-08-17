#!/usr/bin/env python

from random import *

import numpy as np


def generate_ordinal_single_crossing_votes(num_voters: int = None,
                                           num_candidates: int = None,
                                           params: dict = None) -> np.ndarray:
    """ helper function: generate simple single-crossing elections"""

    if params is None:
        params = {}

    domain_id = params.get('domain_id', 'naive')

    votes = np.zeros([num_voters, num_candidates])

    # GENERATE DOMAIN
    if domain_id == 'naive':

        domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)

        domain = [[i for i in range(num_candidates)] for _ in range(domain_size)]

        for line in range(1, domain_size):

            poss = []
            for i in range(num_candidates - 1):
                if domain[line - 1][i] < domain[line - 1][i + 1]:
                    poss.append([domain[line - 1][i], domain[line - 1][i + 1]])

            r = np.random.randint(0, len(poss))  # random swap

            for i in range(num_candidates):

                domain[line][i] = domain[line - 1][i]

                if domain[line][i] == poss[r][0]:
                    domain[line][i] = poss[r][1]

                elif domain[line][i] == poss[r][1]:
                    domain[line][i] = poss[r][0]

    # elif domain_id == 'naive_pf':
    #     domain = naive_sampler.sample()
    elif domain_id == 'uniform':
        return uniform_sampler.sampleElection()
    else:
        print('No such domain!')
        domain = []

    domain_size = len(domain)

    # GENERATE VOTES

    for j in range(num_voters):
        r = np.random.randint(0, domain_size)
        votes[j] = list(domain[r])

    return votes


def get_single_crossing_matrix(num_candidates: int) -> np.ndarray:
    return get_single_crossing_vectors(num_candidates).transpose()


def get_single_crossing_vectors(num_candidates: int) -> np.ndarray:

    matrix = np.zeros([num_candidates, num_candidates])

    for i in range(num_candidates):
        for j in range(num_candidates-i):
            matrix[i][j] = i+j

    for i in range(num_candidates):
        for j in range(i):
            matrix[i][j] += 1

    sums = [1]
    for i in range(num_candidates):
        sums.append(sums[i]+i)

    for i in range(num_candidates):
        matrix[i][i] += sums[i]
        matrix[i][num_candidates-i-1] -= i

    for i in range(num_candidates):
        denominator = sum(matrix[i])
        for j in range(num_candidates):
            matrix[i][j] /= denominator

    return matrix


######################


all_computed = 0  # internal counter

SC_ELECTION = dict()  # function mapping (vote,n) to the number of single-crossing elections that include vote and n-1 other votes


class SCNode:
    def __init__(self, vote):
        self.vote = tuple(vote)
        self.next = []  # all next nodes reachable via single swap
        self.all_next = set()  # all next nodes reachable via any number of swaps
        self.era = 0  # era for search through the graph
        self.domains = -1  # number of different domains that can be
        # generated starting with this vote

    def count_domains(self):
        if self.domains != -1:
            return self.domains
        if len(self.next) == 0:
            self.domains = 1
            return 1

        self.domains = 0
        for node in self.next:
            self.domains += node.count_domains()
        return self.domains

    def sample_domain(self):
        D = [self.vote]
        if len(self.next) == 0: return D
        S = 0
        for node in self.next:
            S += node.count_domains()
        R = randrange(S)
        T = 0
        for node in self.next:
            T += node.count_domains()
            if R < T:
                return D + node.sample_domain()

    def sample_domain_naive(self):
        D = [self.vote]
        if len(self.next) == 0: return D
        node = choice(self.next)
        return D + node.sample_domain_naive()

    def domains_difference(self):
        if len(self.next) == 0: return 0
        A = max(node.count_domains() for node in self.next)
        B = min(node.count_domains() for node in self.next)
        for node in self.next:
            print(f"d: {node.count_domains()}")
        print(f"A: {A}, B:{B}, A-B:{A - B}")
        return A - B

    def generate_all_next(self):
        global all_computed
        if self.all_next: return
        for node in self.next:
            node.generate_all_next()
            self.all_next |= node.all_next
        self.all_next.add(self.vote)
        all_computed += 1
        # print( all_computed, self.all_next )

    def count_elections(self, n, nodes):
        if n == 0: return 0
        if n == 1: return 1
        if (self.vote, n) in SC_ELECTION:
            return SC_ELECTION[(self.vote, n)]

        S = 0
        for vote in self.all_next:
            S += nodes[vote].count_elections(n - 1, nodes)
        SC_ELECTION[(self.vote, n)] = S
        # print( SC_ELECTION[ (self.vote, n) ] )
        return SC_ELECTION[(self.vote, n)]

    def sample_election(self, n, nodes):
        E = [self.vote]
        if n == 1: return E
        S = 0
        for vote in self.all_next:
            S += nodes[vote].count_elections(n - 1, nodes)
        R = randrange(S)
        T = 0
        for vote in self.all_next:
            T += nodes[vote].count_elections(n - 1, nodes)
            if R < T:
                return E + nodes[vote].sample_election(n - 1, nodes)


def makeSCgraph(m):
    nodes = {}
    vote = list(range(m))
    node = SCNode(vote)
    nodes[tuple(vote)] = node

    def recSCgraph(node):
        vote = list(node.vote)
        for i in range(m - 1):
            if vote[i] < vote[i + 1]:
                nv = vote.copy()
                nv[i], nv[i + 1] = nv[i + 1], nv[i]
                tp = tuple(nv)
                if not (tp in nodes):
                    nd = SCNode(nv)
                    nodes[tp] = nd
                    recSCgraph(nd)
                nd = nodes[tp]
                node.next.append(nd)

    recSCgraph(node)
    return nodes


def visitSC(node, f, new_era):
    """visit all votes that can follow the one in node and call f on each"""
    m = len(node.vote)
    v = list(node.vote)
    f(node)
    node.era = new_era
    for new_node in node.next:
        if new_node.era != new_era:
            visitSC(new_node, f, new_era)


def getSCDomainSampler(m):
    nodes = makeSCgraph(m)
    top = tuple(range(m))
    return nodes[top]


### classes for sampling single-crossing domains uniformly
### at random and naively (they work in reasonable time until m = 10
### candidates

class scDomainUniformSampler:
    def __init__(self, m):
        self.nodes = makeSCgraph(m)
        self.top = tuple(range(m))
        self.top_node = self.nodes[self.top]

    def sample(self):
        return self.top_node.sample_domain()

    def prepareElectionSampler(self, n):
        self.n = n
        self.top_node.generate_all_next()
        self.top_node.count_elections(n, self.nodes)

    def sampleElection(self):
        return self.top_node.sample_election(self.n, self.nodes)


class scDomainNaiveSampler:
    def __init__(self, m):
        self.node = getSCDomainSampler(m)

    def sample(self):
        return self.node.sample_domain_naive()


### run an experiment

# def scExperiment(name, sampler, p):
#     experiments = 10000
#     vts = 0
#     pos = 0
#
#     for i in range(experiments):
#         dom = sampler.sample()
#         S = 0
#         for v in dom:
#             # print(v)
#             vts += 1
#             for i in range(m):
#                 if v[i] == p: pos += i + 1
#
#     print(f"{name}: Average {p} position: {pos / vts}")


# if __name__ == "__main__":
#
#     print("============")
#
#     i = 0
#     A = timer()
#
#     m = 7
#     n = 10
#     uni_sampler = scDomainUniformSampler(m)
#     B = timer()
#     print(f"single sampler: {B - A}")
#
#     uni_sampler.prepareElectionSampler(n)
#
#     B = timer()
#     print(f"prepare all next: {B - A}")
#
#     for i in range(30):
#         E = uni_sampler.sampleElection()
#         for v in E:
#             print(v)
#         print()
#
#     B = timer()
#     print(f"sample election: {B - A}")
#
#     exit()
#
#     nai_sampler = scDomainNaiveSampler(m)
#     B = timer()
#     print(f"both samplers : {B - A}")
#
#     for i in range(m):
#         scExperiment("(uni)", uni_sampler, i)
#
#     print("----")
#
#     for i in range(m):
#         scExperiment("(nai)", nai_sampler, i)
#
#     B = timer()
#
#     print("============")
#     print(B - A)

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #

# print('import SC')
# num_candidates = 9
# # naive_sampler = scDomainNaiveSampler(num_candidates)
# # print('halfway')
# uniform_sampler = scDomainUniformSampler(num_candidates)
# print('domain prepared')

# m = 7
# n = 100
# uniform_sampler = scDomainUniformSampler(m)
# # B = timer()
# # print(f"single sampler: {B - A}")
# uniform_sampler.prepareElectionSampler(n)
#
