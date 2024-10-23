
try:
    import gurobipy as gp
    from gurobipy import GRB
except:
    pass

try:
    from mapel.roommates.matching.games import StableRoommates
except:
    pass

from random import shuffle
import itertools
import statistics
import warnings
import sys
import time
import networkx as nx
import copy
import os

sys.setrecursionlimit(10000)
# warnings.filterwarnings("error")

def generate_instance(num_agents):
    instance=[]
    for i in range(num_agents):
        pref = [x for x in range(num_agents) if x != i]
        shuffle(pref)
        instance.append(pref)
    return instance

# number of blocking pairs under weak stability
def number_blockingPairs(instance,matching):
    bps = 0
    num_agents = len(instance)
    for i in range(num_agents):
        for j in range(i+1,num_agents):
            if i == j:
                continue

            partner_i = matching[i]
            partner_j = matching[j]
            partneri_index = get_rank(instance, i, partner_i)
            partnerj_index = get_rank(instance, j, partner_j)

            if num_agents-1 in [partneri_index, partnerj_index]:
                bps += 1
                continue
            
            if get_rank(instance, i, j) < partneri_index:
                if get_rank(instance, j, i) < partnerj_index:
                    bps += 1
    return bps


def compute_stable_SR(votes):
    dict_instance={}
    num_agents=len(votes)
    for i in range(num_agents):
        dict_instance[i]=votes[i]
    game = StableRoommates.create_from_dictionary(dict_instance)
    try:
        matching = game.solve()
        usable_matching = {}
        for m in matching:
            usable_matching[m.name] = matching[m].name
        return usable_matching
    except:
        return 'None'



def spear_distance(instance1,instance2):
    num_agents=len(instance1)
    m = gp.Model("qp")
    #m.setParam('Threads', 6)
    #m.setParam('OutputFlag', False)
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents * num_agents*num_agents)
    for i in range(num_agents):
        m.addConstr(gp.quicksum(x[i, j] for j in range(num_agents)) == 1)
        m.addConstr(gp.quicksum(x[j, i] for j in range(num_agents)) == 1)
    m.addConstr(gp.quicksum(abs(instance1[i].index(k)-instance2[j].index(t)) * x[i, j]*x[k,t] for i in range(num_agents)
                            for j in range(num_agents) for k in [x for x in range(num_agents) if x != i]
                            for t in [x for x in range(num_agents) if x != j]) == opt)
    m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    for i in range(num_agents):
        for j in range(num_agents):
            if x[i, j].X == 1:
                print(i,j)
    return int(m.objVal)


def spear_distance_linear(instance1,instance2):
    num_agents=len(instance1)
    m = gp.Model("mip1")
    #m.setParam('Threads', 6)
    #m.setParam('OutputFlag', False)
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    y = m.addVars(num_agents, num_agents,num_agents,num_agents, lb=0, ub=1, vtype=GRB.CONTINUOUS)
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents * num_agents*num_agents)
    for i in range(num_agents):
        m.addConstr(gp.quicksum(x[i, j] for j in range(num_agents)) == 1)
        m.addConstr(gp.quicksum(x[j, i] for j in range(num_agents)) == 1)
    for i in range(num_agents):
        for j in range(num_agents):
            for k in range(num_agents):
                for l in range(num_agents):
                    m.addConstr(y[i,j,k,l]<=x[i, j])
                    m.addConstr(y[i, j, k, l] <= x[k, l])
                    m.addConstr(y[i, j, k, l] >= x[i, j]+x[k, l]-1)

    m.addConstr(gp.quicksum(abs(instance1[i].index(k)-instance2[j].index(t)) * y[i, j,k,t] for i in range(num_agents)
                            for j in range(num_agents) for k in [x for x in range(num_agents) if x != i]
                            for t in [x for x in range(num_agents) if x != j]) == opt)
    m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    for i in range(num_agents):
        for j in range(num_agents):
            if x[i, j].X == 1:
                print(i,j)
    return int(m.objVal)

#Only call for instances that admit a stable matchig
#summed: Set to true we try to optimize the summed rank of agents for their partner in the matching, set to false we optimize the minimum rank
#Best (only relevant if summed is set to true): Set to true we output the best possible matching, set to false the worst one
def rank_matching(instance,best,summed):
    num_agents=len(instance)
    m = gp.Model("mip1")
    m.setParam('OutputFlag', False)
    m.setParam('Threads', 10)
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents*num_agents)
    for i in range(num_agents):
        m.addConstr(x[i, i]  == 0)
    for i in range(num_agents):
        for j in range(num_agents):
            m.addConstr(x[i, j]  == x[j, i])
    for i in range(num_agents):
        m.addConstr(gp.quicksum(x[i, j] for j in range(num_agents)) == 1)
    for i in range(num_agents):
        for j in range(i+1,num_agents):
            better_pairs=[]
            for t in range(0,instance[i].index(j)+1):
                better_pairs.append([i,instance[i][t]])
            for t in range(0,instance[j].index(i)+1):
                better_pairs.append([j,instance[j][t]])
            m.addConstr(gp.quicksum(x[a[0], a[1]] for a in better_pairs) >= 1)
    if summed:
        #print(instance[0])
        m.addConstr(gp.quicksum(instance[i].index(j)* x[i, j]  for i in range(num_agents) for j in [x for x in range(num_agents) if x != i]) == opt)
        if best:
            m.setObjective(opt, GRB.MAXIMIZE)
        else:
            m.setObjective(opt, GRB.MINIMIZE)
    else:
        for i in range(num_agents):
            m.addConstr(gp.quicksum(instance[j].index(i) * x[i, j] for j in [x for x in range(num_agents) if x != i])<=opt)
        m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    matching={}
    for i in range(num_agents):
        for j in range(num_agents):
            if abs(x[i, j].X - 1) < 0.05:
                matching[i]=j
                matching[j]=i
    return int(m.objVal), matching

def min_num_bps_matching(instance):
    num_agents=len(instance.votes)
    m = gp.Model("mip1")
    m.setParam('OutputFlag', False)
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    y = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents*num_agents)
    m.addConstr(gp.quicksum(y[i, j] for i in range(num_agents) for j in range(num_agents)) <= opt)
    for i in range(num_agents):
        m.addConstr(x[i, i]  == 0)
    for i in range(num_agents):
        for j in range(num_agents):
            m.addConstr(x[i, j]  == x[j, i])
    for i in range(num_agents):
        m.addConstr(gp.quicksum(x[i, j] for j in range(num_agents)) <= 1)
    for i in range(num_agents):
        for j in range(i+1,num_agents):
            better_pairs=[]

            index = instance.votes[i].index(j)+1 if j in instance.votes[i] else num_agents-1
            for t in range(0,index):
                for agnt in instance.votes[i][t]:
                    better_pairs.append([i,agnt])

            index = instance.votes[j].index(i)+1 if i in instance.votes[j] else num_agents-1
            for t in range(0,index):
                for agnt in instance.votes[j][t]:
                    better_pairs.append([j,agnt])
        m.addConstr(gp.quicksum(x[a[0], a[1]] for a in better_pairs) >= 1-y[i,j])

    m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    matching={}
    for i in range(num_agents):
        for j in range(num_agents):
            if abs(x[i, j].X - 1) < 0.05:
                matching[i]=j
                matching[j]=i
    #return int(m.objVal), matching
    return int(m.objVal)

def summed_rank_maximal_matching(instance):
    val,matching= rank_matching(instance.votes,True,True)
    if number_blockingPairs(instance.votes,matching)>0:
        print('ERROR IN summed_rank_maximal_matching')
        exit(0)
    return val


def summed_rank_minimal_matching(instance):
    val,matching= rank_matching(instance.votes,False,True)
    if number_blockingPairs(instance.votes,matching)>0:
        print('ERROR IN summed_rank_minimal_matching')
        exit(0)
    return val

def minimal_rank_maximizing_matching(instance):
    val,matching=rank_matching(instance.votes,True,False)
    if number_blockingPairs(instance.votes,matching)>0:
        print('ERROR IN minimal_rank_maximizing_matching')
        exit(0)
    return val

def avg_num_of_bps_for_random_matching(instance,iterations=100):
    bps=[]
    num_agents=len(instance.votes)
    for _ in range(iterations):
        #create random matching
        agents=list(range(num_agents))
        shuffle(agents)
        matching=[agents[i:i + 2] for i in range(0, num_agents, 2)]
        matching_dict={}
        for m in matching:
            matching_dict[m[0]]=m[1]
            matching_dict[m[1]] = m[0]
        bps.append(number_blockingPairs(instance.votes,matching_dict))
    return statistics.mean(bps), statistics.stdev(bps)


def num_of_bps_maximumWeight(instance) -> int:
    num_agents=len(instance.votes)
    G = nx.Graph()
    for i in range(num_agents):
        for j in range(i+1,num_agents):
            G.add_edge(i,j,weight=2*(num_agents-1) - get_rank(instance.votes, i, j) - get_rank(instance.votes, j, i))
            # G.add_edge(i,j,weight=2*(num_agents-1)-instance.votes[i].index(j)-instance.votes[j].index(i)) esse era o original
            #G.add_edge(i, j, weight=instance[i].index(j) + instance[j].index(i))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    matching_dict = {}
    for p in matching:
        matching_dict[p[0]] = p[1]
        matching_dict[p[1]] = p[0]
    return number_blockingPairs(instance.votes,matching_dict)

#------
def is_stable(rank_matrix, match):
        prefer_be_unmatched = len(rank_matrix)-1
        for i in range(len(rank_matrix)):
            for j in range(i, len(rank_matrix)):
                if i == j:
                    continue
                # Se formam um par em 'match'
                if (i, j) == (i, match[i]):
                    continue

                i_rank_in_j = rank_matrix[i][j]
                i_rank_match = rank_matrix[i][match[i]] if match[i] != -1 else prefer_be_unmatched
                j_rank_in_i = rank_matrix[j][i]
                j_rank_match = rank_matrix[j][match[j]] if match[j] != -1 else prefer_be_unmatched

                if match[i] != -1 and prefer_be_unmatched == i_rank_match:
                    return False
                if match[j] != -1 and prefer_be_unmatched == j_rank_match:
                    return False
                if i_rank_in_j < i_rank_match and j_rank_in_i < j_rank_match:
                    return False
        return True
def get_rank_matrix(mat):
        # out[i][j] == rank of j in i's pref list
        # contains one additional column (the last one), having len(mat[0])+1 values
        out = [[-1 for _ in range(len(mat[0])+1)] for _ in range(len(mat))]
        for i in range(len(out)):
            for j in range(len(out[i])):
                out[i][j] = get_rank(mat, i, j)
        return out
#-------

def export_solutions(instance, file):
    print(f'number of agents: {len(instance.votes)}', file=file)
    print(f'culture: {instance.culture_id}', file=file)
    params = copy.deepcopy(instance.params)
    params.pop('alpha')
    if 'srti_func' in params:
        params.pop('srti_func')
    if 'phi' in params:
        params.pop('phi')
    print(f'params: ', end='', file=file)
    print(' '.join("{}: {}".format(k, v) for k, v in params.items()), file=file)

    print('preference lists:', file=file)
    for pref_list in instance.votes:
        for col in pref_list:
            print(f'{len(col)}: ', end='', file=file)
            for agent in col:
                print(f'{agent} ', end='', file=file)
        print('', file=file)
    print('weakly stable matches:', file=file)


def num_solutions(instance):
    path_to_file = os.path.join(os.getcwd(), 'all-solutions.txt')

    with open(path_to_file, 'a') as file:
        export_solutions(instance, file)

        rank_matrix = get_rank_matrix(instance.votes)
        matching = [-1 for _ in range(len(instance.votes))]
        
        out = _num_solutions(matching, rank_matrix, file=file)

    # print(f'number of solutions: {out}')
    # other = number_of_solutions(instance)
    # if out != other:
    #     print(f'Deu diferente. Otimizado: {out}, ruim: {other}')

        print('', file=file)
    return out

def _num_solutions(matching, rank_matrix, file, pair_added=None):
    num_agents = len(matching)
    minimum = 0 if pair_added is None else min(pair_added[0], pair_added[1])
    
    out = int(is_stable(rank_matrix, matching))
    for ag1 in range(minimum, num_agents):
        for ag2 in range(ag1+1, num_agents):
            if matching[ag1] != -1 or matching[ag2] != -1:
                continue
            updated_matching = copy.deepcopy(matching)
            updated_matching[ag1] = ag2
            updated_matching[ag2] = ag1
            print(str(updated_matching)[1:-1].replace(',', ''), file=file)
            out += _num_solutions(updated_matching, rank_matrix, file, pair_added=(ag1, ag2))
    return out

def get_matching_from_stack(stack):
    return

def __num_solutions(rank_matrix):
    num_agents = len(rank_matrix)
    matching = [-1 for _ in range(len(num_agents))]
    minimum = 0
    out = int(is_stable(rank_matrix, matching))

    stack = []
    ag1 = 0
    ag2 = 1
    while True:
        stack.append((ag1, ag2))
        curr_matching = get_matching_from_stack(stack)
        if is_stable(rank_matrix, curr_matching):
            out += 1
        return out

'''
14 agentes: 135135, 37 segundos com a solução otimizada        

n agentes:
num solucoes com n: prod(n - 2*i), i in [0, n/2]
num solucoes com n-2: prod(n - 2*i), i in [0, n/2 -1]

num solucoes com qualquer tamanho:
sum (  prod(n - 2*i), i in [0, n/2 -j]  ), j in [0, n/2]

'''

# For matchings with unmatched agents, this counts them multiple times
# def number_of_solutions(instance) -> int:
#     def all_pairs(lst):
#         if len(lst) < 2:
#             yield []
#             return
#         if len(lst) % 2 == 1:
#             # Handle odd length list
#             for i in range(len(lst)):
#                 for result in all_pairs(lst[:i] + lst[i+1:]):
#                     yield result
#         else:
#             a = lst[0]
#             for i in range(1,len(lst)):
#                 pair = (a,lst[i])
#                 for rest in all_pairs(lst[1:i]+lst[i+1:]):
#                     yield [pair] + rest    
    
#     num_solutions = 0
#     rank_matrix = get_rank_matrix(instance.votes)
#     diff_sol = {}

#     m = [-1 for _ in range(len(instance.votes))]
#     if is_stable(rank_matrix, m):
#         num_solutions += 1

#     for matching in all_pairs(list(range(len(instance.votes)))):

#         for mask in range(1, 2**(len(instance.votes)//2)):

#             m = [-1 for _ in range(len(instance.votes))]
#             for i, match in enumerate(matching):
#                 if ((mask >> i) & 1) == 1:
#                     m[match[0]] = match[1]
#                     m[match[1]] = match[0]

#             if is_stable(rank_matrix, m):
#                 if str(m) in diff_sol:
#                     diff_sol[str(m)] += 1
#                 else:
#                     diff_sol[str(m)] = 1
#                 num_solutions += 1

#     # print(f'num_solutions: {num_solutions}')
#     for sol in diff_sol.items():
#         if sol[1] > 1:
#             print(sol[0] + " " + str(sol[1]))
#     return num_solutions


# NEW 15.03.2022
def mutuality(instance) -> int:
    vectors = instance.get_retrospective_vectors()
    score = 0
    for vector in vectors:
        for i, v in enumerate(vector):
            score += abs(v-i)
    return score

def dist_from_id_1(instance) -> int:
    vectors = instance.get_retrospective_vectors()
    score = 0
    for vector in vectors:
        for i in range(len(vector)):
            for j in range(i,len(vector)):
                score += abs(vector[j]-vector[i])
    return score

def dist_from_id_2(instance) -> int:
    vectors = instance.get_retrospective_vectors()
    score = 0
    for vector in vectors:
        score += len(set(vector))
    return score

def get_rank(matrix, a1, a2):
        '''
            Returns the rank of a2 in the a1's preference list
            if not present, return len(matrix)-1 (one more than max value)
        '''
        for i, lst in enumerate(matrix[a1]):
            if a2 in lst:
                return i
        return len(matrix)-1