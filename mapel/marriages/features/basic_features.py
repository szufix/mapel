import gurobipy as gp
from gurobipy import GRB
# from mapel.roommates.matching.games import StableRoommates
try:
    from mapel.roommates.matching.games import StableRoommates
except:
    pass

from random import shuffle
import statistics
import warnings
import sys
import time
import networkx as nx

sys.setrecursionlimit(10000)
# warnings.filterwarnings("error")

def generate_instance(num_agents):
    instance=[]
    for i in range(num_agents):
        pref = [x for x in range(num_agents) if x != i]
        shuffle(pref)
        instance.append(pref)
    return instance

def number_blockingPairs(instance,matching):
    bps=0
    num_agents=len(instance)
    for i in range(num_agents):
        for j in range(i+1,num_agents):
            if i!=j:
                partner_i=matching[i]
                partneri_index=instance[i].index(partner_i)
                partner_j = matching[j]
                partnerj_index = instance[j].index(partner_j)
                if instance[i].index(j)<partneri_index:
                    if instance[j].index(i) < partnerj_index:
                        bps+=1
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
        return None



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
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents*num_agents)
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
            if x[i, j].X == 1:
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
            for t in range(0,instance.votes[i].index(j)+1):
                better_pairs.append([i,instance.votes[i][t]])
            for t in range(0,instance.votes[j].index(i)+1):
                better_pairs.append([j,instance.votes[j][t]])
            m.addConstr(gp.quicksum(x[a[0], a[1]] for a in better_pairs) >= 1-y[i,j])

    m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    matching={}
    for i in range(num_agents):
        for j in range(num_agents):
            if x[i, j].X == 1:
                matching[i]=j
                matching[j]=i
    #return int(m.objVal), matching
    return int(m.objVal)

def summed_rank_maximal_matching(instance):
    try:
        val,matching= rank_matching(instance.votes,True,True)
    except:
        return 'None'
    if number_blockingPairs(instance.votes,matching)>0:
        print('ERROR IN summed_rank_maximal_matching')
        exit(0)
    return val

def summed_rank_minimal_matching(instance):
    try:
        val,matching= rank_matching(instance.votes,False,True)
    except:
        return 'None'
    if number_blockingPairs(instance.votes,matching)>0:
        print('ERROR IN summed_rank_minimal_matching')
        exit(0)
    return val

def minimal_rank_maximizing_matching(instance):
    try:
        val,matching=rank_matching(instance.votes,True,False)
    except:
        return 'None'
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
            G.add_edge(i,j,weight=2*(num_agents-1)-instance.votes[i].index(j)-instance.votes[j].index(i))
            #G.add_edge(i, j, weight=instance[i].index(j) + instance[j].index(i))
    matching=nx.max_weight_matching(G, maxcardinality=True)
    matching_dict={}
    for p in matching:
        matching_dict[p[0]]=p[1]
        matching_dict[p[1]] = p[0]
    return number_blockingPairs(instance.votes,matching_dict)

#
# j=0
# for i in range(100):
#     instance=generate_instance(50)
#     print(number_of_bps_maximumWeight(instance))
#     print(min_num_bps_matching(instance))
#
#
# exit()
# instance1=generate_instance(10)
# instance2=generate_instance(10)
# start = time.process_time()
# print(spear_distance_linear(instance1,instance2))
# print(time.process_time() - start)
# exit()