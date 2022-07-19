import gurobipy as gp
from gurobipy import GRB
from random import shuffle
import statistics
import warnings
import sys
import time
import networkx as nx
sys.setrecursionlimit(10000)
# warnings.filterwarnings("error")

def generate_instance(num_agents):
    man=[]
    woman = []
    for i in range(num_agents):
        pref = [x for x in range(num_agents)]
        shuffle(pref)
        man.append(pref)
    for i in range(num_agents):
        pref = [x for x in range(num_agents)]
        shuffle(pref)
        woman.append(pref)
    return [man, woman]

def number_blockingPairs(instance,matching):
    #matching=[[3, 1, 0, 2, 4], [2, 1, 3, 0, 4]]
    bps=0
    num_agents=len(instance[0])
    for i in range(num_agents):
        for j in range(num_agents):
            #print(i,j)
            partner_i=matching[0][i]
            partneri_index=instance[0][i].index(partner_i)
            partner_j = matching[1][j]
            partnerj_index = instance[1][j].index(partner_j)
            if instance[0][i].index(j)<partneri_index:
                if instance[1][j].index(i) < partnerj_index:
                    #print(i,j)
                    bps+=1
    #print(bps)
    return bps




#Only call for instances that admit a stable matchig
#summed: Set to true we try to optimize the summed rank of agents for their partner in the matching, set to false we optimize the minimum rank
#Best (only relevant if summed is set to true): Set to true we output the best possible matching, set to false the worst one
def rank_matching(instance,best,summed):
    num_agents=len(instance[0])
    m = gp.Model("mip1")
    m.setParam('OutputFlag', False)
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents*num_agents)
    opt1 = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents * num_agents)
    opt2 = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents * num_agents)
    for i in range(num_agents):
        m.addConstr(gp.quicksum(x[i, j] for j in range(num_agents)) <= 1)
        m.addConstr(gp.quicksum(x[j, i] for j in range(num_agents)) <= 1)
    for i in range(num_agents):
        for j in range(num_agents):
            #print(i,j)
            better_pairs=[]
            for t in range(0,instance[0][i].index(j)+1):
                better_pairs.append([i,instance[0][i][t]])
            for t in range(0,instance[1][j].index(i)+1):
                better_pairs.append([instance[1][j][t],j])
            #print(better_pairs)
            #exit()
            m.addConstr(gp.quicksum(x[a[0], a[1]] for a in better_pairs) >= 1)
    if summed:
        #print(instance[0])
        m.addConstr(gp.quicksum(
            instance[0][i].index(j) * x[i, j] for i in range(num_agents) for j in range(num_agents)) == opt1)
        m.addConstr(gp.quicksum(
            instance[1][i].index(j) * x[j, i] for i in range(num_agents) for j in range(num_agents)) == opt2)
        m.addConstr(opt1 + opt2 == opt)
        if best:
            m.setObjective(opt, GRB.MAXIMIZE)
        else:
            m.setObjective(opt, GRB.MINIMIZE)
    else:
        for i in range(num_agents):
            m.addConstr(gp.quicksum(instance[0][j].index(i) * x[j, i] for j in range(num_agents))<=opt)
            m.addConstr(gp.quicksum(instance[1][j].index(i) * x[i, j] for j in range(num_agents))<= opt)
        m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    matching1={}
    matching2 = {}
    for i in range(num_agents):
        for j in range(num_agents):
            if x[i, j].X == 1:
                matching1[i]=j
                matching2[j]=i
    return int(m.objVal), [matching1,matching2]



def summed_rank_maximal_matching(instance):
    try:
        val,matching= rank_matching(instance,True,True)
    except:
        return 'None'
    # if number_blockingPairs(instance,matching)>0:
    #     print('ERROR IN summed_rank_maximal_matching')
    #     exit(0)
    return val

def summed_rank_minimal_matching(instance):
    try:
        val,matching= rank_matching(instance,False,True)
    except:
        return 'None'
    # if number_blockingPairs(instance,matching)>0:
    #     print('ERROR IN summed_rank_minimal_matching')
    #     exit(0)
    return val

def minimal_rank_maximizing_matching(instance):
    try:
        val,matching=rank_matching(instance,True,False)
    except:
        return 'None'
    # if number_blockingPairs(instance,matching)>0:
    #     print('ERROR IN minimal_rank_maximizing_matching')
    #     exit(0)
    return val

def avg_number_of_bps_for_random_matching(instance,iterations=100):
    bps=[]
    num_agents=len(instance[0])
    for _ in range(iterations):
        #create random matching
        agents=list(range(num_agents))
        shuffle(agents)
        m2=[]
        for i in range(num_agents):
            m2.append(agents.index(i))
        #print([agents,m2])
        #exit()
        bps.append(number_blockingPairs(instance,[agents,m2]))
    return statistics.mean(bps), statistics.stdev(bps)

def number_of_bps_maximumWeight(instance):
    num_agents=len(instance[0])
    G = nx.Graph()
    for i in range(num_agents):
        for j in range(num_agents):
            G.add_edge(i,j+num_agents,weight=2*(num_agents-1)-instance[0][i].index(j)-instance[1][j].index(i))
            #G.add_edge(i, j, weight=instance[i].index(j) + instance[j].index(i))
    matching=nx.max_weight_matching(G, maxcardinality=True)
    #print(matching)
    #exit()
    matching_dict_m={}
    matching_dict_w = {}
    for p in matching:
        small=min(p[0],p[1])
        big = max(p[0], p[1])
        matching_dict_m[small]=big-num_agents
        matching_dict_w[big-num_agents] = small
    #print(matching_dict_m,matching_dict_w)
    return number_blockingPairs(instance,[matching_dict_m,matching_dict_w])

'''
j=0
for i in range(100):
    instance=generate_instance(50)
    #print('---------------------')
    #print(instance)
    #print(number_of_bps_maximumWeight(instance))
    print('NEW')
    print(summed_rank_maximal_matching(instance))
    print(summed_rank_minimal_matching(instance))
    print(minimal_rank_maximizing_matching(instance))
    #print(average_number_of_bps_for_random_matching(instance))
    #swap_distance_to_stable(instance)
    #delete_distance_to_stable(instance)
    #min_num_blocking_agents_matching(instance)
    #print(number_of_bps_maximumWeight(instance))
    #print(min_num_bps_matching(instance))


exit()
instance1=generate_instance(10)
instance2=generate_instance(10)
start = time.process_time()
print(spear_distance_linear(instance1,instance2))
print(time.process_time() - start)
exit()
'''