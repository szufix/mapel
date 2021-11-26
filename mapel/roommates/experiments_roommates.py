import gurobipy as gp
from gurobipy import GRB
from mapel.roommates.matching.games import StableRoommates
from random import shuffle
import statistics
import warnings
warnings.filterwarnings("error")

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

def compute_stable_SR(instance):
    dict_instance={}
    num_agents=len(instance)
    for i in range(num_agents):
        dict_instance[i]=instance[i]
    game = StableRoommates.create_from_dictionary(dict_instance)
    try:
        matching = game.solve()
    except:
        return None
    usable_matching = {}
    for m in matching:
        usable_matching[m.name] = matching[m].name
    return usable_matching


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

def summed_rank_maximal_matching(instance):
    val,matching= rank_matching(instance,True,True)
    if number_blockingPairs(instance,matching)>0:
        print('ERROR IN summed_rank_maximal_matching')
        exit(0)
    return val

def summed_rank_minimal_matching(instance):
    val,matching= rank_matching(instance,False,True)
    if number_blockingPairs(instance,matching)>0:
        print('ERROR IN summed_rank_minimal_matching')
        exit(0)
    return val

def minimal_rank_maximizing_matching(instance):
    val,matching=rank_matching(instance,True,False)
    if number_blockingPairs(instance,matching)>0:
        print('ERROR IN minimal_rank_maximizing_matching')
        exit(0)
    return val

def average_number_of_bps_for_random_matching(instance,iterations=100):
    bps=[]
    num_agents=len(instance)
    for _ in range(iterations):
        #create random matching
        agents=list(range(num_agents))
        shuffle(agents)
        matching=[agents[i:i + 2] for i in range(0, num_agents, 2)]
        matching_dict={}
        for m in matching:
            matching_dict[m[0]]=m[1]
            matching_dict[m[1]] = m[0]
        bps.append(number_blockingPairs(instance,matching_dict))
    return statistics.mean(bps), statistics.stdev(bps)
