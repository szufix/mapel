import gurobipy as gp
from gurobipy import GRB
from random import shuffle
import statistics
import warnings
import sys
import time
import networkx as nx

def generate_instance(num_agents):
    instance=[]
    for i in range(num_agents):
        pref = [x for x in range(num_agents) if x != i]
        shuffle(pref)
        instance.append(pref)
    return instance


#Only works for even number of agents
def swap_distance_to_stable(instance):
    num_agents=len(instance)
    m = gp.Model("mip1")
    m.setParam('OutputFlag', False)
    #who is matched to whom
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    #am I matched to rank i or better
    gm = m.addVars(num_agents, num_agents-1, lb=0, ub=1, vtype=GRB.BINARY)
    #by how much have my partner been swaped up in my preferences
    yy = m.addVars(num_agents, lb=0, ub=num_agents, vtype=GRB.INTEGER)

    for i in range(num_agents):
        yy[i].start=0

    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents*num_agents)
    m.addConstr(gp.quicksum(yy[i] for i in range(num_agents)) == opt)
    for i in range(num_agents):
        m.addConstr(x[i, i]  == 0)

    for i in range(num_agents):
        for j in range(num_agents):
            m.addConstr(x[i, j]  == x[j, i])

    for i in range(num_agents):
        for j in range(num_agents):
            if i!=j:
                for t in range(num_agents-1):
                    m.addConstr(x[i, j]*instance[i].index(j)  <= t+yy[i]+num_agents*(1-gm[i,t]))


    for i in range(num_agents):
        m.addConstr(gp.quicksum(x[i, j] for j in range(num_agents)) == 1)

    for i in range(num_agents):
        for j in range(i+1,num_agents):
            m.addConstr(gm[i,instance[i].index(j)]+gm[j,instance[j].index(i)]>=1)
    m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()

    return int(m.objVal)

def delete_distance_to_stable(instance):
    num_agents=len(instance)
    m = gp.Model("mip1")
    m.setParam('OutputFlag', False)
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    y = m.addVars(num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    for i in range(num_agents):
        y[i].start=0
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents)
    m.addConstr(gp.quicksum(y[i] for i in range(num_agents)) == opt)
    for i in range(num_agents):
        m.addConstr(x[i, i]  == 0)
    for i in range(num_agents):
        for j in range(num_agents):
            m.addConstr(x[i, j]  == x[j, i])
    for i in range(num_agents):
        m.addConstr(gp.quicksum(x[i, j] for j in range(num_agents)) <= (1-y[i]))
    for i in range(num_agents):
        for j in range(i+1,num_agents):
            better_pairs=[]
            for t in range(0,instance[i].index(j)+1):
                better_pairs.append([i,instance[i][t]])
            for t in range(0,instance[j].index(i)+1):
                better_pairs.append([j,instance[j][t]])
            m.addConstr(gp.quicksum(x[a[0], a[1]] for a in better_pairs) >= 1-y[i]-y[j])

    m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    matching={}
    for i in range(num_agents):
        for j in range(num_agents):
            if x[i, j].X == 1:
                matching[i]=j
                matching[j]=i
    return int(m.objVal)


def min_num_blocking_agents_matching(instance):
    num_agents=len(instance)
    m = gp.Model("mip1")
    m.setParam('OutputFlag', False)
    x = m.addVars(num_agents, num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    y = m.addVars(num_agents, lb=0, ub=1, vtype=GRB.BINARY)
    for i in range(num_agents):
        y[i].start=0
    opt = m.addVar(vtype=GRB.INTEGER, lb=0, ub=num_agents)
    m.addConstr(gp.quicksum(y[i] for i in range(num_agents)) <= opt)
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
            m.addConstr(gp.quicksum(x[a[0], a[1]] for a in better_pairs) >= 1-y[i]-y[j])

    m.setObjective(opt, GRB.MINIMIZE)
    m.optimize()
    matching={}
    for i in range(num_agents):
        for j in range(num_agents):
            if x[i, j].X == 1:
                matching[i]=j
                matching[j]=i
    print(m.objVal)
    return int(m.objVal)

#for i in range(100):
#    instance=generate_instance(50)
#    print(min_num_blocking_agents_matching(instance))
#    print(swap_distance_to_stable(instance))
#    print(delete_distance_to_stable(instance))
