import random


def convert_votes_to_sr(votes):
    new_votes = []
    for j in range(len(votes)):
        vote = list(votes[j])
        vote.remove(j)
        new_votes.append(vote)
    return new_votes


def convert_votes_to_srti(votes):
    num_agents = len(votes)
    votes = convert_votes_to_sr(votes)
    return [[[votes[i][j]] for j in range(num_agents-1)] for i in range(num_agents)]


def remove_first(votes):
    new_votes = []
    for j in range(len(votes)):
        vote = list(votes[j][1:])
        new_votes.append(vote)
    return new_votes


def add_tie(list, index):
    if index == 0:
        return list
    for i in range(len(list[index])):
        list[index-1].append(list[index][i])
    list.pop(index)
    list.append([])

def add_ties_random(list, num_agents, ties_percent):
    ties_to_add = int(num_agents * ties_percent)
    for i in range(num_agents):
        rnd = random.sample(range(0, num_agents-1), ties_to_add)
        # rnd.sort(reverse=True)
        for r in rnd:
            add_tie(list[i], r)
        
def add_incompleteness_at_end(list, num_agents, incomplete_percent):
    agents_to_remove = int(num_agents * incomplete_percent)
    for i in range(num_agents):
        col = num_agents-2
        poped = 0
        while poped < agents_to_remove:
            if list[i][col]:
                list[i][col].pop()
                poped += 1
            else:
                col -= 1