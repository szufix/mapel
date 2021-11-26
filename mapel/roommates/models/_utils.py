
def convert(votes):
    new_votes = []
    for j in range(len(votes)):
        vote = list(votes[j])
        vote.remove(j)
        new_votes.append(vote)
    return new_votes
