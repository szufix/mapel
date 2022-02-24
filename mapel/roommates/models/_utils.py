
def convert(votes):
    new_votes = []
    for j in range(len(votes)):
        vote = list(votes[j])
        vote.remove(j)
        new_votes.append(vote)
    return new_votes


def remove_first(votes):
    new_votes = []
    for j in range(len(votes)):
        vote = list(votes[j][1:])
        new_votes.append(vote)
    return new_votes
