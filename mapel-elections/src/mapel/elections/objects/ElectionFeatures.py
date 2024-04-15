"""
Helper ElectionFeatures object which stores the feature vectors and MOV's for each election.
"""

import math
from itertools import combinations, permutations

import numpy as np

NO_WINNER = []
NUM_ATTRIBUTES = 4
ST_KEY = 'ST-compass-'
AN_KEY = 'AN-compass-'
UN_KEY = 'UN-compass-'
ID_KEY = 'ID-compass-'

"""
Compute winning order from a list of scores
"""


def scores_to_winners(scores: list):
    to_sort = dict()
    for i in range(0, len(scores)):
        if scores[i] in to_sort:
            to_sort[scores[i]].append(i)
        else:
            to_sort[scores[i]] = [i]
    result = []
    for key in sorted(to_sort, reverse=True):
        result += to_sort[key]
    return result


"Compute majority scores and ordering of candidates."


def majority(num_voters, num_candidates, votes):
    scores = num_candidates * [0]
    for vote in votes:
        scores[vote[0]] += 1
    if max(scores) < math.ceil(num_voters / 2):
        return NO_WINNER, scores
    else:
        majority_order = scores_to_winners(scores)
        return majority_order, scores


"Compute plurality scores and ordering of candidates."


def plurality(num_voters, num_candidates, votes):
    scores = num_candidates * [0]
    for vote in votes:
        scores[vote[0]] += 1
    plurality_order = scores_to_winners(scores)
    return plurality_order, scores


"Given positions of candidate a and candidate b, return the point difference between them for plurality."


def plurality_point_difference(num_candidates, position_of_a, position_of_b):
    if position_of_a == 0:
        return 1
    elif position_of_b == 0:
        return -1
    else:
        return 0


"Compute veto scores and ordering of candidates."


def veto(num_voters, num_candidates, votes):
    scores = num_candidates * [num_voters]
    for vote in votes:
        scores[vote[num_candidates - 1]] -= 1
    veto_order = scores_to_winners(scores)
    return veto_order, scores


"Given positions of candidate a and candidate b, return the point difference between them for veto."


def veto_point_difference(num_candidates, position_of_a, position_of_b):
    if position_of_a != num_candidates - 1 and position_of_b != num_candidates - 1:
        return 0
    elif position_of_a == num_candidates - 1:
        return -1
    elif position_of_b == num_candidates - 1:
        return 1
    else:
        'Error - this should never happen.'


"Compute k-approval scores and ordering of candidates."


def k_approval(num_voters, num_candidates, votes, k: int = 1):
    scores = num_candidates * [0]
    for vote in votes:
        for i in range(0, k):
            scores[vote[i]] += 1
    approvel_order = scores_to_winners(scores)
    return approvel_order, scores


"Given positions of candidate a and candidate b, return the point difference between them for k-approval."


def k_approval_point_difference(num_candidates, position_of_a, position_of_b, k, *args, **kwargs):
    if position_of_a < k <= position_of_b:
        return 1
    elif position_of_b < k <= position_of_a:
        return -1
    else:
        return 0


"Compute borda scores and ordering of candidates."


def borda(num_voters, num_candidates, votes):
    scores = num_candidates * [0]
    for vote in votes:
        for i in range(0, num_candidates):
            scores[vote[i]] += num_candidates - 1 - i
    borda_order = scores_to_winners(scores)
    return borda_order, scores


"Given positions of candidate a and candidate b, return the point difference between them for borda."


def borda_point_difference(num_candidates, position_of_a, position_of_b, *args, **kwargs):
    return position_of_b - position_of_a


"Choose the top k votes and change them to [c precedes others precedes d]."


def change_votes(votes, c, d, k):
    changed_votes = []
    for vote in votes:
        changed_votes.append(vote[1])
    for i in range(0, k):
        new_vote = list(changed_votes[i])
        new_vote.remove(c)
        new_vote.remove(d)
        new_vote.insert(0, c)
        new_vote.append(d)
        changed_votes[i] = new_vote
    return changed_votes


"""ElectionFeatures class stores MOV's and feature vectors so that they need be computed onlz oce during a distance 
calculation."""


class ElectionFeatures:
    def __init__(self, id):
        self.compass_points = dict()
        self.parent_instance_ptr = None
        self.id = id
        self.features_vector = None
        self.compass_vector = None
        self.mixed_vector = None
        self.paths_vector = None
        self.mov_feature_vector = None
        self.votes = None
        self.num_candidates = None
        self.num_voters = None

        # scoring vectors
        # majority
        self.majority_order = None
        self.majority_scores = None
        # plurality
        self.plurality_order = None
        self.plurality_scores = None
        self.plurality_mov = None
        self.plurality_mov_scaled = None
        # veto
        self.veto_order = None
        self.veto_scores = None
        self.veto_mov = None
        self.veto_mov_scaled = None
        # kapproval
        self.kapproval_kvals = []
        self.kapproval_order = []  # list of lists
        self.kapproval_scores = []  # list of lists
        self.kapproval_mov = []  # list
        self.kapproval_mov_scaled = []  # list
        # borda
        self.borda_order = None
        self.borda_scores = None
        self.borda_mov = None
        self.borda_mov_scaled = None

    def can_calculate(self):
        if self.votes is not None:
            return True
        print("No votes copied to ElectinonFeatures object")
        return False

    def copeland_experiment(self):
        if not self.can_calculate():
            return
        alphas = [0] + [x / 100.0 for x in range(1, 101)]
        for alpha in alphas:
            self.copeland(alpha)

    def k_approval_scores(self):
        ks = [x for x in range(1, self.num_candidates)]
        for k in ks:
            order, scores = k_approval(self.num_voters, self.num_candidates, self.votes, k)
            self.kapproval_kvals.append(k)
            self.kapproval_order.append(order)
            self.kapproval_scores.append(scores)

    def k_approval_movs(self):
        for k in self.kapproval_kvals:
            self.kapproval_mov.append(self.mov_scoring_protocols(k_approval, k_approval_point_difference, k=k))
            self.kapproval_mov_scaled.append(self.kapproval_mov[-1] / self.num_voters)

    def calculate_all(self):
        self.calculate_voting_scores()
        self.calculate_movs()
        self.prepare_mov_vector()

    def calculate_voting_scores(self):
        if not self.can_calculate():
            return
        self.majority_order, self.majority_scores = majority(self.num_voters, self.num_candidates, self.votes)
        self.plurality_order, self.plurality_scores = plurality(self.num_voters, self.num_candidates, self.votes)
        self.veto_order, self.veto_scores = veto(self.num_voters, self.num_candidates, self.votes)
        self.k_approval_scores()
        self.borda_order, self.borda_scores = borda(self.num_voters, self.num_candidates, self.votes)

    def calculate_movs(self):
        self.plurality_mov = self.mov_scoring_protocols(plurality, plurality_point_difference)
        self.plurality_mov_scaled = self.plurality_mov / self.num_voters
        self.veto_mov = self.mov_scoring_protocols(veto, veto_point_difference)
        self.veto_mov_scaled = self.veto_mov / self.num_voters
        self.k_approval_movs()
        self.borda_mov = self.mov_scoring_protocols(borda, borda_point_difference)
        self.borda_mov_scaled = self.borda_mov / self.num_voters

    def prepare_mov_vector(self):
        vector = np.empty(NUM_ATTRIBUTES)
        vector[0] = self.plurality_mov_scaled
        vector[1] = self.veto_mov_scaled
        vector[2] = self.kapproval_mov_scaled[0]
        vector[3] = self.borda_mov_scaled
        self.mov_feature_vector = vector

    def debug(self):
        print("Votes in election:\n", self.votes)
        print("Majority Scores: ", self.majority_scores, " and winners: ", self.majority_order)
        print("Plurality Scores: ", self.plurality_scores, " winners: ", self.plurality_order, " mov: ",
              self.plurality_mov, "mov scaled: ", self.plurality_mov_scaled)
        print("Veto Scores: ", self.veto_scores, " winners: ", self.veto_order, " mov: ", self.veto_mov, "mov scaled: ",
              self.veto_mov_scaled)
        print("Borda Scores: ", self.borda_scores, " winners: ", self.borda_order, " mov: ", self.borda_mov,
              "mov scaled: ", self.borda_mov_scaled)
        for i in range(0, len(self.kapproval_kvals)):
            pass
            print("K=", self.kapproval_kvals[i], "K-Approval Scores: ", self.kapproval_scores[i], " winners: ",
                  self.kapproval_order[i], " mov: ",
                  self.kapproval_mov[i], "mov scaled: ", self.kapproval_mov_scaled[i])
        print("Feature vector, ", self.mov_feature_vector)

    def testing(self):
        print("official borda winner", self.borda_order, " and scores ", self.borda_scores)
        print("official plurality winner", self.plurality_order, " and scores ", self.borda_scores)
        margin = self.mov_scoring_protocols(borda, borda_point_difference)
        print("MOV borda", margin)
        print("MOV plurality, ", self.mov_scoring_protocols(plurality, plurality_point_difference))
        print("MOV kapproval, k=2, ", self.mov_scoring_protocols(k_approval, k_approval_point_difference, k=2))

    def mov_scoring_protocols(self, rule_to_run, point_difference_to_run, *args, **kwargs):
        output = rule_to_run(self.num_voters, self.num_candidates, self.votes, *args, **kwargs)
        winner = output[0][0]
        for number_k in range(1, self.num_voters + 1):
            for alternative in range(0, self.num_candidates):
                if alternative == winner:
                    continue
                ranked_votes = []
                for vote in self.votes:
                    position_of_a = np.where(vote == winner)[0][0]
                    position_of_b = np.where(vote == alternative)[0][0]
                    difference = point_difference_to_run(self.num_candidates, position_of_a, position_of_b, *args,
                                                         **kwargs)
                    ranked_votes.append((difference, vote))
                ranked_votes.sort(key=lambda a: a[0], reverse=True)
                changed_votes = change_votes(ranked_votes, alternative, winner, number_k)
                new_output = rule_to_run(self.num_voters, self.num_candidates, changed_votes, *args, **kwargs)
                new_winner = new_output[0][0]
                if new_winner != winner:
                    return number_k
        print("Error - MOV algorithm did not terminate correctly")  # in theory, we should never get here
