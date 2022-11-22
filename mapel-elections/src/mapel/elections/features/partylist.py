import sys
import time
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    sys.path.append(os.environ["PATH"])
    from abcvoting import fileio, genprofiles
    import gurobipy as gb
    from abcvoting.preferences import Profile, Voter
    from abcvoting import abcrules
except ImportError:
    pass


def convert_election_to_profile(election):
    profile = Profile(num_cand=election.num_candidates)

    voters = []
    for i, vote in enumerate(election.votes):
        voter = Voter(vote)
        voters.append(voter)

    profile._voters = voters
    return profile


def partylistdistance(election, feature_params=None):

    if 'largepartysize' in feature_params:
        largepartysize = feature_params['largepartysize']
    else:
        largepartysize = 5

    if 'time_limit' in feature_params:
        time_limit = feature_params['time_limit']
    else:
        time_limit = 5

    profile = convert_election_to_profile(election)

    model = gb.Model()
    model.setParam("OutputFlag", False)
    model.setParam('TimeLimit', time_limit)  # in seconds

    same_party = {}
    for c1 in profile.candidates:
        for c2 in range(c1):
            same_party[(c1, c2)] = model.addVar(vtype=gb.GRB.BINARY)

    edit = {}
    for v, _ in enumerate(profile):
        for c in profile.candidates:
            edit[(v, c)] = model.addVar(vtype=gb.GRB.BINARY)

    for v, vote in enumerate(profile):
        for c1 in profile.candidates:
            for c2 in range(c1):
                if c1 in vote.approved and c2 in vote.approved:
                    model.addConstr(
                        (same_party[(c1, c2)] == 1) >> (edit[(v, c1)] == edit[(v, c2)])
                    )
                    model.addConstr(
                        # (same_party[(c1, c2)] == 0) >> (edit[(v, c1)] + edit[(v, c2)] >= 1)
                        (same_party[(c1, c2)] + edit[(v, c1)] + edit[(v, c2)] >= 1)
                    )
                elif c1 not in vote.approved and c2 not in vote.approved:
                    model.addConstr(
                        (same_party[(c1, c2)] == 1) >> (edit[(v, c1)] == edit[(v, c2)])
                    )
                    model.addConstr(
                        # (same_party[(c1, c2)] == 0) >> (edit[(v, c1)] + edit[(v, c2)] <= 1)
                        (edit[(v, c1)] + edit[(v, c2)] <= 1 + same_party[(c1, c2)])
                    )
                else:
                    model.addConstr(
                        (same_party[(c1, c2)] == 1) >> (edit[(v, c1)] + edit[(v, c2)] == 1)
                    )
                    if c1 in vote.approved:
                        model.addConstr(
                            (same_party[(c1, c2)] + edit[(v, c1)] >= edit[(v, c2)])
                            # (same_party[(c1, c2)] == 0) >> (edit[(v, c1)] >= edit[(v, c2)])
                        )
                    else:
                        model.addConstr(
                            (same_party[(c1, c2)] + edit[(v, c2)] >= edit[(v, c1)])
                            # (same_party[(c1, c2)] == 0) >> (edit[(v, c2)] >= edit[(v, c1)])
                        )

    model.setObjective(
        gb.quicksum(edit[(v, c)] for v, _ in enumerate(profile) for c in profile.candidates),
        gb.GRB.MINIMIZE,
    )

    model.optimize()

    newprofile = Profile(num_cand=profile.num_cand)
    for v, vote in enumerate(profile):
        new_approved = {
            c
            for c in profile.candidates
            if (c in vote.approved and edit[(v, c)].X <= 0.1)
               or (c not in vote.approved and edit[(v, c)].X >= 0.9)
        }
        newprofile.add_voter(new_approved)

    # number of parties
    parties = set(profile.candidates)
    for c1 in profile.candidates:
        for c2 in range(c1):
            if same_party[(c1, c2)].X >= 0.9:
                parties.discard(c2)
    num_large_parties = 0
    for party in parties:
        support = len([voter for voter in newprofile if party in voter.approved])
        if support >= largepartysize:
            num_large_parties += 1

    # print(f"new {newprofile}")
    # print([sum(edit[(v, c)].X for c in profile.candidates) for v, _ in enumerate(profile)])
    # print([(c1, c2, same_party[(c1, c2)].X) for c1 in profile.candidates for c2 in range(c1)])

    # return model.objVal, num_large_parties, newprofile
    print(model.objVal, model.objbound, num_large_parties)
    return model.objVal, model.objbound, num_large_parties
    # print(num_large_parties)
    # return num_large_parties


def pav_time(election):

    profile = Profile(election.num_candidates)
    profile.add_voters(election.votes)
    committee_size = 10
    resolute = True
    rule_name = 'pav'
    start = time.time()
    winning_committees = abcrules.compute(rule_name, profile, committee_size,
                                          algorithm="gurobi", resolute=resolute)
    return time.time() - start
