#!/usr/bin/env python

import os
import csv
import ast
import sys
import numpy as np

try:
    sys.path.append('/Users/szufa/PycharmProjects/abcvoting/')
    from abcvoting.preferences import Profile
    from abcvoting import abcrules
except ImportError:
    Profile = None
    abcrules = None


def compute_abcvoting_rule(experiment=None, rule_name=None, committee_size=1, printing=False,
                           resolute=False):
    all_winning_committees = {}
    for election in experiment.elections.values():
        if printing:
            print(election.election_id)
        profile = Profile(election.num_candidates)
        profile.add_voters(election.votes)
        try:
            winning_committees = abcrules.compute(rule_name, profile, committee_size,
                                                  algorithm="gurobi", resolute=resolute)
            # print(winning_committees)
        except Exception:
            try:
                winning_committees = abcrules.compute(rule_name, profile, committee_size)
            except:
                winning_committees = {}
        all_winning_committees[election.election_id] = winning_committees
    store_committees_to_file(experiment.experiment_id, rule_name, all_winning_committees)


def store_committees_to_file(experiment_id, rule_name, all_winning_committees):
    path = os.path.join(os.getcwd(), "experiments", experiment_id, 'features',
                        f'{rule_name}.csv')
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["election_id", "committee"])
        for election_id in all_winning_committees:
            writer.writerow([election_id, all_winning_committees[election_id]])


def import_committees_from_file(experiment_id, rule_name):
    all_winning_committees = {}
    path = os.path.join(os.getcwd(), "experiments", experiment_id, 'features',
                        f'{rule_name}.csv')
    with open(path, 'r', newline='') as csv_file:
        header = [h.strip() for h in csv_file.readline().split(';')]
        reader = csv.DictReader(csv_file, fieldnames=header, delimiter=';')
        for row in reader:
            winning_committees = ast.literal_eval(row['committee'])
            if len(winning_committees) == 0:
                winning_committees = [set()]
            all_winning_committees[row['election_id']] = winning_committees
    return all_winning_committees


def compute_not_abcvoting_rule(experiment=None, rule_name=None, committee_size=1, printing=False,
                           resolute=False):
    all_winning_committees = {}
    for election in experiment.elections.values():
        if printing:
            print(election.election_id)

        winning_committees = compute_borda_c4_rule(election, committee_size)

        all_winning_committees[election.election_id] = winning_committees
    store_committees_to_file(experiment.experiment_id, rule_name, all_winning_committees)


def compute_borda_c4_rule(election, committee_size=1):

    scores = np.zeros(election.num_candidates)
    for i, vote in enumerate(election.votes):
        max_score = 10 # election.approval_cuts[i]
        for pos, c in enumerate(vote):
            scores[c] += pos

    print(scores)

    return {}
