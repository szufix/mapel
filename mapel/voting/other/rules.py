#!/usr/bin/env python

import os
import csv
import ast
import sys

try:
    sys.path.append('/Users/szufa/PycharmProjects/abcvoting/')
    from abcvoting.preferences import Profile
    from abcvoting import abcrules
except ImportError:
    Profile = None
    abcrules = None


try:
    rule_mapping = {
        'av': abcrules.compute_av,
        'pav': abcrules.compute_pav,
        'sav': abcrules.compute_sav,
        'slav': abcrules.compute_slav,
        'cc': abcrules.compute_cc,
        # 'geom2': abcrules.compute_geom2,
        'seqpav': abcrules.compute_seqpav,
        'revseqpav': abcrules.compute_revseqpav,
        'seqslav': abcrules.compute_seqslav,
        'seqcc': abcrules.compute_seqcc,
        'seqphragmen': abcrules.compute_seqphragmen,
        'minimaxphragmen': abcrules.compute_minimaxphragmen,
        'monroe': abcrules.compute_monroe,
        'greedy-monroe': abcrules.compute_greedy_monroe,
        'minimaxav': abcrules.compute_minimaxav,
        'lexminimaxav': abcrules.compute_lexminimaxav,
        'rule-x': abcrules.compute_rule_x,
        'phragmen-enestroem': abcrules.compute_phragmen_enestroem,
        'consensus-rule': abcrules.compute_consensus_rule,
    }
except Exception:
    rule_mapping = {}

def compute_rule(experiment=None, rule_name=None, committee_size=1, printing=False):
    rule = rule_mapping[rule_name]
    all_winning_committees = {}
    for election in experiment.elections.values():
        if printing:
            print(election.election_id)
        profile = Profile(election.num_candidates)
        profile.add_voters(election.votes)
        # try:
        winning_committees = rule(profile, committee_size, algorithm="gurobi")
        # except Exception:
        #     winning_committees = [{}]
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

