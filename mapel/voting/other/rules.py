#!/usr/bin/env python

import os
import csv
import ast

try:
    from abcvoting.preferences import Profile
    from abcvoting import abcrules
except ImportError:
    Profile = None
    abcrules = None

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


def compute_rule(experiment=None, rule_name=None, committee_size=1):
    rule = rule_mapping[rule_name]
    committees = {}
    for election in experiment.elections.values():
        profile = Profile(election.num_candidates)
        profile.add_voters(election.votes)
        try:
            committee = rule(profile, committee_size, resolute=True)[0]
        except Exception:
            committee = {}
        committees[election.election_id] = committee
    store_committees_to_file(experiment.experiment_id, rule_name, committees)


def store_committees_to_file(experiment_id, rule_name, committees):
    path = os.path.join(os.getcwd(), "experiments", experiment_id, 'features',
                        f'{rule_name}.csv')
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["election_id", "committee"])
        for election_id in committees:
            writer.writerow([election_id, committees[election_id]])


def import_committees_from_file(experiment_id, rule_name):
    committees = {}
    path = os.path.join(os.getcwd(), "experiments", experiment_id, 'features',
                        f'{rule_name}.csv')
    with open(path, 'r', newline='') as csv_file:
        header = [h.strip() for h in csv_file.readline().split(';')]
        reader = csv.DictReader(csv_file, fieldnames=header, delimiter=';')
        for row in reader:
            committee = ast.literal_eval(row['committee'])
            if len(committee) == 0:
                committee = set()
            committees[row['election_id']] = committee
    return committees

