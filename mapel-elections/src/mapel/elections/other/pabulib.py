import csv
import os
import numpy as np

from mapel.elections.cultures_ import store_votes_in_a_file


def convert_pb_to_app(experiment, num_candidates=100, num_voters=100, model='pabulib'):

    path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "source")

    for p, name in enumerate(os.listdir(path)):
        print(name)
        path = f'experiments/{experiment.experiment_id}/source/{name}'


        votes = {}
        projects = {}

        # import from .pb file
        with open(path, 'r', newline='', encoding="utf-8") as csvfile:
            meta = {}
            section = ""
            header = []
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                if str(row[0]).strip().lower() in ["meta", "projects", "votes"]:
                    section = str(row[0]).strip().lower()
                    header = next(reader)
                elif section == "meta":
                    meta[row[0]] = row[1].strip()
                elif section == "projects":
                    projects[row[0]] = {}
                    for it, key in enumerate(header[1:]):
                        projects[row[0]][key.strip()] = row[it+1].strip()
                elif section == "votes":
                    votes[row[0]] = {}
                    for it, key in enumerate(header[1:]):
                        votes[row[0]][key.strip()] = row[it+1].strip()

        # extract approval ballots from votes
        approval_votes = []
        for vote in votes.values():
            vote = vote['vote'].split(',')
            approval_votes.append(set(vote))

        # randomly map projects to [0,1,2...,num_projects]
        mapping = {}
        perm = np.random.permutation(len(projects))
        for i, key in enumerate(projects.keys()):
            mapping[key] = perm[i]

        approval_votes = [[mapping[i] for i in vote] for vote in approval_votes]

        # cut projects to [0,1,2...,k]
        approval_votes_cut = []
        for vote in approval_votes:
            vote_cut = set()
            for c in vote:
                if c < num_candidates:
                    vote_cut.add(c)
            approval_votes_cut.append(vote_cut)

        # cut the voters:
        perm = np.random.permutation(len(votes))
        # print(len(perm))
        final_approval_votes_cut = []
        for i, vote in enumerate(approval_votes_cut):
            # print(num_voters)
            if perm[i] < num_voters:
                # print(perm[i])
                final_approval_votes_cut.append(vote)
        # print(len(final_approval_votes_cut))

        # store in .app file
        name = name.replace('.pb', '')
        path = f'experiments/{experiment.experiment_id}/elections/{model}_{p}.app'
        # num_candidates = num_candidates
        # num_voters = len(votes)
        params = {}
        ballot = 'approval'

        election = None
        model_id = model
        store_votes_in_a_file(election, model_id, num_candidates, num_voters,
                          params, path, ballot, votes=final_approval_votes_cut)

