
import csv
import random as rand


def generate_approval_grenoble_votes(num_voters=None, num_candidates=None,
                                         params=None):

    name = 'Grenoble_2017'
    path = f'source/{name}.csv'

    votes = []
    with open(path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')

        for row in reader:
            vote = set()
            vote_raw = list(row.values())[1:12]
            for i, value in enumerate(vote_raw):
                if value == '1':
                    vote.add(i)
            if vote:
                votes.append(vote)

    return rand.sample(votes, num_voters)


def generate_approval_field_votes(num_voters=None, num_candidates=None,
                                         params=None):

    name = params['name']
    path = f'source/{name}.csv'

    votes = []
    with open(path, 'r', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')

        for row in reader:
            vote = set()
            vote_raw = list(row.values())[1:12]
            for i, value in enumerate(vote_raw):
                if value == '1':
                    vote.add(i)
            if vote:
                votes.append(vote)

    return rand.sample(votes, num_voters)

