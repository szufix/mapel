
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


def generate_approval_field_votes(num_voters=None, num_candidates=None, params=None):

    folder = params['folder']
    name = params['name']

    if folder == 'french_2017':

        path = f'source/{folder}/{name}.csv'
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

    elif folder == 'french_2002':

        path = f'source/{folder}/{name}.toc'
        votes = []

        with open(path, 'r', newline='') as toc_file:

            num_candidates = int(toc_file.readline())
            for _ in range(num_candidates):
                toc_file.readline()

            line = toc_file.readline().rstrip("\n").split(',')
            total_num_voters = int(line[0])
            num_options = int(line[2])

            votes = []

            for j in range(num_options):
                line = toc_file.readline().strip()
                quantity = int(line.split(',')[0])
                line = line.split('{')
                if len(line) == 2:
                    raw_vote = set(line[0].split(',')[1])
                elif len(line) == 3:
                    raw_vote = set(line[1].replace('{', '').replace('}', '').split(',')[0:-1])

                vote = set()
                if raw_vote != {''}:
                    for c in raw_vote:
                        vote.add(int(c)-1)

                for _ in range(quantity):
                    votes.append(vote)

    return rand.sample(votes, num_voters)

