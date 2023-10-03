"""
This parser is compatible with Preflib file format in October 2023
Sample code for using it:
from mapel.elections.other.parser import preflib_parser
votes, nr_candidates, nr_votes, label = preflib_parser(file_name="~Input the absolute path to the file~", file_ending=4)
"""

import re
import numpy as np

regex_file_name = r'# FILE NAME:'
regex_title = r'# TITLE:'
regex_data_type = r'# DATA TYPE:'
regex_number_alternatives = r"# NUMBER ALTERNATIVES:"
regex_number_voters = r"# NUMBER VOTERS:"
regex_number_unique_orders = r"# NUMBER UNIQUE ORDERS:"
regex_number_categories = r"# NUMBER CATEGORIES:"


def validate():
    pass


def process_soc_line(line: str, votes: list):
    tokens = line.split(':')
    nr_this_vote = int(tokens[0])
    vote = [int(x) for x in tokens[1].split(',')]
    vote = np.array([x-1 for x in vote])
    for i in range(0, nr_this_vote):
        votes.append(vote)
    pass


def process_soi_line(line: str, votes: list):
    pass


def process_toc_line(line: str, votes: list):
    pass


def process_toi_line(line: str, votes: list):
    pass


def preflib_parser(file_name: str, label: str = None, file_ending: int = 4):
    votes = []
    nr_candidates = 0
    nr_votes = 0
    nr_unique = 0
    alternative_names = list()
    from_file_file_name = ''
    from_file_title = ''
    from_file_data_type = ''
    if label == "" or label is None:
        label = file_name[:-file_ending]
    file = open(file_name, "r")
    # read metadata
    for line in file:
        if line[-1] == '\n':
            line = line[:-1]
        if line[0] != '#':
            if from_file_data_type == 'soc':
                process_soc_line(line, votes)
            elif from_file_data_type == 'soi':
                process_soi_line(line, votes)
            elif from_file_data_type == 'toc':
                process_toc_line(line, votes)
            elif from_file_data_type == 'toi':
                process_toi_line(line, votes)
            break
        elif re.search(regex_file_name, line):
            from_file_file_name = line.split(':')[1][1:-file_ending]
        elif re.search(regex_title, line):
            from_file_title = line.split(':')[1].replace(" ", "")
        elif re.search(regex_data_type, line):
            from_file_data_type = line.split(':')[1][1:]
        elif re.search(regex_number_alternatives, line):
            nr_candidates = int(line.split(':')[1])
        elif re.search(regex_number_voters, line):
            nr_votes = int(line.split(':')[1])
        elif re.search(regex_number_unique_orders, line):
            nr_unique = int(line.split(':')[1])

    label = from_file_title + "_" + from_file_file_name
    # read votes
    if from_file_data_type == 'soc':
        for line in file:
            process_soc_line(line, votes)
    elif from_file_data_type == 'soi':
        for line in file:
            process_soi_line(line, votes)
    elif from_file_data_type == 'toc':
        for line in file:
            process_toc_line(line, votes)
    elif from_file_data_type == 'toi':
        for line in file:
            process_toi_line(line, votes)
    else:
        print(label, " - unknown data format.")

    return np.array(votes), nr_candidates, nr_votes, label
