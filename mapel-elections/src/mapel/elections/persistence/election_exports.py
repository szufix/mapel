import csv
from collections import Counter

from mapel.core.glossary import *
from mapel.core.utils import *


def export_votes_to_file(election,
                         culture_id: str,
                         num_candidates: int,
                         num_voters: int,
                         params: dict,
                         path,
                         ballot_type,
                         votes=None,
                         is_aggregated=True):
    """
    Exports votes to a file.

    Parameters
    ----------
        election
            Election.
        culture_id: str
            Culture id.
        num_candidates : int
            Number of Candidates
        num_voters : int
            Number of Voters
        params : int
            Model params to be exported.
        path:
            Path to the place in which the file should be stored.
        ballot_type:
            Ballot type.
        votes:
            Votes.
        is_aggregated : bool
            If True then votes are stored in aggregated way.
    """
    if votes is None:
        votes = election.votes

    if params is None:
        params = {}

    with open(path, 'w') as file_:
        if ballot_type == 'ordinal':
            file_.write(f'# FILE NAME: {election.election_id}.soc\n')
            file_.write(f'# DATA TYPE: soc \n')
        elif ballot_type == 'approval':
            file_.write(f'# FILE NAME: {election.election_id}.app\n')
            file_.write(f'# DATA TYPE: app \n')
        file_.write(f'# CULTURE ID: {culture_id} \n')
        file_.write(f'# PARAMS: {str(params)} \n')
        file_.write(f'# NUMBER ALTERNATIVES: {num_candidates} \n')
        file_.write(f'# NUMBER VOTERS: {num_voters} \n')

        if is_aggregated:

            c = Counter(map(tuple, votes))
            counted_votes = [[count, list(row)] for row, count in c.items()]
            counted_votes = sorted(counted_votes, reverse=True)

            if ballot_type == 'approval':
                for i in range(len(counted_votes)):
                    file_.write(str(counted_votes[i][0]) + ': {')
                    for j in range(len(counted_votes[i][1])):
                        file_.write(str(int(counted_votes[i][1][j])))
                        if j < len(counted_votes[i][1]) - 1:
                            file_.write(", ")
                    file_.write("}\n")

            elif ballot_type == 'ordinal':
                for i in range(len(counted_votes)):
                    file_.write(str(counted_votes[i][0]) + ': ')
                    for j in range(len(counted_votes[i][1])):
                        file_.write(str(int(counted_votes[i][1][j])))
                        if j < len(counted_votes[i][1]) - 1:
                            file_.write(", ")
                    file_.write("\n")
        else:

            file_.write(str(num_voters) + ', ' + str(num_voters) + ', ' +
                        str(num_voters) + "\n")

            if ballot_type == 'approval':
                for i in range(len(votes)):
                    file_.write('1: {')
                    for j in range(len(votes[i])):
                        file_.write(str(int(list(votes[i])[j])))
                        if j < len(votes[i]) - 1:
                            file_.write(", ")
                    file_.write("}\n")

            elif ballot_type == 'ordinal':
                for i in range(len(votes)):
                    file_.write('1: ')
                    for j in range(len(votes[i])):
                        file_.write(str(int(votes[i][j])))
                        if j < len(votes[i]) - 1:
                            file_.write(", ")
                    file_.write("\n")


def export_approval_election(election,
                             is_aggregated: bool = True):
    """
    Exports approval election in an .app file

    Parameters
    ----------
        election
            Election.
        is_aggregated : bool
            If True then votes are stored in aggregated way.
    """
    path_to_folder = os.path.join(os.getcwd(), "experiments", election.experiment_id, "elections")
    make_folder_if_do_not_exist(path_to_folder)
    path_to_file = os.path.join(path_to_folder, f'{election.election_id}.app')

    if election.culture_id in APPROVAL_FAKE_MODELS:
        file_ = open(path_to_file, 'w')
        file_.write(f'$ {election.culture_id} {election.params} \n')
        file_.write(str(election.num_candidates) + '\n')
        file_.write(str(election.num_voters) + '\n')
        file_.close()

    else:
        export_votes_to_file(election,
                             election.culture_id,
                             election.num_candidates,
                             election.num_voters,
                             election.params,
                             path_to_file,
                             ballot_type='approval',
                             votes=election.votes,
                             is_aggregated=is_aggregated)


def export_ordinal_election(election,
                            is_aggregated: bool = True):
    """
    Exports ordinal election to a .soc file

    Parameters
    ----------
        election
            Election.
        is_aggregated : bool
            If True then votes are stored in aggregated way.
    """

    path_to_folder = os.path.join(os.getcwd(), "experiments", election.experiment_id, "elections")
    make_folder_if_do_not_exist(path_to_folder)
    path_to_file = os.path.join(path_to_folder, f'{election.election_id}.soc')

    if election.culture_id in LIST_OF_FAKE_MODELS:
        file_ = open(path_to_file, 'w')
        file_.write(f'$ {election.culture_id} {election.params} \n')
        file_.write(str(election.num_candidates) + '\n')
        file_.write(str(election.num_voters) + '\n')
        file_.close()
    else:
        export_votes_to_file(election,
                             election.culture_id,
                             election.num_candidates,
                             election.num_voters,
                             election.params,
                             path_to_file,
                             ballot_type='ordinal',
                             votes=election.votes,
                             is_aggregated=is_aggregated)


def export_distances(experiment,
                     object_type: str = 'vote',
                     length: int = None):
    """
    Exports distances to a csv file.

    Parameters
    ----------
        experiment
            Experiment.
        object_type : str
            Object type.
        length : int
            Number of elections.
    """

    file_name = f'{experiment.election_id}_{object_type}.csv'
    path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "distances",
                        file_name)
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["v1", "v2", "distance"])
        for v1 in range(length):
            for v2 in range(length):
                distance = str(experiment.distances[object_type][v1][v2])
                writer.writerow([v1, v2, distance])


def export_coordinates(experiment,
                       object_type: str = 'vote',
                       length: int = None):
    """
    Exports coordinates to a csv file.

    Parameters
    ----------
        experiment
            Experiment.
        object_type : str
            Object type.
        length : int
            Number of elections.
    """

    file_name = f'{experiment.election_id}_{object_type}.csv'
    path = os.path.join(os.getcwd(), "experiments", experiment.experiment_id, "coordinates",
                        file_name)
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(["vote_id", "x", "y"])
        for vote_id in range(length):
            x = str(experiment.coordinates[object_type][vote_id][0])
            y = str(experiment.coordinates[object_type][vote_id][1])
            writer.writerow([vote_id, x, y])
