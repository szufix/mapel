
from mapel.core.glossary import *
from mapel.core.utils import *
from collections import Counter


def export_instance_to_a_file(experiment):
    """ Store votes in a file """

    path_to_folder = os.path.join(os.getcwd(), "election", experiment.experiment_id, "instances")
    make_folder_if_do_not_exist(path_to_folder)
    path_to_file = os.path.join(path_to_folder, f'{experiment.instance_id}.ri')

    with open(path_to_file, 'w') as file_:

        if experiment.culture_id in NICE_NAME:
            file_.write("# " + NICE_NAME[experiment.culture_id] + " " + str(experiment.params) + "\n")
        else:
            file_.write("# " + experiment.culture_id + " " + str(experiment.params) + "\n")

        file_.write(str(experiment.num_agents) + "\n")

        for i in range(experiment.num_agents):
            file_.write(str(i) + ', a' + str(i) + "\n")

        c = Counter(map(tuple, experiment.votes))
        counted_votes = [[count, list(row)] for row, count in c.items()]

        file_.write(str(experiment.num_agents) + ', ' + str(experiment.num_agents) + ', ' +
                    str(len(counted_votes)) + "\n")

        for i in range(len(counted_votes)):
            file_.write(str(counted_votes[i][0]) + ', ')
            for j in range(len(counted_votes[i][1])):
                file_.write(str(int(counted_votes[i][1][j])))
                if j < len(counted_votes[i][1]) - 1:
                    file_.write(", ")
            file_.write("\n")

