
from mapel.core.glossary import *
from mapel.core.utils import *
from collections import Counter


def export_instance_to_a_file(experiment):
    """ Store votes in a file """

    path_to_folder = os.path.join(os.getcwd(), "election", experiment.experiment_id, "instances")
    make_folder_if_do_not_exist(path_to_folder)
    path_to_file = os.path.join(path_to_folder, f'{experiment.instance_id}.mi')

    with open(path_to_file, 'w') as file_:

        if experiment.model_id in NICE_NAME:
            file_.write("# " + NICE_NAME[experiment.model_id] + " " + str(experiment.params) + "\n")
        else:
            file_.write("# " + experiment.model_id + " " + str(experiment.params) + "\n")

        for s in range(2):

            file_.write(str(experiment.num_agents) + "\n")

            for i in range(experiment.num_agents):
                if s == 0:
                    file_.write(str(i) + ', a' + str(i) + "\n")
                else:
                    file_.write(str(i) + ', b' + str(i) + "\n")

            votes = experiment.votes[s]

            file_.write(str(experiment.num_agents) + ', ' + str(experiment.num_agents) + ', ' +
                        str(len(votes)) + "\n")

            for i in range(len(votes)):
                file_.write(str(i) + ', ')
                for j in range(experiment.num_agents):
                    file_.write(str(int(votes[i][j])))
                    if j < len(votes[i]) - 1:
                        file_.write(", ")
                file_.write("\n")
