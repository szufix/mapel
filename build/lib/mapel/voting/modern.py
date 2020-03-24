""" This module contains all the functions that user can use """

from . import objects as obj
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from shutil import copyfile
import itertools


def print_2d(exp_name, num_winners=0, mask=False,
             angle=0, reverse=False, update=False, values="default", coloring="purple",
             num_elections=800, winners_order="approx_cc", main_order=""):
    """ Print the two-dimensional embedding of multi-dimensional map of the elections """

    model = obj.Model_2d(exp_name, num_winners=num_winners, num_elections=num_elections,
                         winners_order=winners_order, main_order=main_order)
    core = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

    if angle != 0:
        model.rotate(angle)

    if reverse:
        model.reverse()

    if update:
        model.update()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')

    if values != "default":

        file_name = os.path.join(core, "experiments", str(exp_name), "controllers", str(values) + ".txt")
        file_ = open(file_name, 'r')
        ctr = 0
        for k in range(model.num_families):
            for _ in range(model.families[k].size):
                if ctr >= num_elections:
                    break
                shade = float(file_.readline())
                color = coloring
                if coloring == "intervals":
                    color = interval_color(shade)
                    shade = 1
                ax.scatter(model.points[model.main_order[ctr]][0], model.points[model.main_order[ctr]][1],
                           label=values, color=color, alpha=shade, s=9)
                ctr += 1

    else:

        for k in range(model.num_families):
            ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                       color=model.families[k].color, label=model.families[k].label,
                       alpha=model.families[k].alpha, s=9)

    #  mark the winners
    for w in model.winners:
        ax.scatter(model.points[w][0], model.points[w][1], color="red", s=50, marker='x')

    if mask:

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.margins(0.25)

        file_name = os.path.join(core, "images", str(exp_name) + "_map.png")
        plt.savefig(file_name)

        background = Image.open(file_name)
        c = 0.59
        x = int(1066*c)
        y = int(675*c)

        file_name = os.path.join(core, "images", "mask.png")
        foreground = Image.open(file_name).resize((x, y))
        background.paste(foreground, (-30, 39), foreground)
        background.show()

        file_name = os.path.join(core, "images", str(exp_name) + "_map_with_mask.png")
        background.save(file_name)

    else:

        text_name = str(model.num_voters) + " x " + str(model.num_candidates)
        text = ax.text(0.0, 1.05, text_name, transform=ax.transAxes)
        file_name = os.path.join(core, "images", str(exp_name) + "_map.png")
        if values == "default":
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(file_name, bbox_extra_artists=(lgd, text), bbox_inches='tight')
        else:
            plt.savefig(file_name, bbox_inches='tight')
        plt.show()


def print_matrix(exp_name, scale=1.):
    """Print the matrix with average distances between each pair of models """

    model = obj.Model_xd(exp_name)
    matrix = np.zeros([model.num_families, model.num_families])
    quantities = np.zeros([model.num_families, model.num_families])

    mapping = [i for i in range(model.num_families) for _ in range(model.families[i].size)]

    for i in range(model.num_points):
        for j in range(i+1, model.num_points):
            matrix[mapping[i]][mapping[j]] += model.distances[i][j]
            quantities[mapping[i]][mapping[j]] += 1

    for i in range(model.num_families):
        for j in range(i, model.num_families):
            matrix[i][j] /= float(quantities[i][j])
            matrix[i][j] *= scale
            matrix[i][j] = int(round(matrix[i][j], 0))
            matrix[j][i] = matrix[i][j]

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    file_name = os.path.join(file_name, "experiments", str(exp_name), "controllers", "matrix.txt")
    file_ = open(file_name, 'r')

    #  everything with _new refers to a new order

    num_families_new = int(file_.readline())
    order = [i for i in range(num_families_new)]

    for i in range(num_families_new):
        line = str(file_.readline().replace("\n", "").replace(" ", ""))
        for j in range(model.num_families):
            if model.families[j].label.replace(" ", "") == line:
                order[i] = j

    fig, ax = plt.subplots()
    matrix_new = np.zeros([num_families_new, num_families_new])

    for i in range(num_families_new):
        for j in range(num_families_new):
            c = int(matrix[order[i]][order[j]])
            matrix_new[i][j] = c
            ax.text(i, j, str(c), va='center', ha='center')

    labels_new = []
    for i in range(num_families_new):
        labels_new.append(model.families[order[i]].label)

    ax.matshow(matrix_new, cmap=plt.cm.Blues)

    x_values = labels_new
    y_values = labels_new
    y_axis = np.arange(0, num_families_new, 1)
    x_axis = np.arange(0, num_families_new, 1)

    plt.yticks(y_axis, y_values)
    plt.xticks(x_axis, x_values, rotation='vertical')

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    file_name = os.path.join(file_name, "images", str(exp_name) + "_matrix.png")
    plt.savefig(file_name)
    plt.show()


def prepare_approx_cc_order(exp_name):
    """ Copy all the elections and the change the order according to approx_cc order """

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    file_name = os.path.join(file_name, "experiments", str(exp_name), "results", "winners", "approx_cc.txt")
    file_ = open(file_name, 'r')

    file_.readline()  # skip this line
    num_elections = int(file_.readline())
    file_.readline()  # skip this line

    for i in range(num_elections):
        target = str(file_.readline().replace("\n", ""))

        src = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
        src = os.path.join(src, "experiments", str(exp_name), "elections", "soc_original",
                           "tmp_" + str(target) + ".soc")

        dst = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
        dst = os.path.join(dst, "experiments", str(exp_name), "elections", "soc_approx_cc",
                           "tmp_" + str(i) + ".soc")

        copyfile(src, dst)


def interval_color(shade):

    if shade > 0.8:
        color = "red"
    elif shade > 0.6:
        color = "orange"
    elif shade > 0.4:
        color = "yellow"
    elif shade > 0.2:
        color = "green"
    else:
        color = "blue"

    return color
