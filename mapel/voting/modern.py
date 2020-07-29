""" This module contains all the functions that user can use """

from . import objects as obj
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from shutil import copyfile
import itertools
import scipy.stats as stats

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def print_2d(exp_name, num_winners=0, mask=False,
             angle=0, reverse=False, update=False, values="default", coloring="purple",
             num_elections=800, winners_order="positionwise_approx_cc", main_order="", metric="positionwise",
             saveas="map_2d", show=True, ms=9, coloring_func=None):
    """ Print the two-dimensional embedding of multi-dimensional map of the elections """

    model = obj.Model_2d(exp_name, num_winners=num_winners, num_elections=num_elections,
                         winners_order=winners_order, main_order=main_order, metric=metric)
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

        file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "controllers", "advanced", str(values) + ".txt")
        file_ = open(file_name, 'r')
        #num_elections = int(file_.readline())

        ctr = 0
        for k in range(model.num_families):
            for _ in range(model.families[k].size):
                if ctr >= num_elections:
                    break
                shade = float(file_.readline())
                marker = "o"
                color = coloring
                if coloring == "intervals":
                    color = interval_color(shade)
                    shade = 1
                if coloring_func is not None:
                    shade, color, marker = coloring_func(shade)
                ax.scatter(model.points[model.main_order[ctr]][0], model.points[model.main_order[ctr]][1],
                           label=values, color=color, alpha=shade, s=ms, marker=marker)
                ctr += 1

    else:

        for k in range(model.num_families):
            if model.families[k].show:
                ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                           color=model.families[k].color, label=model.families[k].label,
                           alpha=model.families[k].alpha, s=ms)

    #  mark the winners
    for w in model.winners:
        ax.scatter(model.points[w][0], model.points[w][1], color="red", s=50, marker='x')

    if mask:

        file_points = os.path.join(os.getcwd(), "images", "tmp", str(exp_name) + "_points.png")
        plt.savefig(file_points)

        x = int(640 * 1)
        y = int(480 * 1.25)

        background = Image.open(file_points).resize((x, y))
        #foreground.show()

        background = add_margin(background, 20, 115, 40, 145, (255, 255, 255))

        file_mask = os.path.join(os.getcwd(), "images", "masks", "mask.png")
        foreground = Image.open(file_mask)#.resize((x, y))
        #background.show()


        background.paste(foreground, (9, 11), foreground)

        if show:
            background.show()

        file_map_with_mask = os.path.join(os.getcwd(), "images", str(saveas) + ".png")
        background.save(file_map_with_mask)

        """
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
        """

    else:

        text_name = str(model.num_voters) + " x " + str(model.num_candidates)
        text = ax.text(0.0, 1.05, text_name, transform=ax.transAxes)
        file_name = os.path.join(os.getcwd(), "images", str(saveas) + ".png")
        if values == "default":
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(file_name, bbox_extra_artists=(lgd, text), bbox_inches='tight')
        else:
            plt.savefig(file_name, bbox_inches='tight')

        if show:
            plt.show()


def print_matrix(exp_name, scale=1., metric="positionwise", saveas="matrix", show=True):
    """Print the matrix with average distances between each pair of models """

    model = obj.Model_xd(exp_name, metric=metric)
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

    file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "controllers", "basic", "matrix.txt")
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

    file_name = os.path.join(os.getcwd(), "images", str(saveas) + ".png")
    plt.savefig(file_name)
    if show:
        plt.show()


def prepare_approx_cc_order(exp_name, metric="positionwise"):
    """ Copy all the elections and the change the order according to approx_cc order """

    file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "results", "winners", str(metric) + "_approx_cc.txt")
    file_ = open(file_name, 'r')

    file_.readline()  # skip this line
    num_elections = int(file_.readline())
    file_.readline()  # skip this line

    for i in range(num_elections):
        target = str(file_.readline().replace("\n", ""))

        src = os.path.join(os.getcwd(), "experiments", str(exp_name), "elections", "soc_original",
                           "core_" + str(target) + ".soc")

        dst = os.path.join(os.getcwd(), "experiments", str(exp_name), "elections", "soc_" + str(metric) + "_approx_cc",
                           "core_" + str(i) + ".soc")

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


def print_param_vs_distance(exp_name, values="", scale="none", metric="positionwise", saveas="correlation", show=True):

    #  only for example_100_100

    file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "controllers", "advanced", str(values) + ".txt")
    file_ = open(file_name, 'r')

    file_name = os.path.join(os.getcwd(), "experiments", str(exp_name), "results", "distances", str(metric) + ".txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    file_.readline()
    file_.readline()

    distances = [0. for _ in range(num_elections)]
    times = [float(file_.readline()) for _ in range(num_elections)]
    target = [i for i in range(0, 30)]   #  IC elections


    for i in range(num_elections):
        for j in range(i+1, num_elections):
            line = file_.readline().replace("\n", "").split(" ")
            #print(i,j, line[2])
            dist = float(line[2])

            if j in target:
                distances[i] += dist

            if i in target:
                distances[j] += dist

    #print(distances)

    #print(len(times), len(distances))

    distances = [x/30. for x in distances]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #plt.axis('off')

    if scale == "log":
        times = np.log(times)
        plt.ylabel("log ( " + str(values) + " )")
    elif scale == "loglog":
        times = np.log(times)
        times = np.log(times)
        plt.ylabel("log ( log ( " + str(values) + " ) )")


    pear = stats.pearsonr(times, distances)
    pear = round(pear[0], 2)

    model = obj.Model_xd(exp_name, metric)

    left = 0
    for k in range(model.num_families):
        right = left + model.families[k].size
        ax.scatter(distances[left:right], times[left:right],
                   color=model.families[k].color, label=model.families[k].label,
                   alpha=model.families[k].alpha, s=9)
        left = right

    title_text = str(model.num_voters) + " voters  x  " + str(model.num_candidates) + " candidates"
    pear_text = "PCC = " + str(pear)
    add_text = ax.text(0.7, 0.8, pear_text, transform=ax.transAxes)
    plt.title(title_text)
    plt.xlabel("average distance from IC elections")
    file_name = os.path.join(os.getcwd(), "images", str(saveas) + ".png")
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(file_name, bbox_extra_artists=(lgd, add_text), bbox_inches='tight')
    if show:
        plt.show()
