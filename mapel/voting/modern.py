""" This module contains all the functions that user can use """

### COMMENT ON SERVER ###
import matplotlib.pyplot as plt
import matplotlib.lines as lines
#########################

from . import objects as obj
import numpy as np
import os
from PIL import Image
from shutil import copyfile
import scipy.stats as stats
import networkx as nx
import csv


# MAIN FUNCTIONS
def print_2d(experiment_id, num_winners=0, mask=False,
             angle=0, reverse=False, update=False, values=None,
             num_elections=800, secondary_order_name="positionwise_approx_cc", main_order_name="default", metric="positionwise",
             saveas="map_2d", show=True, ms=9, normalizing_func=None, xticklabels=None, cmap='Purples_r',
             ignore=None, marker_func=None, tex=False, black=False):
    """ Print the two-dimensional embedding of multi-dimensional map of the elections """

    if xticklabels is None:
        xticklabels = [0., 0.2, 0.4, 0.6, 0.8, 1.]

    model = obj.Model_2d(experiment_id, num_winners=num_winners, num_elections=num_elections,
                         secondary_order_name=secondary_order_name, main_order_name=main_order_name, metric=metric, ignore=ignore)

    if angle != 0:
        model.rotate(angle)

    if reverse:
        model.reverse()

    if update:
        model.update()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')

    if values is not None:

        file_name = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced", str(values) + ".txt")
        file_ = open(file_name, 'r')

        images = []
        shades = []
        xx = []
        yy = []
        markers = []

        ctr = 0
        for k in range(model.num_families):
            for _ in range(model.families[k].size):

                shade = float(file_.readline())
                if normalizing_func is not None:
                    shade = normalizing_func(shade)
                shades.append(shade)

                marker = 'x'
                if marker_func is not None:
                    marker = marker_func(shade)
                markers.append(marker)

                xx.append(model.points[model.main_order[ctr]][0])
                yy.append(model.points[model.main_order[ctr]][1])

                ctr += 1

        xx = np.asarray(xx)
        yy = np.asarray(yy)
        shades = np.asarray(shades)
        markers = np.asarray(markers)

        unique_markers = set(markers)

        for um in unique_markers:
            masks = (markers == um)
            images.append([ax.scatter(xx[masks], yy[masks], c=shades[masks], vmin=0, vmax=1, cmap=cmap, marker=um, s=ms)])

        cb = fig.colorbar(images[0][0], ax=ax, orientation="horizontal", shrink=0.75)
        cb.ax.locator_params(nbins=len(xticklabels)-1)
        cb.ax.tick_params(labelsize=16)
        if xticklabels is not None:
            cb.ax.set_xticklabels(xticklabels)

    else:

        for k in range(model.num_families):
            if model.families[k].show:
                ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                           color=model.families[k].color, label=model.families[k].label,
                           alpha=model.families[k].alpha, s=ms)

    #  mark the orders
    # for w in model.orders:
    #    ax.scatter(model.points[w][0], model.points[w][1], color="red", s=50, marker='x')

    if mask:
        fig.set_size_inches(10, 10)
        corners = [[-1, 1, 1, -1], [1, 1, -1, -1]]
        ax.scatter(corners[0], corners[1], alpha=0)

        if experiment_id == "testbed_100_100":
            add_mask_100_100(fig, ax, black=black)

        file_name = saveas + ".png"
        path = os.path.join(os.getcwd(), "images", file_name)
        plt.savefig(path, bbox_inches='tight')
        plt.grid(True)

        plt.xlim(-1.4, 1.3)
        plt.ylim(-1.4, 1.3)

        if tex:
            import tikzplotlib
            file_name = saveas + ".tex"
            path = os.path.join(os.getcwd(), "images", "tex", file_name)
            tikzplotlib.save(path)

        if show:
            plt.show()

    else:
        text_name = str(model.num_voters) + " x " + str(model.num_candidates)
        text = ax.text(0.0, 1.05, text_name, transform=ax.transAxes)
        file_name = os.path.join(os.getcwd(), "images", str(saveas) + ".png")
        if values is None:
            lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(file_name, bbox_extra_artists=(lgd, text), bbox_inches='tight')
        else:

            plt.savefig(file_name, bbox_inches='tight')

        if show:
            plt.show()


def print_3d(experiment_id, num_winners=0, mask=False,
             angle=0, reverse=False, update=False, values="default", coloring="purple",
             num_elections=800, winners_order="positionwise_approx_cc", order="", metric="positionwise",
             saveas="map_3d", show=True, dot=9):
    """ Print the two-dimensional embedding of multi-dimensional map of the elections """

    model = obj.Model_3d(experiment_id, num_winners=num_winners, num_elections=num_elections,
                         winners_order=winners_order, order=order, metric=metric)
    if angle != 0:
        model.rotate(angle)

    if reverse:
        model.reverse()

    if update:
        model.update()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.axis('off')

    for k in range(model.num_families):
        if model.families[k].show:
            ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1], model.points_by_families[k][2],
                       color=model.families[k].color, label=model.families[k].label,
                       alpha=model.families[k].alpha, s=dot)

    text_name = str(model.num_voters) + " x " + str(model.num_candidates)
    #text = ax.text(0.0, 1.05, text_name, transform=ax.transAxes)
    file_name = os.path.join(os.getcwd(), "images", str(saveas) + ".png")

    #import pickle
    #pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))

    if values == "default":
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.savefig(file_name, bbox_inches='tight')

    if show:
        plt.show()


def print_matrix(experiment_id, scale=1., metric="positionwise", saveas="matrix", show=True):
    """Print the matrix with average distances between each pair of models """

    model = obj.Model_xd(experiment_id, metric=metric)
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

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "controllers", "basic", "matrix.txt")
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

    print(labels_new)

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


def prepare_approx_cc_order(experiment_id, metric="positionwise"):
    """ Copy all the elections and the change the order according to approx_cc order """

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "results", "orders", str(metric) + "_approx_cc.txt")
    file_ = open(file_name, 'r')

    file_.readline()  # skip this line
    num_elections = int(file_.readline())
    file_.readline()  # skip this line

    for i in range(num_elections):
        target = str(file_.readline().replace("\n", ""))

        src = os.path.join(os.getcwd(), "experiments", str(experiment_id), "elections", "soc_original",
                           "core_" + str(target) + ".soc")

        dst = os.path.join(os.getcwd(), "experiments", str(experiment_id), "elections", "soc_" + str(metric) + "_approx_cc",
                           "core_" + str(i) + ".soc")

        copyfile(src, dst)


def print_param_vs_distance(experiment_id, values="", scale="none", metric="positionwise", saveas="correlation", show=True):

    #  only for example_100_100

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "results", "distances", str(metric) + ".txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    file_.readline()
    file_.readline()

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "controllers", "advanced", str(values) + ".txt")
    file_2 = open(file_name, 'r')
    times = [float(file_2.readline()) for _ in range(num_elections)]

    distances = [0. for _ in range(num_elections)]
    target = [i for i in range(0, 30)]   #  IC elections

    for i in range(num_elections):
        for j in range(i+1, num_elections):
            line = file_.readline().replace("\n", "").split(" ")
            dist = float(line[2])

            if j in target:
                distances[i] += dist

            if i in target:
                distances[j] += dist

    distances = [x/30. for x in distances]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if scale == "log":
        times = np.log(times)
        plt.ylabel("log ( " + str(values) + " )")
    elif scale == "loglog":
        times = np.log(times)
        times = np.log(times)
        plt.ylabel("log ( log ( " + str(values) + " ) )")

    pear = stats.pearsonr(times, distances)
    pear = round(pear[0], 2)
    model = obj.Model_xd(experiment_id, metric)

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


def convert_xd_to_2d(experiment, num_iterations=1000, metric="positionwise"):

    model = obj.Model_xd(experiment, metric=metric)
    X = np.zeros((model.num_points, model.num_points))
    for i in range(model.num_points):
        for j in range(i + 1, model.num_points):
            if model.distances[i][j] == 0:
                model.distances[i][j] = 0.01
            X[i][j] = 1. / model.distances[i][j]
            X[j][i] = 1. / model.distances[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    my_pos = nx.spring_layout(G, iterations=num_iterations, dim=2)

    file_name = os.path.join(os.getcwd(), "experiments", experiment, "results", "points", metric + "_2d.csv")
    with open(file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "x", "y"])

        for i in range(model.num_points):
            x = round(my_pos[i][0], 5)
            y = round(my_pos[i][1], 5)
            writer.writerow([i, x, y])


def convert_xd_to_3d(experiment, num_iterations=1000, metric="positionwise"):

    model = obj.Model_xd(experiment, metric=metric)
    X = np.zeros((model.num_points, model.num_points))
    for i in range(model.num_points):
        for j in range(i + 1, model.num_points):
            if model.distances[i][j] == 0:
                model.distances[i][j] = 0.01
            X[i][j] = 1. / model.distances[i][j]
            X[j][i] = 1. / model.distances[i][j]

    dt = [('weight', float)]
    X = X.view(dt)
    G = nx.from_numpy_matrix(X)

    my_pos = nx.spring_layout(G, iterations=num_iterations, dim=3)

    file_name = os.path.join(os.getcwd(), "experiments", experiment, "results", "points", metric + "_3d.csv")
    with open(file_name, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "x", "y", "z"])

        for i in range(model.num_points):
            x = round(my_pos[i][0], 5)
            y = round(my_pos[i][1], 5)
            z = round(my_pos[i][2], 5)
            writer.writerow([i, x, y, z])


# HELPER FUNCTIONS
def custom_div_cmap(num_colors=101, name='custom_div_cmap',
                    colors=None):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """
    if colors is None:
        colors = ["lightgreen", "yellow", "orange", "red", "black"]

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name=name, colors=colors, N=num_colors)
    return cmap


def temporary_shade(shade):
    if shade == 1.:
        return 0.2, "purple", "x"
    elif shade >= 0.91:
        shade = 1-(shade-0.9)*10
        return shade, "purple", "o"
    return 1, "purple", "o"


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def add_mask_100_100(fig, ax, black=False):

            def my_arrow(x1, y1, x2, y2):
                ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.02, head_length=0.05, fc='k', ec='k')

            def my_line(x1, y1, x2, y2):
                ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0., head_length=0., fc='k', ec='k')

            def my_text(x1, y1, text, color="black", alpha=1.):

                if black:
                    color = "black"
                    alpha = 1

                plt.text(x1, y1, text, size=12, rotation=0., ha="center", va="center",
                         color=color, alpha=alpha,
                         bbox=dict(boxstyle="round", ec="black", fc="white"))

            def my_number(x1, y1, number, color="black"):

                if black:
                    color = "black"

                plt.text(x1, y1, number, size=8, rotation=0., ha="center", va="center", color=color, fontweight='bold')

            def my_plot(points):

                x = [p[0] for p in points]
                y = [p[1] for p in points]
                from scipy.interpolate import interp1d

                t = np.arange(len(x))
                ti = np.linspace(0, t.max(), 10 * t.size)

                xi = interp1d(t, x, kind='cubic')(ti)
                yi = interp1d(t, y, kind='cubic')(ti)

                ax.plot(xi, yi, color='black')

            # 1D Euclidean
            points = [[0.31, 0.74], [0.37, 0.8], [0.53, 0.66], [0.69, 0.43], [0.71, 0.26], [0.68, 0.12], [0.59, 0.02], [0.49, -0.04], [0.45, 0.04], [0.44, 0.3], [0.31, 0.74]]
            my_plot(points)
            #ax.add_line(plt.Polygon(points, fill=None, edgecolor='black', lw=2, joinstyle="round", capstyle="round"))
            my_arrow(0.64, 0.71, 0.61, 0.66)
            my_line(0.75, 0.90, 0.64, 0.71)
            my_line(0.95, 0.65, 0.64, 0.71)
            my_text(0.75, 0.90, "1D Interval", color="darkgreen")
            my_text(0.95, 0.65, "SP (Con.)", color="limegreen")

            # Walsh SP
            #points = [[0.04, -0.64], [0, -0.89], [-0.07, -1.02], [-0.16, -1], [-0.28, -0.93], [-0.34, -0.79], [-0.22, -0.69], [0.04, -0.64]]
            points = [[0.04, -0.66], [-0.005, -0.89], [-0.07, -1.02], [-0.16, -1], [-0.28, -0.93], [-0.34, -0.79],
                      [-0.22, -0.69], [0.04, -0.66]]
            my_plot(points)

            ax.arrow(-0.32, -1.2, 0.07, 0.16, head_width=0.02, head_length=0.05, fc='k', ec='k')
            my_text(-0.32, -1.2, "SP (Wal.)", color="OliveDrab")

            # Mallows next to Walsh
            points = [[0.04, -0.66], [0.12, -0.64], [0.15, -0.72], [0.15, -0.98], [0.11, -1.05], [-0.07, -1.02]]
            my_plot(points)

            # Mallows 0.5
            points = [[0.43, -0.95], [0.38, -1.03], [0.32, -1.04], [0.19, -0.98],
                      [0.19, -0.69], [0.24, -0.63], [0.3, -0.69]]
            my_plot(points)

            # Mallows 0.25
            points = [[0.3, -0.69], [0.35, -0.57], [0.42, -0.58], [0.44, -0.71], [0.55, -0.83], [0.54, -0.95], [0.43, -0.95], [0.3, -0.69]]
            my_plot(points)

            # Mallows [many]
            points = [[0.42, -0.58], [0.55, -0.45], [0.67, -0.48], [0.76, -0.59], [0.72, -0.78], [0.65, -0.84],
                      [0.55, -0.83]]

            my_plot(points)

            my_arrow(0.56, -1.2, 0.61, -0.92)
            my_arrow(0.56, -1.2, 0.51, -1.05)
            my_arrow(0.56, -1.2, 0.45, -1.07)
            my_arrow(0.56, -1.2, 0.2, -1.05)
            my_text(0.56, -1.2, "Mallows", color="DodgerBlue")

            # Sphere
            points = [[-0.8, 0.42], [-0.64, 0.71], [-0.43, 0.86], [-0.31, 0.90], [-0.17, 0.88], [-0.14, 0.78],
                      [-0.22, 0.53], [-0.42, 0.10], [-0.62, 0.01], [-0.77, 0.06], [-0.82, 0.22], [-0.8, 0.42]]
            my_plot(points)

            my_arrow(-0.63, 0.94, -0.59, 0.85)
            my_line(-0.9, 0.9, -0.63, 0.94)
            my_line(-0.41, 1.1, -0.63, 0.94)
            my_text(-0.9, 0.9, "SPOC", color="darkred", alpha=0.7)
            my_text(-0.41, 1.1, "2/3/5D H-Sphere", color="black", alpha=0.8)

            # Impartial Culture & Mallows 0.999
            points = [[-0.41, 0.75], [-0.32, 0.68], [-0.29, 0.47], [-0.35, 0.34], [-0.50, 0.21], [-0.61, 0.27],
                      [-0.63, 0.41], [-0.41, 0.75]]
            my_plot(points)

            my_arrow(-0.84, 0.37, -0.66, 0.34)
            my_line(-1.05, 0.32, -0.84, 0.37)
            my_line(-1, 0.55, -0.84, 0.37)
            my_text(-1.05, 0.32, "IC", color="black")
            my_text(-1, 0.55, "Mallows", color="darkblue")

            # 2D
            points = [[0.12, 0.27], [0.3, 0.37], [0.41, 0.18], [0.41, -0.03],
                      [0.31, -0.18]]
            my_plot(points)
            my_arrow(1.07, -0.22, 0.47, -0.08)
            my_text(1.07, -0.22, "2D Square", color="green")

            # 3D
            points = [[0.31, -0.18], [0.16, -0.26], [0.08, -0.03], [0, 0.24], [0.12, 0.27], [0.26, 0.04], [0.31, -0.18]]
            my_plot(points)
            my_arrow(1.0, -0.47, 0.37, -0.22)
            my_text(1.0, -0.47, "3D Cube", color="ForestGreen", alpha=0.7)

            # 5D, 10D, 20D
            points = [[0.16, -0.26], [-0.11, -0.32], [-0.38, -0.22], [-0.43, -0.05], [0, 0.24]]
            my_plot(points)

            my_arrow(-1.05, -0.10, -0.49, -0.12)
            my_text(-1.05, -0.10, "5/10/20D H-Cube", color="MediumSeaGreen", alpha=0.9)

            # Mallows 0.95 (next to SC)
            points = [[-0.23, -0.4], [-0.3, -0.5], [-0.58, -0.47], [-0.64, -0.28],  [-0.23, -0.4]]
            my_plot(points)

            ax.arrow(-0.95, -0.44, 0.24, 0.03, head_width=0.02, head_length=0.05, fc='k', ec='k')
            ax.arrow(-0.95, -0.44, 0.59, 0.5, head_width=0.02, head_length=0.05, fc='k', ec='k')
            my_text(-0.95, -0.44, "Mallows", color="darkblue", alpha=0.8)

            # Single-Crossing
            points = [[-0.23, -0.4], [-0.04, -0.37], [0.24, -0.29], [0.39, -0.28], [0.4, -0.33], [0.14, -0.47],
                      [-0.3, -0.65], [-0.52, -0.6], [-0.58, -0.47]]
            my_plot(points)

            my_arrow(-0.80, -0.84, -0.48, -0.67)
            my_text(-0.80, -0.84, "Single Crossing", color="purple", alpha=0.6)

            # Mallows 0.99
            points = [[-0.26, 0.34], [-0.1, 0.48], [-0.05, 0.41], [-0.04, 0.34], [-0.14, 0.2], [-0.26, 0.1],
                      [-0.34, 0.13], [-0.26, 0.34]]
            my_plot(points)

            # [free] URN
            # ax.arrow(-0.67, -1.77, 0.12, 0.12, head_width=0.02, head_length=0.05, fc='k', ec='k')
            my_arrow(0.11, 1.0, -0.07, 0.72)
            my_arrow(0.11, 1.0, 0.16, 0.64)
            my_text(0.11, 1.0, "Urn", color="Orange")
            my_arrow(1., 0.2, 0.73, 0)
            my_text(1., 0.2, "Urn", color="OrangeRed")

            # Mallows parameters
            my_number(0.78, -0.85, "0.001", color="darkblue")
            my_number(0.82, -0.74, "0.01", color="darkblue")
            my_number(0.84, -0.63, "0.05", color="darkblue")
            my_number(0.55, -0.39, "0.1", color="darkblue")
            my_number(0.35, -0.5, "0.25", color="darkblue")
            my_number(0.23, -0.57, "0.5", color="darkblue")
            my_number(0.08, -0.58, "0.75", color="darkblue")
            my_number(-0.65, -0.52, "0.95", color="darkblue")
            my_number(-0.91, -0.31, "0.99", color="darkblue")
            my_number(-0.94, 0.67, "0.999", color="darkblue")

            # Urn parameters
            my_number(-0.08, 0.88, "0.01", color="orangered")
            my_number(0.07, 0.81, "0.02", color="orangered")
            my_number(0.2, 0.87, "0.05", color="orangered")
            my_number(0.26, 0.75, "0.1", color="orangered")
            my_number(0.77, 0.13, "0.2", color="orangered")
            my_number(0.82, -0.03, "0.5", color="orangered")
