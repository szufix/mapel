
""" This module contains all the functions that user can use """

### COMMENT ON SERVER ###
import matplotlib.pyplot as plt


from . import objects as obj
import numpy as np
import os
from PIL import Image
from shutil import copyfile
import scipy.stats as stats

#import voting.elections as el
#import voting.metrics as metr

from . import metrics as metr
from . import elections as el
#  mark the orders
#  for w in model.orders:
#    ax.scatter(model.points[w][0], model.points[w][1], color="red", s=50, marker='x')

# HELPER FUNCTIONS FOR PRINT_2D
def get_values_from_file(model, experiment_id, values, normalizing_func=None, marker_func=None):
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                        str(values) + ".txt")

    _min = 0
    _max = 0
    values = []
    with open(path, 'r') as txtfile:
        for _ in range(model.num_elections):
            values.append(float(txtfile.readline()))
    _min = min(values)
    _max = max(values)
    print(_min, _max)

    with open(path, 'r') as txtfile:

        shades = []
        xx = []
        yy = []
        markers = []

        ctr = 0
        for k in range(model.num_families):
            for _ in range(model.families[k].size):

                shade = float(txtfile.readline())
                if normalizing_func is not None:
                    shade = normalizing_func(shade)
                else:
                    shade = (shade-_min) / (_max-_min)
                shades.append(shade)

                marker = model.families[k].marker
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
        return xx, yy, shades, markers, _min, _max


def get_values_from_file_3d(model, experiment_id, values, normalizing_func):
    path = os.path.join(os.getcwd(), "experiments", experiment_id, "controllers", "advanced",
                        str(values) + ".txt")

    _min = 0
    _max = 0
    values = []
    with open(path, 'r') as txtfile:
        for _ in range(model.num_elections):
            values.append(float(txtfile.readline()))
    _min = min(values)
    _max = max(values)
    print(_min, _max)

    with open(path, 'r') as txtfile:

        shades = []
        xx = []
        yy = []
        zz = []
        markers = []

        ctr = 0
        for k in range(model.num_families):
            for _ in range(model.families[k].size):

                shade = float(txtfile.readline())
                if normalizing_func is not None:
                    shade = normalizing_func(shade)
                else:
                    shade = (shade-_min) / (_max-_min)
                shades.append(shade)

                marker = model.families[k].marker
                markers.append(marker)

                xx.append(model.points[model.main_order[ctr]][0])
                yy.append(model.points[model.main_order[ctr]][1])
                zz.append(model.points[model.main_order[ctr]][2])

                ctr += 1
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        zz = np.asarray(zz)
        shades = np.asarray(shades)
        markers = np.asarray(markers)
        return xx, yy, zz, shades, markers, _min, _max


def add_advanced_points_to_picture(fig=None, ax=None, model=None, experiment_id=None, values=None, normalizing_func=None,
                                   marker_func=None, xticklabels=None, ms=None, cmap=None, ticks=None):
    xx, yy, shades, markers, _min, _max = get_values_from_file(model, experiment_id, values, normalizing_func, marker_func)
    unique_markers = set(markers)
    images = []

    for um in unique_markers:
        masks = (markers == um)
        images.append(
            [ax.scatter(xx[masks], yy[masks], c=shades[masks], vmin=0, vmax=1, cmap=cmap, marker=um, s=ms)])

    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("bottom", size="5%", pad="5%")

    if xticklabels is None:
        lin = np.linspace(_min, _max, 6)
        xticklabels = [np.round(lin[i]) for i in range(6)]

    cb = fig.colorbar(images[0][0], cax=cax, orientation="horizontal",
                      shrink=1, ticks=ticks)  # shrink not working

    cb.ax.locator_params(nbins=len(xticklabels), tight=True)
    cb.ax.tick_params(labelsize=16)
    if xticklabels is not None:
        cb.ax.set_xticklabels(xticklabels)


def add_advanced_points_to_picture_3d(fig, ax, model, experiment_id,
                                      values=None, cmap=None, ms=None,
                                      normalizing_func=None):
    xx, yy, zz, shades, markers, _min, _max = get_values_from_file_3d(model, experiment_id, values, normalizing_func)
    unique_markers = set(markers)
    images = []

    for um in unique_markers:
        masks = (markers == um)
        images.append(
            [ax.scatter(xx[masks], yy[masks], zz[masks], c=shades[masks], vmin=0, vmax=1, cmap=cmap, marker=um, s=ms)])


def add_basic_points_to_picture(ax=None, model=None, ms=None):
    # TEMPORARY VERSION
    for k in range(model.num_families):
        if model.families[k].show:
            ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                       color=model.families[k].color, label=model.families[k].label,
                       alpha=model.families[k].alpha, s=ms, marker=model.families[k].marker)

LIST_OF_PREFLIB_ELECTIONS = {'sushi', 'irish', 'glasgow', 'skate', 'formula',
                             'tshirt', 'cities_survey', 'aspen', 'ers',
                             'marble', 'cycling_tdf', 'cycling_gdi', 'ice_races',
                             'grenoble'}


def add_tmp_points_to_picture(ax=None, model=None, ms=None, tmp=None, tmp2=None, zorder=None):
    # TEMPORARY VERSION
    for k in range(model.num_families):
        if model.families[k].show:

            if model.families[k].election_model in {'identity', 'uniformity', 'antagonism', 'chess',
                                          'unid', 'anid', 'stid', 'anun', 'stun', 'stan'}:
                ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1], zorder=zorder[0],
                           color=model.families[k].color, label=model.families[k].label,
                           alpha=tmp[0] * model.families[k].alpha, s=ms*tmp2[0], marker=model.families[k].marker)

            elif model.families[k].election_model in LIST_OF_PREFLIB_ELECTIONS:
                ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1], zorder=zorder[1],
                           color=model.families[k].color, label=model.families[k].label,
                           alpha=tmp[1] * model.families[k].alpha, s=ms*tmp2[1], marker=model.families[k].marker)
            else:
                ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                           color=model.families[k].color, label=model.families[k].label, zorder=zorder[2],
                           alpha=tmp[2] * model.families[k].alpha, s=ms*tmp2[2], marker=model.families[k].marker)


def add_mask_to_picture(fig=None, ax=None, black=None, saveas=None, tex=None):
    fig.set_size_inches(10, 10)
    corners = [[-1, 1, 1, -1], [1, 1, -1, -1]]
    ax.scatter(corners[0], corners[1], alpha=0)

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


def add_levels_to_picture(fig=None, ax=None, saveas=None, tex=None):
    fig.set_size_inches(10, 10)

    # corners = [[-1, 1, 1, -1], [1, 1, -1, -1]]
    # ax.scatter(corners[0], corners[1], alpha=0)

    def my_line(x1, y1, x2, y2):
        ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0., head_length=0., fc='k', ec='k')

    def my_plot(points, color='black', type='--'):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        from scipy.interpolate import interp1d

        t = np.arange(len(x))
        ti = np.linspace(0, t.max(), 10 * t.size)

        xi = interp1d(t, x, kind='cubic')(ti)
        yi = interp1d(t, y, kind='cubic')(ti)

        ax.plot(xi, yi, type, color=color)

    """
    points = [[-0.97, 0.95], [-0.37, 0.97], [0.11, 1.33], [0.27, 1.75]]
    my_plot(points)

    points = [[-0.58, 0.22], [0, 0.45], [0.38, 0.87], [0.57, 1.6]]
    my_plot(points)

    points = [[-0.1, -0.02], [0.3, 0.42], [0.57, 0.88], [0.69, 1.42]]
    my_plot(points)

    points = [[0.21, -0.07], [0.46, 0.36], [0.69, 0.86], [0.84, 1.33]]
    my_plot(points)
    """

    """
    # FROM ID
    points = [[0.74, -0.18], [0.92, 0.33], [0.97, 1.06], [0.93, 1.63]]
    my_plot(points)
    points = [[1.55, -0.5], [1.61, 0.15], [1.55, 0.77], [1.38, 1.45]]
    my_plot(points)
    points = [[2.04, -0.11], [1.98, 0.38], [1.92, 0.78], [1.84, 1.2]]
    my_plot(points)
    """

    # FROM UN
    my_plot([[0.55, 0.01], [0.76, 0.48], [0.76, 0.87], [0.5, 1.16]], color='darkblue')
    my_plot([[0.96, -0.28], [1.5, 0.45], [1.32, 1.15], [1.06, 1.54]], color='darkblue')
    my_plot([[1.64, -0.44], [1.8, 0.08], [1.9, 0.67], [1.85, 1.2]], color='darkblue')

    # FROM ST
    my_plot([[1.06, -0.36], [1.36, 0.06], [1.76, 0.23], [2.19, 0.02]], color='darkred')
    my_plot([[0.58, -0.02], [1.05, 0.87], [1.63, 1.03], [2.32, 0.81]], color='darkred')
    my_plot([[0.11, 0.57], [0.52, 0.96], [0.91, 1.3], [1.22, 1.45]], color='darkred')


    # plt.legend(bbox_to_anchor=(1.25, 1.))
    file_name = saveas + ".png"
    path = os.path.join(os.getcwd(), "images", file_name)
    plt.savefig(path, bbox_inches='tight')
    plt.grid(True)

    # plt.xlim(-1.4, 1.3)
    # plt.ylim(-1.4, 1.3)

    if tex:
        import tikzplotlib
        file_name = saveas + ".tex"
        path = os.path.join(os.getcwd(), "images", "tex", file_name)
        tikzplotlib.save(path)

def add_guardians_to_picture(experiment_id, ax=None, values=None, legend=None, saveas=None):

    #fig.set_size_inches(10, 10)
    #corners = [[-1, 1, 1, -1], [1, 1, -1, -1]]
    #ax.scatter(corners[0], corners[1], alpha=0)
    def my_text(x1, y1, text, color="black", alpha=1., size=12):

        plt.text(x1, y1, text, size=size, rotation=0., ha="center", va="center",
                 color=color, alpha=alpha,
                 bbox=dict(boxstyle="round", ec="black", fc="white"))

    if experiment_id == 'ijcai/original+paths':
        my_text(-0.48, 0.30, "UN")
        my_text(0.5, 1.25, "AN")
        my_text(1.8, 0.28, "ID")
        my_text(0.9, -0.83, "ST")
    elif experiment_id == 'ijcai/paths+urn+phi_mallows+preflib':
        my_text(0.03, 0.18, "UN")
        my_text(0.72, 0.94, "AN")
        my_text(2.0, 0.17, "ID")
        my_text(1.3, -0.74, "ST")

    elif experiment_id == 'ijcai/paths+mallows':
        my_text(-0.89, 1.29, "UN", size=24)
        my_text(-0.1, 2.06, "AN", size=24)
        my_text(1.04, 1.29, "ID", size=24)
        my_text(0.32, 0.28, "ST", size=24)
    elif experiment_id == 'ijcai/paths+phi_mallows':
        my_text(-1.18, -0.02, "UN", size=24)
        my_text(-0.37, 0.82, "AN", size=24)
        my_text(0.9, -0.03, "ID", size=24)
        my_text(0.12, -1.06, "ST", size=24)
    elif experiment_id == 'ijcai/paths+urn':
        my_text(-0.03, 0.02, "UN", size=24)
        my_text(0.73, 0.88, "AN", size=24)
        my_text(2.02, 0.03, "ID", size=24)
        my_text(1.28, -0.89, "ST", size=24)

    file_name = os.path.join(os.getcwd(), "images", str(saveas))
    if values is None and legend == True:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.savefig(file_name, bbox_inches='tight')


def add_basic_background_to_picture(ax=None, values=None, legend=None, saveas=None):
    file_name = os.path.join(os.getcwd(), "images", str(saveas))

    if values is None and legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.savefig(file_name, bbox_inches='tight')
    """
    text_name = str(model.num_voters) + " x " + str(model.num_candidates)
    text = ax.text(0.0, 1.05, text_name, transform=ax.transAxes)
    file_name = os.path.join(os.getcwd(), "images", str(saveas))
    if values is None and legend == True:
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(file_name, bbox_extra_artists=(lgd, text), bbox_inches='tight')
    else:
        plt.savefig(file_name, bbox_inches='tight')
    """


def saveas_tex(saveas=None):
    import tikzplotlib
    file_name = saveas + ".tex"
    path = os.path.join(os.getcwd(), "images", "tex", file_name)
    tikzplotlib.save(path)


# MAIN FUNCTIONS
def print_2d(experiment_id, num_winners=0, mask=False,
             angle=0, reverse=False, update=False, values=None, magic=1,
             num_elections=None, secondary_order_name="positionwise_approx_cc", main_order_name="default",
             metric="positionwise", guardians=False, tmp2=[1, 1, 1], zorder=[1, 1, 1], ticks=None,
             saveas="map_2d", show=True, ms=20, normalizing_func=None, xticklabels=None, cmap=None,
             ignore=None, marker_func=None, tex=False, black=False, legend=True, levels=False, tmp=False):
    """ Print the two-dimensional embedding of multi-dimensional map of the elections """


    model = obj.Model_2d(experiment_id, num_elections=num_elections, main_order_name=main_order_name, distance_name=metric,
                         ignore=ignore, attraction_factor=magic)

    if angle != 0:
        model.rotate(angle)

    if reverse:
        model.reverse()

    if update:
        model.update()

    if cmap is None:
        cmap = custom_div_cmap()

    if values is not None:
        fig = plt.figure(figsize=(6.4, 4.8+0.48))
    else:
        fig = plt.figure()
    #import matplotlib.gridspec as gridspec
    #gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1])
    ax = fig.add_subplot()
    plt.axis('off')

    if values is not None:
        add_advanced_points_to_picture(fig=fig, ax=ax, model=model, experiment_id=experiment_id, values=values,
                                       normalizing_func=normalizing_func, marker_func=marker_func,
                                       xticklabels=xticklabels, ms=ms, cmap=cmap, ticks=ticks)
    elif tmp:
        add_tmp_points_to_picture(ax=ax, model=model, ms=ms, tmp=tmp, tmp2=tmp2, zorder=zorder)
    else:
        add_basic_points_to_picture(ax=ax, model=model, ms=ms)

    if mask:
        add_mask_to_picture(fig=fig, ax=ax, black=black, saveas=saveas, tex=tex)
    elif levels:
        add_levels_to_picture(fig=fig, ax=ax, saveas=saveas, tex=tex)
    else:
        add_basic_background_to_picture(ax=ax, values=values, legend=legend, saveas=saveas)

    if guardians:
        add_guardians_to_picture(experiment_id, ax=ax, values=values, legend=legend, saveas=saveas)

    if tex:
        saveas_tex(saveas=saveas)

    if show:
        plt.show()


def print_3d(experiment_id, ms=20, magic=1, ignore=None,
             angle=0, reverse=False, update=False, values=None, coloring="purple",
             num_elections=None, main_order_name="default", order="", metric="positionwise",
             saveas="map_3d", show=True, dot=9, normalizing_func=None, xticklabels=None, cmap=None):
    """ Print the two-dimensional embedding of multi-dimensional map of the elections """

    model = obj.Model_3d(experiment_id, num_elections=num_elections, main_order_name=main_order_name, distance_name=metric,
                         ignore=ignore, attraction_factor=magic)
    if angle != 0:
        model.rotate(angle)

    if reverse:
        model.reverse()

    if update:
        model.update()

    if cmap is None:
        cmap = custom_div_cmap()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.axis('off')

    if values is not None:
        add_advanced_points_to_picture_3d(fig, ax, model, experiment_id,
                                          values=values, ms=ms, cmap=cmap,
                                          normalizing_func=normalizing_func)
    else:
        for k in range(model.num_families):
            if model.families[k].show:
                ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1], model.points_by_families[k][2],
                           color=model.families[k].color, label=model.families[k].label,
                           alpha=model.families[k].alpha, s=dot)

    text_name = str(model.num_voters) + " x " + str(model.num_candidates)
    # text = ax.text(0.0, 1.05, text_name, transform=ax.transAxes)
    file_name = os.path.join(os.getcwd(), "images", str(saveas) + ".png")

    if values == "default":
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.savefig(file_name, bbox_inches='tight')

    if show:
        plt.show()


def print_matrix(experiment_id, scale=1., metric="positionwise", saveas="matrix", show=True):
    """Print the matrix with average distances between each pair of models """

    model = obj.Model_xd(experiment_id, distance_name=metric)
    matrix = np.zeros([model.num_families, model.num_families])
    quantities = np.zeros([model.num_families, model.num_families])

    mapping = [i for i in range(model.num_families) for _ in range(model.families[i].size)]

    for i in range(model.num_elections):
        for j in range(i + 1, model.num_elections):
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
        line = str(file_.readline().replace("\n", "").replace(" ", "").lower())
        for j in range(model.num_families):
            if model.families[j].label.replace(" ", "").lower() == line:
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

    #print(labels_new)

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

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "controllers", "orders",
                             str(metric) + "_approx_cc.txt")
    file_ = open(file_name, 'r')

    file_.readline()  # skip this line
    num_elections = int(file_.readline())
    file_.readline()  # skip this line

    for i in range(num_elections):
        target = str(file_.readline().replace("\n", ""))

        src = os.path.join(os.getcwd(), "experiments", str(experiment_id), "elections", "soc_original",
                           "core_" + str(target) + ".soc")

        dst = os.path.join(os.getcwd(), "experiments", str(experiment_id), "elections",
                           "soc_" + str(metric) + "_approx_cc",
                           "core_" + str(i) + ".soc")

        copyfile(src, dst)

def print_param_vs_distance(experiment_id, values="", scale="none", metric="positionwise", saveas="correlation",
                            show=True, target='identity', ylabel_text=''):

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "controllers", "distances",
                             str(metric) + ".txt")
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    file_.readline()
    file_.readline()

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "controllers", "advanced",
                             str(values) + ".txt")
    file_2 = open(file_name, 'r')
    times = [float(file_2.readline()) for _ in range(num_elections)]

    distance_name = 'positionwise'
    metric_name = 'emd'
    x = 0
    election_2_id = 'guess'

    model = obj.Model(experiment_id)

    distances = [0. for _ in range(num_elections)]

    if target == 'uniformity':
        tag = 'un'
        xlabel_text = "average distance from UN elections"
        election_model = 'uniformity'

    elif target == 'antagonism':
        tag = 'an'
        xlabel_text = "average distance from AN elections"
        election_model = 'antagonism'

    elif target == 'stratification':
        tag = 'st'
        xlabel_text = "average distance from ST elections"
        election_model = 'stratification'

    else:  #target == 'identity':
        tag = 'id'
        xlabel_text = "average distance from ID elections"
        election_model = 'identity'

    el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                          num_voters=model.num_voters, num_candidates=model.num_candidates,
                          special=x)
    election_2 = obj.Election(experiment_id, election_2_id)

    for i in range(model.num_elections):
        election_1_id = 'core_' + str(i)
        election_1 = obj.Election(experiment_id, election_1_id)
        distances[i] = metr.get_distance(election_1, election_2, distance_name=distance_name, metric_name=metric_name)

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
                   alpha=model.families[k].alpha, s=9, marker=model.families[k].marker)
        left = right

    title_text = str(model.num_voters) + " voters  x  " + str(model.num_candidates) + " candidates"
    pear_text = "PCC = " + str(pear)
    add_text = ax.text(0.7, 0.8, pear_text, transform=ax.transAxes)
    plt.title(title_text)
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    file_name = os.path.join(os.getcwd(), "images", "correlation", str(saveas) + '_' + tag + ".png")
    # lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()


def excel_super(experiment_id, values="", scale="none", metric="positionwise", saveas="correlation",
                            show=True, target='identity', ylabel_text=''):
    num_elections = 1000

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "controllers", "advanced",
                             str(values) + ".txt")

    file_2 = open(file_name, 'r')
    times = [float(file_2.readline()) for _ in range(num_elections)]

    distances = [0. for _ in range(num_elections)]
    if target == 'uniformity':
        tag = 'un'
        xlabel_text = "Distance from UN elections"

        model = obj.Model(experiment_id)
        x = 0
        election_2_id = 'guess'
        election_model = 'uniformity'
        distance_name = 'positionwise'
        metric_type = 'emd'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)

        for i in range(model.num_elections):

            election_1_id = 'core_' + str(i)
            elections_ids = [election_1_id, election_2_id]
            distances[i] = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)

    elif target == 'identity':
        tag = 'id'
        xlabel_text = "average distance from ID elections"

        model = obj.Model(experiment_id)
        x = 0
        election_2_id = 'guess'
        election_model = 'identity'
        distance_name = 'positionwise'
        metric_type = 'emd'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        #election_2 = obj.Election(experiment_id, election_2_id)

        for i in range(model.num_elections):
            election_1_id = 'core_' + str(i)
            #election_1 = obj.Election(experiment_id, election_1_id)
            elections_ids = [election_1_id, election_2_id]
            distances[i] = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)

    plt.hist(distances, bins=12)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pear = stats.pearsonr(times, distances)
    pear = round(pear[0], 2)
    model = obj.Model(experiment_id)

    left = 0
    for k in range(model.num_families):
        right = left + model.families[k].size
        ax.scatter(distances[left:right], times[left:right],
                   color=model.families[k].color, label=model.families[k].label,
                   alpha=model.families[k].alpha, s=9, marker=model.families[k].marker)
        left = right

    title_text = str(model.num_voters) + " voters  x  " + str(model.num_candidates) + " candidates"
    pear_text = "PCC = " + str(pear)
    add_text = ax.text(0.7, 0.8, pear_text, transform=ax.transAxes)
    plt.title(title_text)
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    file_name = os.path.join(os.getcwd(), "images", "correlation", str(saveas) + '_' + tag + ".png")
    # lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()

def excel_super_2(experiment_id, values="", scale="none", metric="positionwise", saveas="correlation",
                            show=True, target='identity', ylabel_text=''):
    num_elections = 1000

    file_name = os.path.join(os.getcwd(), "experiments", str(experiment_id), "controllers", "advanced",
                             str(values) + ".txt")

    file_2 = open(file_name, 'r')

    distances = [0. for _ in range(num_elections)]

    if target == 'identity':
        tag = 'id'
        xlabel_text = "average distance from ID elections"

        model = obj.Model(experiment_id)
        x = 0
        election_2_id = 'guess'
        election_model = 'identity'
        distance_name = 'positionwise'
        metric_type = 'emd'
        el.generate_elections(experiment_id, election_model=election_model, election_id=election_2_id,
                              num_voters=model.num_voters, num_candidates=model.num_candidates,
                              special=x)
        #election_2 = obj.Election(experiment_id, election_2_id)

        for i in range(model.num_elections):
            election_1_id = 'core_' + str(i)
            #election_1 = obj.Election(experiment_id, election_1_id)
            elections_ids = [election_1_id, election_2_id]
            distance = metr.get_distance(experiment_id, distance_name, elections_ids, metric_type)
            #print(distance)
            distances[i] = distance


    plt.hist(distances, bins=12)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pear = stats.pearsonr(times, distances)
    pear = round(pear[0], 2)
    model = obj.Model(experiment_id)

    left = 0
    for k in range(model.num_families):
        right = left + model.families[k].size
        ax.scatter(distances[left:right], times[left:right],
                   color=model.families[k].color, label=model.families[k].label,
                   alpha=model.families[k].alpha, s=9, marker=model.families[k].marker)
        left = right

    title_text = str(model.num_voters) + " voters  x  " + str(model.num_candidates) + " candidates"
    pear_text = "PCC = " + str(pear)
    add_text = ax.text(0.7, 0.8, pear_text, transform=ax.transAxes)
    plt.title(title_text)
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    file_name = os.path.join(os.getcwd(), "images", "correlation", str(saveas) + '_' + tag + ".png")
    # lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()

# HELPER FUNCTIONS
def custom_div_cmap(num_colors=101, name='custom_div_cmap',
                    colors=None):
    if colors is None:
        colors = ["lightgreen", "yellow", "orange", "red", "black"]

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name=name, colors=colors, N=num_colors)
    return cmap


def temporary_shade(shade):
    if shade == 1.:
        return 0.2, "purple", "x"
    elif shade >= 0.91:
        shade = 1 - (shade - 0.9) * 10
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
        ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.02, head_length=0.05, fc='k', ec='k')

    def my_line(x1, y1, x2, y2):
        ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0., head_length=0., fc='k', ec='k')

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
    points = [[0.31, 0.74], [0.37, 0.8], [0.53, 0.66], [0.69, 0.43], [0.71, 0.26], [0.68, 0.12], [0.59, 0.02],
              [0.49, -0.04], [0.45, 0.04], [0.44, 0.3], [0.31, 0.74]]
    my_plot(points)
    # ax.add_line(plt.Polygon(points, fill=None, edgecolor='black', lw=2, joinstyle="round", capstyle="round"))
    my_arrow(0.64, 0.71, 0.61, 0.66)
    my_line(0.75, 0.90, 0.64, 0.71)
    my_line(0.95, 0.65, 0.64, 0.71)
    my_text(0.75, 0.90, "1D Interval", color="darkgreen")
    my_text(0.95, 0.65, "SP (Con.)", color="limegreen")

    # Walsh SP
    # points = [[0.04, -0.64], [0, -0.89], [-0.07, -1.02], [-0.16, -1], [-0.28, -0.93], [-0.34, -0.79], [-0.22, -0.69], [0.04, -0.64]]
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
    points = [[0.3, -0.69], [0.35, -0.57], [0.42, -0.58], [0.44, -0.71], [0.55, -0.83], [0.54, -0.95], [0.43, -0.95],
              [0.3, -0.69]]
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
    points = [[-0.23, -0.4], [-0.3, -0.5], [-0.58, -0.47], [-0.64, -0.28], [-0.23, -0.4]]
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



def print_clustering(experiment_id, magic=1, q=0):
    #"""
    model = obj.Model_xd(experiment_id)

    guardians = ['identity', 'uniformity', 'antagonism', 'chess']
    distances = {}

    for guardian in guardians:

        path = os.path.join("experiments", experiment_id, "controllers", "distances", guardian + ".txt")
        file_ = open(path, 'r')

        distances[guardian] = []
        for i in range(model.num_elections):
            distance = float(file_.readline().strip())
            distances[guardian].append(distance)

    path = os.path.join("experiments", str(experiment_id), "controllers", "advanced", 'guardians.txt')
    file_ = open(path, 'w')

    for i in range(model.num_elections):
        order = [0,1,2,3]
        values = [distances['identity'][i], distances['uniformity'][i], distances['antagonism'][i], distances['chess'][i]]
        order = [x for _, x in sorted(zip(values, order))]
        file_.write(str(order[q]) + '\n')
        #file_.write(str(int(order[0]+order[1])) + '\n')
    #"""
    file_.close()

    def normalizing_func(shade):
        print(shade)
        shade /= 3
        return shade

    def reversed_func(arg):
        arg *= 25
        return arg

    #[print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = custom_div_cmap(colors=["blue", "black", "red", "green"], num_colors=4)
    xticklabels = list(['                   identity',
                        '                   uniformity',
                        '                   antagonism',
                        '                   stratification', ''])
    print_2d(experiment_id, saveas="distances/map_clustering_" + str(q), ms=16, values="guardians", magic=magic,
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)


def print_clustering_bis(experiment_id, magic=1, q=0):
    #"""
    model = obj.Model_xd(experiment_id)

    guardians = ['identity', 'uniformity', 'antagonism', 'chess']
    distances = {}

    for guardian in guardians:

        path = os.path.join("experiments", experiment_id, "controllers", "distances", guardian + ".txt")
        file_ = open(path, 'r')

        distances[guardian] = []
        for i in range(model.num_elections):
            distance = float(file_.readline().strip())
            distances[guardian].append(distance)

    path = os.path.join("experiments", str(experiment_id), "controllers", "advanced", 'guardians.txt')
    file_ = open(path, 'w')

    for i in range(model.num_elections):
        order = [1000, 100, 10, 1]
        values = [distances['identity'][i], distances['uniformity'][i], distances['antagonism'][i], distances['chess'][i]]
        order = [x for _, x in sorted(zip(values, order))]
        #file_.write(str(order[q]) + '\n')
        file_.write(str(int(order[0]+order[1])) + '\n')
        values = sorted(values)
        print(values[0], values[1])

    #"""
    file_.close()

    def normalizing_func(shade):
        shade = int(shade)
        if shade == 11: # CH-AN
            shade = 0
        elif shade == 101:   # CH-UN
            shade = 1/5
        elif shade == 1001: # CH-ID
            shade = 2/5
        elif shade == 1010: # AN-ID
            shade = 3/5
        elif shade == 1100: # UN-ID
            shade = 4/5
        elif shade == 110: # AN-UN
            shade = 1
        return shade

    def reversed_func(arg):
        arg *= 25
        return arg

    #[print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = custom_div_cmap(colors=["black", "green", "red", "orange", "purple", "blue"], num_colors=6)
    xticklabels = list(['CH-AN',
                        'CH-UN',
                        'CH-ID',
                        'AN-ID',
                        'UN-ID',
                        'AN-UN'])
    print_2d(experiment_id, saveas="distances/map_clustering_" + str(q), ms=16, values="guardians", magic=magic,
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)



def print_clustering_bis_3d(experiment_id, magic=1, q=0):
    #"""
    model = obj.Model_xd(experiment_id)

    def normalizing_func(shade):
        shade = int(shade)
        if shade == 11: # CH-AN
            shade = 0
        elif shade == 101:   # CH-UN
            shade = 1/5
        elif shade == 1001: # CH-ID
            shade = 2/5
        elif shade == 1010: # AN-ID
            shade = 3/5
        elif shade == 1100: # UN-ID
            shade = 4/5
        elif shade == 110: # AN-UN
            shade = 1
        return shade

    def reversed_func(arg):
        arg *= 25
        return arg

    #[print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = custom_div_cmap(colors=["black", "green", "red", "orange", "purple", "blue"], num_colors=6)

    print_3d(experiment_id, saveas="distances/_miodzio" + str(q), ms=16, values="guardians",
                normalizing_func=normalizing_func, cmap=custom_map)


