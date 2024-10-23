import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import tikzplotlib
except ImportError:
    tikzplotlib = None

from mapel.core.glossary import *
import mapel.core.persistence.experiment_imports as imports


def _get_main_mask(mask):
    if mask == '10x50-fr-swap-color':
        figsize = (6.4, 6.4)
        pad_inches = 1
    elif mask == '10x50-kk-swap-color':
        figsize = (6.4, 6.4)
        pad_inches = 0.8
    elif mask == '100x100-fr-emd-positionwise-color':
        figsize = (6.4, 6.4)
        pad_inches = 1
    elif mask == '100x100-kk-emd-positionwise-color':
        figsize = (6.9, 6.9)
        pad_inches = 1.4
    else:
        logging.warning("Incorrect mask name")

    return figsize, pad_inches


def _get_feature_mask(mask):
    if mask == '10x50-fr-swap-black':
        figsize = (7.7, 7.7)
        pad_inches = 0.4
    elif mask == '10x50-kk-swap-black':
        figsize = (7.2, 7.2)
        pad_inches = 0.5
    elif mask == '100x100-fr-emd-positionwise-black':
        figsize = (7.5, 7.5)
        pad_inches = 0.6
    elif mask == '100x100-kk-emd-positionwise-black':
        figsize = (7.6, 7.6)
        pad_inches = 1.4
    else:
        logging.warning("Incorrect mask name")

    return figsize, pad_inches


def print_map_2d(
        experiment,
        return_map: bool = False,
        xlabel=None,
        shading: bool = False,
        legend_pos=None,
        title_pos=None,
        angle=0,
        reverse: bool = False,
        update: bool = False,
        axis: bool = False,
        individual: bool = False,
        textual: list[str] = None,
        textual_size: int = 16,
        title: str = None,
        bbox_inches='tight',
        saveas: str = None,
        show: bool = True,
        urn_orangered: bool = False,
        tex: bool = False,
        legend: bool = True,
        dpi: int = 250,
        title_size: int = 16,
        figsize=(6.4, 6.4),
        pad_inches=None,
        mask: str = None
):
    """
    Print map of instances on a 2d plane.

    Parameters
    ----------
        experiment : Experiment
            Experiment object.
        return_map : bool
            If True then return: fig, ax
        xlabel
            Pyplot param.
        shading : bool
            If True that use a culture param to adjust the alpha of the point on the map.
        legend_pos
            Pyplot param.
        title_pos
            Pyplot param.
        angle
            Rotate the map by a given angle.
        reverse : bool
            If True then reverse the map.
        update : bool
            If True then the points coordinates are exported.
        axis : bool
            Deprecated param.
        individual : bool
            Deprecated param.
        textual : list[str]
            Print thext over the points with names from the list.
        textual_size
            Size of the 'textual' text.
        title : str
            Pyplot param.
        bbox_inches
            Pyplot param.
        saveas
            Name of file in which the image will be stored.
        show : bool
            If true show the map on screen.
        urn_orangered : bool
            Deprecated param.
        tex : bool
            If True save image in tex format.
        legend
            Pyplot param.
        dpi : int
            Quality of the image.
        title_size : int
            Size of the title.
        figsize
            Size of the figure.
        pad_inches
            Pyplot param.
        mask : str
            Name of the mask to be put on top of the image.

    """

    if textual is None:
        textual = []

    experiment.compute_coordinates_by_families()

    if angle != 0:
        experiment.rotate(angle)

    if reverse:
        experiment.reverse()

    if experiment.is_exported and update:
        experiment.update()

    if mask is not None:
        show = False
        figsize, pad_inches = _get_main_mask(mask)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot()

    plt.axis('equal')

    if not axis:
        plt.axis('off')

    _add_textual(experiment=experiment, textual=textual, ax=ax, size=textual_size)

    if shading:
        basic_coloring_with_shading(experiment=experiment, ax=ax, urn_orangered=urn_orangered)
    elif individual:
        basic_coloring_with_individual(experiment=experiment, ax=ax, individual=individual)
    else:
        basic_coloring(experiment=experiment, ax=ax, dim=2, textual=textual)

    _basic_background(ax=ax,
                      legend=legend,
                      pad_inches=pad_inches,
                      saveas=saveas,
                      xlabel=xlabel,
                      bbox_inches=bbox_inches,
                      title=title,
                      legend_pos=legend_pos,
                      title_size=title_size,
                      title_pos=title_pos,
                      dpi=dpi,
                      mask=mask)

    if return_map:
        return fig, ax
    else:
        if tex:
            _saveas_tex(saveas=saveas)
        if show:
            plt.show()
        plt.close()


def get_map_2d(experiment, **kwargs):
    return print_map_2d(experiment, return_map=True, **kwargs)


def print_map_2d_colored_by_feature(
        experiment,
        return_map=False,
        feature_id=None,
        column_id='value',
        xlabel=None,
        legend_pos=None,
        title_pos=None,
        axis: bool = False,
        rounding: int = 1,
        upper_limit: float = np.infty,
        lower_limit: float = -np.infty,
        ticks=None,
        textual: list[str] = None,
        scale='default',
        title=None,
        bbox_inches='tight',
        saveas=None,
        show: bool = True,
        normalizing_func=None,
        xticklabels=None,
        cmap=None,
        marker_func=None,
        tex: bool = False,
        feature_labelsize: int = 14,
        dpi: int = 250,
        title_size: int = 16,
        ticks_pos=None,
        omit: list = None,
        textual_size: int = 16,
        figsize=(6.4, 6.4),
        strech=None,
        colors=None,
        pad_inches=None,
        mask: str = None
):
    if textual is None:
        textual = []

    if omit is None:
        omit = []

    if mask is not None:
        show = False
        figsize, pad_inches = _get_feature_mask(mask)

    experiment.compute_coordinates_by_families()

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot()

    plt.axis('equal')

    if not axis:
        plt.axis('off')

    shades_dict, cmap = _color_map_by_feature(experiment=experiment,
                                              fig=fig,
                                              ax=ax,
                                              feature_id=feature_id,
                                              rounding=rounding,
                                              normalizing_func=normalizing_func,
                                              ticks_pos=ticks_pos,
                                              marker_func=marker_func,
                                              upper_limit=upper_limit,
                                              lower_limit=lower_limit,
                                              scale=scale,
                                              xticklabels=xticklabels,
                                              cmap=cmap,
                                              omit=omit,
                                              ticks=ticks,
                                              column_id=column_id,
                                              feature_labelsize=feature_labelsize,
                                              strech=strech,
                                              colors=colors)
    _add_textual(experiment=experiment,
                 textual=textual,
                 ax=ax,
                 size=textual_size,
                 shades_dict=shades_dict,
                 cmap=cmap,
                 column_id=column_id,
                 feature_id=feature_id)

    # BACKGROUND
    _basic_background(ax=ax,
                      legend=False,
                      mask=mask,
                      saveas=saveas,
                      xlabel=xlabel,
                      bbox_inches=bbox_inches,
                      title=title,
                      legend_pos=legend_pos,
                      title_size=title_size,
                      title_pos=title_pos,
                      dpi=dpi,
                      pad_inches=pad_inches)

    if return_map:
        return fig, ax
    else:
        if tex:
            _saveas_tex(saveas=saveas)
        if show:
            plt.show()


def get_map_map_2d_colored_by_feature(experiment, **kwargs):
    return print_map_2d_colored_by_feature(experiment, return_map=True, **kwargs)


def print_map_2d_colored_by_features(
        experiment,
        feature_ids=None,
        xlabel=None,
        legend_pos=None,
        title_pos=None,
        axis=False,
        rounding=1,
        upper_limit=np.infty,
        ticks=None,
        textual=None,
        title=None,
        bbox_inches=None,
        saveas=None,
        show=True,
        ms=20,
        normalizing_func=None,
        xticklabels=None,
        cmap=None,
        marker_func=None,
        tex: bool = False,
        feature_labelsize: int = 14,
        dpi: int = 250,
        column_id: str = 'value',
        title_size: int = 16,
        ticks_pos=None,
        textual_size: int = 16,
        figsize=(20, 10)
) -> None:
    if textual is None:
        textual = []

    experiment.compute_coordinates_by_families()

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot()

    plt.axis('equal')

    if not axis:
        plt.axis('off')

    _color_map_by_features(experiment=experiment, fig=fig, ax=ax,
                           feature_ids=feature_ids, rounding=rounding,
                           normalizing_func=normalizing_func, ticks_pos=ticks_pos,
                           marker_func=marker_func, upper_limit=upper_limit,
                           xticklabels=xticklabels, ms=ms, cmap=cmap,
                           ticks=ticks, column_id=column_id, feature_labelsize=feature_labelsize)
    _add_textual(experiment=experiment, textual=textual, ax=ax, size=textual_size)

    _basic_background(ax=ax, legend=False,
                      saveas=saveas, xlabel=xlabel, bbox_inches=bbox_inches,
                      title=title, legend_pos=legend_pos, title_size=title_size,
                      title_pos=title_pos, dpi=dpi)

    if tex:
        _saveas_tex(saveas=saveas)

    if show:
        plt.show()


def print_map_3d(experiment,
                 xlabel=None,
                 feature_id=None,
                 axis=False,
                 ticks=None,
                 dim=3,
                 title=None,
                 shading=False,
                 saveas="map_3d",
                 show=True,
                 ms=20,
                 normalizing_func=None,
                 xticklabels=None,
                 cmap=None,
                 marker_func=None,
                 tex=False,
                 legend=True,
                 title_pos=None):
    """ This function is not updated """
    experiment.compute_coordinates_by_families(dim=dim)

    if cmap is None:
        cmap = custom_div_cmap()

    if feature_id is not None:
        fig = plt.figure(figsize=(6.4, 4.8 + 0.48))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if not axis:
        plt.axis('off')

    # COLORING
    if feature_id is not None:
        _color_map_by_feature(experiment=experiment, fig=fig, ax=ax,
                              feature_id=feature_id,
                              normalizing_func=normalizing_func,
                              marker_func=marker_func,
                              xticklabels=xticklabels, ms=ms, cmap=cmap,
                              ticks=ticks, dim=dim)
    else:
        basic_coloring(experiment=experiment, ax=ax, dim=dim)

    # BACKGROUND
    _basic_background(ax=ax, legend=legend,
                      saveas=saveas, xlabel=xlabel,
                      title=title, title_pos=title_pos)

    if tex:
        _saveas_tex(saveas=saveas)

    if show:
        plt.show()


# HELPER FUNCTIONS FOR PRINT_2D
def convert_none(value):
    if value == 'None':
        return None
    return value


def convert_none_time(value):
    if value == 0.:
        return None
    return value


def _import_values_for_feature(experiment,
                               feature_id=None,
                               upper_limit=None,
                               lower_limit=None,
                               normalizing_func=None,
                               marker_func=None,
                               dim=2,
                               column_id='value',
                               omit=None,
                               scale='default'):
    """ Import values for a feature_id """

    if isinstance(feature_id, str):
        if feature_id in experiment.features:
            values = experiment.features[feature_id][column_id]
            if column_id == 'time':
                values = {k: convert_none_time(v) for k, v in values.items()}
            else:
                values = {k: convert_none(v) for k, v in values.items()}
        else:
            values = imports.get_values_from_csv_file(experiment, feature_id=feature_id,
                                                      feature_long_id=feature_id,
                                                      upper_limit=upper_limit,
                                                      lower_limit=lower_limit,
                                                      column_id=column_id)
            if feature_id == 'partylist' and column_id == 'value':
                bounds = imports.get_values_from_csv_file(experiment, feature_id=feature_id,
                                                          feature_long_id=feature_id,
                                                          upper_limit=upper_limit,
                                                          lower_limit=lower_limit,
                                                          column_id='bound')
    else:
        values = feature_id

    _min = min(x for x in values.values() if x is not None)
    _max = max(x for x in values.values() if x is not None)

    shades = []
    xx = []
    yy = []
    zz = []
    markers = []
    mses = []
    none_xx, none_yy = [], []
    blank_xx, blank_yy = [], []

    ctr = 0

    my_shade = {}
    for family_id in experiment.families:
        for k in range(experiment.families[family_id].size):

            try:
                if experiment.families[family_id].size == 1:
                    election_id = family_id
                else:
                    election_id = family_id + '_' + str(k)
                shade = values[election_id]
            except:
                election_id = family_id + '_' + str(k)
                shade = values[election_id]

            if shade is None or election_id in omit:
                my_shade[election_id] = None
            elif normalizing_func is not None:
                my_shade[election_id] = normalizing_func(shade)
            else:
                my_shade[election_id] = shade

            if scale == 'log':
                if election_id not in omit and my_shade[election_id] is not None:
                    my_shade[election_id] = math.log(my_shade[election_id] + 1)
                    # pass
            elif scale == 'sqrt':
                if election_id not in omit and my_shade[election_id] is not None:
                    my_shade[election_id] = my_shade[election_id] ** 0.5
    local_min = min(x for x in my_shade.values() if x is not None)
    local_max = max(x for x in my_shade.values() if x is not None)
    if feature_id == 'jr':
        local_min = 0.75

    names = []
    for family_id in experiment.families:

        for k in range(experiment.families[family_id].size):

            try:
                if experiment.families[family_id].size == 1:
                    election_id = family_id
                else:
                    election_id = family_id + '_' + str(k)
                shade = my_shade[election_id]
            except:
                election_id = family_id + '_' + str(k)
                shade = my_shade[election_id]

            if shade is None:
                blank_xx.append(experiment.coordinates[election_id][0])
                blank_yy.append(experiment.coordinates[election_id][1])
                continue

            if normalizing_func is not None:
                shade = normalizing_func(shade)

            if local_max - local_min != 0:
                shade = (shade - local_min) / (local_max - local_min)
            shades.append(shade)
            names.append(election_id)

            mses.append(experiment.families[family_id].ms)

            marker = experiment.families[family_id].marker
            if marker_func is not None:
                marker = marker_func(shade)
            markers.append(marker)

            xx.append(experiment.coordinates[election_id][0])
            yy.append(experiment.coordinates[election_id][1])
            if dim == 3:
                zz.append(experiment.coordinates[election_id][2])

            ctr += 1

    xx = np.asarray(xx)
    yy = np.asarray(yy)
    if dim == 3:
        zz = np.asarray(zz)

    shades = np.asarray(shades)
    markers = np.asarray(markers)
    mses = np.asarray(mses)
    # if feature_id == 'partylist' and column_id == 'value':
    #     mses = np.asarray(
    #         [1. + 20. * (b / (v + 1)) for b, v in zip(bounds.values(), values.values())])
    # else:
    #     mses = None
    
    return xx, yy, zz, shades, markers, mses, local_min, local_max, blank_xx, blank_yy, names


def _import_values_for_features(experiment,
                                feature_ids=None,
                                upper_limit=np.infty,
                                normalizing_func=None,
                                marker_func=None,
                                dim=2,
                                column_id='value'):
    """ Import values for a feature_ids """

    values_1 = imports.get_values_from_csv_file(experiment, feature_id=feature_ids[0],
                                                column_id=column_id, upper_limit=upper_limit)

    values_2 = imports.get_values_from_csv_file(experiment, feature_id=feature_ids[1],
                                                column_id=column_id, upper_limit=upper_limit)

    xx = []
    yy = []
    zz = []
    markers = []
    blank_xx, blank_yy = [], []

    ctr = 0

    shades_1 = []
    shades_2 = []

    local_min_1 = min(x for x in values_1.values() if x is not None)
    local_max_1 = max(x for x in values_1.values() if x is not None)

    local_min_2 = min(x for x in values_2.values() if x is not None)
    local_max_2 = max(x for x in values_2.values() if x is not None)

    local_min = min(local_min_1, local_min_2)
    local_max = max(local_max_1, local_max_2)

    for family_id in experiment.families:

        for k in range(experiment.families[family_id].size):
            if experiment.families[family_id].size == 1:
                election_id = family_id
            else:
                election_id = family_id + '_' + str(k)

            shade_1 = values_1[election_id]
            shade_2 = values_2[election_id]

            if shade_1 is None:
                blank_xx.append(experiment.coordinates[election_id][0])
                blank_yy.append(experiment.coordinates[election_id][1])
                continue

            shade_1 = (shade_1 - local_min) / (local_max - local_min)
            shade_2 = (shade_2 - local_min) / (local_max - local_min)
            shades_1.append(shade_1)
            shades_2.append(shade_2)

            marker = experiment.families[family_id].marker
            markers.append(marker)

            xx.append(experiment.coordinates[election_id][0])
            yy.append(experiment.coordinates[election_id][1])

            ctr += 1

    xx = np.asarray(xx)
    yy = np.asarray(yy)

    shades_1 = np.asarray(shades_1)
    shades_2 = np.asarray(shades_2)
    markers = np.asarray(markers)
    mses = None

    return xx, yy, zz, shades_1, shades_2, markers, mses, local_min, local_max, blank_xx, blank_yy


def get_values_from_file_3d(experiment, experiment_id, values, normalizing_func):
    path = os.path.join(os.getcwd(), "experiment", experiment_id, "controllers", "advanced",
                        str(values) + ".txt")
    _min = 0
    _max = 0
    values = []
    with open(path, 'r') as txt_file:
        for _ in range(experiment.num_elections):
            values.append(float(txt_file.readline()))
    _min = min(values)
    _max = max(values)

    with open(path, 'r') as txt_file:

        shades = []
        xx = []
        yy = []
        zz = []
        markers = []

        ctr = 0
        for k in range(experiment.num_families):
            for _ in range(experiment.families[k].size):

                shade = float(txt_file.readline())
                if normalizing_func is not None:
                    shade = normalizing_func(shade)
                else:
                    shade = (shade - _min) / (_max - _min)
                shades.append(shade)

                marker = experiment.families[k].marker
                markers.append(marker)

                xx.append(experiment.coordinates[experiment.main_order[ctr]][0])
                yy.append(experiment.coordinates[experiment.main_order[ctr]][1])
                zz.append(experiment.coordinates[experiment.main_order[ctr]][2])

                ctr += 1
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        zz = np.asarray(zz)
        shades = np.asarray(shades)
        markers = np.asarray(markers)
        return xx, yy, zz, shades, markers, _min, _max


def _color_map_by_feature(experiment=None,
                          fig=None,
                          ax=None,
                          feature_id=None,
                          upper_limit=np.infty,
                          lower_limit=-np.infty,
                          normalizing_func=None,
                          marker_func=None,
                          xticklabels=None,
                          ms=20,
                          cmap=None,
                          ticks=None,
                          dim=2,
                          rounding=1,
                          column_id='value',
                          feature_labelsize=14,
                          ticks_pos=None,
                          omit=None,
                          scale='default',
                          strech=None,
                          colors=None):
    xx, yy, zz, shades, markers, mses, _min, _max, blank_xx, blank_yy, names = \
        _import_values_for_feature(
            experiment,
            feature_id=feature_id,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
            normalizing_func=normalizing_func,
            marker_func=marker_func,
            dim=dim,
            column_id=column_id,
            omit=omit,
            scale=scale)

    vmin = 0
    vmax = 1
    if strech is not None:
        length = _max - _min
        vmin = 0 - (_min - strech[0]) / length
        vmax = 1 + (strech[1] - _max) / length

    unique_markers = set(markers)
    images = []

    if cmap is None:
        if rounding == 0:
            num_colors = int(min(_max - _min + 1, 101))
            if num_colors < 10:
                xticklabels = [str(q) for q in range(int(_min), int(_max) + 1)]
                ticks_pos = [(2 * q + 1) / num_colors / 2 for q in range(num_colors)]
            if colors is None:
                cmap = custom_div_cmap(num_colors=num_colors)
            else:
                cmap = custom_div_cmap(colors=colors, num_colors=num_colors)
        else:
            cmap = custom_div_cmap()

    for um in unique_markers:
        masks = (markers == um)
        if um == '.':
            images.append(ax.scatter(xx[masks],
                                     yy[masks],
                                     c='grey',
                                     alpha=0.6,
                                     marker=um,
                                     s=mses[masks]))

        elif dim == 2:
            images.append(ax.scatter(xx[masks],
                                     yy[masks],
                                     c=shades[masks],
                                     vmin=vmin,
                                     vmax=vmax,
                                     cmap=cmap,
                                     marker=um,
                                     s=mses[masks]))
        elif dim == 3:
            images.append(ax.scatter(xx[masks],
                                     yy[masks],
                                     zz[masks],
                                     c=shades[masks],
                                     vmin=vmin,
                                     vmax=vmax,
                                     cmap=cmap,
                                     marker=um,
                                     s=ms))

    images.append(ax.scatter(blank_xx, blank_yy, marker='.', alpha=0.1))

    if dim == 2:
        if xticklabels is None:
            # if normalizing_func is None:
            if strech is None:
                lin = np.linspace(_min, _max, 6)
            else:
                lin = np.linspace(strech[0], strech[1], 6)

            xticklabels = [lin[i] for i in range(6)]
            if scale == 'log':
                xticklabels = [int(math.e ** x) for x in xticklabels]
            elif scale == 'sqrt':
                xticklabels = [x ** 2 for x in xticklabels]

            if rounding == 0:
                xticklabels = [str(int(x)) for x in xticklabels]
            else:
                xticklabels = [str(np.round(x, rounding)) for x in xticklabels]

            # else:
            #     lin = np.linspace(_min, _max, 6)
            #     if rounding == 0:
            #         xticklabels = [int(_min), '', 'log', 'scale', '', str(int(_max))]
            #     else:
            #         xticklabels = [_min, '', 'log', 'scale', '', str(_max)]
            #
            # if scale_to_interval:
            #     xticklabels = [str(float(x)/_max) for x in xticklabels]

    if upper_limit < np.infty:
        xticklabels[-1] += '+'

    if lower_limit > -np.infty:
        xticklabels[0] = '-' + xticklabels[0]

    cb = fig.colorbar(images[0], orientation="horizontal", pad=0.1, shrink=0.55,
                      ticks=ticks,
                      )

    cb.ax.locator_params(nbins=len(xticklabels), tight=True)
    cb.ax.tick_params(labelsize=feature_labelsize)

    if ticks_pos is None:
        # ticks_pos = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ticks_pos = np.linspace(vmin, vmax, 6)

    cb.ax.xaxis.set_ticks(ticks_pos)

    if xticklabels is not None:
        cb.ax.set_xticklabels(xticklabels)

    shades_dict = {}
    for i, name in enumerate(names):
        shades_dict[name] = shades[i]

    return shades_dict, cmap


def _color_map_by_features(experiment=None, fig=None, ax=None, feature_ids=None,
                           upper_limit=np.infty,
                           normalizing_func=None, marker_func=None, xticklabels=None, ms=None,
                           cmap=None, ticks=None, dim=2, rounding=1, column_id='value',
                           feature_labelsize=14, ticks_pos=None):
    xx, yy, zz, shades_1, shades_2, markers_1, mses_1, _min, _max, blank_xx_1, blank_yy_1 = _import_values_for_features(
        experiment, feature_ids=feature_ids, upper_limit=upper_limit,
        normalizing_func=normalizing_func,
        marker_func=marker_func, dim=dim, column_id=column_id)

    markers = []
    mask_1 = [False] * len(shades_1)
    mask_2 = [False] * len(shades_1)
    mask_3 = [False] * len(shades_1)
    mask_4 = [False] * len(shades_1)
    mask_5 = [False] * len(shades_1)
    mask_6 = [False] * len(shades_1)

    for i in range(len(shades_1)):
        # print(shades_1[i]-shades_2[i])
        if shades_1[i] < shades_2[i]:
            # markers.append('x')
            if shades_1[i] == 0.:
                mask_4[i] = True
            else:
                mask_1[i] = True
        elif shades_1[i] > shades_2[i]:
            # markers.append('o')
            if shades_2[i] == 0.:
                mask_5[i] = True
            else:
                mask_2[i] = True
        else:
            # markers.append('^')
            if shades_1[i] == 0.:
                mask_6[i] = True
            else:
                mask_3[i] = True

        # if max(shades_1[i], shades_2[i]) == 1:
        #     markers.append('o')
        # else:
        #     markers.append('x')

    tmp = []
    for a, b in zip(shades_1, shades_2):
        tmp.append(min(a, b))
    div = max(tmp)
    shades_1 = np.array([x / div for x in shades_1])
    shades_2 = np.array([x / div for x in shades_2])
    _max = _min + (_max - _min) * div

    images = []

    if mses_1 is None:
        mses = np.asarray([ms for _ in range(len(shades_1))])

    cmap_1 = 'Blues_r'  # custom_div_cmap(colors=[(1, 0.9, 0.9), 'red'])
    cmap_2 = 'Reds_r'  # custom_div_cmap(colors=[(0.9, 1, 0.9), 'green'])
    cmap_3 = custom_div_cmap(colors=['green', (0.9, 0.9, 1)])

    # cmap_4 = custom_div_cmap(colors=[(1, 0.9, 0.9), 'red'])
    # cmap_5 = custom_div_cmap(colors=[(0.9, 1, 0.9), 'green'])
    # cmap_6 = custom_div_cmap(colors=[(0.9, 0.9, 1), 'blue'])

    images.append(ax.scatter(xx[mask_1], yy[mask_1], c=shades_1[mask_1], vmin=0, vmax=1,
                             cmap=cmap_1, s=mses[mask_1], marker='o'))
    images.append(ax.scatter(xx[mask_2], yy[mask_2], c=shades_2[mask_2], vmin=0, vmax=1,
                             cmap=cmap_2, s=mses[mask_2], marker='o'))
    images.append(ax.scatter(xx[mask_3], yy[mask_3], c=shades_1[mask_3], vmin=0, vmax=1,
                             cmap=cmap_3, s=mses[mask_3], marker='o'))

    images.append(ax.scatter(xx[mask_4], yy[mask_4], c=shades_1[mask_4], vmin=0, vmax=1,
                             cmap=cmap_1, s=mses[mask_4], marker='x'))
    images.append(ax.scatter(xx[mask_5], yy[mask_5], c=shades_2[mask_5], vmin=0, vmax=1,
                             cmap=cmap_2, s=mses[mask_5], marker='x'))
    images.append(ax.scatter(xx[mask_6], yy[mask_6], c=shades_1[mask_6], vmin=0, vmax=1,
                             cmap=cmap_3, s=mses[mask_6], marker='x'))

    images.append(ax.scatter(blank_xx_1, blank_yy_1, marker='.', alpha=0.1))

    if dim == 2:
        if xticklabels is None:
            if normalizing_func is None:
                lin = np.linspace(_min, _max, 6)
                if rounding == 0:
                    xticklabels = [int(lin[i]) for i in range(6)]
                else:
                    xticklabels = [int(np.round(lin[i], rounding)) for i in range(6)]
            else:
                lin = np.linspace(_min, _max, 6)
                if rounding == 0:
                    xticklabels = [int(_min), '', 'log', 'scale', '', str(int(_max))]
                else:
                    xticklabels = [int(_min), '', 'log', 'scale', '', str(int(_max))]

        if ticks_pos is None:
            ticks_pos = [0, 0.2, 0.4, 0.6, 0.8, 1]

        # VERTICAL VERSION
        cb_0 = fig.colorbar(images[0], orientation="vertical", pad=-0.1, shrink=0.55,
                            ticks=ticks)
        cb_0.ax.locator_params(nbins=len(xticklabels), tight=True)
        cb_0.ax.tick_params(labelsize=feature_labelsize)
        cb_0.ax.yaxis.set_ticks(ticks_pos)
        cb_0.ax.set_yticklabels(xticklabels)

        cb_1 = fig.colorbar(images[1], orientation="vertical", pad=-0.1, shrink=0.55,
                            ticks=ticks)
        # cb_1.ax.xaxis.set_ticks(ticks_pos)
        # cb_1.ax.set_xticklabels(xticklabels)
        cb_1.ax.set_yticklabels([])

        cb_2 = fig.colorbar(images[2], orientation="vertical", pad=0, shrink=0.55,
                            ticks=ticks)
        # cb_2.ax.xaxis.set_ticks(ticks_pos)
        # cb_2.ax.set_xticklabels(xticklabels)
        cb_2.ax.set_yticklabels([])

        # HORIZONTAL VERSION
        # cb_0 = fig.colorbar(images[0], orientation="vertical", pad=-0.1, shrink=0.55,
        #                     ticks=ticks)
        # cb_0.ax.locator_params(nbins=len(xticklabels), tight=True)
        # cb_0.ax.tick_params(labelsize=feature_labelsize)
        # cb_0.ax.xaxis.set_ticks(ticks_pos)
        # cb_0.ax.set_xticklabels(xticklabels)
        # # cb_0.ax..set_ticks([])
        #
        # cb_1 = fig.colorbar(images[1], orientation="vertical", pad=-0.1, shrink=0.55,
        #                     ticks=ticks)
        # # cb.ax.locator_params(nbins=len(xticklabels), tight=True)
        # # cb.ax.tick_params(labelsize=feature_labelsize)
        #
        # cb_1.ax.xaxis.set_ticks(ticks_pos)
        # cb_1.ax.set_xticklabels(xticklabels)
        # cb_1.ax.set_xticklabels([])
        #
        # cb_2 = fig.colorbar(images[2], orientation="vertical", pad=0, shrink=0.55,
        #                     ticks=ticks)
        # # cb.ax.locator_params(nbins=len(xticklabels), tight=True)
        # # cb.ax.tick_params(labelsize=feature_labelsize)
        # cb_2.ax.xaxis.set_ticks(ticks_pos)
        # cb_2.ax.set_xticklabels(xticklabels)
        # cb_2.ax.set_xticklabels([])


# HELPER FUNCTIONS FOR PRINT_3D
def add_advanced_points_to_picture_3d(fig, ax, experiment, experiment_id,
                                      values=None, cmap=None, ms=None,
                                      normalizing_func=None):
    xx, yy, zz, shades, markers, _min, _max = get_values_from_file_3d(
        experiment, experiment_id, values, normalizing_func)
    unique_markers = set(markers)
    images = []

    for um in unique_markers:
        masks = (markers == um)
        images.append(
            [ax.scatter(xx[masks], yy[masks], zz[masks], c=shades[masks],
                        vmin=0, vmax=1, cmap=cmap, marker=um, s=ms)])


# COLORING
def basic_coloring(experiment=None, ax=None, dim=2, textual=None):
    if textual is None:
        textual = []
    for family in experiment.families.values():

        if family.show:
            if dim == 2:
                if family.label in textual:
                    label = '_nolegend_'
                else:
                    label = family.label

                ax.scatter(experiment.coordinates_by_families[family.family_id][0],
                           experiment.coordinates_by_families[family.family_id][1],
                           color=family.color,
                           label=label,
                           alpha=family.alpha,
                           s=family.ms,
                           marker=family.marker)
            elif dim == 3:
                ax.scatter(experiment.coordinates_by_families[family.family_id][0],
                           experiment.coordinates_by_families[family.family_id][1],
                           experiment.coordinates_by_families[family.family_id][2],
                           color=family.color,
                           label=family.label,
                           alpha=family.alpha,
                           s=family.ms,
                           marker=family.marker)


def basic_coloring_with_individual(experiment=None, ax=None, individual=None):
    for family_id in experiment.families:
        if experiment.families[family_id].show:

            for i in range(experiment.families[family_id].size):
                election_id = experiment.families[family_id].election_ids[i]

                ax.scatter(experiment.coordinates_by_families[family_id][0][i],
                           experiment.coordinates_by_families[family_id][1][i],
                           color=individual['color'][election_id],
                           alpha=individual['alpha'][election_id],
                           s=individual['ms'][election_id],
                           marker=individual['marker'][election_id])


def basic_coloring_with_shading(experiment=None,
                                ax=None,
                                urn_orangered=False):
    for family in experiment.families.values():
        if family.show:
            label = family.label
            for i in range(family.size):
                election_id = list(family.instance_ids)[i]

                try:
                    alpha = experiment.instances[election_id].printing_params['alpha']
                except:
                    alpha = None

                if alpha is None:
                    alpha = family.alpha

                ms = family.ms
                color = family.color

                if 'Mallows (triangle)' in label:
                    tint = experiment.instances[election_id].params['tint']
                    tint = 2 * tint
                    color = (0.75 - 0.75 * alpha + 0.125 * tint + 0.875 * alpha * tint,
                             0.75 - 0.75 * alpha,
                             0.75 - 0.75 * alpha + 0.125 * (1 - tint) + 0.875 * alpha * (1 - tint))
                    alpha = 1

                elif 'Urn' in label:

                    color, alpha = get_color_alpha_for_urn(alpha, urn_orangered)

                else:

                    if alpha is None or alpha > 1:
                        alpha = 1

                    if '1D _path' in label:
                        alpha *= 4
                    elif '2D _path' in label:
                        alpha *= 2
                    elif 'scale' in family.path:
                        alpha *= 1. / family.path['scale']

                    # alpha *= family.alpha
                    alpha = (alpha + 0.2) / 1.2

                if i == (family.size - 1):
                    ax.scatter(experiment.coordinates_by_families[family.family_id][0][i],
                               experiment.coordinates_by_families[family.family_id][1][i],
                               color=color,
                               label=label,
                               alpha=alpha,
                               # s=family.ms,
                               s=ms,
                               marker=family.marker)
                else:
                    ax.scatter(experiment.coordinates_by_families[family.family_id][0][i],
                               experiment.coordinates_by_families[family.family_id][1][i],
                               color=color,
                               alpha=alpha,
                               # s=family.ms,
                               s=ms,
                               marker=family.marker)


# BACKGROUNDS
def _basic_background(
        ax=None,
        legend=None,
        saveas=None,
        xlabel=None,
        title=None,
        pad_inches=None,
        legend_pos=None,
        bbox_inches='tight',
        title_size=16,
        title_pos=None,
        dpi=250,
        mask=None
):
    file_name = os.path.join(os.getcwd(), "images", str(saveas))

    if xlabel is not None:
        plt.xlabel(xlabel, size=14)
    if title is not None:
        if title_pos is not None:
            plt.suptitle(title, fontsize=title_size, y=title_pos)
        else:
            plt.suptitle(title, fontsize=title_size)

    if legend:
        if legend_pos is not None:
            ax.legend(bbox_to_anchor=legend_pos, loc="upper center")
        else:
            ax.legend()

    if saveas is not None:

        try:
            os.mkdir(os.path.join(os.getcwd(), "images"))
        except FileExistsError:
            pass

        plt.savefig(file_name, bbox_inches=bbox_inches, dpi=dpi, pad_inches=pad_inches)

        if mask:
            path_1 = file_name + ".png"
            path_2 = os.path.join(os.getcwd(), "images", "masks", mask + ".png")
            path_3 = file_name + ".png"
            _overlay_images(path_1, path_2, path_3)


# TEX
def _saveas_tex(saveas=None):
    try:
        os.mkdir(os.path.join(os.getcwd(), "images", "tex"))
    except FileExistsError:
        pass
    file_name = saveas + ".svg"
    path = os.path.join(os.getcwd(), "images", "tex", file_name)
    tikzplotlib.save(path)


# MAIN FUNCTIONS
def print_matrix(experiment=None,
                 scale=1.,
                 rounding=1,
                 saveas=None,
                 show=True,
                 ms=8,
                 title=None,
                 omit=None,
                 self_distances=False,
                 yticks='left',
                 with_std=False,
                 time=False,
                 dpi=100,
                 vmin=None,
                 vmax=None):
    """Print the matrix with average distances between each pair of election """

    if omit is None:
        omit = []

    selected_families = list(experiment.families.keys())
    for family_id in omit:
        if family_id in selected_families:
            selected_families.remove(family_id)
    num_selected_families = len(selected_families)
    num_selected_instances = 0
    for family_id in selected_families:
        num_selected_instances += len(experiment.families[family_id].instance_ids)

    # CREATE MAPPING FOR BUCKETS
    bucket = []
    for family_id in selected_families:
        for _ in range(experiment.families[family_id].size):
            bucket.append(family_id)

    # CREATE MAPPING FOR ELECTIONS
    mapping = {}
    ctr = 0
    for family_id in selected_families:
        for instance_id in experiment.families[family_id].instance_ids:
            mapping[ctr] = instance_id
            ctr += 1

    # PREPARE EMPTY DICTS
    matrix = {family_id_1: {} for family_id_1 in selected_families}
    quantities = {family_id_1: {} for family_id_1 in selected_families}

    for family_id_1 in selected_families:
        for family_id_2 in selected_families:
            matrix[family_id_1][family_id_2] = 0
            quantities[family_id_1][family_id_2] = 0

    # ADD VALUES
    for i in range(num_selected_instances):
        limit = i + 1
        if self_distances:
            limit = i
        for j in range(limit, num_selected_instances):
            if time:
                matrix[bucket[i]][bucket[j]] += experiment.times[mapping[i]][
                    mapping[j]]
            else:
                matrix[bucket[i]][bucket[j]] += experiment.distances[mapping[i]][mapping[j]]
            quantities[bucket[i]][bucket[j]] += 1
    # NORMALIZE
    # for family_id_1, family_id_2 in combinations(election.families, 2):
    # for family_id_1, family_id_2 in product(election.families, 2):
    for i, family_id_1 in enumerate(selected_families):
        for j, family_id_2 in enumerate(selected_families):
            if i <= j:
                # add normalization for election distances
                if quantities[family_id_1][family_id_2] != 0.:
                    matrix[family_id_1][family_id_2] /= float(quantities[family_id_1][family_id_2])
                matrix[family_id_1][family_id_2] = \
                    round(matrix[family_id_1][family_id_2] * scale, rounding)
                if rounding == 0:
                    matrix[family_id_1][family_id_2] = int(matrix[family_id_1][family_id_2])
                matrix[family_id_2][family_id_1] = matrix[family_id_1][family_id_2]

    # THE REST
    fig, ax = plt.subplots()
    num_families_new = num_selected_families

    matrix_new = np.zeros([num_families_new, num_families_new])

    # FIND MIN & MAX
    _min = min([min(matrix[family_id].values()) for family_id in selected_families])
    _max = max([max(matrix[family_id].values()) for family_id in selected_families])

    if vmin is None:
        vmin = _min
    if vmax is None:
        vmax = _max

    threshold = vmin + 0.75 * (vmax - vmin)

    # PRINT
    if with_std:
        for i, family_id_1 in enumerate(selected_families):
            for j, family_id_2 in enumerate(selected_families):
                c = matrix[family_id_1][family_id_2]
                std = round(experiment.stds[mapping[i]][mapping[j]] * scale, rounding)
                if rounding == 0:
                    std = int(std)
                matrix_new[i][j] = c
                color = "black"
                if c >= threshold:
                    color = "white"
                ax.text(j - 0.1, i + 0.1, str(c), va='bottom', ha='center',
                        color=color, size=12)
                if std >= 10:
                    ax.text(j - 0.3, i + 0.1, '$\pm$' + str(std), va='top',
                            ha='left', color=color, size=9)
                else:
                    ax.text(j - 0.1, i + 0.1, '$\pm$' + str(std), va='top',
                            ha='left', color=color, size=9)
    else:
        for i, family_id_1 in enumerate(selected_families):
            for j, family_id_2 in enumerate(selected_families):
                c = matrix[family_id_1][family_id_2]
                matrix_new[i][j] = c
                color = "black"
                if c >= threshold:
                    color = "white"
                c = str(c)
                try:
                    if c[0] == '0':
                        c = c[1:]
                    if c[-1] == '0' and c[-2] == '.':
                        c = c[:-1]
                except:
                    pass
                ax.text(j, i, str(c), va='center', ha='center', color=color, size=ms)

    labels = []
    for family_id in selected_families:
        if family_id in RULE_NAME_MATRIX:
            labels.append(RULE_NAME_MATRIX[experiment.families[family_id].label])
        elif family_id in SHORT_NICE_NAME:
            labels.append(SHORT_NICE_NAME[experiment.families[family_id].label])
        else:
            labels.append(experiment.families[family_id].label)

    ax.matshow(matrix_new, cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)

    x_values = labels
    y_values = labels
    y_axis = np.arange(0, num_families_new, 1)
    x_axis = np.arange(0, num_families_new, 1)

    if yticks != 'none':
        ax.set_yticks(y_axis)
        if yticks == 'left':
            ax.set_yticklabels(y_values, rotation=25, size=ms + 2)
        if yticks == 'right':
            ax.set_yticklabels(y_values, rotation=-25, size=ms + 2)
            ax.yaxis.tick_right()
    else:
        ax.set_yticks([])

    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_values, rotation=80, size=ms + 2)

    if title:
        plt.title(title)

    if saveas and experiment.is_exported:
        file_name = os.path.join(os.getcwd(), "images", str(saveas) + ".png")
        plt.savefig(file_name, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()


# HELPER FUNCTIONS
def custom_div_cmap(num_colors=101, name='custom_div_cmap', colors=None, reverse=False):
    if colors is None:
        # colors = ["lightgreen", "yellow", "orange", "red", "purple", "black"]
        if reverse:
            colors = ["black", "blue", "purple", "red", "orange", "yellowgreen", "green"]
        else:
            colors = ["green", "yellowgreen", "orange", "red", "purple", "blue", "black"]
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name=name, colors=colors, N=num_colors)


def add_margin(pil_img, top, right, bottom, left, color):
    """ Add margin to the picture """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def map_diameter(c):
    """ Compute the diameter """
    return 1. / 3. * float((c + 1) * (c - 1))


# SKELETON RELATED
def name_from_label(experiment, name):
    for family in experiment.families.values():
        if family.label == name:
            return family.family_id
    return 'None'


def _add_textual(experiment=None, textual=None, ax=None, size=16,
                 shades_dict=None, cmap=None, column_id=None, feature_id=None):
    """ Add textual """

    def my_text(x1, y1, text, color="black", alpha=1., size=size, b_color='black',
                boxstyle="round"):
        ax.text(x1, y1, text, size=size, rotation=0., ha="center",
                va="center",
                color=color, alpha=alpha, zorder=100,
                bbox=dict(boxstyle=boxstyle, ec=b_color, fc="white")
                )

    for name in textual:
        name_id = name_from_label(experiment, name)
        x = .0
        y = .0
        try:
            x, y = experiment.coordinates[name_id]
        except KeyError as err:
            if name_id in ['ST', 'UN']:
                x, y = experiment.coordinates[name_id + "_0"]

        if name in RULE_NAME_MAP:
            name = RULE_NAME_MAP[name]

        if shades_dict is None:
            my_text(x, y, name, color='black')
        else:
            if column_id == 'ejr':
                if name_id in ['rule-x', 'pav']:
                    rgba = cmap(shades_dict[name_id])
                    my_text(x, y, name, color='black', b_color='black', boxstyle='sawtooth')
                else:
                    if shades_dict[name_id] == 1:
                        rgba = cmap(shades_dict[name_id])
                        my_text(x, y, name, color=rgba, b_color=rgba, boxstyle='sawtooth')
                    else:
                        rgba = cmap(shades_dict[name_id])
                        my_text(x, y, name, color=rgba, b_color=rgba)

            elif feature_id in ['priceability']:
                if name_id in ['rule-x', 'seqphragmen']:
                    rgba = cmap(shades_dict[name_id])
                    my_text(x, y, name, color='black', b_color='black', boxstyle='sawtooth')
                else:
                    if shades_dict[name_id] == 1:
                        rgba = cmap(shades_dict[name_id])
                        my_text(x, y, name, color=rgba, b_color=rgba, boxstyle='sawtooth')
                    else:
                        rgba = cmap(shades_dict[name_id])
                        my_text(x, y, name, color=rgba, b_color=rgba)
            elif feature_id in ['core']:
                if name_id == 'cc':
                    my_text(x, y, name, color='grey', b_color='grey')
                elif shades_dict[name_id] == 1:
                    rgba = cmap(shades_dict[name_id])
                    my_text(x, y, name, color=rgba, b_color=rgba, boxstyle='sawtooth')
                else:
                    rgba = cmap(shades_dict[name_id])
                    my_text(x, y, name, color=rgba, b_color=rgba)
            else:
                # rgba = cmap(shades_dict[name_id])
                # my_text(x, y, name, color=rgba, b_color=rgba)
                my_text(x, y, name, color='black', b_color='black')
            # if shades_dict[name_id] == 0:
            #     rgba = cmap(shades_dict[name_id])
            #     my_text(x, y, name, color='green', b_color='green')
            #
            # else:
            #     rgba = cmap(shades_dict[name_id])
            #     my_text(x, y, name, color=rgba, b_color=rgba)


def add_roads(experiment=None, roads=None, ax=None):
    def my_line(x1, y1, x2, y2, text):
        ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0., head_length=0.,
                 fc='k', ec='k')
        dx = x1 + (x2 - x1) / 2
        dy = y1 + (y2 - y1) / 2
        ax.annotate(text, xy=(dx, dy), size=12)

    for road in roads:
        x1 = experiment.coordinates[road[0]][0]
        y1 = experiment.coordinates[road[0]][1]
        x2 = experiment.coordinates[road[1]][0]
        y2 = experiment.coordinates[road[1]][1]
        pos_dist = experiment.distances[road[0]][road[1]]
        pos_dist /= map_diameter(experiment.default_num_candidates)
        pos_dist = round(pos_dist, 2)
        text = str(pos_dist)
        if experiment.default_num_candidates == 10 and road in [['WAL', 'UN'], ['UN', 'WAL']]:
            x1 -= 0.5
            y1 -= 2
            x2 -= 0.5
            y2 -= 2

        my_line(x1, y1, x2, y2, text)


def _skeleton_coloring(experiment=None, ax=None, ms=None, dim=2):
    for family_id in experiment.families:
        if experiment.families[family_id].show:
            if dim == 2:
                if family_id in {'A', 'B', 'C', 'WAL', 'CON', 'D', 'E'}:
                    MAL_COUNT = len(experiment.coordinates_by_families[family_id][0])
                    for i in range(MAL_COUNT):
                        normphi = 1.0 / MAL_COUNT * i
                        if family_id == 'A':  # Mal 0.
                            color = (normphi, normphi, 1)
                        elif family_id == 'B':  # Mal 0.25
                            color = (normphi, 0.75, normphi)
                        elif family_id == 'C':  # Mal 0.5
                            color = (1, normphi, normphi)
                        elif family_id in ['D', 'WAL']:  # Walsh
                            color = (1, normphi, 1)
                        elif family_id in ['E', 'CON']:  # Conitzer
                            color = (1, 0.5, normphi)
                        else:
                            color = 'black'
                        ax.scatter([experiment.coordinates_by_families[family_id][0][i]
                                    for _ in range(2)],
                                   [experiment.coordinates_by_families[family_id][1][i]
                                    for _ in range(2)],
                                   color=color,
                                   alpha=1., s=ms + 6,
                                   marker=experiment.families[family_id].marker)
                else:
                    if len(experiment.coordinates_by_families[family_id][0]) == 1:
                        ax.scatter([experiment.coordinates_by_families[family_id][0][0]
                                    for _ in range(2)],
                                   [experiment.coordinates_by_families[family_id][1][0]
                                    for _ in range(2)],
                                   color=experiment.families[family_id].color,
                                   label=experiment.families[family_id].label,
                                   alpha=1,
                                   s=ms,
                                   marker=experiment.families[
                                       family_id].marker)

                    else:
                        ax.scatter(experiment.coordinates_by_families[family_id][0],
                                   experiment.coordinates_by_families[family_id][1],
                                   color=experiment.families[family_id].color,
                                   label=experiment.families[family_id].label,
                                   alpha=0.8,
                                   s=ms,
                                   marker=experiment.families[family_id].marker)


def get_values_from_file_old(experiment, experiment_id, values,
                             normalizing_func=None, marker_func=None):
    path = os.path.join(os.getcwd(), "election", experiment_id, "features",
                        str(values) + ".txt")

    _min = 0
    _max = 0
    values = []
    with open(path, 'r') as txtfile:
        for _ in range(experiment.num_elections):
            values.append(float(txtfile.readline()))
    _min = min(values)
    _max = max(values)

    with open(path, 'r') as txtfile:

        shades = []
        xx = []
        yy = []
        markers = []

        ctr = 0
        for family_id in experiment.families:
            for k in range(experiment.families[family_id].size):
                election_id = family_id + '_' + str(k)

                shade = float(txtfile.readline())
                if normalizing_func is not None:
                    shade = normalizing_func(shade)
                else:
                    shade = (shade - _min) / (_max - _min)
                shades.append(shade)

                marker = experiment.families[family_id].marker
                if marker_func is not None:
                    marker = marker_func(shade)
                markers.append(marker)

                xx.append(experiment.points[election_id][0])
                yy.append(experiment.points[election_id][1])

                ctr += 1

        xx = np.asarray(xx)
        yy = np.asarray(yy)
        shades = np.asarray(shades)
        markers = np.asarray(markers)
        return xx, yy, shades, markers, _min, _max


def _adjust_the_map_on_three_points_horizontal(experiment, left, right, down, is_down=True) -> None:
    try:
        d_x = experiment.coordinates[right][0] - experiment.coordinates[left][0]
        d_y = experiment.coordinates[right][1] - experiment.coordinates[left][1]
        alpha = math.atan(d_x / d_y)
        experiment.rotate(alpha - math.pi / 2.)
        if experiment.coordinates[left][0] > experiment.coordinates[right][0]:
            experiment.rotate(math.pi)
    except Exception:
        pass

    if is_down:
        if experiment.coordinates[left][1] < experiment.coordinates[down][1]:
            experiment.reverse(axis=0)
    else:
        if experiment.coordinates[left][1] > experiment.coordinates[down][1]:
            experiment.reverse(axis=0)


def _adjust_the_map_on_three_points_vertical(experiment, down, up, left, is_left=True) -> None:

    try:
        d_x = experiment.coordinates[down][0] - experiment.coordinates[up][0]
        d_y = experiment.coordinates[down][1] - experiment.coordinates[up][1]
        alpha = math.atan(d_x / d_y)
        experiment.rotate(alpha)
        if experiment.coordinates[down][1] > experiment.coordinates[up][1]:
            experiment.rotate(math.pi)
    except Exception:
        pass

    if is_left:
        if experiment.coordinates[down][0] < experiment.coordinates[left][0]:
            experiment.reverse(axis=1)
    else:
        if experiment.coordinates[down][0] > experiment.coordinates[left][0]:
            experiment.reverse(axis=1)


def _adjust_the_map_on_two_points_horizontal(experiment, left, right) -> None:
    try:
        d_x = experiment.coordinates[right][0] - experiment.coordinates[left][0]
        d_y = experiment.coordinates[right][1] - experiment.coordinates[left][1]
        alpha = math.atan(d_x / d_y)
        experiment.rotate(alpha - math.pi / 2.)
        if experiment.coordinates[left][0] > experiment.coordinates[right][0]:
            experiment.rotate(math.pi)
    except Exception:
        pass


def _adjust_the_map_on_two_points_vertical(experiment, down, up) -> None:

    try:
        d_x = experiment.coordinates[down][0] - experiment.coordinates[up][0]
        d_y = experiment.coordinates[down][1] - experiment.coordinates[up][1]
        alpha = math.atan(d_x / d_y)
        experiment.rotate(alpha)
        if experiment.coordinates[down][1] > experiment.coordinates[up][1]:
            experiment.rotate(math.pi)
    except Exception:
        pass


def _adjust_the_map_on_three_points_without_down(experiment, left, right, up):
    _adjust_the_map_on_three_points_horizontal(experiment, left, right, up, is_down=False)


def _adjust_the_map_on_three_points_without_left(experiment, down, up, right):
    _adjust_the_map_on_three_points_vertical(experiment, down, up, right, is_left=False)


def _adjust_the_map_on_three_points_without_up(experiment, left, right, down):
    _adjust_the_map_on_three_points_horizontal(experiment, left, right, down, is_down=True)


def _adjust_the_map_on_three_points_without_right(experiment, down, up, left):
    _adjust_the_map_on_three_points_vertical(experiment, down, up, left, is_left=True)


def adjust_the_map(experiment, left=None, up=None, right=None, down=None) -> None:

    # NEW CODE
    try:

        if left is not None and right is not None and up is not None:
            _adjust_the_map_on_three_points_without_down(experiment, left, right, up)
        elif right is not None and down is not None and left is not None:
            _adjust_the_map_on_three_points_without_up(experiment, left, right, down)
        elif up is not None and right is not None and down is not None:
            _adjust_the_map_on_three_points_without_left(experiment, down, up, right)
        elif down is not None and left is not None and up is not None:
            _adjust_the_map_on_three_points_without_right(experiment, down, up, left)

        elif left is not None and right is not None:
            _adjust_the_map_on_two_points_horizontal(experiment, left, right)
        elif down is not None and up is not None:
            _adjust_the_map_on_two_points_vertical(experiment, down, up)

    # OLD CODE
    except:
        if experiment.instance_type == 'ordinal':

            try:
                left = experiment.get_instance_id_from_culture_id('uniformity')
                right = experiment.get_instance_id_from_culture_id('identity')
                down = experiment.get_instance_id_from_culture_id('stratification')
                _adjust_the_map_on_three_points_without_up(experiment, left, right, down)
            except Exception:
                try:
                    left = experiment.get_instance_id_from_culture_id('un_from_matrix')
                    right = experiment.get_instance_id_from_culture_id('real_identity')
                    up = experiment.get_instance_id_from_culture_id('real_antagonism')
                    _adjust_the_map_on_three_points_without_down(experiment, left, right, up)
                except Exception:
                    try:
                        left = 'UN_0'
                        right = 'ID'
                        up = 'AN'
                        _adjust_the_map_on_three_points_horizontal(experiment, left, right, up, is_down=False)
                    except Exception:
                        pass
                    pass
                pass

        elif experiment.instance_type == 'approval':
            try:
                left = 'IC 0.5'
                right = 'ID 0.5'
                down = experiment.get_instance_id_from_culture_id('empty')
                _adjust_the_map_on_three_points_horizontal(experiment, left, right, down)
            except Exception:
                try:
                    left = 'IC 0.25'
                    right = 'ID 0.25'
                    down = experiment.get_instance_id_from_culture_id('empty')
                    _adjust_the_map_on_three_points_horizontal(experiment, left, right, down)
                except Exception:
                    pass

        elif experiment.instance_type == 'rule':
            try:
                left = 'cc'
                right = 'av'
                down = 'minimaxav'
                _adjust_the_map_on_three_points_horizontal(experiment, left, right, down)
            except Exception:
                try:
                    left = 'seqcc'
                    right = 'av'
                    down = 'minimaxav'
                    _adjust_the_map_on_three_points_horizontal(experiment, left, right, down)
                except Exception:
                    pass

        elif experiment.instance_type == 'roommates':

            try:
                left = experiment.get_instance_id_from_culture_id('roommates_asymmetric')
                right = experiment.get_instance_id_from_culture_id('roommates_symmetric')
                # up = election.get_election_id_from_model_name('antagonism')
                down = experiment.get_instance_id_from_culture_id('roommates_id')
                _adjust_the_map_on_three_points_horizontal(experiment, left, right, down)
            except Exception:
                pass


        elif experiment.instance_type == 'marriages':

            try:
                left = experiment.get_instance_id_from_culture_id('asymmetric')
                right = experiment.get_instance_id_from_culture_id('symmetric')
                # up = election.get_election_id_from_model_name('antagonism')
                down = experiment.get_instance_id_from_culture_id('id')
                _adjust_the_map_on_three_points_horizontal(experiment, left, right, down)
            except Exception:
                pass

        elif experiment.instance_type == 'allocation':

            try:
                left = experiment.get_election_id_from_model_name('indifference')
                right = experiment.get_election_id_from_model_name('separability')
                up = experiment.get_election_id_from_model_name('contention')
                _adjust_the_map_on_three_points_horizontal(experiment, left, right, up, is_down=False)
            except Exception:
                pass


def _centroid(array) -> (float, float):
    length = array.shape[0]
    sum_x = np.sum(array[:, 0])
    sum_y = np.sum(array[:, 1])
    return sum_x / length, sum_y / length


def adjust_the_map_on_one_point(experiment) -> None:
    xc, yc = _centroid(np.array(list(experiment.coordinates.values())))
    d_x = experiment.coordinates['all_0'][0] - xc
    d_y = experiment.coordinates['all_0'][1] - yc
    try:
        alpha = math.atan(d_x / d_y)
        experiment.rotate(alpha)
        if experiment.coordinates['all_0'][1] < experiment.coordinates['all_1'][1]:
            experiment.rotate(math.pi)
    except:
        print('Cannot adjust!')


def get_color_alpha_for_urn(alpha, urn_orangered=True):
    if urn_orangered:
        if alpha > 1.07:
            return 'red', 0.9
        elif alpha > 0.53:
            return 'orangered', 0.9
        elif alpha > 0.22:
            return 'orange', 0.9
        else:
            return 'gold', 0.9
    else:
        if alpha > 2:
            return 'orange', 1
        else:
            return 'orange', alpha * 0.35 + 0.3


def print_approvals_histogram(election):
    plt.title(election.election_id, size=20)
    bins = np.linspace(0, 100, 51)
    plt.hist([len(vote) for vote in election.votes], bins=bins)
    # x_axis = np.arange(0, 100, 0.01)
    # plt.plot(x_axis, norm.pdf(x_axis, 50, 2)*2000)
    plt.ylim([0, election.num_voters])
    plt.xlim([-1, election.num_candidates + 1])
    plt.savefig("images/histograms/" + election.election_id + ".png")
    plt.show()


def _overlay_images(background_path, overlay_path, output_path):
    # Open the images
    background = Image.open(background_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA")

    # Resize overlay image to the size of the background image
    overlay = overlay.resize(background.size, Image.LANCZOS)

    # Composite the images
    combined = Image.alpha_composite(background, overlay)

    # Save the result
    combined.save(output_path, 'PNG')


def print_map_1d(experiment, saveas=None):
    experiment.compute_coordinates_by_families()
    all_values = [0]
    for family in experiment.families.values():
        x = float(experiment.coordinates_by_families[family.family_id][0][0])
        all_values.append(x)
    min_ = min(all_values)
    max_ = max(all_values)

    fig, ax = plt.subplots(figsize=(3, 8))
    for family in experiment.families.values():
        x = float(experiment.coordinates_by_families[family.family_id][0][0])
        x = (x - min_) / (max_ - min_)
        x = 1 - x
        ax.scatter(0, x)
        ax.annotate(family.family_id, (0, x), rotation=0, size=10)
    ax.get_xaxis().set_visible(False)
    if saveas:
        plt.savefig(saveas)
    plt.show()
