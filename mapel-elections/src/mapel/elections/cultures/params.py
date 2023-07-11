
import numpy as np
from scipy.stats import gamma
import random as rand
import mapel.elections.cultures.mallows as mallows
from mapel.core.glossary import *


# Ordinal
def update_params_ordinal_mallows(params):
    if 'phi' in params and type(params['phi']) is list:
        params['phi'] = np.random.uniform(low=params['phi'][0], high=params['phi'][1])
    elif 'phi' not in params:
        params['phi'] = np.random.random()


def update_params_ordinal_norm_mallows(params, num_candidates):
    if 'normphi' not in params:
        params['normphi'] = np.random.random()
    params['phi'] = mallows.phi_from_normphi(num_candidates, relphi=params['normphi'])
    if 'weight' not in params:
        params['weight'] = 0.


def update_params_ordinal_urn_model(params):
    if 'alpha' not in params:
        params['alpha'] = gamma.rvs(0.8)


def update_params_ordinal_mallows_matrix_path(params, num_candidates):
    params['normphi'] = params['alpha']
    params['phi'] = mallows.phi_from_normphi(num_candidates, relphi=params['normphi'])


def update_params_ordinal_mallows_triangle(params, num_candidates):
    params['normphi'] = 1 - np.sqrt(np.random.uniform())
    params['phi'] = mallows.phi_from_normphi(num_candidates, relphi=params['normphi'])
    params['weight'] = np.random.uniform(0, 0.5)
    params['alpha'] = params['normphi']
    params['tint'] = params['weight']  # for tint on plots


def update_params_ordinal_alpha(printing_params):
    if 'alpha' not in printing_params:
        printing_params['alpha'] = 1
    elif type(printing_params['alpha']) is list:
        printing_params['alpha'] = np.random.uniform(low=printing_params['alpha'][0],
                                                     high=printing_params['alpha'][1])


def update_params_ordinal_preflib(params, model_id):
    # list of IDs larger than 10
    folder = ''
    if model_id == 'irish':
        folder = 'irish_s1'
        # folder = 'irish_f'
        ids = [1, 3]
    elif model_id == 'glasgow':
        folder = 'glasgow_s1'
        # folder = 'glasgow_f'
        ids = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 19, 21]
    elif model_id == 'formula':
        folder = 'formula_s1'
        # 17 races or more
        ids = [17, 35, 37, 40, 41, 42, 44, 45, 46, 47, 48]
    elif model_id == 'skate':
        folder = 'skate_ic'
        # 9 judges
        ids = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48]
    elif model_id == 'sushi':
        folder = 'sushi_ff'
        ids = [1]
    elif model_id == 'grenoble':
        folder = 'grenoble_ff'
        ids = [1]
    elif model_id == 'tshirt':
        folder = 'tshirt_ff'
        ids = [1]
    elif model_id == 'cities_survey':
        folder = 'cities_survey_s1'
        ids = [1, 2]
    elif model_id == 'aspen':
        folder = 'aspen_s1'
        ids = [1]
    elif model_id == 'marble':
        folder = 'marble_ff'
        ids = [1, 2, 3, 4, 5]
    elif model_id == 'cycling_tdf':
        folder = 'cycling_tdf_s1'
        # ids = [e for e in range(1, 69+1)]
        selection_method = 'random'
        ids = [14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26]
    elif model_id == 'cycling_gdi':
        folder = 'cycling_gdi_s1'
        ids = [i for i in range(2, 23 + 1)]
    elif model_id == 'ers':
        folder = 'ers_s1'
        # folder = 'ers_f'
        # 500 voters or more
        ids = [3, 9, 23, 31, 32, 33, 36, 38, 40, 68, 77, 79, 80]
    elif model_id == 'ice_races':
        folder = 'ice_races_s1'
        # 80 voters or more
        ids = [4, 5, 8, 9, 15, 20, 23, 24, 31, 34, 35, 37, 43, 44, 49]
    else:
        ids = []

    if 'id' not in params:
        params['id'] = rand.choices(ids, k=1)[0]

    params['folder'] = folder


def update_params_ordinal(params, printing_params, variable, culture_id, num_candidates):
    if variable is not None:
        printing_params['alpha'] = params[variable]
        printing_params['variable'] = variable
    else:
        if culture_id.lower() == 'mallows':
            update_params_ordinal_mallows(params)
            printing_params['alpha'] = params['phi']
        elif culture_id.lower() == 'norm_mallows' or culture_id.lower() == 'norm-mallows':
            update_params_ordinal_norm_mallows(params, num_candidates)
            printing_params['alpha'] = params['normphi']
        elif culture_id.lower() == 'urn' or culture_id.lower() == 'urn':
            update_params_ordinal_urn_model(params)
            printing_params['alpha'] = params['alpha']
        elif culture_id.lower() == 'mallows_matrix_path':
            update_params_ordinal_mallows_matrix_path(params, num_candidates)
        elif culture_id.lower() == 'mallows_triangle':
            update_params_ordinal_mallows_triangle(params, num_candidates)
        elif culture_id.lower() in LIST_OF_PREFLIB_MODELS:
            update_params_ordinal_preflib(params, culture_id)
        update_params_ordinal_alpha(printing_params)
    return params, printing_params


# Approval #
def update_params_approval_alpha(printing_params):
    if 'alpha' not in printing_params:
        printing_params['alpha'] = 1
    elif type(printing_params['alpha']) is list:
        printing_params['alpha'] = np.random.uniform(low=printing_params['alpha'][0],
                                                     high=printing_params['alpha'][1])


def update_params_approval_p(params):
    if 'p' not in params:
        params['p'] = np.random.rand()
    elif type(params['p']) is list:
        params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])


def update_params_approval_resampling(params):
    if 'phi' in params and type(params['phi']) is list:
        params['phi'] = np.random.uniform(low=params['phi'][0], high=params['phi'][1])
    elif 'phi' not in params:
        params['phi'] = np.random.random()
    params['alpha'] = params['phi']

    if 'p' in params and type(params['p']) is list:
        params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])
    elif 'p' not in params:
        params['p'] = np.random.random()


def update_params_approval_disjoint(params):
    if 'phi' in params and type(params['phi']) is list:
        params['phi'] = np.random.uniform(low=params['phi'][0], high=params['phi'][1])
    elif 'phi' not in params:
        params['phi'] = np.random.random()
    params['alpha'] = params['phi']

    if 'p' in params and type(params['p']) is list:
        params['p'] = np.random.uniform(low=params['p'][0], high=params['p'][1])
    elif 'p' not in params:
        params['p'] = np.random.random() / params['g']


def update_params_approval(params, printing_params, variable, culture_id, num_candidates):
    printing_params['alpha'] = 0
    if variable is not None:
        if culture_id in APPROVAL_MODELS:
            update_params_approval_p(params)
        printing_params['alpha'] = params[variable]
        printing_params['variable'] = variable
        del params['variable']
    else:
        if culture_id in APPROVAL_MODELS:
            update_params_approval_p(params)
        elif culture_id.lower() == 'resampling':
            update_params_approval_resampling(params)
        elif culture_id.lower() == 'disjoint':
            update_params_approval_disjoint(params)
        update_params_approval_alpha(printing_params)

    return params, printing_params

def get_params_for_crate(j):
    base = []
    my_size = 10
    # with_edge
    for p in range(my_size):
        for q in range(my_size):
            for r in range(my_size):
                a = p / (my_size - 1)
                b = q / (my_size - 1)
                c = r / (my_size - 1)
                d = 1 - a - b - c
                tmp = [a, b, c, d]
                if d >= 0 and sum(tmp) == 1:
                    base.append(tmp)
    params = {'alpha': base[j]}
    return params


def get_params_for_paths(family, j, extremes=False):
    path = family.path

    variable = path['variable']

    if 'extremes' in path:
        extremes = path['extremes']

    params = {'variable': variable}
    if extremes:
        params[variable] = j / (family.size - 1)
    elif not extremes:
        params[variable] = (j + 1) / (family.size + 1)

    if 'scale' in path:
        params[variable] *= path['scale']

    if 'start' in path:
        params[variable] += path['start']
    else:
        path['start'] = 0.

    if 'step' in path:
        params[variable] = path['start'] + j * path['step']

    return params, variable