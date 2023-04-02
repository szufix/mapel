import logging

import numpy as np


def generate_projects_cost(cost_function=None, num_candidates: int = None,
                           budget: float = None, params: dict = None):
    main_models = {'unit_cost': generate_unit_cost,
                   'uniform': generate_uniform_costs,
                   'exponential': generate_exponential_costs}

    if cost_function in main_models:
        return main_models.get(cost_function)(num_candidates=num_candidates,
                                              budget=budget, params=params)

    else:
        if cost_function is not None:
            logging.warning(f'No such model id: {cost_function}, using unit cost as default')

        return generate_unit_cost(num_candidates, budget, params)


def generate_unit_cost(num_candidates: int = None, budget: float = 1,
                       params: dict = None) -> np.array:
    """ Return: unit costs """
    return np.ones(num_candidates)


def generate_uniform_costs(num_candidates: int = None, budget: float = 1,
                           params: dict = None) -> np.array:
    """ Return: uniform costs """
    return np.random.random(size=num_candidates) * budget


def generate_exponential_costs(num_candidates: int = None, budget: float = 1,
                               params: dict = None) -> np.array:
    """ Return: exponential costs """
    if params is None:
        params = {}

    mean_cost = params.get('mean_cost', 0.1 * budget)
    min_cost = params.get('min_cost', 0.01 * budget)

    return min_cost + np.random.exponential(mean_cost - min_cost, size=num_candidates)