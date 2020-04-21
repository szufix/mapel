from .voting import modern as mo


def test():
    print("Welcome to Mapel!")


def print_2d(exp_name, **kwargs):
    mo.print_2d(exp_name, **kwargs)


def print_matrix(exp_name, **kwargs):
    mo.print_matrix(exp_name, **kwargs)


def print_param_vs_distance(exp_name, **kwargs):
    mo.print_param_vs_distance(exp_name, **kwargs)


def prepare_approx_cc_order(exp_name, **kwargs):
    mo.prepare_approx_cc_order(exp_name, **kwargs)
