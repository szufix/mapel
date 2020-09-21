from .voting import modern as mo
from .voting import print as pr


def hello():
    print("Welcome to Mapel!")


def print_2d(exp_name, **kwargs):
    mo.print_2d(exp_name, **kwargs)


def print_matrix(exp_name, **kwargs):
    mo.print_matrix(exp_name, **kwargs)


def print_param_vs_distance(exp_name, **kwargs):
    mo.print_param_vs_distance(exp_name, **kwargs)


def prepare_approx_cc_order(exp_name, **kwargs):
    mo.prepare_approx_cc_order(exp_name, **kwargs)


def print_highest_plurality():
    pr.print_highest_plurality()


def print_highest_copeland():
    pr.print_highest_copeland()


def print_highest_dodgson():
    pr.print_highest_dodgson()


def print_highest_borda():
    pr.print_highest_borda()


# HELPER FUNCTIONS
def custom_div_cmap(num_colors=101, name='custom_div_cmap',
                    colors=None):

    if colors is None:
        colors = ["lightgreen", "yellow", "orange", "red", "black"]

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name=name, colors=colors, N=num_colors)
    return cmap
