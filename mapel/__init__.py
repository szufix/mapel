
import mapel.voting._metrics as metr
import mapel.voting.print as pr
import mapel.voting.development as dev



def hello():
    print("Hello!")


def prepare_experiment(**kwargs):
    return dev.prepare_experiment(**kwargs)


def custom_div_cmap(**kwargs):
    return pr.custom_div_cmap(**kwargs)


def print_matrix(**kwargs):
    pr.print_matrix(**kwargs)


def compute_subelection_by_groups(**kwargs):
    metr.compute_subelection_by_groups(**kwargs)
