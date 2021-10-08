
import mapel.voting._metrics as metr
import mapel.voting._elections as ele
import mapel.voting.print as pr
import mapel.voting.development as dev
import mapel.voting.features as features



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


def compute_spoilers(**kwargs):
    return dev.compute_spoilers(**kwargs)


### WITHOUT EXPERIMENT ###
def generate_election(**kwargs):
    return ele.generate_instances(**kwargs)


def compute_distance(*args, **kwargs):
    return metr.get_distance(*args, **kwargs)

# def shapley(*args, **kwargs):
#     return features.shapley(*args, **kwargs)
#
# def banzhaf(*args, **kwargs):
#     return features.banzhaf(*args, **kwargs)