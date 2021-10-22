
import mapel.voting.metrics_main as metr
import mapel.voting.elections_main as ele
import mapel.voting._print as pr
import mapel.voting.other.development as dev
import mapel.voting.features_main as features


from mapel.voting.objects.ApprovalElectionExperiment import ApprovalElectionExperiment
from mapel.voting.objects.OrdinalElectionExperiment import OrdinalElectionExperiment


def hello():
    print("Hello!")


def prepare_experiment(experiment_id=None, elections=None, distances=None, election_type='ordinal',
                       coordinates=None, distance_id='emd-positionwise', _import=True,
                       shift=False):
    if election_type == 'ordinal':
        return OrdinalElectionExperiment(experiment_id=experiment_id, shift=shift,
                                         elections=elections,
                                         election_type=election_type,
                                         distances=distances, coordinates=coordinates,
                                         distance_id=distance_id)
    elif election_type == 'approval':
        return ApprovalElectionExperiment(experiment_id=experiment_id,
                                          elections=elections, _import=_import,
                                          election_type=election_type,
                                          distances=distances, coordinates=coordinates,
                                          distance_id=distance_id)

def print_approvals_histogram(*args):
    pr.print_approvals_histogram(*args)


def custom_div_cmap(**kwargs):
    return pr.custom_div_cmap(**kwargs)


def print_matrix(**kwargs):
    pr.print_matrix(**kwargs)


# def compute_subelection_by_groups(**kwargs):
#     metr.compute_subelection_by_groups(**kwargs)


def compute_spoilers(**kwargs):
    return dev.compute_spoilers(**kwargs)


### WITHOUT EXPERIMENT ###
def generate_election(**kwargs):
    return ele.generate_election(**kwargs)


def compute_distance(*args, **kwargs):
    return metr.get_distance(*args, **kwargs)

# def shapley(*args, **kwargs):
#     return features.shapley(*args, **kwargs)
#
# def banzhaf(*args, **kwargs):
#     return features.banzhaf(*args, **kwargs)