
import mapel.voting._metrics as metr
import mapel.voting._elections as ele
import mapel.voting.print as pr
import mapel.voting.other.development as dev
import mapel.voting._features as features


from mapel.voting.objects.ApprovalElectionExperiment import ApprovalElectionExperiment
from mapel.voting.objects.OrdinalElectionExperiment import OrdinalElectionExperiment


def hello():
    print("Hello!")



def prepare_experiment(experiment_id=None, elections=None, distances=None, election_type='ordinal',
                       coordinates=None, distance_name='emd-positionwise', attraction_factor=1):
    if election_type == 'ordinal':
        return OrdinalElectionExperiment("virtual", experiment_id=experiment_id,
                                         elections=elections,
                                         election_type=election_type,
                                         attraction_factor=attraction_factor,
                                         distances=distances, coordinates=coordinates,
                                         distance_name=distance_name)
    elif election_type == 'approval':
        return ApprovalElectionExperiment("virtual", experiment_id=experiment_id,
                                          elections=elections,
                                          election_type=election_type,
                                          attraction_factor=attraction_factor,
                                          distances=distances, coordinates=coordinates,
                                          distance_name=distance_name)

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