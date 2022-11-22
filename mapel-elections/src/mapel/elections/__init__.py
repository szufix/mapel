from .metrics_ import get_distance
from .other.development import compute_spoilers
import mapel.core.printing as pr
from .objects.ApprovalElectionExperiment import ApprovalElectionExperiment
from .objects.OrdinalElection import OrdinalElection
from .objects.ApprovalElection import ApprovalElection
from .objects.OrdinalElectionExperiment import OrdinalElectionExperiment


def prepare_online_ordinal_experiment(**kwargs):
    return prepare_experiment(**kwargs, instance_type='ordinal', store=False)

def prepare_offline_ordinal_experiment(**kwargs):
    return prepare_experiment(**kwargs, instance_type='ordinal', store=True)


def prepare_online_approval_experiment(**kwargs):
    return prepare_experiment(**kwargs, instance_type='approval', store=False)


def prepare_offline_approval_experiment(**kwargs):
    return prepare_experiment(**kwargs, instance_type='approval', store=True)


def prepare_experiment(experiment_id=None, instances=None, distances=None, instance_type='ordinal',
                       coordinates=None, distance_id='emd-positionwise', _import=True,
                       shift=False, dim=2, store=True, coordinates_names=None,
                       embedding_id='spring', fast_import=False, with_matrix=False):
    if instance_type == 'ordinal':
        return OrdinalElectionExperiment(experiment_id=experiment_id, shift=shift,
                                         instances=instances, dim=dim, store=store,
                                         instance_type=instance_type,
                                         distances=distances, coordinates=coordinates,
                                         distance_id=distance_id,
                                         coordinates_names=coordinates_names,
                                         embedding_id=embedding_id,
                                         fast_import=fast_import,
                                         with_matrix=with_matrix)
    elif instance_type in ['approval', 'rule']:
        return ApprovalElectionExperiment(experiment_id=experiment_id, shift=shift,
                                          instances=instances, dim=dim, store=store,
                                          instance_type=instance_type,
                                          distances=distances, coordinates=coordinates,
                                          distance_id=distance_id,
                                          coordinates_names=coordinates_names,
                                          embedding_id=embedding_id,
                                          fast_import=fast_import)

def print_approvals_histogram(*args):
    pr.print_approvals_histogram(*args)


def custom_div_cmap(**kwargs):
    return pr.custom_div_cmap(**kwargs)


def print_matrix(**kwargs):
    pr.print_matrix(**kwargs)

def compute_spoilers(**kwargs):
    return compute_spoilers(**kwargs)


### WITHOUT EXPERIMENT ###
def generate_election(**kwargs):
    election = OrdinalElection("virtual", "virtual", **kwargs)
    election.prepare_instance()
    return election

def generate_ordinal_election(**kwargs):
    election = OrdinalElection("virtual", "virtual", **kwargs)
    election.prepare_instance()
    return election

def generate_approval_election(**kwargs):
    election = ApprovalElection("virtual", "virtual", **kwargs)
    election.prepare_instance()
    return election

def generate_election_from_votes(votes=None):
    election = OrdinalElection("virtual", "virtual")
    election.num_candidates = len(votes[0])
    election.num_voters = len(votes)
    election.votes = votes
    return election

def compute_distance(*args, **kwargs):
    return get_distance(*args, **kwargs)
