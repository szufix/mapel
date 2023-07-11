from .distances_ import get_distance
import mapel.core.printing as pr
from .objects.ApprovalElectionExperiment import ApprovalElectionExperiment
from .objects.OrdinalElection import OrdinalElection
from .objects.ApprovalElection import ApprovalElection
from .objects.OrdinalElectionExperiment import OrdinalElectionExperiment


def prepare_online_ordinal_experiment(**kwargs):
    return prepare_experiment(**kwargs, instance_type='ordinal',
                              is_exported=False, is_imported=False)


def prepare_offline_ordinal_experiment(**kwargs):
    return prepare_experiment(**kwargs, instance_type='ordinal',
                              is_exported=True, is_imported=True)


def prepare_online_approval_experiment(**kwargs):
    return prepare_experiment(**kwargs, instance_type='approval',
                              is_exported=False, is_imported=False)


def prepare_offline_approval_experiment(**kwargs):
    return prepare_experiment(**kwargs, instance_type='approval',
                              is_exported=True, is_imported=True)


def prepare_experiment(experiment_id=None,
                       instances=None,
                       distances=None,
                       instance_type=None,
                       coordinates=None,
                       distance_id=None,
                       is_imported=False,
                       is_shifted=False,
                       is_exported=True,
                       coordinates_names=None,
                       embedding_id=None,
                       fast_import=False,
                       with_matrix=False):
    if instance_type == 'ordinal':
        return OrdinalElectionExperiment(experiment_id=experiment_id,
                                         is_shifted=is_shifted,
                                         instances=instances,
                                         is_exported=is_exported,
                                         is_imported=is_imported,
                                         distances=distances,
                                         coordinates=coordinates,
                                         distance_id=distance_id,
                                         coordinates_names=coordinates_names,
                                         embedding_id=embedding_id,
                                         fast_import=fast_import,
                                         with_matrix=with_matrix,
                                         instance_type=instance_type)
    elif instance_type in ['approval', 'rule']:
        return ApprovalElectionExperiment(experiment_id=experiment_id,
                                          is_shifted=is_shifted,
                                          instances=instances,
                                          is_exported=is_exported,
                                          is_imported=is_imported,
                                          distances=distances,
                                          coordinates=coordinates,
                                          distance_id=distance_id,
                                          coordinates_names=coordinates_names,
                                          embedding_id=embedding_id,
                                          fast_import=fast_import,
                                          instance_type=instance_type)


def print_approvals_histogram(*args):
    pr.print_approvals_histogram(*args)


def custom_div_cmap(**kwargs):
    return pr.custom_div_cmap(**kwargs)


def print_matrix(**kwargs):
    pr.print_matrix(**kwargs)


### WITHOUT EXPERIMENT ###
def generate_election(**kwargs):
    election = OrdinalElection(**kwargs)
    election.prepare_instance()
    return election


def generate_ordinal_election(**kwargs):
    election = OrdinalElection(**kwargs)
    election.prepare_instance()
    return election


def generate_approval_election(**kwargs):
    election = ApprovalElection(**kwargs)
    election.prepare_instance()
    return election


def generate_election_from_votes(votes=None):
    election = OrdinalElection()
    election.num_candidates = len(votes[0])
    election.num_voters = len(votes)
    election.votes = votes
    return election


def generate_ordinal_election_from_votes(votes=None):
    election = OrdinalElection()
    election.num_candidates = len(votes[0])
    election.num_voters = len(votes)
    election.votes = votes
    return election


def generate_approval_election_from_votes(votes=None):
    election = ApprovalElection()
    election.num_candidates = len(set().union(*votes))
    election.num_voters = len(votes)
    election.votes = votes
    return election


def compute_distance(*args, **kwargs):
    return get_distance(*args, **kwargs)


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 11.07.2023 #
# # # # # # # # # # # # # # # #
