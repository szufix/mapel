import mapel.elections.features_ as features
import mapel.elections.metrics_ as metr
import mapel.elections.models_ as ele
import mapel.elections.other.development as dev
import mapel.elections.print_ as pr
from mapel.elections.objects.ApprovalElectionExperiment import ApprovalElectionExperiment
from mapel.elections.objects.OrdinalElection import OrdinalElection
from mapel.elections.objects.ApprovalElection import ApprovalElection
from mapel.elections.objects.OrdinalElectionExperiment import OrdinalElectionExperiment

try:
    from mapel.roommates.objects.RoommatesExperiment import RoommatesExperiment
    from mapel.roommates.objects.Roommates import Roommates
    from mapel.marriages.objects.MarriagesExperiment import MarriagesExperiment
    import mapel.roommates.models_ as rom
except:
    pass


def hello():
    print("Hello!")


def prepare_experiment(experiment_id=None, instances=None, distances=None, instance_type='ordinal',
                       coordinates=None, distance_id='emd-positionwise', _import=True,
                       shift=False, dim=2, store=True, coordinates_names=None,
                       embedding_id='kamada', fast_import=False):
    if instance_type == 'ordinal':
        return OrdinalElectionExperiment(experiment_id=experiment_id, shift=shift,
                                         instances=instances, dim=dim, store=store,
                                         instance_type=instance_type,
                                         distances=distances, coordinates=coordinates,
                                         distance_id=distance_id,
                                         coordinates_names=coordinates_names,
                                         embedding_id=embedding_id,
                                         fast_import=fast_import)
    elif instance_type in ['approval', 'rule']:
        return ApprovalElectionExperiment(experiment_id=experiment_id, shift=shift,
                                          instances=instances, dim=dim, store=store,
                                          instance_type=instance_type,
                                          distances=distances, coordinates=coordinates,
                                          distance_id=distance_id,
                                          coordinates_names=coordinates_names,
                                          embedding_id=embedding_id,
                                          fast_import=fast_import)
    elif instance_type == 'roommates':
        return RoommatesExperiment(experiment_id=experiment_id, _import=_import,
                                   distance_id=distance_id, instance_type=instance_type)

    elif instance_type == 'marriages':
        return MarriagesExperiment(experiment_id=experiment_id, _import=_import,
                                   distance_id=distance_id, instance_type=instance_type)


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


def generate_roommates_instance(**kwargs):
    instance = Roommates('virtual', 'tmp', **kwargs)
    instance.prepare_instance()
    return instance


def generate_roommates_votes(**kwargs):
    return rom.generate_votes(**kwargs)


def compute_distance(*args, **kwargs):
    return metr.get_distance(*args, **kwargs)
