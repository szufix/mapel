
import mapel.elections.metrics_main as metr
import mapel.elections.models_main as ele
import mapel.roommates.models_main as rom
import mapel.elections._print as pr
import mapel.elections.other.development as dev
import mapel.elections.features_main as features

from mapel.elections.objects.ApprovalElectionExperiment import ApprovalElectionExperiment
from mapel.elections.objects.OrdinalElectionExperiment import OrdinalElectionExperiment
from mapel.roommates.objects.RoommatesExperiment import RoommatesExperiment
from mapel.roommates.objects.Roommates import Roommates
from mapel.marriages.objects.MarriagesExperiment import MarriagesExperiment


def hello():
    print("Hello!")


def prepare_experiment(experiment_id=None, instances=None, distances=None, instance_type='ordinal',
                       coordinates=None, distance_id='emd-positionwise', _import=True,
                       shift=False, dim=2, store=True, coordinates_names=None):

    if instance_type == 'ordinal':
        return OrdinalElectionExperiment(experiment_id=experiment_id, shift=shift,
                                         instances=instances, dim=dim, store=store,
                                         instance_type=instance_type,
                                         distances=distances, coordinates=coordinates,
                                         distance_id=distance_id,
                                         coordinates_names=coordinates_names)
    elif instance_type in ['approval', 'rule']:
        return ApprovalElectionExperiment(experiment_id=experiment_id, shift=shift,
                                          instances=instances, _import=_import,
                                          instance_type=instance_type,
                                          distances=distances, coordinates=coordinates,
                                          distance_id=distance_id)
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
    return ele.generate_election(**kwargs)


def generate_roommates_instance(**kwargs):
    instance = Roommates('virtual', 'tmp', **kwargs)
    instance.prepare_instance()
    return instance


def generate_roommates_votes(**kwargs):
    return rom.generate_votes(**kwargs)

def compute_distance(*args, **kwargs):
    return metr.get_distance(*args, **kwargs)
