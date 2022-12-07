
from .objects.RoommatesExperiment import RoommatesExperiment
from .objects.Roommates import Roommates
from . import cultures_ as rom

# from mapel.marriages.objects.MarriagesExperiment import MarriagesExperiment


def hello():
    print("Hello!")


def prepare_experiment(experiment_id=None, instances=None, distances=None, instance_type='ordinal',
                       coordinates=None, distance_id='emd-positionwise', _import=True,
                       shift=False, dim=2, store=True, coordinates_names=None,
                       embedding_id='kamada', fast_import=False):
    if instance_type == 'roommates':
        return RoommatesExperiment(experiment_id=experiment_id, _import=_import,
                                   distance_id=distance_id, instance_type=instance_type,
                                   embedding_id=embedding_id)



# def compute_subelection_by_groups(**kwargs):
#     metr.compute_subelection_by_groups(**kwargs)


def generate_roommates_instance(**kwargs):
    instance = Roommates('virtual', 'tmp', **kwargs)
    instance.prepare_instance()
    return instance


def generate_roommates_votes(**kwargs):
    return rom.generate_votes(**kwargs)
