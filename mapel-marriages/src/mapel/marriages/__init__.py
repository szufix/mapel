
from .objects.MarriagesExperiment import MarriagesExperiment
from .objects.Marriages import Marriages
from . import cultures_ as cult
from .distances_ import get_distance


def prepare_online_marriages_experiment(**kwargs):
    return prepare_marriages_experiment(**kwargs, is_exported=False, is_imported=False)


def prepare_offline_marriages_experiment(**kwargs):
    return prepare_marriages_experiment(**kwargs, is_exported=True, is_imported=True)


def prepare_marriages_experiment(experiment_id=None,
                                 instance_type=None,
                                 distance_id=None,
                                 is_imported=None,
                                 is_shifted=False,
                                 is_exported=None,
                                 embedding_id=None):

    return MarriagesExperiment(experiment_id=experiment_id,
                               is_imported=is_imported,
                               distance_id=distance_id,
                               instance_type=instance_type,
                               embedding_id=embedding_id,
                               is_shifted=is_shifted,
                               is_exported=is_exported)


def generate_marriages_instance(**kwargs):
    instance = Marriages('virtual', 'tmp', **kwargs)
    instance.prepare_instance()
    return instance


def generate_marriages_votes(**kwargs):
    return cult.generate_votes(**kwargs)


def compute_distance(*args, **kwargs):
    return get_distance(*args, **kwargs)