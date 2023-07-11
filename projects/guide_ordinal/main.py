from datetime import time

from matplotlib.transforms import Bbox

import mapel.elections as mapel
import itertools
from scipy.stats import stats
import matplotlib.pyplot as plt
import numpy as np
import math

from print import *


if __name__ == "__main__":

    object_type = 'vote'
    size = '10x100'
    culture_id = 'urn'

    experiment_id = f'microscope/{size}/{culture_id}'
    distance_id = 'emd-positionwise'

    if object_type == 'vote':
        distance_id = 'swap'
    elif object_type == 'candidate':
        distance_id = 'position'

    experiment = mapel.prepare_offline_ordinal_experiment(experiment_id=experiment_id,
                                                          distance_id=distance_id,
                                                          )

    # experiment.prepare_elections()
    #
    # for election in experiment.instances.values():
    #     print(election.label)
    #     election.set_default_object_type(object_type)
    #     election.compute_distances(object_type=object_type, distance_id=distance_id)
    #     election.embed(object_type=object_type)
    # # #
    if size == '10x100' and object_type == 'vote':
        print_10x100_vote(experiment)
    elif size == '10x1000' and object_type == 'vote':
        print_10x1000_vote(experiment)
    elif size == '10x100' and object_type == 'candidate':
        print_10x100_candidate(experiment)
    elif size == '100x100' and object_type == 'candidate':
        print_100x100_candidate(experiment)

    experiment.merge_election_images(name=experiment_id, show=False, ncol=5, nrow=1,
                                     object_type=object_type)
