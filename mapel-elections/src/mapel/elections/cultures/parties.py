import numpy as np


def prepare_parties(culture_id=None, params=None):
    parties = []
    if culture_id == '2d_gaussian_party':
        for i in range(params['num_parties']):
            point = np.random.rand(1, 2)
            parties.append(point)
    elif culture_id in ['1d_gaussian_party', 'conitzer_party', 'walsh_party']:
        for i in range(params['num_parties']):
            point = np.random.rand(1, 1)
            parties.append(point)
    return parties