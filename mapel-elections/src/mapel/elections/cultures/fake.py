
from mapel.elections.cultures.mallows import get_mallows_vectors
from mapel.elections.cultures.sp_matrices import get_walsh_vectors, get_conitzer_vectors
from mapel.elections.cultures.params import *


def get_fake_multiplication(num_candidates, params, model):
    params['weight'] = 0.
    params['normphi'] = params['alpha']
    main_matrix = []
    if model == 'conitzer_path':
        main_matrix = get_conitzer_vectors(num_candidates).transpose()
    elif model == 'walsh_path':
        main_matrix = get_walsh_vectors(num_candidates).transpose()
    mallows_matrix = get_mallows_vectors(num_candidates, params).transpose()
    output = np.matmul(main_matrix, mallows_matrix).transpose()
    return output


def get_fake_vectors_single(fake_model_name, num_candidates):
    vectors = np.zeros([num_candidates, num_candidates])

    if fake_model_name == 'identity':
        for i in range(num_candidates):
            vectors[i][i] = 1

    elif fake_model_name == 'uniformity':
        for i in range(num_candidates):
            for j in range(num_candidates):
                vectors[i][j] = 1. / num_candidates

    elif fake_model_name == 'stratification':
        half = int(num_candidates / 2)
        for i in range(half):
            for j in range(half):
                vectors[i][j] = 1. / half
        for i in range(half, num_candidates):
            for j in range(half, num_candidates):
                vectors[i][j] = 1. / half

    elif fake_model_name == 'antagonism':
        for i in range(num_candidates):
            for _ in range(num_candidates):
                vectors[i][i] = 0.5
                vectors[i][num_candidates - i - 1] = 0.5

    return vectors


def get_fake_vectors_crate(num_candidates=None, fake_param=None):
    base_1 = get_fake_vectors_single('uniformity', num_candidates)
    base_2 = get_fake_vectors_single('identity', num_candidates)
    base_3 = get_fake_vectors_single('antagonism', num_candidates)
    base_4 = get_fake_vectors_single('stratification', num_candidates)

    return crate_combination(base_1, base_2, base_3, base_4, length=num_candidates,
                             alpha=fake_param)


def get_fake_convex(fake_model_name, num_candidates, num_voters, params, function_name):
    if fake_model_name == 'unid':
        base_1 = function_name('uniformity', num_candidates)
        base_2 = function_name('identity', num_candidates)
    elif fake_model_name == 'anid':
        base_1 = function_name('antagonism', num_candidates)
        base_2 = function_name('identity', num_candidates)
    elif fake_model_name == 'stid':
        base_1 = function_name('stratification', num_candidates)
        base_2 = function_name('identity', num_candidates)
    elif fake_model_name == 'anun':
        base_1 = function_name('antagonism', num_candidates)
        base_2 = function_name('uniformity', num_candidates)
    elif fake_model_name == 'stun':
        base_1 = function_name('stratification', num_candidates)
        base_2 = function_name('uniformity', num_candidates)
    elif fake_model_name == 'stan':
        base_1 = function_name('stratification', num_candidates)
        base_2 = function_name('antagonism', num_candidates)
    else:
        raise NameError('No such fake vectors/matrix!')

    return convex_combination(base_1, base_2, length=num_candidates, params=params)


def convex_combination(base_1, base_2, length=0, params=None):
    alpha = params['alpha']
    if base_1.ndim == 1:
        output = np.zeros([length])
        for i in range(length):
            output[i] = alpha * base_1[i] + (1 - alpha) * base_2[i]
    elif base_1.ndim == 2:
        output = np.zeros([length, length])
        for i in range(length):
            for j in range(length):
                output[i][j] = alpha * base_1[i][j] + (1 - alpha) * base_2[i][j]
    else:
        raise NameError('Unknown base!')
    return output


def crate_combination(base_1, base_2, base_3, base_4, length=0, alpha=None):
    alpha = alpha['alpha']
    vectors = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            vectors[i][j] = alpha[0] * base_1[i][j] + \
                            alpha[1] * base_2[i][j] + \
                            alpha[2] * base_3[i][j] + \
                            alpha[3] * base_4[i][j]

    return vectors


def get_fake_matrix_single(fake_model_name, num_candidates):
    matrix = np.zeros([num_candidates, num_candidates])

    if fake_model_name == 'identity':
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                matrix[i][j] = 1

    elif fake_model_name in {'uniformity', 'antagonism'}:
        for i in range(num_candidates):
            for j in range(num_candidates):
                if i != j:
                    matrix[i][j] = 0.5

    elif fake_model_name == 'stratification':
        for i in range(int(num_candidates / 2)):
            for j in range(int(num_candidates / 2), num_candidates):
                matrix[i][j] = 1
        for i in range(int(num_candidates / 2)):
            for j in range(int(num_candidates / 2)):
                if i != j:
                    matrix[i][j] = 0.5
        for i in range(int(num_candidates / 2), num_candidates):
            for j in range(int(num_candidates / 2), num_candidates):
                if i != j:
                    matrix[i][j] = 0.5

    return matrix


def get_fake_borda_vector(fake_model_name, num_candidates, num_voters):
    borda_vector = np.zeros([num_candidates])

    m = num_candidates
    n = num_voters

    if fake_model_name == 'identity':
        for i in range(m):
            borda_vector[i] = n * (m - 1 - i)

    elif fake_model_name in {'uniformity', 'antagonism'}:
        for i in range(m):
            borda_vector[i] = n * (m - 1) / 2

    elif fake_model_name == 'stratification':
        for i in range(int(m / 2)):
            borda_vector[i] = n * (m - 1) * 3 / 4
        for i in range(int(m / 2), m):
            borda_vector[i] = n * (m - 1) / 4

    return borda_vector
