
import numpy as np
import numpy.linalg as la
try:
    import pandas as pd
except:
    pass


def min_dim(election):
    if election.fake:
        return 'None'

    election.set_default_object_type('vote')
    election.compute_distances(distance_id='swap')

    df = election.distances['vote']
    df = pd.DataFrame(df, columns=[str(i) for i in range(election.num_voters)])

    # get some distance matrix
    # df = pd.read_csv("tmp.csv")
    A = df.values.T.astype(float)
    # square it
    A = A ** 2

    # centering matrix
    n = A.shape[0]
    J_c = 1. / n * (np.eye(n) - 1 + (n - 1) * np.eye(n))

    # perform double centering
    B = -0.5 * (J_c.dot(A)).dot(J_c)

    # find eigenvalues and eigenvectors
    eigen_val = la.eig(B)[0]
    eigen_vec = la.eig(B)[1].T

    num_zeros = 0
    for i, _ in enumerate(eigen_val):
        if eigen_val[i] > -0.00001 and eigen_val[i] < 0.00001:
            num_zeros += 1

    return election.num_voters-num_zeros