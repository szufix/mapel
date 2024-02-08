from prefsampling.ordinal.euclidean import EuclideanSpace
import prefsampling.ordinal as pref_ordinal


def euclidean_mask(num_voters=None,
                   num_candidates=None,
                   space=None,
                   **kwargs):
    if type(space) is str:
        if space.lower() == 'uniform':
            space = EuclideanSpace.UNIFORM
        elif space.lower() == 'sphere':
            space = EuclideanSpace.SPHERE
        elif space.lower() == 'gaussian':
            space = EuclideanSpace.GAUSSIAN
        elif space.lower() == 'ball':
            space = EuclideanSpace.BALL

    if space is None:
        space = EuclideanSpace.UNIFORM

    return pref_ordinal.euclidean(num_voters=num_voters,
                                  num_candidates=num_candidates,
                                  space=space,
                                  **kwargs)
