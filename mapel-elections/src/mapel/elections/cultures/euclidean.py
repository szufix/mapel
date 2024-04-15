import prefsampling.ordinal as pref_ordinal
import prefsampling.approval as pref_approval

from prefsampling.point import cube, sphere_uniform, gaussian, ball_uniform


def euclidean_ord_mask(num_voters=None,
                       num_candidates=None,
                       space=None,
                       dim=2,
                       **kwargs):

    num_dimensions = dim

    if type(space) is str:
        if space.lower() == 'uniform':
            point_sampler = cube
        elif space.lower() == 'sphere':
            point_sampler = sphere_uniform
        elif space.lower() == 'gaussian':
            point_sampler = gaussian
        elif space.lower() == 'ball':
            point_sampler = ball_uniform

    if space is None:
        point_sampler = cube

    return pref_ordinal.euclidean(
        num_voters=num_voters,
        num_candidates=num_candidates,
        point_sampler=point_sampler,
        point_sampler_args={
                            'num_dimensions': num_dimensions,
                            'center_point': [0 for _ in range(num_dimensions)]},
        **kwargs)


def euclidean_app_mask(num_voters=None,
                       num_candidates=None,
                       space=None,
                       dim=2,
                       radius=0.2,
                       **kwargs):

    num_dimensions = dim

    if type(space) is str:
        if space.lower() == 'uniform':
            point_sampler = cube
        elif space.lower() == 'sphere':
            point_sampler = sphere_uniform
        elif space.lower() == 'gaussian':
            point_sampler = gaussian
        elif space.lower() == 'ball':
            point_sampler = ball_uniform

    if space is None:
        point_sampler = cube

    return pref_approval.euclidean_vcr(
        num_voters=num_voters,
        num_candidates=num_candidates,
        voters_radius=radius,
        candidates_radius=0,
        point_sampler=point_sampler,
        point_sampler_args={
                            'num_dimensions': num_dimensions,
                            'center_point': [0 for _ in range(num_dimensions)]},
        **kwargs)
