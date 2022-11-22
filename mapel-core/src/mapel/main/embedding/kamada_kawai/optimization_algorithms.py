import collections

import numpy as np

from mapel.core.embedding.kamada_kawai.energy_functions import get_energy_dx, get_energy_dy, get_energy_dx_dx, \
    get_energy_dx_dy, get_energy_dy_dx, get_energy_dy_dy


def optimize_bb(func, grad_func, args, x0, max_iter, init_step_size, stop_energy_val=None,
                max_iter_without_improvement=8000, min_improvement_percentage=1.0, percentage_lookup_history=100):
    if isinstance(init_step_size, float):
        init_step_size = [init_step_size, init_step_size]

    init_step_size = np.asarray(init_step_size)
    is_2d = len(x0.shape) == 2

    prev_x = x0.copy()
    x = x0.copy()
    prev_grad = grad_func(prev_x, *args)

    min_energy = 1e15
    min_energy_snap = x0.copy()
    min_energy_iter = 0

    energy_history = collections.deque(maxlen=percentage_lookup_history)

    for i in range(max_iter):
        if i%100 == 0:
            print(f'{i} iterations')

        current_energy = func(x, *args)
        if current_energy < min_energy:
            if len(energy_history) == percentage_lookup_history:
                percentage = current_energy / min_energy
                if 1 - percentage < min_improvement_percentage:
                    return x.copy()

            min_energy = current_energy
            min_energy_snap = x.copy()
            min_energy_iter = i
        elif i - min_energy_iter > max_iter_without_improvement:
            print(f'More than {max_iter_without_improvement} iterations without improvement')
            return min_energy_snap

        # print(f'Energy: {current_energy}: {min_energy}, grad norm: {np.linalg.norm(prev_grad)} {i}')
        if stop_energy_val is not None and current_energy < stop_energy_val:
            return min_energy_snap
        s = x - prev_x
        g = grad_func(x, *args)
        y = g - prev_grad

        if i > 0:
            denominator = abs(np.tensordot(s, y, [0, 0]))
            if is_2d:
                denominator = denominator.diagonal()
            step_size = np.linalg.norm(s, axis=0) ** 2 / denominator
        else:
            step_size = init_step_size

        prev_grad = g
        prev_x = x
        x = x - step_size * g
        energy_history.append(min_energy)

    return min_energy_snap


def _get_delta_energy(positions, k, l, x, y):
    return np.sqrt(get_energy_dx(x, y, k, l, positions) ** 2 + get_energy_dy(x, y, k, l, positions) ** 2)


def _get_pos_k_l_x_y_for_i(positions, k, l, i):
    my_k = np.delete(k[i, :], i)
    my_l = np.delete(l[i, :], i)
    my_positions = np.delete(positions, i, axis=0)

    my_x = positions[i, 0]
    my_y = positions[i, 1]

    return my_positions, my_k, my_l, my_x, my_y


def _optimize_newton(positions, k, l, i, eps=1e-10):
    positions, k, l, x, y = _get_pos_k_l_x_y_for_i(positions, k, l, i)

    delta = _get_delta_energy(positions, k, l, x, y)
    i = 0
    while delta > eps:
        a1 = get_energy_dx_dx(x, y, k, l, positions)
        b1 = get_energy_dx_dy(x, y, k, l, positions)
        c1 = -get_energy_dx(x, y, k, l, positions)

        a2 = get_energy_dy_dx(x, y, k, l, positions)
        b2 = get_energy_dy_dy(x, y, k, l, positions)
        c2 = -get_energy_dy(x, y, k, l, positions)

        dx, dy = np.linalg.solve([[a1, b1], [a2, b2]], [c1, c2])

        x += dx
        y += dy

        if i > 1e4:
            return (x, y), False

        delta = _get_delta_energy(positions, k, l, x, y)
        i += 1
    return (x, y), True


def adam(
        fun,
        jac,
        x0,
        args=(),
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        startiter=0,
        maxiter=1000,
        callback=None,
        **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    best_energy = fun(x, *args), x
    for i in range(startiter, startiter + maxiter):
        energy = fun(x, *args)
        if energy < best_energy[0]:
            best_energy = energy, np.copy(x)

        g = jac(x, *args)

        if callback and callback(x):
            break

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g ** 2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1 ** (i + 1))  # bias correction.
        vhat = v / (1 - beta2 ** (i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

        if i == (startiter + maxiter) // 2:
            learning_rate /= 2

    return best_energy[1]
