import numpy as np


def target_smooth(x):
    """
    Example target function on R.
    """
    return np.sin(3 * x)


def target_sobolev(x):
    """
        Example target function on R.
        """
    return np.exp(-x**2)*np.cos(5*x) + np.maximum(0,x) + np.minimum(0,-x**2)


def target_distance_to_point(x, x0=0.0):
    """
    A 1-Lipschitz target with respect to absolute-value distance.
    """
    return abs(x - x0)

def target_smooth_R2(x):
    """
    Example target function on R^2:

        f(x1, x2) = sin(2x1 + x2)

    Works for one point of shape (2,) or many points of shape (n, 2).
    """
    x = np.asarray(x)
    return np.sin(2 * x[..., 0] + x[..., 1])
