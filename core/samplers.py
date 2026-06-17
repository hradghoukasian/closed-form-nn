import numpy as np


def sample_uniform_R(n, low=-1.0, high=1.0, seed=None):
    """
    Sample n points uniformly from a compact interval [low, high].
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(low, high, size=n)
    return np.sort(X)


def make_grid_R(n, low=-1.0, high=1.0):
    """
    Make a fixed test grid on [low, high].
    """
    return np.linspace(low, high, n)

def sample_uniform_R2(n, low=-1.0, high=1.0, seed=None):
    """
    Sample n points uniformly from [low, high]^2.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 2))


def make_grid_R2(n_per_axis, low=-1.0, high=1.0):
    """
    Make a 2D grid on [low, high]^2.

    Returns
    -------
    X_grid : np.ndarray, shape (n_per_axis^2, 2)
        Flattened grid points.
    X1 : np.ndarray
        Meshgrid x-coordinates.
    X2 : np.ndarray
        Meshgrid y-coordinates.
    """
    x1 = np.linspace(low, high, n_per_axis)
    x2 = np.linspace(low, high, n_per_axis)

    X1, X2 = np.meshgrid(x1, x2)

    X_grid = np.column_stack([X1.ravel(), X2.ravel()])

    return X_grid, X1, X2
