import numpy as np


def abs_distance(x, y):
    """
    Distance on R.
    """
    return abs(x - y)


def estimate_lipschitz_R(X, Y, eps=1e-12):
    """
    Estimate the Lipschitz constant on R from training data.

    Uses:
        L_hat = max_{i != j} |Y_i - Y_j| / |X_i - X_j|

    Since X is one-dimensional, after sorting it is enough to check
    neighboring points.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    order = np.argsort(X)
    X_sorted = X[order]
    Y_sorted = Y[order]

    dx = np.diff(X_sorted)
    dy = np.diff(Y_sorted)

    valid = np.abs(dx) > eps

    if not np.any(valid):
        raise ValueError("Cannot estimate Lipschitz constant: all X values are identical or too close.")

    slopes = np.abs(dy[valid] / dx[valid])

    return np.max(slopes)


def l2_distance(x, y):
    """
    Euclidean distance on R^d.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.linalg.norm(x - y)

def l1_distance(x, y):
    """
    L1 distance on R^d.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum(np.abs(x - y))


def estimate_lipschitz_pairwise(X, Y, rho, eps=1e-12):
    """
    Estimate Lipschitz constant from data using

        L_hat = max_{i != j} |Y_i - Y_j| / rho(X_i, X_j)

    This works for general metric spaces.
    """
    X = list(X)
    Y = np.asarray(Y)

    n = len(X)
    L_hat = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dist = rho(X[i], X[j])

            if dist > eps:
                slope = abs(Y[i] - Y[j]) / dist
                L_hat = max(L_hat, slope)

    return L_hat

