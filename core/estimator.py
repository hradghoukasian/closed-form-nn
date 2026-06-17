import numpy as np


def closed_form_predict(x, X_train, Y_train, L, rho):
    """
    Evaluate the noiseless closed-form Whitney-McShane midpoint estimator
    at a single point x.

    Parameters
    ----------
    x : object
        Test input.
    X_train : list or np.ndarray
        Training inputs.
    Y_train : np.ndarray
        Training labels.
    L : float
        Lipschitz constant.
    rho : callable
        Distance function rho(x, y).

    Returns
    -------
    float
        Prediction f_hat(x).
    """
    dists = np.array([rho(x, xi) for xi in X_train])

    lower = np.max(Y_train - L * dists)
    upper = np.min(Y_train + L * dists)

    return 0.5 * (lower + upper)


def closed_form_predict_many(X_test, X_train, Y_train, L, rho):
    """
    Evaluate the estimator on many test points.
    """
    return np.array([
        closed_form_predict(x, X_train, Y_train, L, rho)
        for x in X_test
    ])
