import numpy as np


def wm_train_predict_R(X_train, a, b, c):
    """
    Train-point predictions for the trainable Whitney-McShane midpoint model
    in the 1D case.

    Model:
        f_theta(x) = 0.5 * [
            min_m { a_m + c_m |x - b_m| }
            +
            max_m { a_m - c_m |x - b_m| }
        ]

    Parameters
    ----------
    X_train : np.ndarray, shape (N,)
        Training inputs.
    a : np.ndarray, shape (M,)
        Height parameters.
    b : np.ndarray, shape (M,)
        Center/location parameters.
    c : np.ndarray, shape (M,)
        Slope/Lipschitz parameters.

    Returns
    -------
    y_hat : np.ndarray, shape (N,)
        Predictions at training points.
    i_active : np.ndarray, shape (N,)
        Active upper-envelope index for each training point.
    j_active : np.ndarray, shape (N,)
        Active lower-envelope index for each training point.
    dists : np.ndarray, shape (N, M)
        Distance matrix |X_n - b_m|.
    """
    X_train = np.asarray(X_train)
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    # dists[n, m] = |X_n - b_m|
    dists = np.abs(X_train[:, None] - b[None, :])

    upper_values = a[None, :] + c[None, :] * dists
    lower_values = a[None, :] - c[None, :] * dists

    i_active = np.argmin(upper_values, axis=1)
    j_active = np.argmax(lower_values, axis=1)

    n_idx = np.arange(len(X_train))

    upper_active = upper_values[n_idx, i_active]
    lower_active = lower_values[n_idx, j_active]

    y_hat = 0.5 * (upper_active + lower_active)

    return y_hat, i_active, j_active, dists


def wm_predict_many_R(X_test, a, b, c):
    """
    Evaluate the trainable Whitney-McShane midpoint model on test points.
    """
    X_test = np.asarray(X_test)

    dists = np.abs(X_test[:, None] - b[None, :])

    upper_values = a[None, :] + c[None, :] * dists
    lower_values = a[None, :] - c[None, :] * dists

    upper = np.min(upper_values, axis=1)
    lower = np.max(lower_values, axis=1)

    return 0.5 * (upper + lower)


def gd_loss_R(X_train, Y_train, a, b, c, X_star, Y_star, L, lam):
    """
    Regularized empirical loss:

        sum_n (f_theta(X_n) - Y_n)^2
        +
        lambda sum_m [
            (b_m - X_m)^2 + (a_m - Y_m)^2 + (c_m - L)^2
        ].
    """
    y_hat, _, _, _ = wm_train_predict_R(X_train, a, b, c)

    data_loss = np.sum((y_hat - Y_train) ** 2)

    reg_loss = lam * np.sum(
        (b - X_star) ** 2
        + (a - Y_star) ** 2
        + (c - L) ** 2
    )

    return data_loss + reg_loss, data_loss, reg_loss


def gd_step_R(X_train, Y_train, a, b, c, X_star, Y_star, L, lam, eta):
    """
    One full-batch subgradient descent step for the 1D model.

    This follows the draft's active-index update rule.

    In 1D:
        d_{n,m} = |X_n - b_m|
        s_{n,m} = sign(X_n - b_m)

    The update uses the active upper index i_n and lower index j_n.
    """
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    N = len(X_train)
    M = len(a)

    y_hat, i_active, j_active, dists = wm_train_predict_R(X_train, a, b, c)
    residuals = y_hat - Y_train

    # s[n, m] = sign(X_n - b_m)
    signs = np.sign(X_train[:, None] - b[None, :])

    grad_a = np.zeros(M)
    grad_b = np.zeros(M)
    grad_c = np.zeros(M)

    for n in range(N):
        r = residuals[n]
        i = i_active[n]
        j = j_active[n]

        # Gradient wrt a
        grad_a[i] += r
        grad_a[j] += r

        # Gradient wrt b
        # Matches the draft formula:
        # r * c_m * (1{j=m} - 1{i=m}) * sign(X_n - b_m)
        grad_b[i] += r * c[i] * (-1.0) * signs[n, i]
        grad_b[j] += r * c[j] * (+1.0) * signs[n, j]

        # Gradient wrt c
        # Matches:
        # r * (1{i=m} - 1{j=m}) * d_{n,m}
        grad_c[i] += r * dists[n, i]
        grad_c[j] += -r * dists[n, j]

    # Regularizer gradients
    grad_a += 2.0 * lam * (a - Y_star)
    grad_b += 2.0 * lam * (b - X_star)
    grad_c += 2.0 * lam * (c - L)

    # GD update
    a_next = a - eta * grad_a
    b_next = b - eta * grad_b
    c_next = c - eta * grad_c

    return a_next, b_next, c_next, y_hat, residuals, i_active, j_active


def train_gd_R(
    X_train,
    Y_train,
    L,
    init_radius=1e-3,
    lam=1.0,
    eta=1e-3,
    num_steps=5000,
    seed=0,
    X_test=None,
    Y_closed_test=None,
    log_every=10,
):
    """
    Run GD initialized near the closed-form parameter configuration

        theta_star = (Y_m, X_m, L)_{m=1}^N.

    Parameters
    ----------
    X_train : np.ndarray, shape (N,)
    Y_train : np.ndarray, shape (N,)
    L : float
        Lipschitz/design constant used in the estimator and regularizer.
    init_radius : float
        Size of random perturbation around theta_star.
    lam : float
        Regularization parameter lambda.
    eta : float
        GD step size.
    num_steps : int
        Number of GD iterations.
    seed : int
        Random seed.
    X_test : np.ndarray or None
        Optional test grid.
    Y_closed_test : np.ndarray or None
        Closed-form predictions on the test grid, used to measure
        function-space distance to the closed-form solution.
    log_every : int
        Store diagnostics every log_every steps.

    Returns
    -------
    result : dict
        Contains final parameters and training history.
    """
    rng = np.random.default_rng(seed)

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    N = len(X_train)

    # Closed-form parameters theta_star
    a_star = Y_train.copy()
    b_star = X_train.copy()
    c_star = L * np.ones(N)

    # Initialize near theta_star
    a = a_star + init_radius * rng.standard_normal(N)
    b = b_star + init_radius * rng.standard_normal(N)
    c = c_star + init_radius * rng.standard_normal(N)

    # Optional: keep slopes positive initially/throughout.
    c = np.maximum(c, 1e-8)

    history = {
        "step": [],
        "loss": [],
        "data_loss": [],
        "reg_loss": [],
        "param_l2_error": [],
        "param_inf_error": [],
        "train_max_residual": [],
        "function_inf_error": [],
        "active_fraction": [],
    }

    for t in range(num_steps + 1):
        if t % log_every == 0:
            loss, data_loss, reg_loss = gd_loss_R(
                X_train, Y_train, a, b, c, b_star, a_star, L, lam
            )

            y_hat, i_active, j_active, _ = wm_train_predict_R(X_train, a, b, c)
            residuals = y_hat - Y_train

            param_l2_error = np.sqrt(
                np.sum((a - a_star) ** 2)
                + np.sum((b - b_star) ** 2)
                + np.sum((c - c_star) ** 2)
            )

            param_inf_error = max(
                np.max(np.abs(a - a_star)),
                np.max(np.abs(b - b_star)),
                np.max(np.abs(c - c_star)),
            )

            active_fraction = np.mean(
                (i_active == np.arange(N)) & (j_active == np.arange(N))
            )

            if X_test is not None and Y_closed_test is not None:
                Y_gd_test = wm_predict_many_R(X_test, a, b, c)
                function_inf_error = np.max(np.abs(Y_gd_test - Y_closed_test))
            else:
                function_inf_error = np.nan

            history["step"].append(t)
            history["loss"].append(loss)
            history["data_loss"].append(data_loss)
            history["reg_loss"].append(reg_loss)
            history["param_l2_error"].append(param_l2_error)
            history["param_inf_error"].append(param_inf_error)
            history["train_max_residual"].append(np.max(np.abs(residuals)))
            history["function_inf_error"].append(function_inf_error)
            history["active_fraction"].append(active_fraction)

        if t == num_steps:
            break

        a, b, c, _, _, _, _ = gd_step_R(
            X_train=X_train,
            Y_train=Y_train,
            a=a,
            b=b,
            c=c,
            X_star=b_star,
            Y_star=a_star,
            L=L,
            lam=lam,
            eta=eta,
        )

        # Keep c positive. This is not essential near theta_star, but helps numerically.
        c = np.maximum(c, 1e-8)

    return {
        "a": a,
        "b": b,
        "c": c,
        "a_star": a_star,
        "b_star": b_star,
        "c_star": c_star,
        "history": history,
    }


def empirical_lipschitz_margin_R(X_train, Y_train, L, eps=1e-12):
    """
    Compute the empirical strict Lipschitz margin

        gamma = min_{m != n} L |X_m - X_n| - |Y_m - Y_n|.

    Assumption 4.7 needs gamma > 0.
    """
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    N = len(X_train)
    gamma = np.inf

    for i in range(N):
        for j in range(i + 1, N):
            dist = abs(X_train[i] - X_train[j])
            if dist > eps:
                gap = L * dist - abs(Y_train[i] - Y_train[j])
                gamma = min(gamma, gap)

    return gamma