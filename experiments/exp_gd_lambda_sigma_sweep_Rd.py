
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Global plot style
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Metrics on R^d
# ============================================================

def l2_distance_many(X, B):
    """
    Pairwise l2 distances.

    X: shape (N, d)
    B: shape (M, d)

    returns D: shape (N, M), D[n,m] = ||X_n - B_m||_2
    """
    diff = X[:, None, :] - B[None, :, :]
    return np.linalg.norm(diff, axis=2)


def l1_distance_many(X, B):
    """
    Pairwise l1 distances.

    X: shape (N, d)
    B: shape (M, d)

    returns D: shape (N, M), D[n,m] = ||X_n - B_m||_1
    """
    diff = X[:, None, :] - B[None, :, :]
    return np.sum(np.abs(diff), axis=2)


def distance_many(X, B, metric):
    if metric == "l2":
        return l2_distance_many(X, B)
    elif metric == "l1":
        return l1_distance_many(X, B)
    else:
        raise ValueError("metric must be either 'l1' or 'l2'")


def grad_distance_wrt_b(X, B, metric, eps=1e-12):
    """
    Gradient/subgradient of rho(X_n, B_m) with respect to B_m.

    returns G: shape (N, M, d)
    """
    diff = X[:, None, :] - B[None, :, :]

    if metric == "l2":
        dists = np.linalg.norm(diff, axis=2)
        denom = np.maximum(dists, eps)

        # grad_b ||X - b||_2 = (b - X) / ||X - b||_2
        G = -diff / denom[:, :, None]
        return G

    elif metric == "l1":
        # subgrad_b ||X - b||_1 = sign(b - X)
        G = np.sign(B[None, :, :] - X[:, None, :])
        return G

    else:
        raise ValueError("metric must be either 'l1' or 'l2'")


# ============================================================
# Random exactly 1-Lipschitz tanh target
# ============================================================

def make_random_1_lip_tanh_target(
    d,
    metric="l2",
    low=-1.0,
    high=1.0,
    seed=None,
):
    """
    Generate

        f(x) = tanh(W_norm^T x + b)

    so that f is exactly 1-Lipschitz with respect to the chosen metric.

    For l2:
        ||W_norm||_2 = 1.

    For l1:
        ||W_norm||_infty = 1.

    We choose b = - W_norm^T x0, so tanh' reaches its maximum
    value 1 at x0.
    """
    rng = np.random.default_rng(seed)

    W = rng.standard_normal(d)

    if metric == "l2":
        W = W / np.linalg.norm(W, ord=2)

    elif metric == "l1":
        W = W / np.max(np.abs(W))

    else:
        raise ValueError("metric must be either 'l1' or 'l2'")

    # Use an interior point so that the point of maximum slope is inside the domain.
    margin = 0.1 * (high - low)
    x0 = rng.uniform(low + margin, high - margin, size=d)

    b = -np.dot(W, x0)

    def f(X):
        X = np.asarray(X)
        return np.tanh(X @ W + b)

    return f, W, b, x0


# ============================================================
# Whitney--McShane trainable model on R^d
# ============================================================

def wm_train_predict_Rd(X_train, a, b, c, metric):
    """
    Train-point predictions for

        f_theta(x) = 1/2 [
            min_m {a_m + c_m rho(x,b_m)}
            +
            max_m {a_m - c_m rho(x,b_m)}
        ]

    X_train: shape (N, d)
    a: shape (M,)
    b: shape (M, d)
    c: shape (M,)
    """
    dists = distance_many(X_train, b, metric)

    upper_values = a[None, :] + c[None, :] * dists
    lower_values = a[None, :] - c[None, :] * dists

    i_active = np.argmin(upper_values, axis=1)
    j_active = np.argmax(lower_values, axis=1)

    n_idx = np.arange(X_train.shape[0])

    upper_active = upper_values[n_idx, i_active]
    lower_active = lower_values[n_idx, j_active]

    y_hat = 0.5 * (upper_active + lower_active)

    return y_hat, i_active, j_active, dists


def gd_loss_Rd(X_train, Y_train, a, b, c, a_star, b_star, c_star, lam, metric):
    y_hat, _, _, _ = wm_train_predict_Rd(X_train, a, b, c, metric)

    data_loss = np.sum((y_hat - Y_train) ** 2)

    reg_loss = lam * (
        np.sum((a - a_star) ** 2)
        + np.sum((b - b_star) ** 2)
        + np.sum((c - c_star) ** 2)
    )

    return data_loss + reg_loss, data_loss, reg_loss


def gd_step_Rd(
    X_train,
    Y_train,
    a,
    b,
    c,
    a_star,
    b_star,
    c_star,
    lam,
    eta,
    metric,
):
    """
    One full-batch subgradient step.
    """
    N = X_train.shape[0]
    M = len(a)
    d = X_train.shape[1]

    y_hat, i_active, j_active, dists = wm_train_predict_Rd(
        X_train, a, b, c, metric
    )
    residuals = y_hat - Y_train

    grad_dist = grad_distance_wrt_b(X_train, b, metric)

    grad_a = np.zeros(M)
    grad_b = np.zeros((M, d))
    grad_c = np.zeros(M)

    for n in range(N):
        r = residuals[n]
        i = i_active[n]
        j = j_active[n]

        # Because y_hat = 0.5 * (upper + lower), each active
        # envelope contributes a factor 1/2 to the gradient.
        grad_a[i] += r
        grad_a[j] += r

        grad_b[i] += r * c[i] * grad_dist[n, i, :]
        grad_b[j] += -r * c[j] * grad_dist[n, j, :]

        grad_c[i] += r * dists[n, i]
        grad_c[j] += -r * dists[n, j]

    # Regularization gradients
    grad_a += 2.0 * lam * (a - a_star)
    grad_b += 2.0 * lam * (b - b_star)
    grad_c += 2.0 * lam * (c - c_star)

    a_next = a - eta * grad_a
    b_next = b - eta * grad_b
    c_next = c - eta * grad_c

    # Keep slopes positive.
    c_next = np.maximum(c_next, 1e-8)

    return a_next, b_next, c_next


def parameter_errors(a, b, c, a_star, b_star, c_star):
    """
    Compute l1 and l2 parameter errors.
    """
    err_a = a - a_star
    err_b = b - b_star
    err_c = c - c_star

    l1_error = (
        np.sum(np.abs(err_a))
        + np.sum(np.abs(err_b))
        + np.sum(np.abs(err_c))
    )

    l2_error = np.sqrt(
        np.sum(err_a ** 2)
        + np.sum(err_b ** 2)
        + np.sum(err_c ** 2)
    )

    return l1_error, l2_error


def train_gd_Rd(
    X_train,
    Y_train,
    metric,
    lam,
    eta,
    num_steps,
    init_sigma,
    seed,
):
    """
    Train GD initialized near theta_star = (Y_m, X_m, 1)_m.
    """
    rng = np.random.default_rng(seed)

    N, d = X_train.shape

    # theta_star = (Y_m, X_m, 1)_m
    a_star = Y_train.copy()
    b_star = X_train.copy()
    c_star = np.ones(N)

    # initialization near theta_star
    a = a_star + init_sigma * rng.standard_normal(N)
    b = b_star + init_sigma * rng.standard_normal((N, d))
    c = c_star + init_sigma * rng.standard_normal(N)
    c = np.maximum(c, 1e-8)

    for _ in range(num_steps):
        a, b, c = gd_step_Rd(
            X_train=X_train,
            Y_train=Y_train,
            a=a,
            b=b,
            c=c,
            a_star=a_star,
            b_star=b_star,
            c_star=c_star,
            lam=lam,
            eta=eta,
            metric=metric,
        )

    l1_error, l2_error = parameter_errors(
        a, b, c, a_star, b_star, c_star
    )

    return {
        "a": a,
        "b": b,
        "c": c,
        "a_star": a_star,
        "b_star": b_star,
        "c_star": c_star,
        "l1_error": l1_error,
        "l2_error": l2_error,
    }


# ============================================================
# Shared statistics and plotting
# ============================================================

def mean_and_ci(values):
    """
    values shape: (num_trials, num_grid_values)

    returns mean, lower, upper using 95% normal CI.
    """
    values = np.asarray(values)

    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1)
    se = std / np.sqrt(values.shape[0])

    ci = 1.96 * se

    lower = mean - ci
    upper = mean + ci

    # Avoid negative lower bounds when plotting on log scale.
    lower = np.maximum(lower, 1e-16)

    return mean, lower, upper


def plot_lambda_sweep(lambdas, l1_all, l2_all, init_sigma, save_path):
    """
    Single plot with l1 and l2 parameter errors versus lambda,
    with 95% confidence intervals.
    """
    lambdas = np.asarray(lambdas)

    l1_mean, l1_lower, l1_upper = mean_and_ci(l1_all)
    l2_mean, l2_lower, l2_upper = mean_and_ci(l2_all)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        lambdas,
        l1_mean,
        marker="o",
        linewidth=2.5,
        label=r"$\|\theta_T-\theta_\star\|_1$",
    )
    ax.fill_between(lambdas, l1_lower, l1_upper, alpha=0.2)

    ax.plot(
        lambdas,
        l2_mean,
        marker="o",
        linewidth=2.5,
        label=r"$\|\theta_T-\theta_\star\|_2$",
    )
    ax.fill_between(lambdas, l2_lower, l2_upper, alpha=0.2)

    ax.set_xlabel(r"regularization parameter $\lambda$")
    ax.set_ylabel(r"final parameter error")
    ax.set_title(rf"GD recovery for fixed $\sigma={init_sigma}$")

    # Allows lambda = 0 while keeping log-like scale for positive lambda.
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_yscale("log")

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_sigma_sweep(sigmas, l1_all, l2_all, fixed_lambda, save_path):
    """
    Single plot with l1 and l2 parameter errors versus sigma,
    with 95% confidence intervals.
    """
    sigmas = np.asarray(sigmas)

    l1_mean, l1_lower, l1_upper = mean_and_ci(l1_all)
    l2_mean, l2_lower, l2_upper = mean_and_ci(l2_all)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        sigmas,
        l1_mean,
        marker="o",
        linewidth=2.5,
        label=r"$\|\theta_T-\theta_\star\|_1$",
    )
    ax.fill_between(sigmas, l1_lower, l1_upper, alpha=0.2)

    ax.plot(
        sigmas,
        l2_mean,
        marker="o",
        linewidth=2.5,
        label=r"$\|\theta_T-\theta_\star\|_2$",
    )
    ax.fill_between(sigmas, l2_lower, l2_upper, alpha=0.2)

    ax.set_xlabel(r"initialization gap $\sigma$")
    ax.set_ylabel(r"final parameter error")
    ax.set_title(rf"GD recovery for fixed $\lambda={fixed_lambda}$")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Lambda sweep
# ============================================================

def run_one_trial_lambda_sweep(
    trial_seed,
    lambdas,
    d=2,
    N_train=100,
    low=-1.0,
    high=1.0,
    metric="l2",
    init_sigma=5e-4,
    eta=1e-3,
    num_steps=3000,
):
    """
    One random function + one random training set.
    Then sweep lambda using the same problem instance.
    """
    rng = np.random.default_rng(trial_seed)

    # Random exactly 1-Lipschitz target
    f, W, bias, x0 = make_random_1_lip_tanh_target(
        d=d,
        metric=metric,
        low=low,
        high=high,
        seed=trial_seed + 10,
    )

    # Random training data
    X_train = rng.uniform(low, high, size=(N_train, d))
    Y_train = f(X_train)

    l1_errors = []
    l2_errors = []

    for lam in lambdas:
        result = train_gd_Rd(
            X_train=X_train,
            Y_train=Y_train,
            metric=metric,
            lam=lam,
            eta=eta,
            num_steps=num_steps,
            init_sigma=init_sigma,
            seed=trial_seed + 1000,
        )

        l1_errors.append(result["l1_error"])
        l2_errors.append(result["l2_error"])

    return {
        "l1_errors": np.array(l1_errors),
        "l2_errors": np.array(l2_errors),
        "W": W,
        "bias": bias,
        "x0": x0,
    }


def run_lambda_sweep(
    d,
    N_train,
    low,
    high,
    metric,
    init_sigma,
    eta,
    num_steps,
    lambdas,
    num_trials,
    base_seed,
):
    print("\n" + "=" * 70)
    print("Running GD lambda sweep")
    print("=" * 70)
    print("dimension d:", d)
    print("metric:", metric)
    print("N_train:", N_train)
    print("fixed init_sigma:", init_sigma)
    print("eta:", eta)
    print("num_steps:", num_steps)
    print("num_trials:", num_trials)

    l1_all = []
    l2_all = []

    for trial in range(num_trials):
        print(f"\nLambda sweep trial {trial + 1}/{num_trials}")

        result = run_one_trial_lambda_sweep(
            trial_seed=base_seed + 100 * trial,
            lambdas=lambdas,
            d=d,
            N_train=N_train,
            low=low,
            high=high,
            metric=metric,
            init_sigma=init_sigma,
            eta=eta,
            num_steps=num_steps,
        )

        l1_all.append(result["l1_errors"])
        l2_all.append(result["l2_errors"])

        print("l1 errors:", result["l1_errors"])
        print("l2 errors:", result["l2_errors"])

    return np.array(l1_all), np.array(l2_all)


# ============================================================
# Sigma sweep
# ============================================================

def run_one_trial_sigma_sweep(
    trial_seed,
    sigmas,
    fixed_lambda,
    d=2,
    N_train=100,
    low=-1.0,
    high=1.0,
    metric="l2",
    eta=1e-3,
    num_steps=3000,
):
    """
    One random function + one random training set.
    Then sweep sigma using the same problem instance.
    """
    rng = np.random.default_rng(trial_seed)

    # Random exactly 1-Lipschitz target
    f, W, bias, x0 = make_random_1_lip_tanh_target(
        d=d,
        metric=metric,
        low=low,
        high=high,
        seed=trial_seed + 10,
    )

    # Random training data
    X_train = rng.uniform(low, high, size=(N_train, d))
    Y_train = f(X_train)

    l1_errors = []
    l2_errors = []

    for sigma in sigmas:
        result = train_gd_Rd(
            X_train=X_train,
            Y_train=Y_train,
            metric=metric,
            lam=fixed_lambda,
            eta=eta,
            num_steps=num_steps,
            init_sigma=sigma,
            seed=trial_seed + 1000,
        )

        l1_errors.append(result["l1_error"])
        l2_errors.append(result["l2_error"])

    return {
        "l1_errors": np.array(l1_errors),
        "l2_errors": np.array(l2_errors),
        "W": W,
        "bias": bias,
        "x0": x0,
    }


def run_sigma_sweep(
    d,
    N_train,
    low,
    high,
    metric,
    fixed_lambda,
    eta,
    num_steps,
    sigmas,
    num_trials,
    base_seed,
):
    print("\n" + "=" * 70)
    print("Running GD sigma sweep")
    print("=" * 70)
    print("dimension d:", d)
    print("metric:", metric)
    print("N_train:", N_train)
    print("fixed lambda:", fixed_lambda)
    print("eta:", eta)
    print("num_steps:", num_steps)
    print("num_trials:", num_trials)

    l1_all = []
    l2_all = []

    for trial in range(num_trials):
        print(f"\nSigma sweep trial {trial + 1}/{num_trials}")

        result = run_one_trial_sigma_sweep(
            trial_seed=base_seed + 100 * trial,
            sigmas=sigmas,
            fixed_lambda=fixed_lambda,
            d=d,
            N_train=N_train,
            low=low,
            high=high,
            metric=metric,
            eta=eta,
            num_steps=num_steps,
        )

        l1_all.append(result["l1_errors"])
        l2_all.append(result["l2_errors"])

        print("l1 errors:", result["l1_errors"])
        print("l2 errors:", result["l2_errors"])

    return np.array(l1_all), np.array(l2_all)


# ============================================================
# Main: run both sweeps
# ============================================================

def main():
    # --------------------------------------------------------
    # Main settings
    # --------------------------------------------------------
    d = 2
    N_train = 100
    low, high = -1.0, 1.0

    # Choose "l2" or "l1"
    metric = "l1"

    # GD settings
    eta = 1e-3
    num_steps = 3000

    # Repetitions
    num_trials = 2

    # --------------------------------------------------------
    # Lambda sweep settings
    # --------------------------------------------------------
    init_sigma_for_lambda_sweep = 5e-4

    lambdas = np.array([
        0.0,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        2e-1,
        3e-1,
        5e-1,
        7e-1,
        1.0,
    ])

    # --------------------------------------------------------
    # Sigma sweep settings
    # --------------------------------------------------------
    fixed_lambda_for_sigma_sweep = 1.0

    sigmas = np.array([
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    ])

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------
    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Run lambda sweep
    # --------------------------------------------------------
    lambda_l1_all, lambda_l2_all = run_lambda_sweep(
        d=d,
        N_train=N_train,
        low=low,
        high=high,
        metric=metric,
        init_sigma=init_sigma_for_lambda_sweep,
        eta=eta,
        num_steps=num_steps,
        lambdas=lambdas,
        num_trials=num_trials,
        base_seed=12345,
    )

    lambda_results_path = results_dir / f"exp_gd_lambda_sweep_Rd_{metric}.npz"
    np.savez(
        lambda_results_path,
        lambdas=lambdas,
        l1_all=lambda_l1_all,
        l2_all=lambda_l2_all,
        d=d,
        N_train=N_train,
        init_sigma=init_sigma_for_lambda_sweep,
        eta=eta,
        num_steps=num_steps,
        metric=metric,
    )

    lambda_fig_path = figures_dir / f"exp_gd_lambda_sweep_Rd_{metric}.pdf"
    plot_lambda_sweep(
        lambdas=lambdas,
        l1_all=lambda_l1_all,
        l2_all=lambda_l2_all,
        init_sigma=init_sigma_for_lambda_sweep,
        save_path=lambda_fig_path,
    )

    # --------------------------------------------------------
    # Run sigma sweep
    # --------------------------------------------------------
    sigma_l1_all, sigma_l2_all = run_sigma_sweep(
        d=d,
        N_train=N_train,
        low=low,
        high=high,
        metric=metric,
        fixed_lambda=fixed_lambda_for_sigma_sweep,
        eta=eta,
        num_steps=num_steps,
        sigmas=sigmas,
        num_trials=num_trials,
        base_seed=54321,
    )

    sigma_results_path = (
        results_dir
        / f"exp_gd_sigma_sweep_Rd_{metric}_lambda_{fixed_lambda_for_sigma_sweep}.npz"
    )
    np.savez(
        sigma_results_path,
        sigmas=sigmas,
        l1_all=sigma_l1_all,
        l2_all=sigma_l2_all,
        d=d,
        N_train=N_train,
        fixed_lambda=fixed_lambda_for_sigma_sweep,
        eta=eta,
        num_steps=num_steps,
        metric=metric,
    )

    sigma_fig_path = (
        figures_dir
        / f"exp_gd_sigma_sweep_Rd_{metric}_lambda_{fixed_lambda_for_sigma_sweep}.pdf"
    )
    plot_sigma_sweep(
        sigmas=sigmas,
        l1_all=sigma_l1_all,
        l2_all=sigma_l2_all,
        fixed_lambda=fixed_lambda_for_sigma_sweep,
        save_path=sigma_fig_path,
    )

    # --------------------------------------------------------
    # Final messages
    # --------------------------------------------------------
    print("\nSaved lambda sweep:")
    print(lambda_results_path)
    print(lambda_fig_path)

    print("\nSaved sigma sweep:")
    print(sigma_results_path)
    print(sigma_fig_path)


if __name__ == "__main__":
    main()