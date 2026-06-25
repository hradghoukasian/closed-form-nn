import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

from core.estimator import closed_form_predict_many
from core.metrics import abs_distance, estimate_lipschitz_R
from core.targets import target_sobolev
from core.samplers import sample_uniform_R, make_grid_R
from core.gd import (
    train_gd_R,
    wm_predict_many_R,
    empirical_lipschitz_margin_R,
)


# ------------------------------------------------------------
# Global plot style: larger fonts for paper figures
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_gd_convergence(history, save_path):
    """
    Plot the main GD convergence diagnostics.
    """
    steps = np.array(history["step"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].semilogy(steps, history["loss"], linewidth=2.5)
    axes[0, 0].set_title("Regularized loss")
    axes[0, 0].set_xlabel("GD step")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogy(steps, history["param_l2_error"], linewidth=2.5)
    axes[0, 1].set_title(r"Parameter error $\|\theta_t-\theta^\star\|_2$")
    axes[0, 1].set_xlabel("GD step")
    axes[0, 1].set_ylabel("parameter L2 error")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].semilogy(steps, history["train_max_residual"], linewidth=2.5)
    axes[1, 0].set_title("Max training residual")
    axes[1, 0].set_xlabel("GD step")
    axes[1, 0].set_ylabel(r"$\max_n |\hat f_{\theta_t}(X_n)-Y_n|$")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].semilogy(steps, history["function_inf_error"], linewidth=2.5)
    axes[1, 1].set_title("Function error to closed form")
    axes[1, 1].set_xlabel("GD step")
    axes[1, 1].set_ylabel(
        r"$\|\hat f_{\theta_t}-\hat f_{\theta^\star}\|_{\infty,\mathrm{test}}$"
    )
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, axes


def plot_final_fit(
    X_train,
    Y_train,
    X_test,
    Y_test,
    Y_closed_test,
    Y_gd_test,
    save_path,
):
    """
    Compare target, closed-form estimator, and GD-trained model.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(X_test, Y_test, label=r"target $f$", linewidth=2.5)
    ax.plot(
        X_test,
        Y_closed_test,
        label=r"closed form $\hat f_{\theta^\star}$",
        linewidth=2.5,
    )
    ax.plot(
        X_test,
        Y_gd_test,
        "--",
        label=r"GD model $\hat f_{\theta_T}$",
        linewidth=2.5,
    )
    ax.scatter(X_train, Y_train, label="training data", s=25)

    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("Final GD model vs closed form")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, ax


def plot_initialization_sweep(results_by_radius, save_path):
    """
    Show when GD succeeds/fails as initialization radius changes.
    """
    radii = []
    final_param_errors = []
    final_function_errors = []
    final_train_residuals = []

    for radius, result in results_by_radius.items():
        h = result["history"]
        radii.append(radius)
        final_param_errors.append(h["param_l2_error"][-1])
        final_function_errors.append(h["function_inf_error"][-1])
        final_train_residuals.append(h["train_max_residual"][-1])

    radii = np.array(radii)
    order = np.argsort(radii)

    radii = radii[order]
    final_param_errors = np.array(final_param_errors)[order]
    final_function_errors = np.array(final_function_errors)[order]
    final_train_residuals = np.array(final_train_residuals)[order]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.loglog(
        radii,
        final_param_errors,
        marker="o",
        linewidth=3.5,
        markersize=7,
        label="final parameter error",
    )
    ax.loglog(
        radii,
        final_function_errors,
        marker="o",
        linewidth=3.5,
        markersize=7,
        label="final function error",
    )
    ax.loglog(
        radii,
        final_train_residuals,
        marker="o",
        linewidth=3.5,
        markersize=7,
        label="final train residual",
    )

    ax.set_xlabel("initialization radius")
    ax.set_ylabel("final error")
    ax.set_title("Basin-of-attraction numerical test")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig, ax


def run_single_gd_experiment():
    """
    Main R^1 GD-recovery experiment.

    This checks:

        GD initialized near theta_star = (Y_m, X_m, L)_m

    converges back to theta_star.
    """
    # -----------------------------
    # Settings
    # -----------------------------
    N_train = 1000
    N_test = 1000
    low, high = -1.0, 1.0
    seed = 0

    # GD settings
    init_radius = 5e-4
    lam = 1.0
    eta = 1e-3
    num_steps = 3000
    log_every = 10

    # Safety factor for strict empirical Lipschitz margin
    lipschitz_safety = 1.10

    # -----------------------------
    # Data
    # -----------------------------
    rho = abs_distance
    f = target_sobolev

    X_train = sample_uniform_R(N_train, low=low, high=high, seed=seed)
    Y_train = f(X_train)

    X_test = make_grid_R(N_test, low=low, high=high)
    Y_test = f(X_test)

    # -----------------------------
    # Lipschitz constant used in optimization
    # -----------------------------
    L_hat = estimate_lipschitz_R(X_train, Y_train)
    L = lipschitz_safety * L_hat

    gamma = empirical_lipschitz_margin_R(X_train, Y_train, L)

    print("Estimated empirical Lipschitz constant L_hat:", L_hat)
    print("Used Lipschitz constant L:", L)
    print("Empirical strict Lipschitz margin gamma:", gamma)

    if gamma <= 0:
        print("WARNING: gamma <= 0. Assumption 4.7 is not satisfied.")

    # -----------------------------
    # Closed-form estimator theta_star
    # -----------------------------
    Y_closed_test = closed_form_predict_many(
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        L=L,
        rho=rho,
    )

    # -----------------------------
    # GD training initialized near theta_star
    # -----------------------------
    result = train_gd_R(
        X_train=X_train,
        Y_train=Y_train,
        L=L,
        init_radius=init_radius,
        lam=lam,
        eta=eta,
        num_steps=num_steps,
        seed=seed + 123,
        X_test=X_test,
        Y_closed_test=Y_closed_test,
        log_every=log_every,
    )

    a_final = result["a"]
    b_final = result["b"]
    c_final = result["c"]
    history = result["history"]

    Y_gd_test = wm_predict_many_R(X_test, a_final, b_final, c_final)

    # -----------------------------
    # Print final diagnostics
    # -----------------------------
    print("\nFinal diagnostics:")
    print("Final loss:", history["loss"][-1])
    print("Final data loss:", history["data_loss"][-1])
    print("Final reg loss:", history["reg_loss"][-1])
    print("Final parameter L2 error:", history["param_l2_error"][-1])
    print("Final parameter infinity error:", history["param_inf_error"][-1])
    print("Final max training residual:", history["train_max_residual"][-1])
    print("Final function infinity error to closed form:", history["function_inf_error"][-1])
    print("Final active fraction:", history["active_fraction"][-1])

    # -----------------------------
    # Save figures
    # -----------------------------
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_gd_convergence(
        history,
        save_path=figures_dir / "exp_gd_R_convergence.png",
    )

    plot_final_fit(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        Y_closed_test=Y_closed_test,
        Y_gd_test=Y_gd_test,
        save_path=figures_dir / "exp_gd_R_final_fit.png",
    )

    print("\nSaved figures:")
    print(figures_dir / "exp_gd_R_convergence.png")
    print(figures_dir / "exp_gd_R_final_fit.png")


def run_initialization_sweep():
    """
    Try several initialization radii and show that small radii recover
    the closed-form solution better than large radii.
    """
    # -----------------------------
    # Settings
    # -----------------------------
    N_train = 1000
    N_test = 1000
    low, high = -1.0, 1.0
    seed = 0

    lam = 1.0
    eta = 1e-3
    num_steps = 3000
    log_every = 10

    lipschitz_safety = 1.10

    init_radii = [
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
    ]

    # -----------------------------
    # Data
    # -----------------------------
    rho = abs_distance
    f = target_sobolev

    X_train = sample_uniform_R(N_train, low=low, high=high, seed=seed)
    Y_train = f(X_train)

    X_test = make_grid_R(N_test, low=low, high=high)

    L_hat = estimate_lipschitz_R(X_train, Y_train)
    L = lipschitz_safety * L_hat

    gamma = empirical_lipschitz_margin_R(X_train, Y_train, L)

    print("\nInitialization sweep")
    print("Estimated empirical Lipschitz constant L_hat:", L_hat)
    print("Used Lipschitz constant L:", L)
    print("Empirical strict Lipschitz margin gamma:", gamma)

    Y_closed_test = closed_form_predict_many(
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        L=L,
        rho=rho,
    )

    # -----------------------------
    # Run sweep
    # -----------------------------
    results_by_radius = {}

    for radius in init_radii:
        print(f"\nRunning init_radius = {radius}")

        result = train_gd_R(
            X_train=X_train,
            Y_train=Y_train,
            L=L,
            init_radius=radius,
            lam=lam,
            eta=eta,
            num_steps=num_steps,
            seed=seed + 777,
            X_test=X_test,
            Y_closed_test=Y_closed_test,
            log_every=log_every,
        )

        h = result["history"]

        print("Final parameter L2 error:", h["param_l2_error"][-1])
        print("Final function infinity error:", h["function_inf_error"][-1])
        print("Final max training residual:", h["train_max_residual"][-1])
        print("Final active fraction:", h["active_fraction"][-1])

        results_by_radius[radius] = result

    # -----------------------------
    # Save sweep figure
    # -----------------------------
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_initialization_sweep(
        results_by_radius,
        save_path=figures_dir / "exp_gd_R_init_sweep.png",
    )

    print("\nSaved figure:")
    print(figures_dir / "exp_gd_R_init_sweep.png")


def main():
    run_single_gd_experiment()

    # Optional:
    # run_initialization_sweep()


if __name__ == "__main__":
    main()