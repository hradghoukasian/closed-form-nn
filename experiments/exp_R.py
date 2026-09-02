

import os
import numpy as np
import matplotlib.pyplot as plt

from core.estimator import closed_form_predict_many
from core.metrics import abs_distance, estimate_lipschitz_R
from core.targets import target_smooth, target_sobolev
from core.samplers import sample_uniform_R, make_grid_R
from core.evaluation import max_error, mse, plot_R_results
from pathlib import Path


# ------------------------------------------------------------
# Global plot style: Calibri + larger fonts for paper figures
# ------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri"],

    "font.size": 20,
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 18,
    "figure.titlesize": 28,

    "lines.linewidth": 2.4,
})


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------

    N_train = 200
    N_test = 1000
    low, high = -1.0, 1.0
    seed = 0

    # --------------------------------------------------------
    # Space: X = [low, high] subset R
    # Metric: rho(x, y) = |x - y|
    # --------------------------------------------------------

    rho = abs_distance

    # --------------------------------------------------------
    # Target function
    # --------------------------------------------------------

    # f = target_smooth
    f = target_sobolev

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train = sample_uniform_R(
        N_train,
        low=low,
        high=high,
        seed=seed
    )

    Y_train = f(X_train)

    # --------------------------------------------------------
    # Lipschitz constant
    # --------------------------------------------------------

    # Lipschitz constant for sin(3x) is at most 3
    # L = 3.0

    # Estimate Lipschitz constant from training data.
    # 1.05 is a safety margin.
    L = 1.05 * estimate_lipschitz_R(
        X_train,
        Y_train
    )

    print("Estimated Lipschitz constant:", L)

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    X_test = make_grid_R(
        N_test,
        low=low,
        high=high
    )

    Y_test = f(X_test)

    # --------------------------------------------------------
    # Estimator and Lipschitz envelopes
    # --------------------------------------------------------

    dists = np.abs(
        X_test[:, None] - X_train[None, :]
    )

    Y_lower_test = np.max(
        Y_train[None, :] - L * dists,
        axis=1
    )

    Y_upper_test = np.min(
        Y_train[None, :] + L * dists,
        axis=1
    )

    Y_hat_test = 0.5 * (
        Y_lower_test + Y_upper_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print(
        "Max error:",
        max_error(Y_test, Y_hat_test)
    )

    print(
        "MSE:",
        mse(Y_test, Y_hat_test)
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    figures_dir = (
        PROJECT_ROOT
        / "results"
        / "figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_R_results(
        X_train,
        Y_train,
        X_test,
        Y_test,
        Y_hat_test,
        Y_lower_test=Y_lower_test,
        Y_upper_test=Y_upper_test,
        show_training=False,
        save_path=figures_dir / "exp_R.pdf",
    )


if __name__ == "__main__":
    main()