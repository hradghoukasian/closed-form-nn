import os
import numpy as np

from core.estimator import closed_form_predict_many
from core.metrics import abs_distance, estimate_lipschitz_R
from core.targets import target_smooth, target_sobolev
from core.samplers import sample_uniform_R, make_grid_R
from core.evaluation import max_error, mse, plot_R_results
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    # Settings
    N_train = 100
    N_test = 1000
    low, high = -1.0, 1.0
    seed = 0

    # Space: X = [low, high] subset R
    # Metric: rho(x,y) = |x-y|
    rho = abs_distance

    # Target function
    # f = target_smooth
    f = target_sobolev


    # Data
    X_train = sample_uniform_R(N_train, low=low, high=high, seed=seed)
    Y_train = f(X_train)

    # # Lipschitz constant for sin(3x) is at most 3
    # L = 3.0

    # Estimate Lipschitz constant from training data (1.05 is the safety margin)
    L = 1.05 * estimate_lipschitz_R(X_train, Y_train)
    print("Estimated Lipschitz constant:", L)

    X_test = make_grid_R(N_test, low=low, high=high)
    Y_test = f(X_test)

    # Estimator
    Y_hat_test = closed_form_predict_many(
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        L=L,
        rho=rho,
    )

    # Metrics
    print("Max error:", max_error(Y_test, Y_hat_test))
    print("MSE:", mse(Y_test, Y_hat_test))

    # Plot
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_R_results(
        X_train,
        Y_train,
        X_test,
        Y_test,
        Y_hat_test,
        save_path=figures_dir / "exp_R.png",
    )


if __name__ == "__main__":
    main()