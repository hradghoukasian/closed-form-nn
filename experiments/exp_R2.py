import numpy as np
from pathlib import Path

from core.estimator import closed_form_predict_many
from core.metrics import l2_distance, estimate_lipschitz_pairwise
from core.targets import target_smooth_R2
from core.samplers import sample_uniform_R2, make_grid_R2
from core.evaluation import max_error, mse, plot_R2_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # Settings
    N_train = 1000
    n_test_per_axis = 60
    low, high = -1.0, 1.0
    seed = 0

    # Space: X = [low, high]^2 subset R^2
    # Metric: rho(x,y) = ||x-y||_2
    rho = l2_distance

    # Target function
    f = target_smooth_R2

    # Training data
    X_train = sample_uniform_R2(N_train, low=low, high=high, seed=seed)
    Y_train = f(X_train)

    # Estimate Lipschitz constant from training data
    L_hat = estimate_lipschitz_pairwise(X_train, Y_train, rho)
    L = 1.05 * L_hat

    print("Estimated Lipschitz constant:", L_hat)
    print("Used Lipschitz constant:", L)

    # Test grid
    X_test, X1, X2 = make_grid_R2(n_test_per_axis, low=low, high=high)
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

    # Reshape for contour plotting
    Y_test_grid = Y_test.reshape(X1.shape)
    Y_hat_grid = Y_hat_test.reshape(X1.shape)

    # Plot
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_R2_results(
        X_train,
        X1,
        X2,
        Y_test_grid,
        Y_hat_grid,
        save_path=figures_dir / "exp_R2.png",
    )


if __name__ == "__main__":
    main()