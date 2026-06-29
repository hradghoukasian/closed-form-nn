import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from core.estimator import closed_form_predict_many
# from core.metrics import l2_distance, estimate_lipschitz_pairwise
from core.metrics import l1_distance, estimate_lipschitz_pairwise
from core.targets import target_smooth_R2
from core.samplers import sample_uniform_R2, make_grid_R2
from core.evaluation import max_error, mse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_R2_results_no_dots(
    X1,
    X2,
    Y_true_grid,
    Y_hat_grid,
    save_path,
    title=None,
):
    """
    Plot target, closed-form estimator, and absolute error
    without showing the training sample dots.
    """
    error_grid = np.abs(Y_true_grid - Y_hat_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Use common scale for true function and estimator
    vmin = min(Y_true_grid.min(), Y_hat_grid.min())
    vmax = max(Y_true_grid.max(), Y_hat_grid.max())

    c0 = axes[0].contourf(X1, X2, Y_true_grid, levels=50, vmin=vmin, vmax=vmax)
    axes[0].set_title("Target function $f$")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    fig.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(X1, X2, Y_hat_grid, levels=50, vmin=vmin, vmax=vmax)
    axes[1].set_title("Closed-form estimator $\\hat f$")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    fig.colorbar(c1, ax=axes[1])

    c2 = axes[2].contourf(X1, X2, error_grid, levels=50)
    axes[2].set_title("Absolute error $|f-\\hat f|$")
    axes[2].set_xlabel("$x_1$")
    axes[2].set_ylabel("$x_2$")
    fig.colorbar(c2, ax=axes[2])

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_one_experiment(
    N_train,
    n_test_per_axis=60,
    low=-1.0,
    high=1.0,
    seed=0,
):
    # Space: X = [low, high]^2 subset R^2
    # # Metric: rho(x,y) = ||x-y||_2
    # rho = l2_distance

    # Metric: rho(x,y) = ||x-y||_1
    rho = l1_distance

    # Target function
    f = target_smooth_R2

    # Training data
    X_train = sample_uniform_R2(N_train, low=low, high=high, seed=seed)
    Y_train = f(X_train)

    # Estimate Lipschitz constant from training data
    L_hat = estimate_lipschitz_pairwise(X_train, Y_train, rho)
    L = 1.05 * L_hat

    print(f"\nRunning N_train = {N_train}")
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
    max_err = max_error(Y_test, Y_hat_test)
    test_mse = mse(Y_test, Y_hat_test)

    print("Max error:", max_err)
    print("MSE:", test_mse)

    # Reshape for contour plotting
    Y_test_grid = Y_test.reshape(X1.shape)
    Y_hat_grid = Y_hat_test.reshape(X1.shape)

    # Plot
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_path = figures_dir / f"exp_R2_N{N_train}.pdf"

    plot_R2_results_no_dots(
        X1=X1,
        X2=X2,
        Y_true_grid=Y_test_grid,
        Y_hat_grid=Y_hat_grid,
        save_path=save_path,
        title=f"Closed-form reconstruction on $[-1,1]^2$, $N_{{train}}={N_train}$",
    )

    return {
        "N_train": N_train,
        "L_hat": L_hat,
        "L_used": L,
        "max_error": max_err,
        "mse": test_mse,
        "figure": save_path,
    }


def main():
    # Settings
    N_train_list = [50,100,1000]
    n_test_per_axis = 60
    low, high = -1.0, 1.0
    seed = 0

    results = []

    for N_train in N_train_list:
        result = run_one_experiment(
            N_train=N_train,
            n_test_per_axis=n_test_per_axis,
            low=low,
            high=high,
            seed=seed,
        )
        results.append(result)

    print("\nSummary:")
    for r in results:
        print(
            f"N_train={r['N_train']}, "
            f"L_hat={r['L_hat']:.6f}, "
            f"L_used={r['L_used']:.6f}, "
            f"max_error={r['max_error']:.6e}, "
            f"mse={r['mse']:.6e}, "
            f"figure={r['figure']}"
        )


if __name__ == "__main__":
    main()