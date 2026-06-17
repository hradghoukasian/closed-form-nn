import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt



def max_error(Y_true, Y_pred):
    return np.max(np.abs(Y_true - Y_pred))


def mse(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)


def plot_R_results(X_train, Y_train, X_test, Y_test, Y_hat_test, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(X_test, Y_test, label=r"$f$ target", linewidth=1.5)
    ax.scatter(X_train, Y_train, label="Training data", s=20)

    ax.plot(X_test, Y_hat_test, label=r"$\hat f$ estimator", linewidth=1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    return fig, ax


def plot_R2_results(X_train, X1, X2, Y_test_grid, Y_hat_grid, save_path=None):
    """
    Plot true function and estimator as two contour plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    contour0 = axes[0].contourf(X1, X2, Y_test_grid, levels=30)
    axes[0].scatter(X_train[:, 0], X_train[:, 1], s=10, color="black")
    axes[0].set_title(r"Target $f$")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    fig.colorbar(contour0, ax=axes[0])

    contour1 = axes[1].contourf(X1, X2, Y_hat_grid, levels=30)
    axes[1].scatter(X_train[:, 0], X_train[:, 1], s=10, color="black")
    axes[1].set_title(r"Estimator $\hat f$")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    fig.colorbar(contour1, ax=axes[1])

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)

    return fig, axes
