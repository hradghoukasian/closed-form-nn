"""
Recreate noisy-recovery figures from saved experiment results.

Creates:

1) exp_noisy_R_simple_mse_vs_sigma
2) exp_noisy_R_simple_beta_vs_sigma
3) exp_noisy_R_simple_recovery_sigma_0p8
4) exp_noisy_R_simple_recovery_sigma_0p05

5) exp_noisy_R_frequency_simple_beta_vs_omega
6) exp_noisy_R_frequency_simple_mse_vs_omega
7) exp_noisy_R_frequency_simple_recovery_omega_50
8) exp_noisy_R_frequency_simple_recovery_omega_5

Each figure is saved as both .pdf and .png.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri"],

    "font.size": 20,
    "axes.titlesize": 25,
    "axes.labelsize": 23,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 18,
    "figure.titlesize": 25,

    "lines.linewidth": 2.8,
})


# ============================================================
# Saved result files
# ============================================================

SIMPLE_RAW_PATH = (
    RESULTS_DIR
    / "exp_noisy_R_simple_raw.csv"
)

SIMPLE_SUMMARY_PATH = (
    RESULTS_DIR
    / "exp_noisy_R_simple_summary.csv"
)

FREQ_RAW_PATH = (
    RESULTS_DIR
    / "exp_noisy_R_frequency_simple_raw.csv"
)

FREQ_SUMMARY_PATH = (
    RESULTS_DIR
    / "exp_noisy_R_frequency_simple_summary.csv"
)


# ============================================================
# Original experiment settings
# ============================================================

LOW = -1.0
HIGH = 1.0

N_TRAIN = 2000
N_TEST = 2000

MLP_WIDTH = 128
MLP_DEPTH = 3
MLP_EPOCHS = 3000
MLP_LR = 1e-3

L_SAFETY = 1.05
BASE_SEED = 12345


# ============================================================
# Utilities
# ============================================================

def save_figure(fig, filename):
    """
    Save figure as PDF only.
    """

    pdf_path = FIGURES_DIR / f"{filename}.pdf"

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
    )

    print("Saved:", pdf_path)


def mse(y, yhat):
    return float(
        np.mean(
            (np.asarray(y) - np.asarray(yhat)) ** 2
        )
    )


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        layers = []
        d = 1

        for _ in range(MLP_DEPTH):
            layers += [
                nn.Linear(d, MLP_WIDTH),
                nn.ReLU(),
            ]

            d = MLP_WIDTH

        layers += [
            nn.Linear(d, 1)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X, Y, seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    X_t = torch.tensor(
        X[:, None],
        dtype=torch.float32,
        device=device,
    )

    Y_t = torch.tensor(
        Y[:, None],
        dtype=torch.float32,
        device=device,
    )

    model = MLP().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=MLP_LR,
    )

    loss_fn = nn.MSELoss()

    for _ in range(MLP_EPOCHS):

        optimizer.zero_grad()

        prediction = model(X_t)

        loss = loss_fn(
            prediction,
            Y_t,
        )

        loss.backward()

        optimizer.step()

    return model


def predict_mlp(model, X):

    device = next(
        model.parameters()
    ).device

    X_t = torch.tensor(
        np.asarray(X)[:, None],
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():

        prediction = (
            model(X_t)
            .cpu()
            .numpy()
            .ravel()
        )

    return prediction


# ============================================================
# Universal formula
# ============================================================

def local_average(X, Y, beta):

    D = np.abs(
        X[:, None]
        - X[None, :]
    )

    A = D <= beta

    return (
        (A @ Y)
        / A.sum(axis=1)
    )


def lipschitz_estimate(X, Ybar):

    order = np.argsort(X)

    Xs = X[order]
    Ys = Ybar[order]

    dx = np.diff(Xs)
    dy = np.diff(Ys)

    mask = np.abs(dx) > 1e-12

    if not np.any(mask):
        return 0.0

    return float(
        np.max(
            np.abs(
                dy[mask] / dx[mask]
            )
        )
    )


def fit_universal(X, Y, beta):

    Ybar = local_average(
        X,
        Y,
        beta,
    )

    L = (
        L_SAFETY
        * lipschitz_estimate(
            X,
            Ybar,
        )
    )

    return Ybar, L


def universal_predict(
    X_test,
    X_train,
    Ybar,
    L,
    chunk=5000,
):

    out = np.empty(
        len(X_test)
    )

    for start in range(
        0,
        len(X_test),
        chunk,
    ):

        end = min(
            start + chunk,
            len(X_test),
        )

        D = np.abs(
            X_test[start:end, None]
            - X_train[None, :]
        )

        upper = np.min(
            Ybar[None, :]
            + L * D,
            axis=1,
        )

        lower = np.max(
            Ybar[None, :]
            - L * D,
            axis=1,
        )

        out[start:end] = (
            0.5
            * (upper + lower)
        )

    return out


# ============================================================
# Plot 1: MSE versus sigma
# ============================================================

def plot_simple_mse(summary):

    fig, ax = plt.subplots(
        figsize=(10, 6.2)
    )

    ax.errorbar(
        summary["sigma"],
        summary["mlp_mse_mean"],
        yerr=summary["mlp_mse_std"],
        marker="o",
        markersize=8,
        capsize=5,
        label="ReLU MLP (Adam)",
    )

    ax.errorbar(
        summary["sigma"],
        summary["uf_mse_mean"],
        yerr=summary["uf_mse_std"],
        marker="s",
        markersize=8,
        capsize=5,
        label="Universal formula",
    )

    ax.set_xlabel(
        r"Noise standard deviation $\sigma$"
    )

    ax.set_ylabel(
        r"Clean test MSE to $f$"
    )

    ax.set_title(
        "Clean recovery error versus noise",
        pad=12,
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(loc="lower left")

    fig.tight_layout()

    save_figure(
        fig,
        "exp_noisy_R_simple_mse_vs_sigma",
    )

    plt.close(fig)


# ============================================================
# Plot 2: beta versus sigma
# ============================================================

def plot_simple_beta(summary):

    fig, ax = plt.subplots(
        figsize=(10, 6.2)
    )

    ax.errorbar(
        summary["sigma"],
        summary["beta_star_mean"],
        yerr=summary["beta_star_std"],
        marker="o",
        markersize=8,
        capsize=5,
    )

    ax.set_xlabel(
        r"Noise standard deviation $\sigma$"
    )

    ax.set_ylabel(
        r"Selected bandwidth $\beta^*$"
    )

    ax.set_title(
        r"Selected $\beta^*$ versus noise",
        pad=12,
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "exp_noisy_R_simple_beta_vs_sigma",
    )

    plt.close(fig)


# ============================================================
# Recreate simple-noise recovery run
# ============================================================

def reconstruct_simple_recovery(
    sigma,
    raw_df,
):

    # Same run used for the original displayed recovery
    run = 0

    row = raw_df[
        np.isclose(
            raw_df["sigma"],
            sigma,
        )
        &
        (raw_df["run"] == run)
    ]

    if len(row) != 1:
        raise ValueError(
            f"Could not find unique result "
            f"for sigma={sigma}, run={run}"
        )

    beta_star = float(
        row.iloc[0]["beta_star"]
    )

    seed = (
        BASE_SEED
        + 10000 * run
        + int(1000 * sigma)
    )

    rng = np.random.default_rng(seed)

    X = np.sort(
        rng.uniform(
            LOW,
            HIGH,
            N_TRAIN,
        )
    )

    Y_clean = np.sin(
        3.0 * X
    )

    Y = (
        Y_clean
        + rng.normal(
            0.0,
            sigma,
            N_TRAIN,
        )
    )

    X_test = np.linspace(
        LOW,
        HIGH,
        N_TEST,
    )

    Y_test = np.sin(
        3.0 * X_test
    )

    # MLP
    mlp = train_mlp(
        X,
        Y,
        seed + 1,
    )

    pred_mlp = predict_mlp(
        mlp,
        X_test,
    )

    # Universal formula using saved beta*
    Ybar, L = fit_universal(
        X,
        Y,
        beta_star,
    )

    pred_uf = universal_predict(
        X_test,
        X,
        Ybar,
        L,
    )

    return (
        X,
        Y,
        X_test,
        Y_test,
        pred_mlp,
        pred_uf,
        beta_star,
    )


# ============================================================
# Plot 3/4: recovery versus sigma
# ============================================================

def plot_simple_recovery(
    sigma,
    raw_df,
):

    (
        X,
        Y,
        X_test,
        Y_test,
        pred_mlp,
        pred_uf,
        beta_star,
    ) = reconstruct_simple_recovery(
        sigma,
        raw_df,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6.2)
    )

    ax.plot(
        X_test,
        Y_test,
        label=r"Ground truth $f$",
        linewidth=3.4,
    )

    ax.plot(
        X_test,
        pred_mlp,
        "--",
        label="ReLU MLP (Adam)",
        linewidth=2.7,
    )

    ax.plot(
        X_test,
        pred_uf,
        ":",
        label=(
            rf"Universal formula, "
            rf"$\beta^*={beta_star:.3g}$"
        ),
        linewidth=3.4,
    )

    # Same readable subsampling of noisy observations
    idx = np.linspace(
        0,
        len(X) - 1,
        min(400, len(X)),
    ).astype(int)

    ax.scatter(
        X[idx],
        Y[idx],
        s=22,
        alpha=0.25,
        label="Noisy samples",
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("Value")

    ax.set_title(
        rf"Recovery at $\sigma={sigma:g}$",
        pad=12,
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(loc="lower left")

    fig.tight_layout()

    tag = str(sigma).replace(
        ".",
        "p"
    )

    save_figure(
        fig,
        f"exp_noisy_R_simple_recovery_sigma_{tag}",
    )

    plt.close(fig)


# ============================================================
# Plot 5: beta versus omega
# ============================================================

def plot_frequency_beta(summary):

    fig, ax = plt.subplots(
        figsize=(10, 6.2)
    )

    ax.errorbar(
        summary["omega"],
        summary["beta_star_mean"],
        yerr=summary["beta_star_std"],
        marker="o",
        markersize=8,
        capsize=5,
    )

    ax.set_xlabel(
        r"Frequency $\omega$"
    )

    ax.set_ylabel(
        r"Selected bandwidth $\beta^*$"
    )

    ax.set_title(
        r"Selected bandwidth versus frequency, $\sigma=0.1$",
        pad=12,
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "exp_noisy_R_frequency_simple_beta_vs_omega",
    )

    plt.close(fig)


# ============================================================
# Plot 6: MSE versus omega
# ============================================================

def plot_frequency_mse(summary):

    fig, ax = plt.subplots(
        figsize=(10, 6.2)
    )

    ax.errorbar(
        summary["omega"],
        summary["mlp_mse_mean"],
        yerr=summary["mlp_mse_std"],
        marker="o",
        markersize=8,
        capsize=5,
        label="ReLU MLP (Adam)",
    )

    ax.errorbar(
        summary["omega"],
        summary["uf_mse_mean"],
        yerr=summary["uf_mse_std"],
        marker="s",
        markersize=8,
        capsize=5,
        label="Universal formula",
    )

    ax.set_xlabel(
        r"Frequency $\omega$"
    )

    ax.set_ylabel(
        r"Clean test MSE to $f_\omega$"
    )

    ax.set_title(
        r"Clean recovery error versus frequency, $\sigma=0.1$",
        pad=12,
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(loc="lower left")

    fig.tight_layout()

    save_figure(
        fig,
        "exp_noisy_R_frequency_simple_mse_vs_omega",
    )

    plt.close(fig)


# ============================================================
# Recreate frequency recovery run
# ============================================================

def reconstruct_frequency_recovery(
    omega,
    raw_df,
):

    sigma = 0.1
    run = 0

    row = raw_df[
        np.isclose(
            raw_df["omega"],
            omega,
        )
        &
        np.isclose(
            raw_df["sigma"],
            sigma,
        )
        &
        (raw_df["run"] == run)
    ]

    if len(row) != 1:
        raise ValueError(
            f"Could not find unique result "
            f"for omega={omega}, sigma={sigma}, run={run}"
        )

    beta_star = float(
        row.iloc[0]["beta_star"]
    )

    # Frequency experiment uses the same deterministic
    # run convention, with sigma fixed to 0.1.
    #
    # If your original frequency experiment used a different
    # seed formula, change only this line.
    seed = (
        BASE_SEED
        + 10000 * run
        + int(1000 * sigma)
        + int(100 * omega)
    )

    rng = np.random.default_rng(seed)

    X = np.sort(
        rng.uniform(
            LOW,
            HIGH,
            N_TRAIN,
        )
    )

    Y_clean = np.sin(
        omega * X
    )

    Y = (
        Y_clean
        + rng.normal(
            0.0,
            sigma,
            N_TRAIN,
        )
    )

    X_test = np.linspace(
        LOW,
        HIGH,
        N_TEST,
    )

    Y_test = np.sin(
        omega * X_test
    )

    mlp = train_mlp(
        X,
        Y,
        seed + 1,
    )

    pred_mlp = predict_mlp(
        mlp,
        X_test,
    )

    Ybar, L = fit_universal(
        X,
        Y,
        beta_star,
    )

    pred_uf = universal_predict(
        X_test,
        X,
        Ybar,
        L,
    )

    return (
        X,
        Y,
        X_test,
        Y_test,
        pred_mlp,
        pred_uf,
        beta_star,
    )


# ============================================================
# Plot 7/8: frequency recovery
# ============================================================

def plot_frequency_recovery(
    omega,
    raw_df,
):

    (
        X,
        Y,
        X_test,
        Y_test,
        pred_mlp,
        pred_uf,
        beta_star,
    ) = reconstruct_frequency_recovery(
        omega,
        raw_df,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6.2)
    )

    ax.plot(
        X_test,
        Y_test,
        label=(
            rf"Ground truth $f_\omega$, "
            rf"$\omega={omega:g}$"
        ),
        linewidth=3.4,
    )

    ax.plot(
        X_test,
        pred_mlp,
        "--",
        label="ReLU MLP (Adam)",
        linewidth=2.7,
    )

    ax.plot(
        X_test,
        pred_uf,
        ":",
        label=(
            rf"Universal formula, "
            rf"$\beta^*={beta_star:.3g}$"
        ),
        linewidth=3.4,
    )

    idx = np.linspace(
        0,
        len(X) - 1,
        min(400, len(X)),
    ).astype(int)

    ax.scatter(
        X[idx],
        Y[idx],
        s=22,
        alpha=0.25,
        label="Noisy samples",
    )

    ax.set_xlabel("$x$")
    ax.set_ylabel("Value")

    ax.set_title(
        rf"Recovery at $\omega={omega:g}$, $\sigma=0.1$",
        pad=12,
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(loc="lower left")

    fig.tight_layout()

    save_figure(
        fig,
        f"exp_noisy_R_frequency_simple_recovery_omega_{omega:g}",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load saved results
    # --------------------------------------------------------

    simple_raw = pd.read_csv(
        SIMPLE_RAW_PATH
    )

    simple_summary = pd.read_csv(
        SIMPLE_SUMMARY_PATH
    )

    freq_raw = pd.read_csv(
        FREQ_RAW_PATH
    )

    freq_summary = pd.read_csv(
        FREQ_SUMMARY_PATH
    )

    # --------------------------------------------------------
    # Noise experiment figures
    # --------------------------------------------------------

    plot_simple_mse(
        simple_summary
    )

    plot_simple_beta(
        simple_summary
    )

    plot_simple_recovery(
        0.8,
        simple_raw,
    )

    plot_simple_recovery(
        0.05,
        simple_raw,
    )

    # --------------------------------------------------------
    # Frequency experiment figures
    # --------------------------------------------------------

    plot_frequency_beta(
        freq_summary
    )

    plot_frequency_mse(
        freq_summary
    )

    plot_frequency_recovery(
        50,
        freq_raw,
    )

    plot_frequency_recovery(
        5,
        freq_raw,
    )

    print(
        "\nAll figures generated successfully."
    )

    print(
        "Saved in:",
        FIGURES_DIR,
    )


if __name__ == "__main__":
    main()