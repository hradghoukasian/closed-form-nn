"""
Experiment 2: effect of sinusoidal frequency at fixed noise

Compare:
  1) ReLU MLP ERM trained on noisy labels.
  2) Noisy universal formula: local averaging + closed-form midpoint.

The noise level is fixed, and the target frequency omega changes:
    f_omega(x) = sin(omega x).
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


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "results" / "figures"
RES_DIR = PROJECT_ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
LOW, HIGH = -1.0, 1.0
DIAMETER = HIGH - LOW

N_TRAIN = 2000
N_TEST = 2000
NUM_RUNS = 10
VAL_FRAC = 0.2

# Fixed noise level for this experiment.
SIGMA = 0.1

# Frequencies to test.
OMEGAS = [1, 5,10,15,20]
RECOVERY_OMEGAS = [1, 10, 20]

NUM_BETAS = 30
L_SAFETY = 1.05

MLP_WIDTH = 128
MLP_DEPTH = 3
MLP_EPOCHS = 3000
MLP_LR = 1e-3

BASE_SEED = 54321


# ---------------------------------------------------------------------
# Target and basic utilities
# ---------------------------------------------------------------------
def f(x, omega):
    """Clean target f_omega(x) = sin(omega x)."""
    return np.sin(float(omega) * np.asarray(x))


def mse(y, yhat):
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def min_spacing(X):
    """Minimum positive distance between sorted data points."""
    diffs = np.diff(np.sort(X))
    diffs = diffs[diffs > 1e-15]
    return float(np.min(diffs))


def beta_grid(X):
    """Beta range: [minimum data spacing, diameter of X]."""
    beta_min = max(min_spacing(X), 1e-12)
    beta_max = DIAMETER
    return np.exp(np.linspace(np.log(beta_min), np.log(beta_max), NUM_BETAS))


# ---------------------------------------------------------------------
# MLP ERM
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        d = 1
        for _ in range(MLP_DEPTH):
            layers += [nn.Linear(d, MLP_WIDTH), nn.ReLU()]
            d = MLP_WIDTH
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X, Y, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.tensor(X[:, None], dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y[:, None], dtype=torch.float32, device=device)

    model = MLP().to(device)
    opt = optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = nn.MSELoss()

    for _ in range(MLP_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X_t), Y_t)
        loss.backward()
        opt.step()

    return model


def predict_mlp(model, X):
    device = next(model.parameters()).device
    X_t = torch.tensor(np.asarray(X)[:, None], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        return model(X_t).cpu().numpy().ravel()


# ---------------------------------------------------------------------
# Noisy universal formula
# ---------------------------------------------------------------------
def local_average(X, Y, beta):
    """Ybar_i = average of Y_j over |X_j - X_i| <= beta."""
    D = np.abs(X[:, None] - X[None, :])
    A = D <= beta
    return (A @ Y) / A.sum(axis=1)


def lipschitz_estimate(X, Ybar):
    """1D empirical Lipschitz estimate using neighboring sorted points."""
    order = np.argsort(X)
    Xs, Ys = X[order], Ybar[order]
    dx = np.diff(Xs)
    dy = np.diff(Ys)
    mask = np.abs(dx) > 1e-12
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(dy[mask] / dx[mask])))


def universal_predict(X_test, X_train, Ybar, L, chunk=5000):
    """
    Closed-form midpoint:
        1/2 [ min_i {Ybar_i + L|x-X_i|} + max_i {Ybar_i - L|x-X_i|} ].
    """
    out = np.empty(len(X_test))
    for s in range(0, len(X_test), chunk):
        e = min(s + chunk, len(X_test))
        D = np.abs(X_test[s:e, None] - X_train[None, :])
        upper = np.min(Ybar[None, :] + L * D, axis=1)
        lower = np.max(Ybar[None, :] - L * D, axis=1)
        out[s:e] = 0.5 * (upper + lower)
    return out


def fit_universal(X, Y, beta):
    Ybar = local_average(X, Y, beta)
    L = L_SAFETY * lipschitz_estimate(X, Ybar)
    return Ybar, L


def choose_beta_holdout_cv(X, Y, seed):
    """
    Choose beta by hold-out validation MSE on noisy validation labels.

    This is a simple validation split, not k-fold CV.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(VAL_FRAC * len(X))

    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]

    X_fit, Y_fit = X[fit_idx], Y[fit_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    rows = []
    for beta in beta_grid(X):
        Ybar, L = fit_universal(X_fit, Y_fit, beta)
        pred_val = universal_predict(X_val, X_fit, Ybar, L)
        rows.append((beta, mse(Y_val, pred_val), L))

    cv = pd.DataFrame(rows, columns=["beta", "val_mse", "L_used"])
    beta_star = float(cv.loc[cv["val_mse"].idxmin(), "beta"])
    return beta_star, cv


# ---------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------
def run_one(omega, run, X_test, Y_test):
    seed = BASE_SEED + 10000 * run + int(100 * omega)
    rng = np.random.default_rng(seed)

    # data
    X = np.sort(rng.uniform(LOW, HIGH, N_TRAIN))
    Y_clean = f(X, omega)
    Y = Y_clean + rng.normal(0.0, SIGMA, N_TRAIN)

    # MLP ERM
    mlp = train_mlp(X, Y, seed + 1)
    pred_mlp = predict_mlp(mlp, X_test)

    # Universal formula with hold-out CV beta
    beta_star, cv = choose_beta_holdout_cv(X, Y, seed + 2)
    Ybar, L = fit_universal(X, Y, beta_star)
    pred_uf = universal_predict(X_test, X, Ybar, L)

    row = {
        "omega": omega,
        "sigma": SIGMA,
        "run": run,
        "mlp_mse": mse(Y_test, pred_mlp),
        "uf_mse": mse(Y_test, pred_uf),
        "mlp_maxerr": float(np.max(np.abs(Y_test - pred_mlp))),
        "uf_maxerr": float(np.max(np.abs(Y_test - pred_uf))),
        "beta_min": beta_grid(X)[0],
        "beta_max": beta_grid(X)[-1],
        "beta_star": beta_star,
        "L_used": L,
    }

    recovery = None
    if run == 0 and omega in RECOVERY_OMEGAS:
        recovery = (omega, X, Y, X_test, Y_test, pred_mlp, pred_uf, beta_star)

    cv.insert(0, "run", run)
    cv.insert(0, "omega", omega)
    cv.insert(0, "sigma", SIGMA)
    return row, cv, recovery


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_mse(summary):
    plt.figure(figsize=(9, 5.5))
    plt.errorbar(summary["omega"], summary["mlp_mse_mean"], yerr=summary["mlp_mse_std"], marker="o", capsize=4, label="ReLU MLP ERM")
    plt.errorbar(summary["omega"], summary["uf_mse_mean"], yerr=summary["uf_mse_std"], marker="s", capsize=4, label="Universal formula")
    plt.xlabel(r"Frequency $\omega$")
    plt.ylabel(r"Clean test MSE to $f_\omega$")
    plt.title(rf"Clean recovery error versus frequency, $\sigma={SIGMA}$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp_noisy_R_frequency_simple_mse_vs_omega.png", dpi=300)
    plt.close()


def plot_beta(summary):
    plt.figure(figsize=(9, 5.5))
    plt.errorbar(summary["omega"], summary["beta_star_mean"], yerr=summary["beta_star_std"], marker="o", capsize=4)
    plt.xlabel(r"Frequency $\omega$")
    plt.ylabel(r"Selected bandwidth $\beta^*$")
    plt.title(rf"Selected bandwidth versus frequency, $\sigma={SIGMA}$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp_noisy_R_frequency_simple_beta_vs_omega.png", dpi=300)
    plt.close()


def plot_recovery(payload):
    omega, X, Y, X_test, Y_test, pred_mlp, pred_uf, beta_star = payload

    plt.figure(figsize=(9, 5.5))
    plt.plot(X_test, Y_test, label=rf"Ground truth $f_{{\omega}}$, $\omega={omega}$", linewidth=3)
    plt.plot(X_test, pred_mlp, "--", label="ReLU MLP ERM")
    plt.plot(X_test, pred_uf, ":", label=rf"Universal formula, $\beta^*={beta_star:.3g}$", linewidth=3)

    idx = np.linspace(0, len(X) - 1, min(400, len(X))).astype(int)
    plt.scatter(X[idx], Y[idx], s=12, alpha=0.25, label="Noisy samples")

    plt.xlabel("x")
    plt.ylabel("value")
    plt.title(rf"Recovery at $\omega={omega}$, $\sigma={SIGMA}$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIG_DIR / f"exp_noisy_R_frequency_simple_recovery_omega_{omega}.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    rows, cvs, recoveries = [], [], []

    for omega in OMEGAS:
        X_test = np.linspace(LOW, HIGH, N_TEST)
        Y_test = f(X_test, omega)

        for run in range(NUM_RUNS):
            print(f"omega={omega}, sigma={SIGMA}, run={run + 1}/{NUM_RUNS}")
            row, cv, recovery = run_one(omega, run, X_test, Y_test)
            rows.append(row)
            cvs.append(cv)
            if recovery is not None:
                recoveries.append(recovery)
            print(f"  MLP MSE={row['mlp_mse']:.3e}, UF MSE={row['uf_mse']:.3e}, beta*={row['beta_star']:.3g}")

    raw = pd.DataFrame(rows)
    cv_all = pd.concat(cvs, ignore_index=True)

    summary = raw.groupby("omega").agg(
        mlp_mse_mean=("mlp_mse", "mean"),
        mlp_mse_std=("mlp_mse", "std"),
        uf_mse_mean=("uf_mse", "mean"),
        uf_mse_std=("uf_mse", "std"),
        beta_star_mean=("beta_star", "mean"),
        beta_star_std=("beta_star", "std"),
    ).reset_index()

    raw.to_csv(RES_DIR / "exp_noisy_R_frequency_simple_raw.csv", index=False)
    cv_all.to_csv(RES_DIR / "exp_noisy_R_frequency_simple_beta_cv.csv", index=False)
    summary.to_csv(RES_DIR / "exp_noisy_R_frequency_simple_summary.csv", index=False)

    plot_mse(summary)
    plot_beta(summary)
    for payload in recoveries:
        plot_recovery(payload)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nSaved results in:", RES_DIR)
    print("Saved figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
