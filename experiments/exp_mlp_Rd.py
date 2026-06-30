import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from core.estimator import closed_form_predict_many
from core.evaluation import mse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Metrics on R^d
# ============================================================

def l2_distance(x, y):
    """
    Euclidean distance on R^d.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.linalg.norm(x - y, ord=2)


def l1_distance(x, y):
    """
    L1 distance on R^d.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum(np.abs(x - y))


def get_metric(metric):
    if metric == "l2":
        return l2_distance
    elif metric == "l1":
        return l1_distance
    else:
        raise ValueError("metric must be either 'l1' or 'l2'")


def pairwise_distances(X, metric):
    """
    Pairwise distances between rows of X.

    Returns D where D[i,j] = rho(X_i, X_j).
    """
    X = np.asarray(X)

    diff = X[:, None, :] - X[None, :, :]

    if metric == "l2":
        return np.linalg.norm(diff, axis=2)

    elif metric == "l1":
        return np.sum(np.abs(diff), axis=2)

    else:
        raise ValueError("metric must be either 'l1' or 'l2'")


def estimate_lipschitz_pairwise_fast(X, Y, metric, eps=1e-12):
    """
    Estimate empirical Lipschitz constant:

        max_{i != j} |Y_i - Y_j| / rho(X_i, X_j).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    D = pairwise_distances(X, metric)
    DY = np.abs(Y[:, None] - Y[None, :])

    mask = D > eps

    if not np.any(mask):
        raise ValueError("Cannot estimate Lipschitz constant: all distances are zero.")

    return np.max(DY[mask] / D[mask])


def empirical_lipschitz_on_points(X, Y_pred, metric, eps=1e-12):
    """
    Empirical Lipschitz estimate of predictions on a finite point cloud.
    """
    return estimate_lipschitz_pairwise_fast(X, Y_pred, metric, eps=eps)


# ============================================================
# Random exactly 1-Lipschitz target
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

    If metric == "l2":
        normalize W so ||W||_2 = 1.

    If metric == "l1":
        normalize W so ||W||_infty = 1.

    We choose b = - W_norm^T x0 for a random interior point x0,
    so tanh' reaches its maximum value 1 somewhere inside the domain.
    """
    rng = np.random.default_rng(seed)

    W = rng.standard_normal(d)

    if metric == "l2":
        W = W / np.linalg.norm(W, ord=2)

    elif metric == "l1":
        W = W / np.max(np.abs(W))

    else:
        raise ValueError("metric must be either 'l1' or 'l2'")

    margin = 0.1 * (high - low)
    x0 = rng.uniform(low + margin, high - margin, size=d)

    b = -np.dot(W, x0)

    def f(X):
        X = np.asarray(X)
        return np.tanh(X @ W + b)

    return f, W, b, x0


# ============================================================
# Sampling
# ============================================================

def sample_uniform_Rd(n, d, low=-1.0, high=1.0, seed=None):
    """
    Sample n points uniformly from [low, high]^d.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, d))


# ============================================================
# MLP model on R^d
# ============================================================

class ReLUMlpRd(nn.Module):
    """
    Vanilla ReLU MLP for d-dimensional regression.

    Architecture:
        d -> width -> width -> ... -> width -> 1

    depth = number of hidden ReLU layers.
    """

    def __init__(self, input_dim, width=128, depth=3):
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(depth):
            layers.append(nn.Linear(current_dim, width))
            layers.append(nn.ReLU())
            current_dim = width

        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_Rd(
    X_train,
    Y_train,
    width=128,
    depth=3,
    lr=1e-3,
    num_epochs=5000,
    seed=0,
    print_every=None,
):
    """
    Train vanilla ReLU MLP on noiseless R^d training data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    input_dim = X_train.shape[1]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1).to(device)

    model = ReLUMlpRd(
        input_dim=input_dim,
        width=width,
        depth=depth,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs + 1):
        model.train()

        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, Y_train_t)
        loss.backward()
        optimizer.step()

        if print_every is not None and epoch % print_every == 0:
            print(f"epoch={epoch}, train_mse={loss.item():.6e}")

    return model


def predict_mlp_Rd(model, X):
    """
    Evaluate trained MLP on numpy inputs.
    """
    device = next(model.parameters()).device

    X = np.asarray(X)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        Y_pred = model(X_t).cpu().numpy().reshape(-1)

    return Y_pred


# ============================================================
# One experiment
# ============================================================

def run_one_experiment(
    run_seed,
    d=2,
    N_train=50,
    N_test=2000,
    N_lip=1000,
    low=-1.0,
    high=1.0,
    metric="l2",
    lipschitz_safety=1.05,
    mlp_widths=(64, 128),
    mlp_depth=3,
    mlp_lr=1e-3,
    mlp_epochs=5000,
):
    """
    Compare closed-form estimator and MLPs of several widths for one random run.

    Randomness:
      - target function f
      - training data
      - test data
      - MLP initialization
    """
    rho = get_metric(metric)

    # --------------------------------------------------------
    # Random exactly 1-Lipschitz target
    # --------------------------------------------------------
    f, W, b, x0 = make_random_1_lip_tanh_target(
        d=d,
        metric=metric,
        low=low,
        high=high,
        seed=run_seed + 11,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    X_train = sample_uniform_Rd(
        N_train,
        d=d,
        low=low,
        high=high,
        seed=run_seed + 101,
    )
    Y_train = f(X_train)

    X_test = sample_uniform_Rd(
        N_test,
        d=d,
        low=low,
        high=high,
        seed=run_seed + 202,
    )
    Y_test = f(X_test)

    X_lip = sample_uniform_Rd(
        N_lip,
        d=d,
        low=low,
        high=high,
        seed=run_seed + 303,
    )

    rows = []

    # --------------------------------------------------------
    # Closed-form estimator
    # --------------------------------------------------------
    L_hat = estimate_lipschitz_pairwise_fast(X_train, Y_train, metric=metric)
    L_used = lipschitz_safety * L_hat

    Y_cf_test = closed_form_predict_many(
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        L=L_used,
        rho=rho,
    )

    Y_cf_lip = closed_form_predict_many(
        X_test=X_lip,
        X_train=X_train,
        Y_train=Y_train,
        L=L_used,
        rho=rho,
    )

    cf_test_mse = mse(Y_test, Y_cf_test)
    cf_emp_lip = empirical_lipschitz_on_points(
        X_lip,
        Y_cf_lip,
        metric=metric,
    )

    rows.append({
        "run": run_seed,
        "method": "Closed form",
        "d": d,
        "metric": metric,
        "N_train": N_train,
        "N_test": N_test,
        "N_lip": N_lip,
        "test_mse": cf_test_mse,
        "empirical_lipschitz": cf_emp_lip,
        "L_hat_train": L_hat,
        "L_used_cf": L_used,
    })

    # --------------------------------------------------------
    # MLPs
    # --------------------------------------------------------
    for width in mlp_widths:
        model = train_mlp_Rd(
            X_train=X_train,
            Y_train=Y_train,
            width=width,
            depth=mlp_depth,
            lr=mlp_lr,
            num_epochs=mlp_epochs,
            seed=run_seed + 404 + width,
            print_every=None,
        )

        Y_mlp_test = predict_mlp_Rd(model, X_test)
        Y_mlp_lip = predict_mlp_Rd(model, X_lip)

        mlp_test_mse = mse(Y_test, Y_mlp_test)
        mlp_emp_lip = empirical_lipschitz_on_points(
            X_lip,
            Y_mlp_lip,
            metric=metric,
        )

        rows.append({
            "run": run_seed,
            "method": f"MLP W={width}",
            "d": d,
            "metric": metric,
            "N_train": N_train,
            "N_test": N_test,
            "N_lip": N_lip,
            "test_mse": mlp_test_mse,
            "empirical_lipschitz": mlp_emp_lip,
            "L_hat_train": np.nan,
            "L_used_cf": np.nan,
        })

    return rows


# ============================================================
# Summary
# ============================================================

def summarize_results(df):
    """
    Summary table with mean and std for test MSE and empirical Lipschitzness.
    """
    method_order = ["Closed form", "MLP W=64", "MLP W=128"]

    rows = []

    for method in method_order:
        sub = df[df["method"] == method]

        test_mse_mean = sub["test_mse"].mean()
        test_mse_std = sub["test_mse"].std(ddof=1)

        lip_mean = sub["empirical_lipschitz"].mean()
        lip_std = sub["empirical_lipschitz"].std(ddof=1)

        rows.append({
            "method": method,
            "test_mse_mean": test_mse_mean,
            "test_mse_std": test_mse_std,
            "test_mse_mean_pm_std": f"{test_mse_mean:.6e} ± {test_mse_std:.6e}",
            "empirical_lipschitz_mean": lip_mean,
            "empirical_lipschitz_std": lip_std,
            "empirical_lipschitz_mean_pm_std": f"{lip_mean:.6f} ± {lip_std:.6f}",
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Experiment settings
    # --------------------------------------------------------
    d = 2
    metric = "l2"   # choose "l2" or "l1"

    N_train = 50
    N_test = 2000
    N_lip = 1000

    low, high = -1.0, 1.0

    num_runs = 20
    base_seed = 12345

    lipschitz_safety = 1.05

    # MLP architectures:
    # d -> W -> W -> W -> 1
    mlp_widths = (64, 128)
    mlp_depth = 3
    mlp_lr = 1e-3
    mlp_epochs = 1000

    print("Running MLP vs closed-form experiment on R^d")
    print("d:", d)
    print("metric:", metric)
    print("N_train:", N_train)
    print("N_test:", N_test)
    print("N_lip:", N_lip)
    print("num_runs:", num_runs)
    print("MLP widths:", mlp_widths)
    print("MLP depth:", mlp_depth)
    print("MLP epochs:", mlp_epochs)

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------
    all_rows = []

    for run in range(num_runs):
        run_seed = base_seed + 1000 * run

        print(f"\nRun {run + 1}/{num_runs}")

        rows = run_one_experiment(
            run_seed=run_seed,
            d=d,
            N_train=N_train,
            N_test=N_test,
            N_lip=N_lip,
            low=low,
            high=high,
            metric=metric,
            lipschitz_safety=lipschitz_safety,
            mlp_widths=mlp_widths,
            mlp_depth=mlp_depth,
            mlp_lr=mlp_lr,
            mlp_epochs=mlp_epochs,
        )

        for row in rows:
            print(
                row["method"],
                "test MSE:",
                row["test_mse"],
                "empirical Lip:",
                row["empirical_lipschitz"],
            )

        all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(raw_df)

    # --------------------------------------------------------
    # Save only CSV tables, no PDF, no plots
    # --------------------------------------------------------
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / f"exp_mlp_Rd_{metric}_raw.csv"
    summary_path = results_dir / f"exp_mlp_Rd_{metric}_summary.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary table:")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()