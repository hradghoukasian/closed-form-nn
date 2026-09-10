"""Place at experiments/exp_mlp_Rd.py; run from the repository root with
PYTHONPATH=. python3 experiments/exp_mlp_Rd.py

Five-fold training-only CV selects additive Lipschitz slack on 20 grid points.
Each budget trains vanilla, spectral-normalized, and orthogonal ReLU networks.
Constrained predictors use the CV-selected formula L as an upper bound, not as
a claim that their actual Lipschitz constant equals L. Spectral bounds also
bound input-l1 Lipschitz constants, since ||x||2 <= ||x||1.

All three families have equal stored dense weights/biases for each width.
SN has extra temporary power-iteration buffers during training; these are
removed before evaluation. Orthogonal constraints reduce independent degrees
of freedom despite identical storage counts. Adam state is excluded throughout.
The anchor's epoch loop (num_epochs + 1 updates) is retained.
"""

import math
import json

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import spectral_norm, orthogonal
from torch.nn.utils.parametrize import remove_parametrizations

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

    best = None
    for i in range(0, len(X), 128):
        for j in range(i, len(X), 128):
            D = cross_distances(X[i:i+128], X[j:j+128], metric)
            DY = np.abs(Y[i:i+128, None] - Y[None, j:j+128])
            mask = D > eps
            if np.any(mask):
                value = float(np.max(DY[mask] / D[mask]))
                best = value if best is None else max(best, value)
    if best is None:
        raise ValueError("Cannot estimate Lipschitz constant: all distances are zero.")
    return best


def cross_distances(A, B, metric):
    diff = A[:, None, :] - B[None, :, :]
    if metric == "l2":
        return np.linalg.norm(diff, axis=2)
    if metric == "l1":
        return np.abs(diff).sum(axis=2)
    raise ValueError("metric must be l1 or l2")


def formula_grid_predict(X, X_train, Y_train, candidates, metric):
    """Same midpoint formula, bounded memory; reuse distances for every L."""
    output = np.empty((len(candidates), len(X)))
    for i in range(0, len(X), 64):
        lower = np.full((len(candidates), min(64, len(X)-i)), -np.inf)
        upper = np.full_like(lower, np.inf)
        for j in range(0, len(X_train), 128):
            D = cross_distances(X[i:i+64], X_train[j:j+128], metric)
            labels = Y_train[j:j+128]
            for k, L in enumerate(candidates):
                lower[k] = np.maximum(lower[k], (labels - L*D).max(axis=1))
                upper[k] = np.minimum(upper[k], (labels + L*D).min(axis=1))
        output[:, i:i+64] = 0.5*(lower+upper)
    return output


def select_lipschitz_cv(X, Y, metric, seed, folds=5, grid_size=20, span=2.0):
    """CV additive slack using fold-local L_D, then refit L on all training data.

    Using full-data L_D inside folds would leak validation labels. Every fold
    therefore evaluates L_D(fold train) + linspace(0, span, grid_size).
    Weighted validation MSE selects the offset (ties favor smaller L).
    """
    if not 2 <= folds <= len(X)//2 or grid_size < 2 or not np.isfinite(span) or span < 0:
        raise ValueError("Need >=2 training points per fold, >=2 grid points, span >=0")
    offsets = np.linspace(0.0, span, grid_size)
    splits = np.array_split(np.random.default_rng(seed).permutation(len(X)), folds)
    squared_errors = np.zeros(grid_size)
    fold_bases = []
    for fold, validation in enumerate(splits):
        train = np.concatenate([part for i, part in enumerate(splits) if i != fold])
        base = estimate_lipschitz_pairwise_fast(X[train], Y[train], metric)
        fold_bases.append(base)
        prediction = formula_grid_predict(X[validation], X[train], Y[train], base+offsets, metric)
        squared_errors += ((prediction-Y[validation])**2).sum(axis=1)
    scores = squared_errors/len(X)
    best = int(np.argmin(scores))
    full_base = estimate_lipschitz_pairwise_fast(X, Y, metric)
    return full_base, full_base+float(offsets[best]), {
        "cv_offset": float(offsets[best]), "cv_mse": float(scores[best]),
        "cv_offsets": json.dumps(offsets.tolist()),
        "cv_scores": json.dumps(scores.tolist()),
        "cv_fold_L_D": json.dumps(fold_bases), "cv_folds": folds,
    }


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
        return getattr(self, "output_scale", 1.0) * self.net(x)


def materialize_constraints(model, family, L):
    """Bake weights and gain into ordinary Linear layers: no inference buffers.

    SN power iteration is approximate during training. Full SVD normalizes each
    evaluated SN matrix; the final gain gives a numerical norm-product bound L.
    Orthogonal matrices are semi-orthogonal when rectangular. Their stored
    entries match vanilla counts, although independent degrees of freedom differ.
    """
    model.eval()
    layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for layer in layers:
            remove_parametrizations(layer, "weight", leave_parametrized=True)
            if family == "spectral":
                norm = torch.linalg.matrix_norm(layer.weight.double(), ord=2).item()
                if norm > 0:
                    layer.weight.div_(norm)
        product = math.prod(torch.linalg.matrix_norm(layer.weight.double(), ord=2).item() for layer in layers)
        # Absorb rounding correction into the output gain, preserving orthogonality
        # of the hidden matrices. Bias has no effect on Lipschitz bounds.
        scale = L / max(1.0, product) * (1.0 - 1e-6)
        layers[-1].weight.mul_(scale)
        layers[-1].bias.mul_(L)
    model.output_scale = 1.0
    return model


def train_mlp_Rd(
    X_train,
    Y_train,
    width=128,
    depth=3,
    lr=1e-3,
    num_epochs=5000,
    seed=0,
    print_every=None,
    family="unconstrained",
    lipschitz_bound=1.0,
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

    if family not in ("unconstrained", "spectral", "orthogonal"):
        raise ValueError("Unknown MLP family")
    if not math.isfinite(lipschitz_bound) or lipschitz_bound < 0:
        raise ValueError("lipschitz_bound must be finite and nonnegative")
    if family != "unconstrained":
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                if family == "spectral":
                    spectral_norm(layer, n_power_iterations=5)
                else:
                    orthogonal(layer, orthogonal_map="householder", use_trivialization=False)
        model.output_scale = lipschitz_bound

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

    if family != "unconstrained":
        materialize_constraints(model, family, lipschitz_bound)
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

def width_for_parameter_budget(target, input_dim, depth):
    """Return the positive integer width nearest a scalar parameter budget.

    A scalar-output MLP with `depth` equal-width ReLU hidden layers has
    (depth - 1) * width**2 + (input_dim + depth + 1) * width + 1
    weights and biases. Ties are resolved in favor of fewer parameters.
    """
    if input_dim < 1 or depth < 1:
        raise ValueError("input_dim and depth must be at least 1")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target must be finite and positive")

    a = depth - 1
    b = input_dim + depth + 1
    if a == 0:
        continuous_width = (target - 1) / b
    else:
        # Stable form of the positive quadratic root; widths below 1
        # are handled by the minimum-width constraint below.
        c = max(target - 1, 0)
        continuous_width = 2 * c / (b + math.sqrt(b * b + 4 * a * c))

    lower = max(1, math.floor(continuous_width))

    def parameter_count(width):
        return a * width**2 + b * width + 1

    return min(
        (lower, lower + 1),
        key=lambda width: (
            abs(parameter_count(width) - target),
            parameter_count(width),
        ),
    )


def run_one_experiment(
    run_seed,
    d=2,
    N_train=50,
    N_test=2000,
    N_lip=1000,
    low=-1.0,
    high=1.0,
    metric="l2",
    cv_folds=5,
    cv_grid_size=20,
    cv_span=2.0,
    mlp_ratios=(0.5, 1.0, 1.5, 2.0),
    mlp_depth=3,
    mlp_lr=1e-3,
    mlp_epochs=5000,
):
    """
    Compare the formula with MLPs targeting fractions of its stored scalars.

    Formula storage is N_train * (d + 1) + 1: coordinates, labels and one L.
    MLP storage counts all weights and biases, excluding optimizer state.
    Widths are integers, so actual ratios can differ from target ratios.

    Randomness:
      - target function f
      - training data
      - test data
      - MLP initialization
    """
    rho = get_metric(metric)
    mlp_ratios = tuple(float(ratio) for ratio in mlp_ratios)
    if not mlp_ratios or any(
        not math.isfinite(ratio) or ratio <= 0 for ratio in mlp_ratios
    ):
        raise ValueError("mlp_ratios must contain finite positive ratios")

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
    formula_scalars = int(X_train.size + Y_train.size + 1)

    # --------------------------------------------------------
    # Closed-form estimator
    # --------------------------------------------------------
    L_hat, L_used, cv_info = select_lipschitz_cv(
        X_train, Y_train, metric, run_seed+505,
        folds=cv_folds, grid_size=cv_grid_size, span=cv_span,
    )

    Y_cf_test = formula_grid_predict(X_test, X_train, Y_train, [L_used], metric)[0]
    Y_cf_lip = formula_grid_predict(X_lip, X_train, Y_train, [L_used], metric)[0]

    cf_test_mse = mse(Y_test, Y_cf_test)
    cf_emp_lip = empirical_lipschitz_on_points(
        X_lip,
        Y_cf_lip,
        metric=metric,
    )

    rows.append({
        "run": run_seed,
        "method": "Closed form",
        "family": "formula",
        "lipschitz_bound": L_used,
        **cv_info,
        "d": d,
        "metric": metric,
        "N_train": N_train,
        "N_test": N_test,
        "N_lip": N_lip,
        "test_mse": cf_test_mse,
        "empirical_lipschitz": cf_emp_lip,
        "L_hat_train": L_hat,
        "L_used_cf": L_used,
        "width": None,
        "mlp_depth": None,
        "target_scalar_ratio": 1.0,
        "stored_scalars": formula_scalars,
        "formula_scalars": formula_scalars,
        "scalar_ratio_to_formula": 1.0,
    })

    # --------------------------------------------------------
    # MLPs
    # --------------------------------------------------------
    for target_ratio in mlp_ratios:
        width = width_for_parameter_budget(
            target=target_ratio * formula_scalars,
            input_dim=d,
            depth=mlp_depth,
        )
        for family, family_name in (("unconstrained", "MLP"), ("spectral", "SN MLP"), ("orthogonal", "Orthogonal MLP")):
            model = train_mlp_Rd(
                X_train=X_train,
                Y_train=Y_train,
                width=width,
                depth=mlp_depth,
                lr=mlp_lr,
                num_epochs=mlp_epochs,
                seed=run_seed + 404 + width,
                print_every=None,
                family=family,
                lipschitz_bound=L_used,
            )
            mlp_scalars = sum(parameter.numel() for parameter in model.parameters())
            expected_scalars = (mlp_depth-1)*width**2 + (d+mlp_depth+1)*width + 1
            assert mlp_scalars == expected_scalars
            assert sum(value.numel() for value in model.state_dict().values()) == expected_scalars
    
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
                "method": f"{family_name} target={target_ratio}x",
                "d": d,
                "metric": metric,
                "N_train": N_train,
                "N_test": N_test,
                "N_lip": N_lip,
                "test_mse": mlp_test_mse,
                "empirical_lipschitz": mlp_emp_lip,
                "L_hat_train": L_hat,
                "L_used_cf": L_used,
                "family": family,
                "lipschitz_bound": math.prod(
                    torch.linalg.matrix_norm(layer.weight.detach().double(), ord=2).item()
                    for layer in model.net if isinstance(layer, nn.Linear)
                ),
                "width": width,
                "mlp_depth": mlp_depth,
                "target_scalar_ratio": target_ratio,
                "stored_scalars": mlp_scalars,
                "formula_scalars": formula_scalars,
                "scalar_ratio_to_formula": mlp_scalars / formula_scalars,
            })

    return rows


# ============================================================
# Summary
# ============================================================

def summarize_results(df):
    """
    Summarize errors, empirical Lipschitz estimates, and scalar budgets.
    """
    method_order = df["method"].drop_duplicates().tolist()

    rows = []

    for method in method_order:
        sub = df[df["method"] == method]

        test_mse_mean = sub["test_mse"].mean()
        test_mse_std = sub["test_mse"].std(ddof=1)

        lip_mean = sub["empirical_lipschitz"].mean()
        lip_std = sub["empirical_lipschitz"].std(ddof=1)

        rows.append({
            "method": method,
            "family": sub["family"].iloc[0],
            "L_used_cf_mean": sub["L_used_cf"].mean(),
            "lipschitz_bound_mean": sub["lipschitz_bound"].mean(),
            "width": sub["width"].iloc[0],
            "mlp_depth": sub["mlp_depth"].iloc[0],
            "target_scalar_ratio": float(sub["target_scalar_ratio"].iloc[0]),
            "stored_scalars": int(sub["stored_scalars"].iloc[0]),
            "formula_scalars": int(sub["formula_scalars"].iloc[0]),
            "scalar_ratio_to_formula": float(
                sub["scalar_ratio_to_formula"].iloc[0]
            ),
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

def latex_number(value, scientific=False):
    """Format one finite number for use inside LaTeX math mode."""
    if not math.isfinite(value):
        return r"\mathrm{NA}"
    if scientific and value != 0:
        mantissa, exponent = f"{value:.3e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return f"{value:.3f}"


def summary_to_latex(summary_df, caption, label="tab:mlp_formula_budgets"):
    """Return a complete table environment using standard LaTeX tabular."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Method & Scalars $P$ & $P/P_{\mathrm{formula}}$ & "
        r"Test MSE $\pm$ std & Empirical Lip. $\pm$ std \\",
        r"\hline",
    ]
    for row in summary_df.itertuples(index=False):
        method = "Closed form" if pd.isna(row.width) else rf"{dict(unconstrained='MLP', spectral='SN MLP', orthogonal='Orth. MLP')[row.family]} $W={int(row.width)}$"
        mse_cell = (
            latex_number(row.test_mse_mean, scientific=True)
            + r"\pm " + latex_number(row.test_mse_std, scientific=True)
        )
        lip_cell = (
            latex_number(row.empirical_lipschitz_mean)
            + r"\pm " + latex_number(row.empirical_lipschitz_std)
        )
        lines.append(
            f"{method} & {int(row.stored_scalars)} & "
            f"{row.scalar_ratio_to_formula:.4f} & "
            f"${mse_cell}$ & ${lip_cell}$ " + r"\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"



def main():
    # --------------------------------------------------------
    # Experiment settings
    # --------------------------------------------------------
    d = 100
    metric = "l2"   # choose "l2" or "l1"

    N_train = 1*(10**3)
    N_test = 2000
    N_lip = 1000

    low, high = -1.0, 1.0

    num_runs = 20
    base_seed = 12345

    cv_folds = 5
    cv_grid_size = 20
    cv_span = 2.0

    # Target 50%, 100%, 150%, and 200% of formula storage.
    # Each equal-width architecture uses the nearest integer width.
    mlp_ratios = (0.5, 1.0, 1.5, 2.0)
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
    print("MLP target scalar ratios:", mlp_ratios)
    print("MLP depth:", mlp_depth)
    print("MLP epochs:", mlp_epochs)
    print("CV folds / grid points / offset span:", cv_folds, cv_grid_size, cv_span)

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
            cv_folds=cv_folds,
            cv_grid_size=cv_grid_size,
            cv_span=cv_span,
            mlp_ratios=mlp_ratios,
            mlp_depth=mlp_depth,
            mlp_lr=mlp_lr,
            mlp_epochs=mlp_epochs,
        )
        print("L_D:", rows[0]["L_hat_train"], "selected L:", rows[0]["L_used_cf"],
              "CV offset:", rows[0]["cv_offset"], "CV MSE:", rows[0]["cv_mse"])

        for row in rows:
            print(
                row["method"],
                "test MSE:",
                row["test_mse"],
                "empirical Lip:",
                row["empirical_lipschitz"],
                "width:",
                row["width"],
                "stored scalars:",
                row["stored_scalars"],
                "target ratio:",
                row["target_scalar_ratio"],
                "actual ratio:",
                row["scalar_ratio_to_formula"],
            )

        all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(raw_df)

    # --------------------------------------------------------
    # Save CSV results and a readable text summary table
    # --------------------------------------------------------
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ratio_tag = "-".join(f"{ratio:g}" for ratio in mlp_ratios)
    output_stem = (
        f"exp_mlp_Rd_{metric}_d{d}_N{N_train}_depth{mlp_depth}"
        f"_ratios{ratio_tag}_cv{cv_folds}x{cv_grid_size}_span{cv_span:g}_constrained"
    )
    raw_path = results_dir / f"{output_stem}_raw.csv"
    summary_path = results_dir / f"{output_stem}_summary.csv"
    text_path = results_dir / f"{output_stem}_summary.txt"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Show actual parameter ratios: integer widths only approximate the targets.
    text_table = pd.DataFrame({
        "Method": [
            "Closed form" if pd.isna(row.width) else (
                f"{row.family} MLP W={int(row.width)} "
                f"(target {100 * row.target_scalar_ratio:g}%)"
            )
            for row in summary_df.itertuples(index=False)
        ],
        "Scalars": summary_df["stored_scalars"].astype(int),
        "Actual ratio to formula": summary_df["scalar_ratio_to_formula"].map(
            lambda ratio: f"{ratio:.4f}x ({100 * ratio:.2f}%)"
        ),
        "Test MSE ± std": summary_df["test_mse_mean_pm_std"],
        "Empirical Lip. ± std": summary_df["empirical_lipschitz_mean_pm_std"],
    })
    formula_scalars = int(summary_df["formula_scalars"].iloc[0])
    text_content = (
        "Comparison of the closed-form estimator and ReLU MLPs\n\n"
        f"d={d}; metric={metric}; domain=[{low:g}, {high:g}]^d; "
        f"N_train={N_train}; N_test={N_test}; N_lip={N_lip}; runs={num_runs}.\n"
        f"MLP hidden depth={mlp_depth}; Adam learning rate={mlp_lr:g}; "
        f"epoch setting={mlp_epochs} ({mlp_epochs + 1} optimizer updates).\n\n"
        + text_table.to_string(index=False)
        + "\n\n"
        + f"Formula scalars = N_train * (d + 1) + 1 = {formula_scalars}.\n"
        + "These are the sample coordinates, labels, and one shared L.\n"
        + "MLP scalars count all weights and biases, excluding optimizer state.\n"
        + "Actual ratio = stored scalars of the method / formula scalars.\n"
        + "For example, 1.5x means 150% of the formula's scalar count.\n"
        + "Target percentages determine the nearest integer hidden width; "
        + "the table reports the achieved percentages.\n"
        + "With d, hidden depth, and target ratios fixed, N_train determines "
        + "the scalar budgets and selected widths.\n"
        + "Counts measure scalar values, not bytes; the formula count refers "
        + "to its direct stored representation.\n"
        + "The reported standard deviations are sample standard deviations "
        + "across runs. Empirical Lipschitz estimates are not certified upper bounds.\n"
    )
    readable_path = results_dir / f"{output_stem}_readable.txt"
    readable_path.write_text(text_content, encoding="utf-8")
    caption = (
        rf"Formula with {cv_folds}-fold CV and ReLU benchmarks; $d={d}$, "
        rf"$N_{{\mathrm{{train}}}}={N_train}$, {num_runs} runs. "
        rf"CV selects one of {cv_grid_size} offsets in $[0,{cv_span:g}]$ above fold-local "
        r"$L_D$, then refits on all training data. SN and orthogonal MLPs use "
        r"the selected formula bound. $P$ counts stored scalars; SN auxiliary "
        r"buffers and the fixed output gain are removed for inference."
    )
    latex = summary_to_latex(summary_df, caption)
    text_path.write_text(latex, encoding="utf-8")
    tex_path = text_path.with_suffix(".tex")
    tex_path.write_text(latex, encoding="utf-8")

    print("\nSummary table:")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(raw_path)
    print(summary_path)
    print(text_path)
    print(tex_path)
    print(readable_path)


if __name__ == "__main__":
    main()
