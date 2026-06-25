import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from core.estimator import closed_form_predict_many
from core.metrics import abs_distance, estimate_lipschitz_R
from core.targets import target_sobolev
from core.samplers import sample_uniform_R, make_grid_R
from core.evaluation import mse, max_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReLUMlpR1(nn.Module):
    """
    Vanilla ReLU MLP for one-dimensional regression.

    Architecture:
        1 -> W -> W -> W -> 1

    where W is the width.
    """

    def __init__(self, width=128, depth=3):
        super().__init__()

        layers = []
        input_dim = 1

        for _ in range(depth):
            layers.append(nn.Linear(input_dim, width))
            layers.append(nn.ReLU())
            input_dim = width

        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_R(
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
    Train a vanilla ReLU MLP on noiseless 1D training data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1).to(device)

    model = ReLUMlpR1(width=width, depth=depth).to(device)

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


def predict_mlp_R(model, X):
    """
    Evaluate trained MLP on numpy inputs.
    """
    device = next(model.parameters()).device

    X_t = torch.tensor(X, dtype=torch.float32).reshape(-1, 1).to(device)

    model.eval()
    with torch.no_grad():
        Y_pred = model(X_t).cpu().numpy().reshape(-1)

    return Y_pred


def run_one_experiment(
    N_train,
    N_test=2000,
    low=-1.0,
    high=1.0,
    seed=0,
    lipschitz_safety=1.10,
    mlp_width=128,
    mlp_depth=3,
    mlp_lr=1e-3,
    mlp_epochs=5000,
):
    """
    Compare closed-form estimator and vanilla ReLU MLP
    for one N_train, one width, and one seed.
    """
    rho = abs_distance
    f = target_sobolev

    # -----------------------------
    # Data
    # -----------------------------
    X_train = sample_uniform_R(N_train, low=low, high=high, seed=seed)
    Y_train = f(X_train)

    X_test = make_grid_R(N_test, low=low, high=high)
    Y_test = f(X_test)

    # -----------------------------
    # Closed-form estimator
    # -----------------------------
    L_hat = estimate_lipschitz_R(X_train, Y_train)
    L = lipschitz_safety * L_hat

    Y_cf_train = closed_form_predict_many(
        X_test=X_train,
        X_train=X_train,
        Y_train=Y_train,
        L=L,
        rho=rho,
    )

    Y_cf_test = closed_form_predict_many(
        X_test=X_test,
        X_train=X_train,
        Y_train=Y_train,
        L=L,
        rho=rho,
    )

    cf_train_mse = mse(Y_train, Y_cf_train)
    cf_test_mse = mse(Y_test, Y_cf_test)
    cf_max_error = max_error(Y_test, Y_cf_test)

    # -----------------------------
    # Vanilla ReLU MLP
    # -----------------------------
    model = train_mlp_R(
        X_train=X_train,
        Y_train=Y_train,
        width=mlp_width,
        depth=mlp_depth,
        lr=mlp_lr,
        num_epochs=mlp_epochs,
        seed=seed,
        print_every=None,
    )

    Y_mlp_train = predict_mlp_R(model, X_train)
    Y_mlp_test = predict_mlp_R(model, X_test)

    mlp_train_mse = mse(Y_train, Y_mlp_train)
    mlp_test_mse = mse(Y_test, Y_mlp_test)
    mlp_max_error = max_error(Y_test, Y_mlp_test)

    mlp_vs_cf_test_mse = mse(Y_cf_test, Y_mlp_test)
    mlp_vs_cf_max_error = max_error(Y_cf_test, Y_mlp_test)

    rows = [
        {
            "seed": seed,
            "N_train": N_train,
            "method": "Closed-form",
            "mlp_width": np.nan,
            "mlp_depth": np.nan,
            "mlp_lr": np.nan,
            "mlp_epochs": np.nan,
            "train_mse": cf_train_mse,
            "test_mse_vs_f": cf_test_mse,
            "max_error_vs_f": cf_max_error,
            "mse_vs_closed_form": 0.0,
            "max_error_vs_closed_form": 0.0,
            "L_hat": L_hat,
            "L_used": L,
        },
        {
            "seed": seed,
            "N_train": N_train,
            "method": f"ReLU MLP (W={mlp_width})",
            "mlp_width": mlp_width,
            "mlp_depth": mlp_depth,
            "mlp_lr": mlp_lr,
            "mlp_epochs": mlp_epochs,
            "train_mse": mlp_train_mse,
            "test_mse_vs_f": mlp_test_mse,
            "max_error_vs_f": mlp_max_error,
            "mse_vs_closed_form": mlp_vs_cf_test_mse,
            "max_error_vs_closed_form": mlp_vs_cf_max_error,
            "L_hat": L_hat,
            "L_used": L,
        },
    ]

    return rows


def summarize_results(df):
    """
    Compute mean and standard deviation over seeds.

    The grouping is by:
        N_train, method, mlp_width

    For the closed-form rows, mlp_width is NaN.
    """
    metric_cols = [
        "train_mse",
        "test_mse_vs_f",
        "max_error_vs_f",
        "mse_vs_closed_form",
        "max_error_vs_closed_form",
        "L_hat",
        "L_used",
    ]

    summary = (
        df.groupby(
            ["N_train", "method", "mlp_width"],
            as_index=False,
            dropna=False,
        )[metric_cols]
        .agg(["mean", "std"])
    )

    # Flatten multi-index columns
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]

    return summary


def print_results_table(df, title="Results"):
    """
    Print individual run results in a readable table.
    """
    display_cols = [
        "seed",
        "N_train",
        "method",
        "mlp_width",
        "train_mse",
        "test_mse_vs_f",
        "max_error_vs_f",
        "mse_vs_closed_form",
        "max_error_vs_closed_form",
    ]

    print(f"\n{title}:")
    print(
        df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4e}",
        )
    )


def print_summary_table(summary_df):
    """
    Print mean/std summary in a readable table.
    """
    display_cols = [
        "N_train",
        "method",
        "mlp_width",
        "train_mse_mean",
        "train_mse_std",
        "test_mse_vs_f_mean",
        "test_mse_vs_f_std",
        "max_error_vs_f_mean",
        "max_error_vs_f_std",
        "mse_vs_closed_form_mean",
        "mse_vs_closed_form_std",
        "max_error_vs_closed_form_mean",
        "max_error_vs_closed_form_std",
    ]

    print("\nSummary over seeds:")
    print(
        summary_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4e}",
        )
    )


def main():
    # -----------------------------
    # Experiment settings
    # -----------------------------
    N_train_list = [50, 200, 1000, 5000]

    num_runs = 20
    seeds = list(range(num_runs))

    # MLP settings
    mlp_width_list = [64, 128]
    mlp_depth = 3
    mlp_lr = 1e-3
    mlp_epochs = 5000

    N_test = 2000
    low, high = -1.0, 1.0
    lipschitz_safety = 1.10

    all_rows = []

    # -----------------------------
    # Run all experiments
    # -----------------------------
    for N_train in N_train_list:
        for mlp_width in mlp_width_list:
            for seed in seeds:
                print(
                    f"\nRunning N_train = {N_train}, "
                    f"width = {mlp_width}, seed = {seed}"
                )

                rows = run_one_experiment(
                    N_train=N_train,
                    N_test=N_test,
                    low=low,
                    high=high,
                    seed=seed,
                    lipschitz_safety=lipschitz_safety,
                    mlp_width=mlp_width,
                    mlp_depth=mlp_depth,
                    mlp_lr=mlp_lr,
                    mlp_epochs=mlp_epochs,
                )

                all_rows.extend(rows)

                temp_df = pd.DataFrame(rows)
                print_results_table(
                    temp_df,
                    title=(
                        f"Results for N_train={N_train}, "
                        f"width={mlp_width}, seed={seed}"
                    ),
                )

    df = pd.DataFrame(all_rows)
    summary_df = summarize_results(df)

    # -----------------------------
    # Save results
    # -----------------------------
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = results_dir / "mlp_vs_closed_form_R_all_runs.csv"
    summary_csv_path = results_dir / "mlp_vs_closed_form_R_summary.csv"

    df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    # -----------------------------
    # Print final tables
    # -----------------------------
    print_results_table(df, title="Full individual-run results")
    print_summary_table(summary_df)

    print("\nSaved raw results CSV:")
    print(raw_csv_path)

    print("\nSaved summary CSV:")
    print(summary_csv_path)


if __name__ == "__main__":
    main()