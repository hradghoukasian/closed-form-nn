"""Formula, CV and neural routines retained from the validated benchmark.

Training retains EPOCHS+1 full-batch Adam updates. Constrained models are
materialized before prediction; saved weights have ordinary Linear layers.
Only Euclidean distance evaluation is accelerated with scipy.cdist.
"""
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import spectral_norm, orthogonal
from torch.nn.utils.parametrize import remove_parametrizations
from scipy.spatial.distance import cdist


def cross_distances(a, b, metric):
    if metric != "l2":
        raise ValueError("The grouped experiments use the Euclidean metric (l2).")
    return cdist(a, b, metric="euclidean")


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


def sample_uniform_Rd(n, d, low=-1.0, high=1.0, seed=None):
    """
    Sample n points uniformly from [low, high]^d.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, d))


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
