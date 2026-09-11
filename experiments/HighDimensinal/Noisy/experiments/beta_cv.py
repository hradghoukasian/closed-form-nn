"""Radius averaging selected by training-only K-fold CV; fixed formula L.

Only noisy training labels enter CV. Every fold constructs its own averages
from its fitting subset. Final fitting recomputes averages at every training
site using all training observations and the selected *absolute* radius.
"""
import numpy as np
from scipy.spatial.distance import cdist


def make_beta_grid(d, size=30):
    """Predefined grid covering small, typical and all-point neighborhoods.

    RMS pairwise distance for uniform [-1,1]^d inputs is sqrt(2d/3).
    Grid does not depend on train, validation or test observations/labels.
    Zero allows no averaging; the cube diameter includes all points.
    """
    if d < 1 or size < 4:
        raise ValueError("Need d >= 1 and at least four beta candidates")
    typical = np.sqrt(2*d/3)
    return np.r_[0., np.linspace(0.5*typical, 1.5*typical, size-2), 2*np.sqrt(d)]


def local_average_from_distances(distances, y, beta):
    neighbors = (distances <= beta).astype(float)
    counts = neighbors.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every averaging site must have a neighbor")
    return (neighbors @ np.asarray(y)) / counts, counts.astype(int)


def predict_from_distances(distances, values, L):
    return 0.5*((values[None, :] + L*distances).min(axis=1)
                + (values[None, :] - L*distances).max(axis=1))


def select_beta_cv(x, noisy_y, seed, L=1., folds=5, grid_size=30, betas=None):
    x, noisy_y = np.asarray(x), np.asarray(noisy_y)
    if not (2 <= folds <= len(x)//2):
        raise ValueError("CV needs at least two folds and two points per fold")
    betas = make_beta_grid(x.shape[1], grid_size) if betas is None else np.asarray(betas)
    if betas.ndim != 1 or len(betas) == 0 or not np.isfinite(betas).all() or np.any(betas < 0):
        raise ValueError("Beta candidates must be finite and nonnegative")
    betas = np.unique(betas)
    splits = np.array_split(np.random.default_rng(seed).permutation(len(x)), folds)
    distance = cdist(x, x)
    scores = np.zeros((folds, len(betas)))
    fitting_sizes, validation_sizes = [], []
    for k, validation in enumerate(splits):
        fitting = np.concatenate([v for j, v in enumerate(splits) if j != k])
        fitting_sizes.append(len(fitting)); validation_sizes.append(len(validation))
        fit_distance = distance[np.ix_(fitting, fitting)]
        query_distance = distance[np.ix_(validation, fitting)]
        for j, beta in enumerate(betas):
            values, _ = local_average_from_distances(fit_distance, noisy_y[fitting], beta)
            predictions = predict_from_distances(query_distance, values, L)
            scores[k, j] = float(np.mean((predictions-noisy_y[validation])**2))
    means = np.average(scores, axis=0, weights=validation_sizes)
    best = int(np.argmin(means))  # Ascending grid: exact ties choose smaller beta.
    return float(betas[best]), dict(cv_betas=betas.tolist(), cv_scores=means.tolist(),
                                  cv_fold_scores=scores.tolist(), cv_best_index=best,
                                  cv_validation_mse=float(means[best]), cv_folds=folds,
                                  cv_fitting_sizes=fitting_sizes,
                                  cv_validation_sizes=validation_sizes)


def refit_full_training(x, noisy_y, beta, L=1.):
    """Return inference state (all X, averaged values, fixed L) and fit diagnostics."""
    x, noisy_y = np.asarray(x), np.asarray(noisy_y)
    distance = cdist(x, x)
    values, counts = local_average_from_distances(distance, noisy_y, beta)
    return values, dict(final_refit_n=len(x), final_average_count_min=int(counts.min()),
                        final_average_count_mean=float(counts.mean()),
                        final_average_count_max=int(counts.max()), beta_selected=float(beta),
                        final_L=float(L))
