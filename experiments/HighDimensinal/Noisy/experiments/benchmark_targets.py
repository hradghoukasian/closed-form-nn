"""The six selected noiseless targets on [-1,1]^d with Euclidean distance."""
from dataclasses import dataclass
import numpy as np


GROUPS = (
    "(Expected) Clear wins",
    "(Expected) Clear near ties [85%-115%]",
    "(Expected) Clear losses",
)

CASES = [
    dict(case="bumps_K64_d2", name="64 signed bumps", group=GROUPS[0],
         family="bumps", level=8, d=2),
    dict(case="folds_T4_d2", name="Folding: 4 layers", group=GROUPS[0],
         family="folds", level=4, d=2),
    dict(case="folds_T1_d100", name="Folding: 1 layer", group=GROUPS[1],
         family="folds", level=1, d=100),
    dict(case="folds_T4_d100", name="Folding: 4 layers", group=GROUPS[1],
         family="folds", level=4, d=100),
    dict(case="affine_l1_d100", name="Affine, ||w||_1=1", group=GROUPS[2],
         family="affine", level=1, d=100),
    dict(case="tanh_ridge_d100", name="Tanh ridge", group=GROUPS[2],
         family="tanh_ridge", level=1, d=100),
]


def random_orthogonal(rng, d):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


@dataclass
class Target:
    family: str
    d: int
    level: int
    signs: np.ndarray | None = None
    matrices: tuple = ()
    centers: np.ndarray | None = None
    radius: float | None = None
    w: np.ndarray | None = None
    b: float = 0.0
    x0: np.ndarray | None = None

    def bump_components(self, x):
        x = np.asarray(x, dtype=float)
        distance = np.linalg.norm(x[..., None, :] - self.centers, axis=-1)
        index = distance.argmin(axis=-1)
        height = np.maximum(self.radius - distance.min(axis=-1), 0.0)
        return index, height

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if self.family == "affine":
            return x @ self.w + self.b
        if self.family == "tanh_ridge":
            return np.tanh(x @ self.w + self.b)
        if self.family == "bumps":
            index, height = self.bump_components(x)
            return self.signs[index] * height
        for q in self.matrices:
            x = np.abs(x @ q.T)
        return x @ self.signs / np.sqrt(self.d)


def make_target(case, seed):
    """Independent RNG streams reproduce the preceding two-seed experiments."""
    family, d, level = case["family"], case["d"], case["level"]
    if family == "tanh_ridge":
        # Original ridge teacher: ||w||_2=1 and an interior zero crossing.
        ridge_rng = np.random.default_rng(seed)
        w = ridge_rng.standard_normal(d)
        w /= np.linalg.norm(w)
        x0 = ridge_rng.uniform(-0.8, 0.8, size=d)
        return Target(family, d, level, w=w, b=-float(w @ x0), x0=x0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    signs_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    if family == "affine":
        w = rng.standard_normal(d)
        w /= np.abs(w).sum()
        b_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        return Target(family, d, level, w=w, b=float(b_rng.uniform(-0.5, 0.5)))
    if family == "bumps":
        if d != 2:
            raise ValueError("This selected bump target uses a two-dimensional grid.")
        axis = -1.0 + (2 * np.arange(level) + 1) / level
        centers = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        return Target(family, d, level, signs_rng.choice([-1., 1.], len(centers)),
                      centers=centers, radius=1.0 / level)
    if family != "folds":
        raise ValueError(f"Unknown target family: {family}")
    matrices = tuple(random_orthogonal(rng, d) for _ in range(level))
    return Target(family, d, level, signs_rng.choice([-1., 1.], d), matrices)


def target_diagnostics(target, x_train, x_test):
    y_train, y_test = target(x_train), target(x_test)
    result = dict(train_mean=float(y_train.mean()), test_mean=float(y_test.mean()),
                  test_variance=float(y_test.var()),
                  train_mean_baseline_mse=float(np.mean((y_test-y_train.mean())**2)),
                  target_euclidean_L=float(np.linalg.norm(target.w))
                  if target.family == "affine" else 1.0)
    if target.family == "affine":
        result.update(w_l1_norm=float(np.abs(target.w).sum()),
                      w_l2_norm=float(np.linalg.norm(target.w)), intercept=target.b,
                      weights=target.w.tolist())
    if target.family == "tanh_ridge":
        z = x_test @ target.w + target.b
        result.update(w_l2_norm=float(np.linalg.norm(target.w)), intercept=target.b,
                      weights=target.w.tolist(), zero_crossing=target.x0.tolist(),
                      test_preactivation_mean=float(z.mean()),
                      test_preactivation_std=float(z.std()),
                      test_fraction_abs_preactivation_le_0_25=float(np.mean(np.abs(z) <= 0.25)),
                      test_tanh_vs_linear_mse=float(np.mean((np.tanh(z)-z)**2)))
    if target.family == "bumps":
        train_index, train_height = target.bump_components(x_train)
        test_index, test_height = target.bump_components(x_test)
        visited = np.unique(train_index[train_height > 0])
        unseen = (test_height > 0) & ~np.isin(test_index, visited)
        result.update(bumps=len(target.centers), radius=target.radius,
                      train_support_fraction=float(np.mean(train_height > 0)),
                      test_support_fraction=float(np.mean(test_height > 0)),
                      bumps_observed=len(visited),
                      test_unseen_bump_fraction=float(unseen.mean()))
    return result
