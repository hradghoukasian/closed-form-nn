"""100D noisy-study targets: one adapted local-sign target and two unchanged folds.

Voronoi tents explicitly replace the earlier disjoint Euclidean-ball bumps,
whose supports received no uniform 100D samples. No test-driven rescaling,
projection, or sampling change is used. The four-layer fold appears once.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist
from benchmark_targets import make_target as original_target

GROUPS = ("Hard", "Tied")
CASES = [
    dict(case="voronoi_K64_d100", name="64 signed Voronoi tents", group="Hard",
         family="voronoi", level=64, d=100),
    dict(case="folds_T1_d100", name="Folding: 1 layer", group="Tied",
         family="folds", level=1, d=100),
    dict(case="folds_T4_d100", name="Folding: 4 layers", group="Tied",
         family="folds", level=4, d=100),
]


@dataclass
class VoronoiTents:
    centers: np.ndarray
    signs: np.ndarray
    center_distances: np.ndarray

    def components(self, x):
        x = np.asarray(x, dtype=float)
        d = self.centers.shape[1]
        flat = x.reshape(-1, d)
        squared = cdist(flat, self.centers, "sqeuclidean")
        nearest = squared.argmin(axis=1)
        gap = squared - squared[np.arange(len(flat)), nearest, None]
        denominator = 2*self.center_distances[nearest].copy()
        denominator[np.arange(len(flat)), nearest] = 1.
        distance_to_faces = gap / denominator
        distance_to_faces[np.arange(len(flat)), nearest] = np.inf
        height = np.maximum(distance_to_faces.min(axis=1), 0.)
        return nearest.reshape(x.shape[:-1]), height.reshape(x.shape[:-1])

    def __call__(self, x):
        index, height = self.components(x)
        return self.signs[index] * height


def make_target(case, seed):
    if case["family"] == "folds":
        return original_target(case, seed)
    if case["family"] != "voronoi":
        raise ValueError(case)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    signs_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    centers = rng.uniform(-1., 1., (case["level"], case["d"]))
    distances = cdist(centers, centers)
    assert np.min(distances + np.eye(len(centers))*1e9) > 0
    return VoronoiTents(centers, signs_rng.choice([-1., 1.], len(centers)), distances)


def target_diagnostics(f, case, x, test):
    y, yt = f(x), f(test)
    out = dict(target_euclidean_L=1., clean_train_mean=float(y.mean()),
               clean_test_mean=float(yt.mean()), clean_test_variance=float(yt.var()),
               clean_train_nonzero_count=int(np.count_nonzero(y)),
               clean_test_nonzero_count=int(np.count_nonzero(yt)))
    if case["family"] == "voronoi":
        train_cells, h = f.components(x)
        test_cells, ht = f.components(test)
        seen = np.unique(train_cells[h > 0])
        out.update(target_definition="signed distance to the boundary of each Voronoi cell",
                   centers=case["level"], train_cells_observed=len(seen),
                   test_cells_observed=len(np.unique(test_cells)),
                   test_fraction_in_unseen_cell=float(np.mean(~np.isin(test_cells, seen))),
                   test_mean_tent_height=float(ht.mean()),
                   test_max_tent_height=float(ht.max()))
    return out
