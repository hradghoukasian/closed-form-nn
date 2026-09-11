"""Check beta CV independently and audit a completed noisy-study archive.

Usage: python3 experiments/verify_noisy_results.py noisy_example_results
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from beta_cv import select_beta_cv, refit_full_training
import benchmark_helpers as bench
from noisy_targets import make_target, CASES
from noisy_tables import sigma_tag


def check_cv_independently():
    rng = np.random.default_rng(711)
    x, y = rng.normal(size=(13, 2)), rng.normal(size=13)
    betas, seed, folds = np.array([0., .4, 1., 2., 20.]), 42, 3
    beta, info = select_beta_cv(x, y, seed, folds=folds, betas=betas)
    splits = np.array_split(np.random.default_rng(seed).permutation(len(x)), folds)
    scores = np.zeros((folds, len(betas)))
    # Separate, direct Euclidean loops; no reuse of CV's distance/average helper.
    for i, valid in enumerate(splits):
        fit = np.concatenate([s for j, s in enumerate(splits) if j != i])
        for j, radius in enumerate(betas):
            values = np.array([np.mean([y[t] for t in fit
                                        if np.linalg.norm(x[t]-x[s]) <= radius]) for s in fit])
            predictions = []
            for q in valid:
                up = min(values[n]+np.linalg.norm(x[q]-x[s]) for n, s in enumerate(fit))
                lo = max(values[n]-np.linalg.norm(x[q]-x[s]) for n, s in enumerate(fit))
                predictions.append((up+lo)/2)
            scores[i, j] = np.mean((np.array(predictions)-y[valid])**2)
    assert np.allclose(scores, info["cv_fold_scores"], atol=1e-12)
    assert beta == betas[np.argmin(np.average(scores, axis=0, weights=list(map(len, splits))))]
    values, fitted = refit_full_training(x, y, beta)
    direct = np.array([y[np.linalg.norm(x-p, axis=1) <= beta].mean() for p in x])
    assert len(values) == fitted["final_refit_n"] == len(x)
    assert np.allclose(values, direct, atol=1e-12)
    assert np.allclose(refit_full_training(x, y, 0.)[0], y)
    assert np.allclose(refit_full_training(x, y, 20.)[0], y.mean())


def audit(output):
    bench.torch.set_num_threads(1)
    manifest = json.loads((output/"config.json").read_text())
    assert manifest["status"] == "complete"
    config = manifest["settings"]
    raw, summary = pd.read_csv(output/"raw.csv"), pd.read_csv(output/"summary.csv")
    expected = len(CASES)*config["num_runs"]*len(config["sigmas"])*(2+3*len(config["mlp_ratios"]))
    assert len(raw) == expected
    assert raw.final_refit_n.eq(config["n_train"]).all()
    assert raw.d.eq(100).all() and raw.formula_L.eq(config["formula_l"]).all()
    for filename, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((Path(__file__).parent/filename).read_bytes()).hexdigest() == digest
    checked_states, checked_formulas = 0, 0
    for case in CASES:
        for seed in manifest["run_seeds"]:
            f = make_target(case, seed+11)
            x = bench.sample_uniform_Rd(config["n_train"], 100, seed=seed+101)
            test = bench.sample_uniform_Rd(config["n_test"], 100, seed=seed+202)
            clean_y, clean_test = f(x), f(test)
            noise = np.random.default_rng(seed+606).normal(size=len(x))
            for sigma in config["sigmas"]:
                key = f"{case['case']}_seed{seed}_sigma{sigma_tag(sigma)}"
                g = raw[raw.case.eq(case["case"]) & raw.run.eq(seed) & raw.sigma.eq(sigma)]
                predictions = np.load(output/"predictions"/f"{key}.npz")
                y = clean_y+sigma*noise
                assert np.array_equal(predictions["clean_test"], clean_test)
                assert np.array_equal(predictions["noisy_train"], y)
                diag = json.loads((output/"records"/f"{key}_diagnostics.json").read_text())
                assert diag["final_refit_n"] == len(x)
                if config["use_beta_cv"]:
                    assert diag["beta_selected"] == diag["cv_betas"][np.argmin(diag["cv_scores"])]
                    assert sum(diag["cv_validation_sizes"]) == len(x)
                    assert all(a+b == len(x) for a, b in zip(
                        diag["cv_validation_sizes"], diag["cv_fitting_sizes"]))
                if config["save_models"]:
                    state = np.load(output/"models"/f"{key}_formula.npz")
                    assert np.array_equal(state["X"], x)
                    # Independent full-data neighborhood averages.
                    direct = np.array([y[np.linalg.norm(x-p, axis=1) <= diag["beta_selected"]].mean()
                                       for p in x])
                    assert np.allclose(state["values"], direct, atol=1e-12)
                    restored = bench.formula_grid_predict(test, state["X"], state["values"],
                                                          [float(state["L"])], "l2")[0]
                    assert np.array_equal(restored, predictions["formula"])
                    checked_formulas += 1
                for row in g.itertuples():
                    name = row.family if row.family in ("formula", "constant") else (
                        f"{row.family}_ratio{row.target_scalar_ratio:g}")
                    assert np.isclose(np.mean((predictions[name]-clean_test)**2), row.test_mse,
                                      rtol=1e-12, atol=1e-14)
                    assert np.isclose(row.scalar_ratio_to_formula,
                                      row.stored_scalars/row.formula_scalars)
                    if row.family == "constant":
                        assert np.allclose(predictions[name], y.mean())
                    elif row.family != "formula" and config["save_models"]:
                        state = bench.torch.load(output/"models"/f"{key}_{name}.pt", weights_only=True)
                        assert sum(v.numel() for v in state.values()) == row.stored_scalars
                        model = bench.ReLUMlpRd(input_dim=100, width=int(row.width), depth=int(row.mlp_depth))
                        model.load_state_dict(state)
                        assert np.array_equal(bench.predict_mlp_Rd(model, test), predictions[name])
                        checked_states += 1
                    if row.family in ("spectral", "orthogonal"):
                        assert row.lipschitz_bound <= config["nn_lipschitz"]+1e-6
    for row in summary.itertuples():
        g = raw[raw.case.eq(row.case) & raw.sigma.eq(row.sigma)]
        v = g[g.family.eq(row.family)]
        if row.family not in ("formula", "constant"):
            v = v[v.target_scalar_ratio.eq(row.target_scalar_ratio)]
        assert len(v) == config["num_runs"] == v.run.nunique()
        assert np.isclose(row.test_mse_mean, v.test_mse.mean(), rtol=1e-12)
        assert np.isclose(row.test_mse_std, v.test_mse.std(ddof=1), equal_nan=True)
        cf = g[g.family.eq("formula")].test_mse.mean()
        assert np.isclose(row.mse_percent_of_formula, 100*v.test_mse.mean()/cf)
    return dict(status="passed", rows=len(raw), summary_rows=len(summary),
                restored_neural_states=checked_states, restored_formula_states=checked_formulas,
                independent_cv_reference=True, full_training_refits=True,
                clean_test_mse_recomputed=True, paired_noise=True, scalar_counts=True,
                recorded_lipschitz_bounds=True, source_hashes_match=True,
                ratios_of_means=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check_cv_independently()
    result = audit(args.output)
    (args.output/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
