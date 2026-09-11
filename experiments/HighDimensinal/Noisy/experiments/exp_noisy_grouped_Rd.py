"""Noisy 100D Hard/Tied study: python3 experiments/exp_noisy_grouped_Rd.py.

Edit the settings below. Beta is the Euclidean local-averaging radius.
CV sees only noisy training labels; final fitting uses the entire training set.
The original noiseless entry point, exp_mlp_Rd.py, is independent of this file.
"""

# ============================================================
# USER SETTINGS
# ============================================================
SIGMAS = (0.1, 0.2, 0.8)
NUM_RUNS = 2
N_TRAIN = 1000
N_TEST = 2000
N_LIP = 1000
USE_BETA_CV = True
CV_FOLDS = 5
BETA_GRID_SIZE = 30
FIXED_BETA = 0.0              # Used only when USE_BETA_CV = False.
FORMULA_L = 1.0               # Fixed oracle bound; never estimated from noise.
NN_LIPSCHITZ = 1.0
MLP_RATIOS = (1.0,)
MLP_DEPTH = 3
MLP_LR = 1e-3
MLP_EPOCHS = 1000             # Original helper uses epochs + 1 updates.
BASE_SEED = 12345
WORKERS = 6
SAVE_MODELS = True

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import scipy
import benchmark_helpers as bench
from beta_cv import make_beta_grid, select_beta_cv, refit_full_training
from noisy_targets import CASES, GROUPS, make_target, target_diagnostics
from noisy_tables import export_results, sigma_tag

ROOT = HERE.parent
FAMILIES = ("unconstrained", "spectral", "orthogonal")
MODEL_NAMES = dict(formula="Formula", unconstrained="ReLU MLP", spectral="Spectral MLP",
                   orthogonal="Orthogonal MLP", constant="Training-mean constant")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def init_worker():
    bench.torch.set_num_threads(1)
    bench.torch.set_num_interop_threads(1)


def settings_from_args(args):
    config = dict(sigmas=list(SIGMAS), num_runs=NUM_RUNS, n_train=N_TRAIN,
                  n_test=N_TEST, n_lip=N_LIP, use_beta_cv=USE_BETA_CV,
                  cv_folds=CV_FOLDS, beta_grid_size=BETA_GRID_SIZE,
                  fixed_beta=FIXED_BETA, formula_l=FORMULA_L, nn_lipschitz=NN_LIPSCHITZ,
                  mlp_ratios=list(MLP_RATIOS), mlp_depth=MLP_DEPTH, mlp_lr=MLP_LR,
                  mlp_epochs=MLP_EPOCHS, base_seed=BASE_SEED,
                  save_models=SAVE_MODELS, metric="l2", d=100)
    for key in ("sigmas", "num_runs", "n_train", "n_test", "n_lip", "use_beta_cv",
                "cv_folds", "beta_grid_size", "fixed_beta", "mlp_epochs"):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    for key, minimum in (("num_runs", 1), ("n_train", 4), ("n_test", 1),
                         ("n_lip", 2), ("mlp_depth", 1), ("mlp_epochs", 0),
                         ("beta_grid_size", 4)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("fixed_beta", "formula_l", "nn_lipschitz"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(config["mlp_lr"]) or config["mlp_lr"] <= 0:
        raise ValueError("MLP_LR must be finite and positive")
    for key, positive in (("sigmas", False), ("mlp_ratios", True)):
        v = config[key]
        if not v or len(set(v)) != len(v) or any(
                not math.isfinite(x) or (x <= 0 if positive else x < 0) for x in v):
            raise ValueError(f"Invalid {key}: require distinct finite values")
    if type(config["use_beta_cv"]) is not bool:
        raise ValueError("USE_BETA_CV must be True or False")
    if config["use_beta_cv"] and not 2 <= config["cv_folds"] <= config["n_train"]//2:
        raise ValueError("CV needs at least two folds and two validation points per fold")
    return config


def run_one_experiment(case, run_seed, sigma, config, output):
    started = time.perf_counter()
    output = Path(output)
    key = f"{case['case']}_seed{run_seed}_sigma{sigma_tag(sigma)}"
    f = make_target(case, run_seed+11)
    x = bench.sample_uniform_Rd(config["n_train"], case["d"], seed=run_seed+101)
    test = bench.sample_uniform_Rd(config["n_test"], case["d"], seed=run_seed+202)
    lip = bench.sample_uniform_Rd(config["n_lip"], case["d"], seed=run_seed+303)
    clean_y, clean_test = f(x), f(test)
    # Pair inputs, target, standardized noise and initialization across sigmas.
    noise = np.random.default_rng(run_seed+606).normal(size=len(x))
    noisy_y = clean_y + sigma*noise
    diagnostics = target_diagnostics(f, case, x, test)
    cf_start = time.perf_counter()
    L = config["formula_l"]
    if config["use_beta_cv"]:
        beta, cv_info = select_beta_cv(x, noisy_y, run_seed+505, L=L,
                                      folds=config["cv_folds"],
                                      grid_size=config["beta_grid_size"])
    else:
        beta = config["fixed_beta"]
        cv_info = dict(cv_betas=[], cv_scores=[], cv_fold_scores=[], cv_best_index=None,
                       cv_validation_mse=np.nan, cv_folds=0,
                       cv_fitting_sizes=[], cv_validation_sizes=[])
    # Fresh averages at ALL training sites, using ALL noisy training labels.
    averaged_y, refit_info = refit_full_training(x, noisy_y, beta, L=L)
    cf_test = bench.formula_grid_predict(test, x, averaged_y, [L], "l2")[0]
    cf_train = bench.formula_grid_predict(x, x, averaged_y, [L], "l2")[0]
    cf_lip = bench.formula_grid_predict(lip, x, averaged_y, [L], "l2")[0]
    cf_seconds = time.perf_counter()-cf_start
    noisy_L_D = bench.estimate_lipschitz_pairwise_fast(x, noisy_y, "l2")
    baseline_mse = float(np.mean((noisy_y.mean()-clean_test)**2))
    diagnostics.update(case=case["case"], run=run_seed, sigma=sigma,
                       noisy_L_D=noisy_L_D, **cv_info, **refit_info)
    write_json(output/"records"/f"{key}_diagnostics.json", diagnostics)
    # Inference uses X, fitted values and L; beta is fit metadata, not inference state.
    formula_scalars = int(x.size + averaged_y.size + 1)
    common = dict(case=case["case"], experiment=case["name"], group=case["group"],
                  target_family=case["family"], target_level=case["level"],
                  run=run_seed, sigma=sigma, d=case["d"], N_train=len(x), N_test=len(test),
                  formula_scalars=formula_scalars, metric="l2", formula_L=L,
                  beta_mode="cv" if config["use_beta_cv"] else "fixed",
                  noisy_L_D=noisy_L_D, cv_folds=cv_info["cv_folds"],
                  cv_validation_mse=cv_info["cv_validation_mse"], **refit_info,
                  clean_test_variance=diagnostics["clean_test_variance"],
                  train_mean_baseline_mse=baseline_mse)
    rows, predictions = [], {"clean_test": clean_test, "clean_train": clean_y,
                             "noisy_train": noisy_y}

    def add_row(family, p, train_p, lip_p, scalars, bound, width=None, ratio=1., seconds=0.):
        mse = float(np.mean((p-clean_test)**2))
        train_mse = float(np.mean((train_p-noisy_y)**2))
        if not np.isfinite([mse, train_mse, bound]).all():
            raise FloatingPointError(f"Nonfinite {key}, {family}")
        rows.append(dict(common, family=family, method=MODEL_NAMES[family],
                         target_scalar_ratio=ratio, width=width, stored_scalars=scalars,
                         scalar_ratio_to_formula=scalars/formula_scalars,
                         mlp_depth=config["mlp_depth"] if family in FAMILIES else None,
                         test_mse=mse, train_noisy_mse=train_mse,
                         train_clean_mse=float(np.mean((train_p-clean_y)**2)),
                         mse_over_constant=mse/baseline_mse if baseline_mse else np.nan,
                         lipschitz_bound=bound,
                         empirical_lipschitz=bench.empirical_lipschitz_on_points(lip, lip_p, "l2"),
                         optimizer_updates=config["mlp_epochs"]+1 if family in FAMILIES else 0,
                         training_seconds=seconds))
        prediction_key = family if family not in FAMILIES else f"{family}_ratio{ratio:g}"
        predictions[prediction_key] = p

    add_row("formula", cf_test, cf_train, cf_lip, formula_scalars, L, seconds=cf_seconds)
    add_row("constant", np.full(len(test), noisy_y.mean()), np.full(len(x), noisy_y.mean()),
            np.full(len(lip), noisy_y.mean()), 1, 0., ratio=np.nan)
    if config["save_models"]:
        np.savez_compressed(output/"models"/f"{key}_formula.npz", X=x, values=averaged_y, L=L)

    for ratio in config["mlp_ratios"]:
        width = bench.width_for_parameter_budget(ratio*formula_scalars, case["d"], config["mlp_depth"])
        expected = ((config["mlp_depth"]-1)*width**2
                    + (case["d"]+config["mlp_depth"]+1)*width + 1)
        for family in FAMILIES:
            model_key = f"{key}_{family}_ratio{ratio:g}"
            nn_start = time.perf_counter()
            with (output/"logs"/f"{model_key}.txt").open("w") as log, redirect_stdout(log):
                model = bench.train_mlp_Rd(
                    x, noisy_y, width=width, depth=config["mlp_depth"], lr=config["mlp_lr"],
                    num_epochs=config["mlp_epochs"], seed=run_seed+404+width, print_every=100,
                    family=family, lipschitz_bound=config["nn_lipschitz"])
            nn_seconds = time.perf_counter()-nn_start
            actual = sum(p.numel() for p in model.parameters())
            assert actual == expected == sum(v.numel() for v in model.state_dict().values())
            bound = math.prod(bench.torch.linalg.matrix_norm(layer.weight.detach().double(), ord=2).item()
                              for layer in model.net if isinstance(layer, bench.nn.Linear))
            if family != "unconstrained":
                assert bound <= config["nn_lipschitz"]+1e-6
            add_row(family, bench.predict_mlp_Rd(model, test), bench.predict_mlp_Rd(model, x),
                    bench.predict_mlp_Rd(model, lip), actual, bound, width, ratio, nn_seconds)
            if config["save_models"]:
                bench.torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                                 output/"models"/f"{model_key}.pt")
    np.savez_compressed(output/"predictions"/f"{key}.npz", **predictions)
    write_json(output/"records"/f"{key}.json", rows)
    return dict(case=case["case"], seed=run_seed, sigma=sigma, beta=beta,
                seconds=time.perf_counter()-started, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", nargs="+", type=float)
    parser.add_argument("--runs", dest="num_runs", type=int)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--n-lip", type=int)
    parser.add_argument("--epochs", dest="mlp_epochs", type=int)
    cv = parser.add_mutually_exclusive_group()
    cv.add_argument("--cv", dest="use_beta_cv", action="store_true")
    cv.add_argument("--no-cv", dest="use_beta_cv", action="store_false")
    parser.set_defaults(use_beta_cv=None)
    parser.add_argument("--cv-folds", type=int)
    parser.add_argument("--beta-grid-size", type=int)
    parser.add_argument("--fixed-beta", type=float)
    parser.add_argument("--output", type=Path, help="New directory; existing directories are rejected.")
    args = parser.parse_args()
    config = settings_from_args(args)
    if args.workers < 1:
        parser.error("--workers must be positive")
    seeds = [config["base_seed"]+1000*i for i in range(config["num_runs"])]
    jobs = [(c, seed, sigma) for sigma in config["sigmas"] for c in CASES for seed in seeds]
    workers = min(args.workers, os.cpu_count() or 1, len(jobs))
    if bench.torch.cuda.is_available():
        workers = 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or ROOT/"results"/f"noisy_d100_N{config['n_train']}_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("records", "logs", "models", "predictions"):
        (output/name).mkdir()
    manifest = dict(settings=config, cases=CASES, run_seeds=seeds, workers=workers,
                    beta_grid=make_beta_grid(100, config["beta_grid_size"]).tolist(),
                    python=platform.python_version(), torch=bench.torch.__version__,
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    device="cuda" if bench.torch.cuda.is_available() else "cpu",
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(HERE.glob("*.py"))}, status="running")
    write_json(output/"config.json", manifest)
    print(f"Results: {output}\n{len(jobs)} target/seed/noise jobs; {workers} workers; "
          f"beta CV={config['use_beta_cv']}; formula L={config['formula_l']}", flush=True)
    start, all_rows = time.perf_counter(), []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=init_worker) as pool:
            futures = [pool.submit(run_one_experiment, c, seed, sigma, config, str(output))
                       for c, seed, sigma in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                all_rows.extend(result["rows"])
                pd.DataFrame(all_rows).to_csv(output/"raw_partial.csv", index=False)
                print(f"[{i}/{len(jobs)}] {result['case']}, seed={result['seed']}, "
                      f"sigma={result['sigma']:g}, beta={result['beta']:.4g}: "
                      f"{result['seconds']:.1f}s", flush=True)
        assert len(all_rows) == len(jobs)*(2+3*len(config["mlp_ratios"]))
        report = export_results(all_rows, CASES, config, output)
        (output/"raw_partial.csv").unlink(missing_ok=True)
        manifest.update(status="complete", rows=len(all_rows), elapsed_seconds=time.perf_counter()-start)
        write_json(output/"config.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", completed_rows=len(all_rows), error=repr(exc))
        write_json(output/"config.json", manifest)
        raise
    print("\n"+report, flush=True)
    print(f"\nSaved text, CSV and LaTeX tables in: {output}", flush=True)


if __name__ == "__main__":
    main()
