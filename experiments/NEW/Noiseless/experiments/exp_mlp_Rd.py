"""Run the six grouped benchmarks: python3 experiments/exp_mlp_Rd.py.

Edit USER SETTINGS below. Every invocation creates a separate results folder.
The expected groups describe the earlier results; measured outcomes can change.
"""

# ============================================================
# USER SETTINGS
# ============================================================
NUM_RUNS = 2                   # Set to 1, 2, 20, etc.
N_TRAIN = 1000
N_TEST = 2000
N_LIP = 1000                  # Points for empirical predictor-Lipschitz estimates.
USE_CV = False                # False: formula uses FIXED_L. True: training-only CV.
FIXED_L = 1.0
CV_FOLDS = 5
CV_GRID_SIZE = 20
CV_SPAN = 2.0                 # Fold-local [L_D, L_D + 2], 20 grid points.
NN_LIPSCHITZ = 1.0            # SN/orthogonal bound; independent of formula CV.
MLP_RATIOS = (1.0,)           # Approximately equal storage; or (0.5, 1., 1.5, 2.).
MLP_DEPTH = 3                 # Number of equal-width hidden ReLU layers.
MLP_LR = 1e-3
MLP_EPOCHS = 1000             # Original loop: epochs + 1 optimizer updates.
BASE_SEED = 12345             # Run seeds: BASE_SEED + 1000 * run_index.
WORKERS = 6                  # Independent CPU processes; one thread each.
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

# Direct execution and python -m both resolve the sibling helper modules.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import scipy
import benchmark_helpers as bench
from benchmark_targets import CASES, GROUPS, make_target, target_diagnostics
from benchmark_tables import export_results

ROOT = HERE.parent
FAMILIES = ("unconstrained", "spectral", "orthogonal")
MODEL_NAMES = dict(formula="Formula", unconstrained="ReLU MLP",
                   spectral="Spectral MLP", orthogonal="Orthogonal MLP",
                   constant="Training-mean constant")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def init_worker():
    bench.torch.set_num_threads(1)
    bench.torch.set_num_interop_threads(1)


def settings_from_args(args):
    config = dict(num_runs=NUM_RUNS, n_train=N_TRAIN, n_test=N_TEST, n_lip=N_LIP,
                  use_cv=USE_CV, fixed_l=FIXED_L, cv_folds=CV_FOLDS,
                  cv_grid_size=CV_GRID_SIZE, cv_span=CV_SPAN,
                  nn_lipschitz=NN_LIPSCHITZ, mlp_ratios=list(MLP_RATIOS),
                  mlp_depth=MLP_DEPTH, mlp_lr=MLP_LR, mlp_epochs=MLP_EPOCHS,
                  base_seed=BASE_SEED, save_models=SAVE_MODELS, metric="l2")
    for key in ("num_runs", "n_train", "n_test", "n_lip", "mlp_epochs", "use_cv",
                "fixed_l", "cv_folds", "cv_grid_size", "cv_span"):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    for key, minimum in (("num_runs", 1), ("n_train", 2), ("n_test", 1),
                         ("n_lip", 2), ("mlp_depth", 1), ("mlp_epochs", 0)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if type(config["use_cv"]) is not bool:
        raise ValueError("USE_CV must be True or False")
    for key in ("fixed_l", "nn_lipschitz", "cv_span"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(config["mlp_lr"]) or config["mlp_lr"] <= 0:
        raise ValueError("MLP_LR must be finite and positive")
    ratios = config["mlp_ratios"]
    if not ratios or len(set(ratios)) != len(ratios) or any(
            not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("MLP_RATIOS must contain distinct, finite positive values")
    if config["use_cv"] and not (2 <= config["cv_folds"] <= config["n_train"]//2
                                and config["cv_grid_size"] >= 2):
        raise ValueError("CV needs >=2 folds, >=2 points per fold and >=2 grid points")
    return config


def run_one_experiment(case, run_seed, config, output):
    """Train on this seed's data; the held-out test set never selects L or weights."""
    started = time.perf_counter()
    output = Path(output)
    key = f"{case['case']}_seed{run_seed}"
    f = make_target(case, run_seed + 11)
    x = bench.sample_uniform_Rd(config["n_train"], case["d"], seed=run_seed+101)
    test = bench.sample_uniform_Rd(config["n_test"], case["d"], seed=run_seed+202)
    lip = bench.sample_uniform_Rd(config["n_lip"], case["d"], seed=run_seed+303)
    y, y_test = f(x), f(test)
    diagnostics = target_diagnostics(f, x, test)
    if config["use_cv"]:
        l_hat, l_used, cv_info = bench.select_lipschitz_cv(
            x, y, "l2", run_seed+505, folds=config["cv_folds"],
            grid_size=config["cv_grid_size"], span=config["cv_span"])
    else:
        l_hat = bench.estimate_lipschitz_pairwise_fast(x, y, "l2")
        l_used = float(config["fixed_l"])
        cv_info = dict(cv_folds=0, cv_offset=np.nan, cv_mse=np.nan,
                       cv_offsets="[]", cv_scores="[]", cv_fold_L_D="[]")
    diagnostics.update(case=case["case"], run=run_seed, L_hat_train=l_hat,
                       L_used_cf=l_used, **cv_info)
    write_json(output / "records" / f"{key}_diagnostics.json", diagnostics)
    p_formula = int(x.size + y.size + 1)
    base_mse = diagnostics["train_mean_baseline_mse"]
    common = dict(case=case["case"], experiment=case["name"], group=case["group"],
                  target_family=case["family"], target_level=case["level"],
                  run=run_seed, d=case["d"], N_train=len(x), N_test=len(test),
                  N_lip=len(lip), formula_scalars=p_formula, metric="l2",
                  formula_L_mode="cv" if config["use_cv"] else "fixed",
                  L_used_cf=l_used, L_hat_train=l_hat, **cv_info,
                  target_euclidean_L=diagnostics["target_euclidean_L"],
                  train_mean_baseline_mse=base_mse)
    rows = []

    def add_row(family, prediction, training_prediction, lip_prediction,
                stored_scalars, bound, width=None, ratio=1.0, seconds=0.0):
        error = float(np.mean((prediction-y_test)**2))
        train_error = float(np.mean((training_prediction-y)**2))
        if not np.isfinite([error, train_error, bound]).all():
            raise FloatingPointError(f"Nonfinite result in {key}, {family}")
        row = dict(common, family=family, method=MODEL_NAMES[family],
                   target_scalar_ratio=ratio, width=width,
                   mlp_depth=config["mlp_depth"] if family in FAMILIES else None,
                   stored_scalars=stored_scalars,
                   scalar_ratio_to_formula=stored_scalars/p_formula,
                   test_mse=error, train_mse=train_error,
                   normalized_mse=error/base_mse if base_mse > 0 else np.nan,
                   lipschitz_bound=bound,
                   requested_nn_L=config["nn_lipschitz"]
                   if family in ("spectral", "orthogonal") else np.nan,
                   empirical_lipschitz=bench.empirical_lipschitz_on_points(
                       lip, lip_prediction, "l2"),
                   optimizer_updates=config["mlp_epochs"]+1 if family in FAMILIES else 0,
                   training_seconds=seconds)
        rows.append(row)

    cf_start = time.perf_counter()
    cf_test = bench.formula_grid_predict(test, x, y, [l_used], "l2")[0]
    cf_train = bench.formula_grid_predict(x, x, y, [l_used], "l2")[0]
    cf_lip = bench.formula_grid_predict(lip, x, y, [l_used], "l2")[0]
    if l_used >= l_hat:
        assert np.allclose(cf_train, y, atol=1e-12)
    add_row("formula", cf_test, cf_train, cf_lip, p_formula, l_used,
            seconds=time.perf_counter()-cf_start)
    # One scalar: the training-label mean. No test labels determine this value.
    add_row("constant", np.full(len(test), y.mean()), np.full(len(x), y.mean()),
            np.full(len(lip), y.mean()), 1, 0.0, ratio=np.nan)

    for ratio in config["mlp_ratios"]:
        width = bench.width_for_parameter_budget(ratio*p_formula, case["d"], config["mlp_depth"])
        expected = ((config["mlp_depth"]-1)*width**2
                    + (case["d"]+config["mlp_depth"]+1)*width + 1)
        for family in FAMILIES:
            model_key = f"{key}_{family}_ratio{ratio:g}"
            training_start = time.perf_counter()
            with (output / "logs" / f"{model_key}.txt").open("w") as log, redirect_stdout(log):
                model = bench.train_mlp_Rd(
                    x, y, width=width, depth=config["mlp_depth"], lr=config["mlp_lr"],
                    num_epochs=config["mlp_epochs"], seed=run_seed+404+width,
                    print_every=100, family=family, lipschitz_bound=config["nn_lipschitz"])
            seconds = time.perf_counter()-training_start
            actual = sum(p.numel() for p in model.parameters())
            assert actual == expected == sum(v.numel() for v in model.state_dict().values())
            bound = math.prod(bench.torch.linalg.matrix_norm(layer.weight.detach().double(), ord=2).item()
                              for layer in model.net if isinstance(layer, bench.nn.Linear))
            if family != "unconstrained":
                assert bound <= config["nn_lipschitz"] + 1e-6
            add_row(family, bench.predict_mlp_Rd(model, test), bench.predict_mlp_Rd(model, x),
                    bench.predict_mlp_Rd(model, lip), actual, bound, width, ratio, seconds)
            if config["save_models"]:
                # Save ordinary CPU inference weights, not optimizer state or parametrization buffers.
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bench.torch.save(state, output / "models" / f"{model_key}.pt")
    write_json(output / "records" / f"{key}.json", rows)
    return dict(case=case["case"], seed=run_seed, rows=rows,
                seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", dest="num_runs", type=int)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--n-lip", type=int)
    parser.add_argument("--epochs", dest="mlp_epochs", type=int)
    cv = parser.add_mutually_exclusive_group()
    cv.add_argument("--cv", dest="use_cv", action="store_true")
    cv.add_argument("--no-cv", dest="use_cv", action="store_false")
    parser.set_defaults(use_cv=None)
    parser.add_argument("--fixed-l", type=float)
    parser.add_argument("--cv-folds", type=int)
    parser.add_argument("--cv-grid-size", type=int)
    parser.add_argument("--cv-span", type=float)
    parser.add_argument("--output", type=Path, help="New output directory; an existing directory is rejected.")
    args = parser.parse_args()
    config = settings_from_args(args)
    if args.workers < 1:
        parser.error("--workers must be positive")
    workers = min(args.workers, os.cpu_count() or 1, len(CASES)*config["num_runs"])
    if bench.torch.cuda.is_available():
        workers = 1  # Avoid concurrent jobs competing for one accelerator.
    tag = "CV" if config["use_cv"] else f"L{config['fixed_l']:g}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or ROOT / "results" /
              f"grouped_N{config['n_train']}_runs{config['num_runs']}_{tag}_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("records", "logs", "models"):
        (output/name).mkdir()
    seeds = [config["base_seed"]+1000*i for i in range(config["num_runs"])]
    manifest = dict(settings=config, cases=CASES, run_seeds=seeds, workers=workers,
                    python=platform.python_version(), torch=bench.torch.__version__,
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    device="cuda" if bench.torch.cuda.is_available() else "cpu",
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(HERE.glob("*.py"))}, status="running")
    write_json(output/"config.json", manifest)
    print(f"Results: {output}\nFormula: {tag}; NN bound: {config['nn_lipschitz']}; "
          f"runs: {config['num_runs']}; workers: {workers}", flush=True)
    for group in GROUPS:
        print(group + ": " + "; ".join(f"{c['name']} (d={c['d']})"
                                       for c in CASES if c["group"] == group), flush=True)
    start = time.perf_counter()
    all_rows = []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=init_worker) as pool:
            futures = [pool.submit(run_one_experiment, c, seed, config, str(output))
                       for c in CASES for seed in seeds]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                all_rows.extend(result["rows"])
                # Save partial records, but only final tables compare complete paired means.
                pd.DataFrame(all_rows).to_csv(output/"raw_partial.csv", index=False)
                print(f"[{i}/{len(futures)}] {result['case']}, seed {result['seed']}: "
                      f"{result['seconds']:.1f}s", flush=True)
        expected = len(CASES)*config["num_runs"]*(2+3*len(config["mlp_ratios"]))
        assert len(all_rows) == expected
        report = export_results(all_rows, CASES, config, output)
        (output/"raw_partial.csv").unlink(missing_ok=True)
        manifest.update(status="complete", rows=len(all_rows), elapsed_seconds=time.perf_counter()-start)
        write_json(output/"config.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", completed_rows=len(all_rows), error=repr(exc))
        write_json(output/"config.json", manifest)
        raise
    print("\n" + report, flush=True)
    print(f"\nSaved in: {output}\nText table: {output/'summary.txt'}\n"
          f"LaTeX tables: {output/'tables.tex'}\nPaper table: {output/'paper_table.tex'}\n"
          f"Raw CSV: {output/'raw.csv'}", flush=True)


if __name__ == "__main__":
    main()
