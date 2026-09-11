# Noisy 100D Hard/Tied benchmarks

Run the three noise levels with training-only cross-validation of the formula's
local-averaging radius beta, followed by fresh fitting on all training data:

```bash
python3 -m pip install -r requirements-benchmarks.txt
python3 experiments/exp_noisy_grouped_Rd.py
```

Run these commands in the unzipped `grouped_benchmarks` directory. For your
existing GitHub checkout, copy all Python files from this package's
`experiments/` directory into its `experiments/` directory, then run the same
command from the repository root. The earlier noiseless entry point remains
`experiments/exp_mlp_Rd.py`.

## Settings at the top of experiments/exp_noisy_grouped_Rd.py

```python
SIGMAS = (0.1, 0.2, 0.8)
NUM_RUNS = 2
N_TRAIN = 1000
N_TEST = 2000
N_LIP = 1000
USE_BETA_CV = True
CV_FOLDS = 5
BETA_GRID_SIZE = 30
FIXED_BETA = 0.0       # Only used when USE_BETA_CV = False.
FORMULA_L = 1.0       # Fixed oracle Lipschitz bound, not selected by this CV.
NN_LIPSCHITZ = 1.0
MLP_RATIOS = (1.0,)
MLP_DEPTH = 3
MLP_LR = 1e-3
MLP_EPOCHS = 1000     # Previous helper's convention: 1001 optimizer updates.
BASE_SEED = 12345
WORKERS = 6
SAVE_MODELS = True
```

All targets are in dimension 100. The default study uses two seeds, as in the
previous exploratory comparisons. Use more seeds to assess whether the observed
differences are stable before making publication claims. CLI examples:

```bash
python3 experiments/exp_noisy_grouped_Rd.py --runs 20
python3 experiments/exp_noisy_grouped_Rd.py --runs 1 --sigmas 0.2 --workers 3
python3 experiments/exp_noisy_grouped_Rd.py --no-cv --fixed-beta 0
```

Every invocation writes a new timestamped subdirectory of `results/`. An
explicit `--output` must name a directory that does not already exist.

## Data and fitting protocol

For every target and seed, sample 1000 training and 2000 test points uniformly
and independently from `[-1,1]^100`. The training labels are
`Y_i = f(X_i) + sigma * Z_i`, with independent standard Gaussian `Z_i`. Test MSE
uses the **clean** values `f(X_test)`. There is no additive sigma-squared floor
in this evaluation, because the test labels are clean.

Seeds are `12345 + 1000*r`. Teacher, training-input, test-input, Lipschitz-probe,
CV-split and noise RNG seeds add 11, 101, 202, 303, 505 and 606, respectively.
Neural initialization adds `404 + width`. Inputs, teacher, standardized noise,
CV splits and initialization seeds are paired across noise levels. Noise is
also shared across methods, so every method fits exactly the same noisy data.

Beta is a Euclidean radius. For a fitting index set I, the formula first forms

```
V_i(beta; I) = mean { Y_j : j in I, ||X_j-X_i||_2 <= beta },  i in I,
```

including the observation at the center. It then predicts

```
g(x) = 0.5 * [ min_i (V_i + L*||x-X_i||_2)
             + max_i (V_i - L*||x-X_i||_2) ],  i in I.
```

The previous noisy 1D code uses the same radius-averaging definition. This
100D study keeps the earlier high-dimensional oracle choice **L=1**, instead
of reestimating L from the noisy or averaged labels.

For CV, divide the training data into five folds. For each radius and fold,
compute all averages using **only that fold's fitting observations**, build
the formula on those observations, and score it against the untouched noisy
validation labels. The validation-label noise adds the same expected
sigma-squared term to every candidate's score. Choose the smallest mean
validation MSE, weighting folds by their sizes. Exact ties choose the smaller
radius. Clean labels and the test set never select beta.

After selection, discard the fold models. **Recompute the averages at all
1000 training sites from all 1000 noisy training observations**, using the
selected absolute radius; this is the formula's final refit. Do not rescale
beta for the change from 800 to 1000 fitting observations. Inference retains
all training sites, their final averaged values and L.

The predefined 30-value grid in `experiments/beta_cv.py` is

```
[0] + linspace(0.5*sqrt(2*d/3), 1.5*sqrt(2*d/3), 28) + [2*sqrt(d)]
```

It is specified before observing labels or test data. At d=100 the interior
range is approximately 4.082 to 12.247, centered around the cube's RMS pairwise
distance 8.165. Radius zero allows no averaging. The cube diameter 20 includes
every fitting point, providing the constant predictor as a candidate. The
30 entries can induce fewer than 30 distinct neighborhood configurations.

The formula is globally L-Lipschitz for any fitted values V: each envelope
is L-Lipschitz, and their midpoint is L-Lipschitz. Averaged or noisy values
need not be interpolated when they are incompatible with the fixed bound.

## Targets and the 100D adaptation

| Group | Target | d |
|---|---|---:|
| Hard | 64 signed Voronoi tents | 100 |
| Tied | Folding: 1 layer | 100 |
| Tied | Folding: 4 layers | 100 |

The names Hard and Tied identify the experimental groups; they do not assert
the observed ranking under noise. The fold targets retain the previous
definitions and seeds. Moving the earlier two-dimensional four-layer fold to
100 dimensions makes it identical to the existing four-layer Tied target,
so it appears once here.

The earlier disjoint Euclidean-ball bumps received no nonzero uniformly
sampled labels in 100 dimensions. This study **explicitly replaces them with
signed Voronoi tents**. It changes the target, without restricting the sampling
distribution or projecting inputs into a low-dimensional subspace. These are
a constructed stress test, not a claim to reproduce a standard named dataset.

Sample K=64 centers independently and uniformly from the 100D cube. Independently
assign Rademacher signs s_j. Let j be the nearest center to x and define

```
h_j(x) = min_{k != j} (||x-c_k||_2^2 - ||x-c_j||_2^2) / (2*||c_k-c_j||_2)
f(x) = s_j * h_j(x).
```

Inside the Voronoi cell of j, h_j is the distance to that cell's boundary.
Each affine expression in the minimum has gradient of norm one. Hence each
signed cell function is 1-Lipschitz. It vanishes on every cell boundary, so
adjacent pieces join continuously. Subdivide any line segment at its cell
crossings and sum the piecewise bounds to obtain the global bound one.
The gradient attains norm one inside an open affine piece. Thus this is an
exactly 1-Lipschitz, piecewise affine target. It is nonzero away from the cell
boundaries, and all 1000 training and 2000 test labels are nonzero in both
archived seeds. Diagnostics record cell coverage and target variance.

For folding, draw independent orthogonal matrices Q_l by Gaussian QR with
diagonal signs corrected. Let h_0(x)=x, h_l(x)=abs(Q_l h_{l-1}(x)) coordinatewise,
and f(x)=s^T h_T(x)/sqrt(d), for independent Rademacher signs s. Orthogonal maps
and absolute values are nonexpansive. Within an open linear region, the
gradient has norm one, so the target is exactly 1-Lipschitz.

## Models, fairness and scalar counts

All three neural families reuse the previous training routines: unconstrained
ReLU, spectrally normalized ReLU, and orthogonal-weight ReLU MLPs. Each has
architecture `100 -> 200 -> 200 -> 200 -> 1`, uses full-batch Adam at 0.001,
and receives 1001 updates. There is **no early stopping or neural hyperparameter
CV** in this continuation. Only the formula's averaging radius is selected
by CV. Results compare those specified training pipelines; they do not
establish that a tuned or early-stopped MLP must perform the same way.

Constrained models are materialized into ordinary weights before evaluation.
Full SVD checks the product-of-spectral-norms bound of at most one. Rectangular
orthogonal layers are semi-orthogonal; the final layer incorporates the gain
and a numerical margin. Counts refer to stored inference scalars, not
independent degrees of freedom or transient normalization/optimizer state.

| Model | Stored scalars | Ratio to formula |
|---|---:|---:|
| Formula | 101001 | 1.000000 |
| Each MLP | 100801 | 0.998020 |
| Noisy-training-mean constant | 1 | 0.000009901 |

For N samples and dimension d, the formula stores `N*(d+1)+1` scalars. Beta is
fit metadata and is no longer needed once the averaged values are stored.
For hidden depth D and common width W, each MLP stores
`(D-1)*W**2 + (d+D+1)*W + 1` scalars. Width is chosen automatically to be
nearest to the requested formula-storage ratio when N changes.

The constant baseline predicts the mean of the noisy training labels. Include
it when interpreting results: matching this baseline can mean successful
noise suppression with little recovery of target structure.

## Outputs

The completed two-seed run is in `noisy_example_results/`:

- `summary.txt`, `summary.md`: three noise-level comparisons, with Hard/Tied
  groups, raw clean test MSE and sample standard deviation, and percentages.
- `paper_sigma_0p1.tex`, `paper_sigma_0p2.tex`, `paper_sigma_0p8.tex`: actual
  paper-ready LaTeX floats with two captioned subtables each. Identical `.txt`
  copies are provided. Add `\usepackage{booktabs,subcaption,graphicx}` to your
  paper preamble.
- `paper_tables.tex`, `paper_tables.txt`: all three floats together.
- `paper_tables_preview.pdf`: checked rendering of all three tables.
- `raw.csv`, `summary.csv`: seed-level and aggregated numeric results.
- `beta_selections.csv`: selected radii, validation scores, final refit sizes
  and averaging-neighborhood sizes for every target, seed and noise level.
- `parameter_counts.csv`: inference scalar counts and ratios.
- `records/`: per-job metrics and full CV/target diagnostics.
- `models/`: fitted formula states and materialized neural weights.
- `predictions/`: clean test labels, clean/noisy training labels and every
  model's test predictions for independent checking of reported MSEs.
- `config.json`: settings, actual beta grid, seeds, software versions, source
  hashes, runtime and completion status.

Percentages are `100 * mean(model MSE) / mean(formula MSE)`, not the average
of per-seed ratios. **Above 100% favors the formula**; below 100% favors the
other method. One seed cannot provide a sample standard deviation, so that
quantity is NA when `NUM_RUNS=1`.

The archived study completed 18 target/seed/noise jobs (54 trained MLPs) in
187 seconds on this CPU environment. The formula's mean MSE is lower than all
three neural means in each of the nine target/noise comparisons. However,
its mean MSE stays within 1% of the constant baseline in every comparison.
These results primarily demonstrate noise suppression under the specified
training schedules; they provide little evidence of recovering detailed
target structure. The group names therefore should not be read as proven
hardness or guaranteed performance classifications.

Run `python3 experiments/verify_noisy_results.py noisy_example_results` to
check an independent CV reference, full-data refits, restored model predictions,
clean-label MSEs, inference counts, source hashes and ratios of paired means.

Formula inference can be restored without beta or CV:

```python
state = np.load("path_to_saved_formula.npz")
pred = benchmark_helpers.formula_grid_predict(
    X_query, state["X"], state["values"], [float(state["L"])], "l2"
)[0]
```
