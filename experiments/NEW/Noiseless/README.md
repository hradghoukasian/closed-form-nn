# Grouped formula / MLP benchmarks

Six selected targets, in the agreed order. All use noiseless uniform inputs on
`[-1,1]^d` and Euclidean distance.

| Expected group | Target | Dimension |
|---|---|---:|
| Clear wins | 64 signed bumps | 2 |
| Clear wins | Four folding layers | 2 |
| Near ties [85%, 115%] | One folding layer | 100 |
| Near ties [85%, 115%] | Four folding layers | 100 |
| Clear losses | Affine, `||w||_1 = 1` | 100 |
| Clear losses | `tanh(w^T x + b)`, `||w||_2 = 1` | 100 |

Expected groups were chosen from the preliminary comparisons. They stay fixed
even if new runs change the measured rankings. `summary.csv` also reports each
MLP's observed classification using the inclusive 85–115% near-tie interval.

## Run

Unzip this package, open a terminal in the `grouped_benchmarks` folder, and run:

```bash
python3 -m pip install -r requirements-benchmarks.txt
python3 experiments/exp_mlp_Rd.py
```

Python 3.10 or newer is required. The script works as a standalone package.
To use your existing GitHub checkout, copy **all four Python files** from this
package's `experiments/` into its `experiments/` folder. This replaces
`experiments/exp_mlp_Rd.py` and adds its three helper modules. Run the same command
from the repository root. Dependencies are listed above.

## Settings at the top of experiments/exp_mlp_Rd.py

```python
NUM_RUNS = 2            # 1, 2, 20, etc.
N_TRAIN = 1000
N_TEST = 2000
N_LIP = 1000
USE_CV = False
FIXED_L = 1.0
CV_FOLDS = 5
CV_GRID_SIZE = 20
CV_SPAN = 2.0
NN_LIPSCHITZ = 1.0
MLP_RATIOS = (1.0,)
MLP_DEPTH = 3
MLP_LR = 1e-3
MLP_EPOCHS = 1000
BASE_SEED = 12345
WORKERS = 6
SAVE_MODELS = True
```

The default is the previous protocol: **two runs, 1,000 training samples,
formula L=1, CV off, and 1,001 full-batch Adam updates**. The original training
loop performs `MLP_EPOCHS + 1` updates; this convention is retained explicitly.

Command-line overrides are also available:

```bash
python3 experiments/exp_mlp_Rd.py --runs 20
python3 experiments/exp_mlp_Rd.py --runs 1 --no-cv --fixed-l 2
python3 experiments/exp_mlp_Rd.py --runs 2 --cv --cv-grid-size 20 --cv-span 2
```

With CV on, each fold evaluates `L_D(fold training) + linspace(0,2,20)`.
Validation MSE selects the additive offset. The final formula uses **all
training observations**, with `L_D(all training) + selected_offset`. Test data
never select the offset. The constrained neural bound stays at
`NN_LIPSCHITZ`, independently of the selected formula bound.

CPU workers use one thread each; lower `WORKERS` if needed. CUDA availability
automatically reduces the pool to one worker. Changing settings creates a new
dated output folder, so results from different configurations are not mixed.

## Models and counts

Every target compares the formula, an unconstrained ReLU MLP, a spectrally
normalized ReLU MLP, an orthogonal ReLU MLP, and a training-mean constant
predictor. The last is a one-scalar diagnostic baseline.

The formula stores `P_formula = N_TRAIN * (d + 1) + 1` scalars: the input sites,
their scalar labels, and one Lipschitz bound. An MLP of hidden depth `D` and
constant width `W` stores `(D-1)*W**2 + (d+D+1)*W + 1` weights and biases. The
nearest positive integer width is chosen separately for each dimension and
requested storage ratio; a tie chooses the smaller count. All three neural
families use the same widths, sample sets, initialization seeds, optimizer,
and update count.

At the defaults:

| d | Formula scalars | Each MLP's scalars | Hidden width | MLP / formula |
|---|---:|---:|---:|---:|
| 2 | 3,001 | 2,961 | 37 | 0.986671 |
| 100 | 101,001 | 100,801 | 200 | 0.998020 |

These match stored inference scalars. Orthogonality reduces independent degrees
of freedom; spectral normalization uses temporary power-iteration buffers during
training. Normalization buffers and optimizer state are excluded from the count.
Constraints are materialized into ordinary inference weights before evaluation.
Full SVD checks the final numerical norm-product bounds. Hidden orthogonal
matrices are semi-orthogonal when rectangular; the output weight incorporates
the requested Lipschitz gain and numerical rounding correction.

## Exact targets

- **Signed bumps:** centers form the 8-by-8 grid with coordinates
  `-1 + (2*j+1)/8`, `j=0,...,7`. Set `r=1/8` and independently sample signs
  `s_j` uniformly from `{-1,+1}`. The target is
  `f(x) = sum_j s_j * max(r - ||x-c_j||_2, 0)`.
  Disjoint support interiors give a global Lipschitz bound one; slopes attain
  one inside supports. This probes learning many independent local signs.
- **Folding:** independently sample orthogonal matrices by Gaussian QR with
  the diagonal signs corrected. Set `h_0(x)=x`,
  `h_l(x)=abs(Q_l h_(l-1)(x))` coordinatewise, and
  `f(x)=s^T h_T(x)/sqrt(d)`, with independent uniform random signs `s`.
  The Lipschitz constant is one: each map is nonexpansive, and within any open
  linear region the gradient has norm one. Comparing T=4 in d=2 versus d=100
  probes dimension and coverage with the same definition.
- **Affine:** sample `g ~ N(0,I_d)`, set `w=g/||g||_1`, and independently sample
  `b ~ Uniform[-0.5,0.5]`. Set `f(x)=w^T x+b`. Its Euclidean Lipschitz constant
  is `||w||_2`, approximately 0.123 in the two default seeds; the default
  formula bound one is therefore loose. The infinity-metric constant is one.
- **Tanh ridge:** sample `g ~ N(0,I_d)`, set `w=g/||g||_2`, and sample
  `x0 ~ Uniform[-0.8,0.8]^d`. Set `b=-w^T x0` and
  `f(x)=tanh(w^T x+b)`. Its Euclidean Lipschitz constant is exactly one because
  `tanh'(0)=1` at the interior point `x0`. It depends on one linear projection
  and is approximately affine near the hyperplane `w^T x+b=0`. The uniform
  design is not restricted to that local region. Diagnostics report how many
  test preactivations satisfy `|w^T x+b| <= 0.25`.

Target, training, test, and empirical-Lipschitz seeds are respectively the run
seed plus 11, 101, 202, and 303. Neural initialization uses `run_seed+404+width`;
CV uses `run_seed+505`. Run seeds are `12345+1000*i`. The existing five target
definitions and model-training routines retain the preceding experiment's
randomness and normalization conventions. This is the noiseless midpoint
formula; it does not add local averaging or a beta sweep.

## Find the tables

The script prints the complete output directory when it starts and finishes.
By default it is a new subfolder of `results/`. Files are:

- `summary.txt` and `summary.md`: grouped raw MSE with sample standard
  deviation, MSE percentages versus the formula and constant predictor,
  empirical Lipschitz statistics, and scalar counts/ratios.
- `tables.tex`: actual LaTeX `table` / `tabular` environments with those tables.
- `tables_latex.txt`: the identical LaTeX source saved as `.txt`.
- `report.tex`: a standalone landscape LaTeX document containing the tables.
- `paper_table.tex`: one paper-ready table with three stacked subtables captioned
  **Hard**, **Tied**, and **MLP exploits different inductive bias**. Cells contain
  raw test MSE with sample standard deviation and, below it, the percentage of
  the formula's mean MSE. It includes the training-mean constant baseline.
- `paper_table_latex.txt`: identical copy-paste LaTeX; add
  `\usepackage{booktabs,subcaption,graphicx}` to your paper's preamble.
- `paper_table_preview.tex`: standalone portrait source for that table.
- `raw.csv`: every seed's metrics, L values, counts and CV information.
- `summary.csv`: grouped means/stds, ratios, and observed classifications.
- `parameter_counts.csv`: actual MLP/formula scalar counts and ratios.
- `records/`: individual completed experiments and target/CV diagnostics.
- `models/`: materialized PyTorch inference weights, when `SAVE_MODELS=True`.
- `config.json`: settings, software versions and source hashes for the run.

Percentages use the **ratio of mean MSEs**, not the mean of per-run percentages.
Below 100% favors the MLP; above 100% favors the formula. Near ties include both
85% and 115%. Zero denominators are N/A. With one run, sample std is undefined
and reported as N/A, not zero. The 100D near ties in the earlier experiments
were worse than the constant baseline; check that baseline in future runs too.

`example_results/` contains the completed default two-run results generated
while checking this package. Running the script creates fresh results.
`example_results/paper_table_preview.pdf` is the checked rendering of the new
three-subtable export. The existing experimental records are reused unchanged;
`table_export_provenance.json` records the later table-export update.
