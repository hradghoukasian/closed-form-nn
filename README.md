# A Closed-Form Formula for Consistent Lipschitz Regression on Metric Spaces with Sparse Neural Network Realizations

This repository contains the code used to reproduce the numerical experiments in the paper:

> **A Closed-Form Formula for Consistent Lipschitz Regression on Metric Spaces with Sparse Neural Network Realizations**  
> Ruiyang Hong, Hrad Ghoukasian, and Anastasis Kratsios

The experiments illustrate the behavior of the proposed closed-form Lipschitz regression estimator in noiseless and noisy settings and compare it with standard ReLU MLP baselines.

---

## 📂 Repository Structure

- `core/` — core implementation of the closed-form estimator and supporting utilities.
- `experiments/` — scripts used to run the numerical experiments reported in the paper.
- `results/` — generated numerical results and figures.

---

## 🧪 Experiments

The scripts in `experiments/` correspond to the numerical experiments in the paper:

- **`exp_R.py`** — illustrates the closed-form reconstruction of a one-dimensional noiseless target function on $[-1,1]$.

- **`exp_R2.py`** — evaluates the closed-form estimator on a two-dimensional noiseless target over $[-1,1]^2$ for different training sample sizes.

- **`exp_mlp_Rd_l2.py`** — compares the closed-form estimator with ReLU MLP baselines on random exactly $1$-Lipschitz target functions in $\mathbb{R}^d$, reporting test MSE and empirical Lipschitz constants.

- **`exp_noisy_R_simple.py`** — studies recovery of $f(x)=\sin(3x)$ under increasing Gaussian noise and evaluates the selected neighborhood bandwidth, test MSE, and representative reconstructions.

- **`exp_noisy_R_frequency_simple.py`** — studies recovery of $f_\omega(x)=\sin(\omega x)$ as the target frequency increases under a fixed noise level, including the selected bandwidth and comparison with a ReLU MLP.

---

## 📊 Outputs

Running the experiment scripts generates the corresponding numerical results and figures in the `results/` directory, including the tables and plots reported in the paper.

---

## ⚙️ Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The experiments use standard scientific Python packages including NumPy, pandas, Matplotlib, and PyTorch.

