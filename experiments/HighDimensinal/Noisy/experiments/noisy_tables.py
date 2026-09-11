"""Separate noise-level tables with Hard/Tied subtables and paired-mean ratios."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_tables import latex_escape, markdown_table, observed_comparison
from noisy_targets import GROUPS

FAMILIES = ("formula", "unconstrained", "spectral", "orthogonal", "constant")
NAMES = ("Formula", "ReLU MLP", "SN MLP", "Orthogonal MLP", "Constant")


def sigma_tag(sigma):
    return f"{sigma:g}".replace(".", "p")


def latex_table(summary, cases, config, sigma, ratio):
    selected = summary[summary.sigma.eq(sigma) &
                       (summary.family.isin(("formula", "constant")) |
                        summary.target_scalar_ratio.eq(ratio))]
    tag = f"{sigma_tag(sigma)}-ratio{ratio:g}".replace(".", "p")
    cv_text = (f"The formula selects the averaging radius $\\beta$ by {config['cv_folds']}-fold "
               f"CV over {config['beta_grid_size']} predefined candidates using noisy training "
               "labels, then recomputes all averages using the full training set. "
               if config["use_beta_cv"] else f"The averaging radius is fixed at $\\beta={config['fixed_beta']:g}$. ")
    caption = (f"Clean test MSE with additive Gaussian noise $\\sigma={sigma:g}$; "
               f"mean $\\pm$ sample standard deviation over {config['num_runs']} seeds. "
               "Brackets give $100\\,\\mathrm{MSE}_{\\mathrm{method}}/"
               "\\mathrm{MSE}_{\\mathrm{formula}}$ using the respective seed means. "
               f"Inputs are uniform on $[-1,1]^{{100}}$, with $N_{{\\mathrm{{train}}}}={config['n_train']}$ "
               f"and $N_{{\\mathrm{{test}}}}={config['n_test']}$. " + cv_text +
               f"The formula uses fixed $L={config['formula_l']:g}$; SN and orthogonal networks "
               f"have bound {config['nn_lipschitz']:g}. All MLPs use {config['mlp_depth']} hidden ReLU layers "
               f"and {config['mlp_epochs']+1} full-batch Adam updates with learning rate "
               f"${config['mlp_lr']:g}$, without early stopping.")
    lines = ["% Preamble: \\usepackage{booktabs,subcaption,graphicx}",
             r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}", r"\caption{"+caption+"}",
             rf"\label{{tab:noisy-{tag}}}"]

    def cell(row):
        mean, std = row.test_mse_mean, row.test_mse_std
        exponent = int(math.floor(math.log10(abs(mean)))) if mean else 0
        scale = 10.**exponent
        s = f"{std/scale:.3f}" if np.isfinite(std) else r"\mathrm{NA}"
        pct = (f"{row.mse_percent_of_formula:.1f}" + r"\%"
               if np.isfinite(row.mse_percent_of_formula) else r"\mathrm{NA}")
        return (r"\shortstack{$(" + f"{mean/scale:.3f}" + r"\pm " + s +
                r")\times10^{" + str(exponent) + r"}$\\$[" + pct + r"]$}")

    for i, group in enumerate(GROUPS):
        if i:
            lines.append(r"\par\medskip")
        lines.extend([r"\begin{subtable}{\textwidth}", r"\centering",
                      r"\caption{"+group+"}", rf"\label{{tab:noisy-{tag}-{i+1}}}",
                      r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{@{}lccccc@{}}",
                      r"\toprule", r"Target & Formula & ReLU MLP & SN MLP & Orthogonal MLP & Constant \\",
                      r"\midrule"])
        for case in [c for c in cases if c["group"] == group]:
            g = selected[selected.case.eq(case["case"])].set_index("family")
            lines.append(latex_escape(case["name"]) + " &\n  " +
                         " &\n  ".join(cell(g.loc[f]) for f in FAMILIES) + r" \\")
            lines.append(r"\addlinespace[3pt]")
        lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{subtable}"])
    r = selected[selected.family.eq("unconstrained")].iloc[0]
    note = (r"\emph{Notes.} Hard and Tied are target-group labels, not conclusions about noisy results. "
            "Signed Voronoi tents replace the earlier disjoint-ball bumps to obtain nonzero "
            "uniform samples in 100 dimensions; each target is 1-Lipschitz. "
            f"Stored scalars: formula {int(r.formula_scalars):,}; each MLP {int(r.stored_scalars):,} "
            f"(ratio {r.scalar_ratio_to_formula:.4f}, width {int(r.width)}). "
            "The constant predicts the noisy training-label mean. Only the formula's radius "
            "is cross-validated; the neural training schedule is fixed.")
    lines.extend([r"\par\smallskip", r"\begin{minipage}{\textwidth}", r"\footnotesize",
                  note, r"\end{minipage}", r"\end{table*}", ""])
    return "\n".join(lines)


def export_results(records, cases, config, output):
    output = Path(output)
    raw = pd.DataFrame(records).sort_values(["sigma", "case", "run", "family", "target_scalar_ratio"])
    summary_rows = []
    for sigma in config["sigmas"]:
        for case in cases:
            sub = raw[raw.sigma.eq(sigma) & raw.case.eq(case["case"])]
            cf_mean = sub.loc[sub.family.eq("formula"), "test_mse"].mean()
            const_mean = sub.loc[sub.family.eq("constant"), "test_mse"].mean()
            for family in FAMILIES:
                for ratio in ([None] if family in ("formula", "constant") else config["mlp_ratios"]):
                    g = sub[sub.family.eq(family)]
                    if ratio is not None:
                        g = g[g.target_scalar_ratio.eq(ratio)]
                    assert len(g) == config["num_runs"] == g.run.nunique()
                    first = g.iloc[0]
                    row = {k: first[k] for k in ("sigma", "case", "experiment", "group", "d", "family",
                                                "method", "target_scalar_ratio", "width", "mlp_depth",
                                                "stored_scalars", "formula_scalars", "scalar_ratio_to_formula",
                                                "formula_L", "beta_mode", "cv_folds", "final_refit_n",
                                                "optimizer_updates")}
                    row["completed_runs"] = len(g)
                    for column in ("test_mse", "train_noisy_mse", "train_clean_mse", "empirical_lipschitz",
                                   "lipschitz_bound", "training_seconds", "beta_selected", "cv_validation_mse",
                                   "final_average_count_mean", "clean_test_variance"):
                        row[column+"_mean"] = g[column].mean()
                        row[column+"_std"] = g[column].std(ddof=1)
                    row["mse_percent_of_formula"] = 100*row["test_mse_mean"]/cf_mean if cf_mean else np.nan
                    row["mse_percent_of_constant"] = 100*row["test_mse_mean"]/const_mean if const_mean else np.nan
                    row["observed_comparison"] = (observed_comparison(row["mse_percent_of_formula"])
                                                  if family not in ("formula", "constant") else "Reference")
                    summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    raw.to_csv(output/"raw.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    raw[raw.family.eq("formula")][["sigma", "case", "run", "beta_selected", "cv_validation_mse",
                                    "final_refit_n", "final_average_count_min", "final_average_count_mean",
                                    "final_average_count_max"]].to_csv(output/"beta_selections.csv", index=False)
    summary[["method", "width", "stored_scalars", "formula_scalars", "scalar_ratio_to_formula"]].drop_duplicates().to_csv(
        output/"parameter_counts.csv", index=False)
    report = [f"Noisy 100D targets; {config['num_runs']} seeds, N_train={config['n_train']}, "
              f"N_test={config['n_test']}. Clean test MSE.\n"
              "Cell: mean +/- sample std [percentage of formula's mean MSE].\n"
              "Above 100% favors the formula. Groups identify targets, not measured outcomes.\n"]
    tex = []
    for sigma in config["sigmas"]:
        for ratio in config["mlp_ratios"]:
            report.append(f"\nSigma={sigma:g}; target MLP storage ratio={ratio:g}\n")
            selected = summary[summary.sigma.eq(sigma) &
                               (summary.family.isin(("formula", "constant")) |
                                summary.target_scalar_ratio.eq(ratio))]
            for group in GROUPS:
                rows = []
                for case in [c for c in cases if c["group"] == group]:
                    g = selected[selected.case.eq(case["case"])].set_index("family")
                    cells = [f"{g.loc[f].test_mse_mean:.5f} +/- {g.loc[f].test_mse_std:.5f} "
                             f"[{g.loc[f].mse_percent_of_formula:.1f}%]" for f in FAMILIES]
                    rows.append([case["name"], *cells])
                report.append(group+"\n"+markdown_table(["Target", *NAMES], rows))
            table = latex_table(summary, cases, config, sigma, ratio)
            name = f"paper_sigma_{sigma_tag(sigma)}" + (f"_ratio{ratio:g}" if len(config["mlp_ratios"]) > 1 else "")
            (output/f"{name}.tex").write_text(table)
            (output/f"{name}.txt").write_text(table)
            tex.append(table)
    report = "\n\n".join(report)+"\n"
    (output/"summary.txt").write_text(report)
    (output/"summary.md").write_text(report)
    (output/"paper_tables.tex").write_text("\n\n".join(tex))
    (output/"paper_tables.txt").write_text("\n\n".join(tex))
    return report
