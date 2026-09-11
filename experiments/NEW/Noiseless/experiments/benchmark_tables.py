"""Grouped exports. All percentages compare complete, paired per-model means."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_targets import GROUPS

FAMILIES = ("formula", "unconstrained", "spectral", "orthogonal", "constant")
NAMES = ("Formula", "ReLU MLP", "Spectral", "Orthogonal", "Constant")


def observed_comparison(percent):
    if not np.isfinite(percent):
        return "Undefined"
    if percent > 115:
        return "Formula win"
    if percent < 85:
        return "Formula loss"
    return "Near tie"


def latex_escape(text):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                     "_": r"\_", "#": r"\#", "{": r"\{", "}": r"\}"}
    return "".join(substitutions.get(c, c) for c in str(text))


def number(x, scientific=False):
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.3e}" if scientific else f"{x:.1f}"


def tex_number(x, scientific=False):
    if not np.isfinite(x):
        return r"\mathrm{NA}"
    if scientific and x != 0:
        mantissa, exponent = f"{x:.3e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return f"{x:.3g}" if scientific else f"{x:.1f}"


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"]*len(headers)) + " |",
                       *["| " + " | ".join(str(x).replace("|", r"\|") for x in r) + " |"
                         for r in rows]])


def paper_table_latex(summary, cases, config, ratio):
    """One paper-ready float with three vertically stacked, captioned subtables."""
    captions = ("Hard", "Tied", "MLP exploits different inductive bias")
    selected = summary[summary.family.isin(("formula", "constant"))
                       | summary.target_scalar_ratio.eq(ratio)]
    bounds = ("The formula selects its bound by training-only cross-validation. "
              if config["use_cv"] else f"The formula uses $L={config['fixed_l']:g}$, with CV off. ")
    caption = (f"Test MSE (mean $\\pm$ sample standard deviation over {config['num_runs']} runs). "
               "Bracketed values express each method's mean MSE as a percentage "
               "of the formula's mean MSE; lower is better. "
               f"Uniform inputs on $[-1,1]^d$; $N_{{\\mathrm{{train}}}}={config['n_train']}$, "
               f"$N_{{\\mathrm{{test}}}}={config['n_test']}$. " + bounds +
               f"MLPs have {config['mlp_depth']} hidden ReLU layers and use "
               f"{config['mlp_epochs']+1} full-batch Adam updates at learning rate "
               f"${config['mlp_lr']:g}$. SN and orthogonal MLP bounds are "
               f"${config['nn_lipschitz']:g}$.")
    tag = f"{ratio:g}".replace(".", "p")
    lines = ["% Preamble: \\usepackage{booktabs,subcaption,graphicx}",
             r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\newcommand{\cfbenchcell}[4]{\shortstack{$(#1\pm#2)\times10^{#3}$\\$[#4]$}}",
             r"\caption{" + caption + "}",
             rf"\label{{tab:grouped-benchmarks-{tag}}}"]

    def cell(row):
        mean, std = row.test_mse_mean, row.test_mse_std
        exponent = int(math.floor(math.log10(abs(mean)))) if mean else 0
        scale = 10.0**exponent
        m = f"{mean/scale:.3f}"
        s = f"{std/scale:.3f}" if np.isfinite(std) else r"\mathrm{NA}"
        percent = (f"{row.mse_percent_of_formula:.1f}" + r"\%"
                   if np.isfinite(row.mse_percent_of_formula) else r"\mathrm{NA}")
        return rf"\cfbenchcell{{{m}}}{{{s}}}{{{exponent}}}{{{percent}}}"

    for i, (group, subcaption) in enumerate(zip(GROUPS, captions)):
        if i:
            lines.extend([r"\par\medskip", ""])
        lines.extend([r"\begin{subtable}{\textwidth}", r"\centering",
                      r"\caption{" + subcaption + "}",
                      rf"\label{{tab:grouped-benchmarks-{tag}-{i+1}}}",
                      r"\resizebox{\linewidth}{!}{%",
                      r"\begin{tabular}{@{}lrccccc@{}}", r"\toprule",
                      r"Target & $d$ & Formula & ReLU MLP & SN MLP & Orthogonal MLP & Constant \\",
                      r"\midrule"])
        for c in [c for c in cases if c["group"] == group]:
            g = selected[selected.case.eq(c["case"])].set_index("family")
            label = r"Affine, $\|w\|_1=1$" if c["family"] == "affine" else latex_escape(c["name"])
            lines.append(f"{label} & {c['d']} &")
            for j, family in enumerate(FAMILIES):
                lines.append("  " + cell(g.loc[family]) + (" &" if j < len(FAMILIES)-1 else r" \\"))
            lines.append(r"\addlinespace[3pt]")
        lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{subtable}"])
    counts = []
    for d in dict.fromkeys(c["d"] for c in cases):
        r = selected[(selected.d.eq(d)) & selected.family.eq("unconstrained")].iloc[0]
        def integer(x):
            return f"{int(x):,}".replace(",", r"\,")
        counts.append(f"$d={d}$: $P_{{\\mathrm{{formula}}}}={integer(r.formula_scalars)}$, "
                      f"$P_{{\\mathrm{{MLP}}}}={integer(r.stored_scalars)}$ "
                      f"(ratio ${r.scalar_ratio_to_formula:.4f}$)")
    note = (r"\emph{Notes.} " + "; ".join(counts) + ". "
            "All three MLP families match these stored-scalar counts. "
            "The constant predictor stores the training-label mean. "
            "The panel labels denote the expected behavior; a measured tie means all three "
            r"MLP percentages lie in $[85\%,115\%]$. "
            r"The affine target has Euclidean Lipschitz constant $\|w\|_2<1$, "
            r"whereas the tanh target uses $\|w\|_2=1$ and an interior zero crossing.")
    lines.extend([r"\par\smallskip", r"\begin{minipage}{\textwidth}", r"\footnotesize",
                  note, r"\end{minipage}", r"\end{table*}", ""])
    return "\n".join(lines)


def export_results(records, cases, config, output):
    output = Path(output)
    case_order = {c["case"]: i for i, c in enumerate(cases)}
    family_order = {f: i for i, f in enumerate(FAMILIES)}
    raw = pd.DataFrame(records)
    raw["_case"] = raw.case.map(case_order)
    raw["_family"] = raw.family.map(family_order)
    raw = raw.sort_values(["_case", "run", "target_scalar_ratio", "_family"],
                          kind="stable").drop(columns=["_case", "_family"])
    summary_rows = []
    for case in cases:
        sub = raw[raw.case.eq(case["case"])]
        formula_mean = sub.loc[sub.family.eq("formula"), "test_mse"].mean()
        baseline_mean = sub.loc[sub.family.eq("constant"), "test_mse"].mean()
        for family in FAMILIES:
            for ratio in config["mlp_ratios"] if family not in ("formula", "constant") else [None]:
                g = sub[sub.family.eq(family)]
                if ratio is not None:
                    g = g[g.target_scalar_ratio.eq(ratio)]
                assert len(g) == config["num_runs"] == g.run.nunique()
                first = g.iloc[0]
                item = {k: first[k] for k in (
                    "case", "experiment", "group", "d", "family", "method", "width",
                    "mlp_depth", "target_scalar_ratio", "stored_scalars", "formula_scalars",
                    "scalar_ratio_to_formula", "formula_L_mode", "optimizer_updates")}
                item["completed_runs"] = len(g)
                for column in ("test_mse", "train_mse", "empirical_lipschitz", "L_used_cf",
                               "L_hat_train", "target_euclidean_L", "lipschitz_bound",
                               "normalized_mse", "training_seconds"):
                    item[column+"_mean"] = g[column].mean()
                    item[column+"_std"] = g[column].std(ddof=1)
                item["mse_percent_of_formula"] = (
                    100*item["test_mse_mean"]/formula_mean if formula_mean > 0 else np.nan)
                item["mse_percent_of_constant"] = (
                    100*item["test_mse_mean"]/baseline_mean if baseline_mean > 0 else np.nan)
                item["observed_comparison"] = (
                    observed_comparison(item["mse_percent_of_formula"])
                    if family not in ("formula", "constant") else "Reference")
                summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    raw.to_csv(output/"raw.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    l_mode = (f"CV with {config['cv_folds']} folds, {config['cv_grid_size']} offsets in "
              f"[0, {config['cv_span']:g}]" if config["use_cv"] else f"fixed L={config['fixed_l']:g}")
    protocol = (f"{config['num_runs']} runs; N_train={config['n_train']}, "
                f"N_test={config['n_test']}, N_lip={config['n_lip']}; Euclidean metric. "
                f"Formula: {l_mode}; constrained NN bound={config['nn_lipschitz']:g}. "
                f"{config['mlp_depth']} hidden layers, Adam lr={config['mlp_lr']:g}, "
                f"{config['mlp_epochs']+1} full-batch updates.")
    intro = (f"{len(cases)} grouped experiments\n\n" + protocol + "\n\n"
             "Percentages are 100 * model mean MSE / reference mean MSE. "
             "Above 100% versus the formula favors the formula. Near ties are inclusive [85%,115%]. "
             "Expected groups are fixed from the earlier exploratory runs; they do not force outcomes. "
             "The constant predictor is the training-label mean. "
             "A single run has no sample standard deviation (shown as N/A).\n")
    report = [intro]
    tex_tables = []
    # Every budget gets its own complete grouped comparison.
    for ratio in config["mlp_ratios"]:
        for mode, title in (("test_mse", "Raw test MSE: mean +/- sample std"),
                            ("formula_percent", "MSE as a percentage of the formula"),
                            ("constant_percent", "MSE as a percentage of the constant predictor"),
                            ("empirical_lipschitz", "Empirical Lipschitz estimate: mean +/- sample std")):
            report.append(f"\n{title}; target storage ratio {ratio:g}\n")
            latex = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
                     r"\setlength{\tabcolsep}{3pt}",
                     r"\caption{" + latex_escape(f"{title}. Target MLP/formula storage ratio {ratio:g}. "
                                                 + protocol) + "}",
                     r"\begin{tabular}{lrccccc}", r"\hline",
                     r"Target & $d$ & Formula & ReLU MLP & Spectral & Orthogonal & Constant \\",
                     r"\hline"]
            for group in GROUPS:
                report.append(group)
                latex.append(r"\multicolumn{7}{l}{\textbf{" + latex_escape(group) + r"}} \\")
                display_rows = []
                for case in [c for c in cases if c["group"] == group]:
                    g = summary[summary.case.eq(case["case"])]
                    cells, tex_cells = [], []
                    for family in FAMILIES:
                        s = g[g.family.eq(family)]
                        if family not in ("formula", "constant"):
                            s = s[s.target_scalar_ratio.eq(ratio)]
                        assert len(s) == 1
                        row = s.iloc[0]
                        if mode in ("test_mse", "empirical_lipschitz"):
                            mean, std = row[mode+"_mean"], row[mode+"_std"]
                            scientific = mode == "test_mse"
                            cells.append(number(mean, scientific) + " +/- " + number(std, scientific))
                            tex_cells.append("$" + tex_number(mean, scientific) + r"\pm "
                                             + tex_number(std, scientific) + "$")
                        else:
                            val = row["mse_percent_of_formula" if mode == "formula_percent"
                                      else "mse_percent_of_constant"]
                            cells.append(number(val) + ("%" if np.isfinite(val) else ""))
                            tex_cells.append("$" + tex_number(val)
                                             + (r"\%" if np.isfinite(val) else "") + "$")
                    display_rows.append([case["name"], case["d"], *cells])
                    latex.append(" & ".join([latex_escape(case["name"]), str(case["d"]), *tex_cells]) + r" \\")
                report.append(markdown_table(["Target", "d", *NAMES], display_rows))
                latex.append(r"\hline")
            latex.extend([r"\end{tabular}", r"\end{table}"])
            tex_tables.append("\n".join(latex))
    storage = summary[["d", "method", "target_scalar_ratio", "width", "stored_scalars",
                       "formula_scalars", "scalar_ratio_to_formula"]].drop_duplicates()
    storage.to_csv(output/"parameter_counts.csv", index=False)
    storage_rows = [[int(r.d), r.method, "N/A" if pd.isna(r.width) else int(r.width),
                     int(r.stored_scalars), int(r.formula_scalars), f"{r.scalar_ratio_to_formula:.6f}"]
                    for r in storage.itertuples()]
    report.append("\nStored scalars\n" + markdown_table(
        ["d", "Method", "Width", "Scalars", "Formula scalars", "P / P_formula"], storage_rows))
    count_tex = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
                 r"\caption{Stored inference scalars: coordinates, labels and one bound for the formula; "
                 r"weights and biases for MLPs; one training-label mean for the constant predictor.}",
                 r"\begin{tabular}{rlrrrr}", r"\hline",
                 r"$d$ & Method & Width & $P$ & $P_{\mathrm{formula}}$ & $P/P_{\mathrm{formula}}$ \\",
                 r"\hline"]
    count_tex.extend(" & ".join(latex_escape(x) for x in row) + r" \\" for row in storage_rows)
    count_tex.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    tex_tables.append("\n".join(count_tex))
    report.append("\nThe affine target has ||w||_1=1; its Euclidean Lipschitz constant is ||w||_2, "
                  "recorded per seed in records/*_diagnostics.json. Formula and NN bounds are upper bounds, "
                  "not assertions that the actual predictor constant equals the bound. Scalar storage "
                  "excludes optimizer state and temporary normalization buffers. Orthogonal constraints "
                  "reduce independent degrees of freedom despite matching dense scalar storage. "
                  "The tanh ridge uses ||w||_2=1 and b=-w^T x0 for an interior point x0, "
                  "so its Euclidean Lipschitz constant is exactly one. It is approximately affine "
                  "locally where |w^T x+b| is small; the uniform sample is not restricted to that region.\n")
    text = "\n".join(report)
    (output/"summary.txt").write_text(text)
    (output/"summary.md").write_text(text)
    latex_text = "\n\n".join(tex_tables) + "\n"
    (output/"tables.tex").write_text(latex_text)
    (output/"tables_latex.txt").write_text(latex_text)
    # Standalone source uses only standard LaTeX. Each table fits a landscape page.
    (output/"report.tex").write_text(
        "\\documentclass{article}\n\\usepackage[a4paper,landscape,margin=1.5cm]{geometry}\n"
        "\\begin{document}\n" + "\n\\clearpage\n".join(tex_tables) + "\n\\end{document}\n")
    paper_tables = "\n\n".join(paper_table_latex(summary, cases, config, ratio)
                                for ratio in config["mlp_ratios"])
    (output/"paper_table.tex").write_text(paper_tables)
    (output/"paper_table_latex.txt").write_text(paper_tables)
    (output/"paper_table_preview.tex").write_text(
        "\\documentclass{article}\n\\usepackage[margin=2cm]{geometry}\n"
        "\\usepackage{booktabs,subcaption,graphicx}\n\\begin{document}\n"
        "\\input{paper_table.tex}\n\\end{document}\n")
    return text
