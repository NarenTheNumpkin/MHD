"""Manuscript figures in the figures4papers house style (github.com/ChenLiu-1996/figures4papers).

Style: Helvetica, top/right spines off, thick axes, frameless legends, no grid, semantic palette (blue = proposed,
reds = published baselines, greens = residual-based controls, neutral = references). Canvases are drawn at twice the
IEEE print width (14.3 in for full-width, 7.0 in for single-column figures) with 15 pt base text, so text prints at
about 7.5 pt. Every figure is exported as vector PDF (used by the manuscript) and 300 dpi PNG.

  python figures_paper.py <input csv folder> <output figure folder>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_1": "#DDF3DE", "green_2": "#AADCA9",
           "green_3": "#8BCF8B", "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
           "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
           "gray_dark": "#4D4D4D", "gray_mid": "#767676"}
GREEN_DARK = "#3E8E41"
FULL, COLUMN = 14.3, 7.0


def apply_style():
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 15,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2,
        "axes.labelsize": 16, "axes.titlesize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "xtick.major.width": 2, "ytick.major.width": 2, "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.width": 1.2, "ytick.minor.width": 1.2, "xtick.minor.size": 3, "ytick.minor.size": 3,
        "legend.frameon": False, "legend.fontsize": 14, "pdf.fonttype": 42, "svg.fonttype": "none",
        "lines.linewidth": 3, "lines.markersize": 9, "axes.titlelocation": "left", "axes.titlepad": 10,
    })


def finalize(fig, name, pad=1.0):
    fig.tight_layout(pad=pad)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(name)


# 1. Pipeline schematic ---------------------------------------------------------------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(FULL, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 33)
    ax.axis("off")
    boxes = [
        ("Real clinical table", "laboratory values, categories,\nmissingness indicators,\nclass label", "neutral"),
        ("Fit-only coding", "normal-quantile scores,\nrecorded resolution,\nrelation residuals", "ours"),
        ("Diffusion model", "Gaussian and masked\ndiffusion; transformer with\ngraph bias and hyperedges", "ours"),
        ("Decoding", "round to the recorded grid,\nrebuild targets, re-derive\nexactly reported values", "ours"),
        ("Evaluation", "realism, relation diagnostics,\nutility and membership on\nheld-out records", "eval"),
    ]
    styles = {"neutral": ("#F2F2F2", PALETTE["gray_dark"]), "ours": ("#E3ECF7", PALETTE["blue_main"]),
              "eval": (PALETTE["green_1"], GREEN_DARK)}
    width, gap, y0, height = 17.0, 3.4, 18.0, 13.0
    xs = [1.0 + i * (width + gap) for i in range(len(boxes))]
    for i, ((title, body, kind), x) in enumerate(zip(boxes, xs)):
        face, edge = styles[kind]
        ax.add_patch(FancyBboxPatch((x, y0), width, height, boxstyle="round,pad=0.2,rounding_size=1.4",
                                    facecolor=face, edgecolor=edge, linewidth=2.2))
        ax.text(x + width / 2, y0 + height - 2.6, title, ha="center", va="center", fontsize=15, fontweight="bold", color=edge)
        ax.text(x + width / 2, y0 + height / 2 - 1.6, body, ha="center", va="center", fontsize=12.5, color="#1A1A1A",
                linespacing=1.4)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.4, y0 + height / 2), (x + width + gap - 0.4, y0 + height / 2),
                                         arrowstyle="-|>", mutation_scale=22, linewidth=2.2, color=PALETTE["gray_dark"]))
    bx0, bx1, by0, by1 = xs[1], xs[3] + width, 0.8, 13.6
    ax.add_patch(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, boxstyle="round,pad=0.2,rounding_size=1.4",
                                facecolor="white", edgecolor=PALETTE["blue_secondary"], linewidth=1.8, linestyle=(0, (5, 3))))
    for x in (xs[1], xs[3]):
        ax.add_patch(FancyArrowPatch((x + width / 2, y0 - 0.3), (x + width / 2, by1 + 0.35), arrowstyle="-",
                                     linewidth=1.6, color=PALETTE["blue_secondary"], linestyle=(0, (2, 2))))
    tx = bx0 + 1.4
    ax.text(tx, by1 - 2.2, "Relation reparameterization", ha="left", va="center", fontsize=14.5, fontweight="bold",
            color=PALETTE["blue_main"])
    ax.text(tx, by1 - 5.9, r"Encode:  $r_k = x_{t_k} - g_k(x_{P_k \setminus t_k})$", ha="left", va="center", fontsize=13.5)
    ax.text(tx, by1 - 9.6, r"Decode:  $\hat{x}_{t_k} = \rho\,(\,g_k(\hat{x}_{P_k \setminus t_k}) + \hat{r}_k\,)$", ha="left",
            va="center", fontsize=13.5)
    ex = bx0 + (bx1 - bx0) * 0.52
    ax.plot([ex - 1.2, ex - 1.2], [by0 + 1.6, by1 - 1.6], color=PALETTE["neutral"], linewidth=1.5)
    ax.text(ex, by1 - 2.2, "Example: blood count", ha="left", va="center", fontsize=13.5, fontweight="bold", color=PALETTE["gray_dark"])
    ax.text(ex, by1 - 5.9, "Hct = RBC × MCV / 10 holds up to rounding", ha="left", va="center", fontsize=12.5, color="#1A1A1A")
    ax.text(ex, by1 - 9.6, "generate Hct, MCV and residual r;\nrebuild RBC = 10 Hct / MCV + r", ha="left", va="center", fontsize=12.5,
            color="#1A1A1A", linespacing=1.35)
    ax.text(xs[0] + width / 2, 7.2, "Target choice", ha="center", va="center", fontsize=13.5, fontweight="bold", color=PALETTE["gray_dark"])
    ax.text(xs[0] + width / 2, 3.6, "reconstruct the participant\nwith the widest spread on\nthe additive scale", ha="center",
            va="center", fontsize=12, color="#1A1A1A", linespacing=1.35)
    ax.text(xs[4] + width / 2, 7.2, "Same knowledge, simple controls", ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=GREEN_DARK)
    ax.text(xs[4] + width / 2, 3.6, "re-derivation in either direction,\nindependent or joint\nresampling of real residuals",
            ha="center", va="center", fontsize=12, color="#1A1A1A", linespacing=1.35)
    finalize(fig, "fig_pipeline", pad=0.2)


# 2. Case-study ablation ----------------------------------------------------------------------------------------------
def ablation():
    part = pd.read_csv(DATA / "ablation_partition_means.csv")
    refs = pd.read_csv(DATA / "reference_rescored.csv")
    metrics = ["c2st", "numeric_w", "association", "violation_gap"]
    refs = refs.groupby(["arm", "inner_seed"])[metrics].mean().reset_index()
    frame = pd.concat([part, refs], ignore_index=True)
    frame["violation_gap"] = frame["violation_gap"] * 1000
    arms = [("a0_legacy_full_prior", "Original recipe", PALETTE["red_strong"], None),
            ("a1_fixed_steps_ema", "+ corrected training", PALETTE["blue_main"], 0.35),
            ("a3_plus_T100", "+ quantile coding, 100 steps", PALETTE["blue_main"], 0.6),
            ("a4_plus_precision", "+ recorded resolution (corrected)", PALETTE["blue_main"], 1.0),
            ("a4_but_epoch_selection", "Corrected, original training loop", PALETTE["red_2"], None),
            ("copula_repair", "Repaired copula", PALETTE["neutral"], None),
            ("bootstrap", "Bootstrap copy control", "white", None)]
    titles = {"c2st": "C2ST AUROC", "numeric_w": "Numeric Wasserstein", "association": "Association RMSE",
              "violation_gap": r"Rule violation gap ($\times 10^{-3}$)"}
    fig = plt.figure(figsize=(FULL, 5.0))
    grid = fig.add_gridspec(2, 4, height_ratios=[0.22, 1])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    handles = []
    for m, metric in enumerate(metrics):
        ax = fig.add_subplot(grid[1, m])
        stats = frame.groupby("arm")[metric].agg(["mean", "std"]).reindex([a[0] for a in arms])
        span = (stats["mean"] + stats["std"].fillna(0)).max()
        for i, (arm, name, color, alpha) in enumerate(arms):
            mean, sd = stats.loc[arm, "mean"], stats.loc[arm, "std"]
            bar = ax.bar(i, mean, yerr=sd, color=color, alpha=alpha if alpha else 1, edgecolor="black", linewidth=1.2,
                         hatch="//" if arm == "bootstrap" else None, width=0.78,
                         error_kw={"elinewidth": 1.6, "capthick": 1.6, "capsize": 4})
            base = 0.45 if metric == "c2st" else 0
            ax.text(i, mean + (sd if np.isfinite(sd) else 0) + 0.025 * (span - base),
                    f"{mean:.2f}" if metric != "violation_gap" else f"{mean:.1f}", ha="center", va="bottom", fontsize=11.5)
            if m == 0:
                handles.append(Patch(facecolor=color, alpha=alpha if alpha else 1, edgecolor="black", linewidth=1.2,
                                     hatch="//" if arm == "bootstrap" else None, label=name))
        if metric == "c2st":
            ax.set_ylim(0.45, 1.0)
            ax.axhline(0.5, color=PALETTE["gray_mid"], linewidth=1.5, linestyle=(0, (4, 3)))
        else:
            ax.set_ylim(0, span * 1.18)
        ax.set_xticks([])
        ax.set_title(titles[metric])
    legend_ax.legend(handles=handles, loc="center", ncol=4, fontsize=13.5, columnspacing=1.4, handlelength=1.6)
    finalize(fig, "fig_ablation", pad=0.6)


# 3. Ranking reversals --------------------------------------------------------------------------------------------------
def reversals():
    s = pd.read_csv(DATA / "a7_pipeline_summary.csv")
    order = [("Fidelity: common vs corrected practice", "Realism ranking", PALETTE["blue_main"]),
             ("Fidelity, privacy-aware ranking", "Privacy-aware ranking", PALETTE["green_3"]),
             ("Rules: violation count vs gap to real", "Rule scoring", PALETTE["red_2"])]
    settings = [("nhanes", 250), ("nhanes", 1000), ("thyroid", 250), ("thyroid", 1000)]
    names = ["NHANES\n$n$ = 250", "NHANES\n$n$ = 1,000", "Thyroid\n$n$ = 250", "Thyroid\n$n$ = 1,000"]
    fig, ax = plt.subplots(figsize=(COLUMN, 4.3))
    width = 0.26
    for k, (key, label, color) in enumerate(order):
        rows = [s[(s.comparison == key) & (s.dataset == d) & (s["size"] == n)].iloc[0] for d, n in settings]
        x = np.arange(len(settings)) + (k - 1) * width
        mean = np.array([r.reversal for r in rows]) * 100
        low = mean - np.array([r.reversal_low for r in rows]) * 100
        high = np.array([r.reversal_high for r in rows]) * 100 - mean
        ax.bar(x, mean, width=width, color=color, edgecolor="black", linewidth=1.2, label=label,
               yerr=[low, high], error_kw={"elinewidth": 1.4, "capthick": 1.4, "capsize": 3})
    ax.set_xticks(np.arange(len(settings)), names, fontsize=13)
    ax.set_ylabel("Generator pairs reversed (%)")
    ax.set_ylim(0, 62)
    ax.legend(loc="upper left", ncol=2, fontsize=12.5, columnspacing=1.0, handlelength=1.4)
    finalize(fig, "fig_reversals", pad=0.5)


# 4. Recording resolution ----------------------------------------------------------------------------------------------
def precision():
    s = pd.read_csv(DATA / "a3_precision_c2st.csv")
    order = ["mhd_corrected", "mhd_earlystop", "mhd_affine", "tabddpm_upstream", "tabddpm_adapter", "tabdiff", "ctgan", "tvae"]
    names = {"mhd_corrected": "MedHyperDiff", "mhd_earlystop": "MedHyperDiff, legacy", "mhd_affine": "MedHyperDiff, affine",
             "tabddpm_upstream": "TabDDPM, upstream", "tabddpm_adapter": "TabDDPM, adapter", "tabdiff": "TabDiff",
             "ctgan": "CTGAN", "tvae": "TVAE"}
    settings = [("nhanes", 250), ("nhanes", 1000), ("thyroid", 250), ("thyroid", 1000)]
    fig = plt.figure(figsize=(FULL, 5.4))
    grid = fig.add_gridspec(2, 4, height_ratios=[0.12, 1], width_ratios=[1.32, 1, 1, 1])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    y = np.arange(len(order))[::-1]
    h = 0.38
    for p, (d, n) in enumerate(settings):
        ax = fig.add_subplot(grid[1, p])
        g = s[(s.dataset == d) & (s["size"] == n)].set_index("variant").reindex(order)
        ax.barh(y + h / 2, g.c2st_raw, height=h, color=PALETTE["red_1"], edgecolor="black", linewidth=1.0, label="Raw output")
        ax.barh(y - h / 2, g.c2st_banded, height=h, color=PALETTE["blue_main"], edgecolor="black", linewidth=1.0,
                label="Decoded to recorded resolution")
        ax.set_xlim(0.5, 1.03)
        ax.set_xticks([0.6, 0.8, 1.0], ["0.6", "0.8", "1.0"])
        ax.set_yticks(y, [names[v] for v in order] if p == 0 else [""] * len(order), fontsize=13)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{'NHANES' if d == 'nhanes' else 'Thyroid'}, $n$ = {n:,}")
        ax.set_xlabel("C2ST AUROC")
        if p == 0:
            handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=2, fontsize=14)
    finalize(fig, "fig_precision", pad=0.6)


# 5. Realism against membership ----------------------------------------------------------------------------------------
def memorization():
    d = pd.read_csv(DATA / "all_jobs.csv")
    if "status" in d:
        d = d[d.status == "complete"]
    means = d.assign(membership=d[["membership_nn", "membership_density"]].max(axis=1)).groupby(
        ["dataset", "size", "variant"])[["c2st", "membership"]].mean().reset_index()
    settings = [("nhanes", 250), ("nhanes", 1000), ("thyroid", 250), ("thyroid", 1000)]
    roles = {"mhd_corrected": ("MedHyperDiff", PALETTE["blue_main"], "o", 190),
             "tabdiff": ("TabDiff", PALETTE["violet"], "D", 130),
             "tabddpm_upstream": ("TabDDPM", PALETTE["red_strong"], "s", 130),
             "copula": ("Gaussian copula", PALETTE["teal"], "^", 140),
             "bootstrap": ("Bootstrap copy", "black", "X", 150)}
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 4.4), sharey=True)
    for ax, (ds, n) in zip(axes, settings):
        g = means[(means.dataset == ds) & (means["size"] == n)]
        ax.axhspan(0.6, 1.05, color=PALETTE["red_1"], alpha=0.45, linewidth=0)
        ax.axhline(0.6, color=PALETTE["red_strong"], linewidth=1.6, linestyle=(0, (4, 3)))
        other = g[~g.variant.isin(roles)]
        ax.scatter(other.c2st, other.membership, s=90, color=PALETTE["neutral"], edgecolor=PALETTE["gray_dark"], linewidth=1.0, zorder=2)
        for variant, (name, color, marker, size) in roles.items():
            row = g[g.variant == variant]
            ax.scatter(row.c2st, row.membership, s=size, color=color, marker=marker, edgecolor="white", linewidth=1.2, zorder=3, label=name)
        ax.set_xlim(0.42, 1.04)
        ax.set_ylim(0.38, 1.04)
        ax.set_title(f"{'NHANES' if ds == 'nhanes' else 'Thyroid'}, $n$ = {n:,}")
        ax.set_xlabel("C2ST AUROC")
    axes[0].set_ylabel("Membership AUROC")
    handles = [Line2D([], [], linestyle="", marker=m, markersize=10, markerfacecolor=c, markeredgecolor="white", label=nm)
               for _, (nm, c, m, _) in roles.items()]
    handles.append(Line2D([], [], linestyle="", marker="o", markersize=9, markerfacecolor=PALETTE["neutral"],
                          markeredgecolor=PALETTE["gray_dark"], label="Other variants"))
    handles.append(Patch(facecolor=PALETTE["red_1"], alpha=0.6, label="Above 0.60 threshold"))
    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=13, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_memorization.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("fig_memorization")


# 6. Evaluation size ---------------------------------------------------------------------------------------------------
def evaluation_size():
    s = pd.read_csv(DATA / "a6_agreement_by_size.csv")
    fig, ax = plt.subplots(figsize=(COLUMN, 4.3))
    for ds, color, marker, name in (("nhanes", PALETTE["blue_main"], "o", "NHANES"), ("thyroid", PALETTE["red_strong"], "s", "Thyroid")):
        g = s[s.dataset == ds].sort_values("eval_size")
        ax.plot(g.eval_size, g.agreement * 100, color=color, marker=marker, label=name, markeredgecolor="white", markeredgewidth=1.2)
    ax.axhline(90, color=PALETTE["gray_mid"], linewidth=1.6, linestyle=(0, (4, 3)))
    ax.text(55, 90.6, "90% agreement", fontsize=12, color=PALETTE["gray_dark"])
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500, 1000], ["50", "100", "200", "500", "1,000"])
    ax.minorticks_off()
    ax.set_ylim(75, 98)
    ax.set_xlabel("Evaluation records per source")
    ax.set_ylabel("Pairs ordered as full reference (%)")
    ax.legend(loc="lower right")
    finalize(fig, "fig_evaluation_size", pad=0.5)


# 7. First relation test (2017-18 blood counts) ------------------------------------------------------------------------
def first_test():
    d = pd.read_csv(DATA / "registry_iteration13.csv")
    published = ["tabdiff", "tabddpm_upstream", "ctgan", "tvae", "copula"]
    families = [("As\ngenerated", "primary", PALETTE["red_strong"]), ("Rule\nre-der.", "derived", PALETTE["red_2"]),
                ("Textbook\nre-der.", "textbook", PALETTE["red_1"]), ("Indep.\nresiduals", "resampled", PALETTE["green_2"]),
                ("Joint\nresiduals", "joint", PALETTE["green_3"])]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 4.6), sharey=True)
    for ax, size in zip(axes, (250, 1000)):
        cell = d[d["size"] == size]
        means = cell.groupby(["variant", "processing"]).relation_w1.mean()
        labels, values, lows, highs, colors = [], [], [], [], []
        for name, proc, color in families:
            vals = np.array([means.loc[(v, proc)] for v in published])
            labels.append(name); values.append(vals.min()); lows.append(0); highs.append(np.median(vals) - vals.min()); colors.append(color)
        labels.append("TabDiff +\nreparam."); values.append(means.loc[("tabdiff_reparam", "primary")]); lows.append(0); highs.append(0)
        colors.append(PALETTE["teal"])
        labels.append("Ours"); values.append(means.loc[("mhd_reparam", "primary")]); lows.append(0); highs.append(0)
        colors.append(PALETTE["blue_main"])
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=1.2, width=0.72,
               yerr=[lows, highs], error_kw={"elinewidth": 1.4, "capthick": 1.4, "capsize": 4, "ecolor": PALETTE["gray_dark"]})
        for xi, v, hi in zip(x, values, highs):
            ax.text(xi, (v + hi) * 1.18, f"{v:.3f}" if v < 1 else f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom",
                    fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(0.08, 400)
        ax.set_xticks(x, labels, fontsize=12.5)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"Blood count 2017-18, $n$ = {size:,}")
    axes[0].set_ylabel("Relation distance (log scale)")
    finalize(fig, "fig_first_test", pad=0.6)


# 8. Robustness across cycles and sizes --------------------------------------------------------------------------------
def robustness():
    d = pd.read_csv(DATA / "registry_iteration14.csv")
    datasets = [("nhanes_cbc_h", "Blood count 2013-14"), ("nhanes_cbc_l", "Blood count 2021-23"),
                ("nhanes_chem_g", "Chemistry 2011-12"), ("nhanes_chem_l", "Chemistry 2021-23")]
    lines = [(("mhd:c1", "primary"), "MedHyperDiff", PALETTE["blue_main"], "-", "o", 1.0),
             (("mhd:c0", "primary"), "MedHyperDiff, fixed 5,000 updates", PALETTE["blue_secondary"], (0, (4, 2)), "o", 0.55),
             (("tabdiff", "primary"), "TabDiff, as generated", PALETTE["red_2"], (0, (1.5, 1.5)), "s", 1.0),
             (("tabdiff", "derived"), "TabDiff, rule-direction re-derivation", PALETTE["red_strong"], "-", "s", 1.0),
             (("tabdiff", "textbook"), "TabDiff, textbook re-derivation", PALETTE["violet"], "-", "v", 1.0),
             (("tabdiff", "joint"), "TabDiff, joint residuals", GREEN_DARK, "-", "D", 1.0),
             (("copula", "joint"), "Copula, joint residuals", PALETTE["teal"], "-", "^", 1.0)]
    fig = plt.figure(figsize=(FULL, 8.4))
    grid = fig.add_gridspec(3, 4, height_ratios=[0.2, 1, 1])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    for c, (ds, title) in enumerate(datasets):
        cell = d[d.dataset == ds]
        sizes = sorted(int(n) for n in cell["size"].unique())
        for r, (metric, ylabel) in enumerate((("relation_w1", "Relation distance"), ("c2st", "C2ST AUROC"))):
            ax = fig.add_subplot(grid[r + 1, c])
            for key, name, color, dash, marker, alpha in lines:
                if metric == "relation_w1" and key[0] == "mhd:c0":
                    continue
                g = cell[(cell.variant == key[0]) & (cell.processing == key[1])].groupby("size")[metric].mean()
                ax.plot([sizes.index(n) for n in g.index], g.values, linestyle=dash, color=color, marker=marker, alpha=alpha, label=name,
                        markeredgecolor="white", markeredgewidth=1.0, linewidth=2.6, markersize=8)
            ax.set_xticks(range(len(sizes)), [f"{n:,}" for n in sizes])
            ax.set_xlim(-0.25, len(sizes) - 0.75)
            if metric == "relation_w1":
                ax.set_yscale("log")
                ax.set_title(title)
            else:
                ax.set_xlabel("Nominal sample size $n$")
            if c == 0:
                ax.set_ylabel(ylabel + (" (log)" if metric == "relation_w1" else ""))
            if r == 1 and c == 0:
                handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=4, fontsize=13, columnspacing=1.2, handlelength=2.6)
    finalize(fig, "fig_robustness", pad=0.6)


if __name__ == "__main__":
    apply_style()
    for function in (pipeline, ablation, reversals, precision, memorization, evaluation_size, first_test, robustness):
        function()
