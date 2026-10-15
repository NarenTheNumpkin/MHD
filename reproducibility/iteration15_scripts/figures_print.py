"""Print-size versions of earlier report figures for the IEEE manuscript (7.16 in wide, 7 pt serif text).

Reads only saved summary tables (no runs):
  python scripts/figures_print.py <folder with the CSV tables> <output folder>
The plotted values are those of the original report figures; only layout, titles and labels change.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import report_iteration11 as r11
import report_iteration12 as r12
import report_iteration13 as r13

CSV, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, BLUE, GRAY, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#1f5fbf", "#8a8984", "#d9622b"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.8, "font.family": "serif"})
PANEL = lambda ds, n: f"{'NHANES' if ds == 'nhanes' else 'Thyroid'}, $n$ = {n:,}"


def style(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5, colors=INK2)


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)


def ablation():
    labels = {"a0_legacy_full_prior": "Original recipe", "a1_fixed_steps_ema": "+ corrected training recipe",
              "a2_plus_quantile": "+ quantile coding", "a3_plus_T100": "+ 100 diffusion steps",
              "a4_plus_precision": "+ recorded resolution (corrected)", "a4_but_epoch_selection": "Corrected, original training loop",
              "a4_no_guidance": "Corrected, no guidance", "a4_backbone": "Corrected, backbone only",
              "copula_repair": "Copula + repair", "copula": "Gaussian copula", "tvae_compact100": "TVAE",
              "ctgan_default300": "CTGAN", "bootstrap": "Bootstrap copy control"}
    partition = pd.read_csv(CSV / "ablation_partition_means.csv")
    refs = pd.read_csv(CSV / "reference_rescored.csv").groupby(["arm", "inner_seed"])[r11.KEY].mean().reset_index()
    frame = pd.concat([partition, refs[refs.arm.isin(["copula_repair", "copula", "tvae_compact100", "ctgan_default300", "bootstrap"])]])
    order = [a for a in labels if a in set(frame.arm)]
    metrics = [("c2st", "C2ST", .5), ("numeric_w", "Numeric W", None), ("association", "Association RMSE", None),
               ("violation_gap", "Rule gap", 0.)]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.75), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (metric, title, ref) in zip(axes, metrics):
        stats = frame.groupby("arm")[metric].agg(["mean", "std"]).reindex(order)
        colors = [BLUE if a == "a4_plus_precision" else GRAY for a in order]
        ax.errorbar(stats["mean"], y, xerr=stats["std"].fillna(0), fmt="none", ecolor=GRID, elinewidth=1.4, zorder=1)
        ax.scatter(stats["mean"], y, s=14, color=colors, zorder=3, edgecolor="white", linewidth=.5)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=.7, ls=(0, (3, 3)))
        ax.set_yticks(y, [labels[a] for a in order])
        ax.set_title(title, loc="left")
        style(ax)
    fig.tight_layout(w_pad=.6)
    save(fig, "fig_ablation.png")


def precision():
    s = pd.read_csv(CSV / "a3_precision_c2st.csv")
    order = list(r12.CONTINUOUS)
    short = {"mhd_corrected": "MedHyperDiff", "mhd_earlystop": "MedHyperDiff, legacy", "mhd_affine": "MedHyperDiff, affine",
             "tabddpm_upstream": "TabDDPM, upstream", "tabddpm_adapter": "TabDDPM, adapter", "tabdiff": "TabDiff", "ctgan": "CTGAN", "tvae": "TVAE"}
    groups = [(ds, n) for ds in r12.it.DATASETS for n in r12.it.SIZES]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.3), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (ds, n) in zip(axes, groups):
        g = s[(s.dataset == ds) & (s["size"] == n)].set_index("variant").reindex(order)
        for yi, (_, row) in zip(y, g.iterrows()):
            ax.plot([row.c2st_raw, row.c2st_banded], [yi, yi], color=GRID, lw=1.6, zorder=1)
        ax.scatter(g.c2st_raw, y, s=14, color=GRAY, zorder=3, edgecolor="white", linewidth=.5, label="raw output")
        ax.scatter(g.c2st_banded, y, s=14, color=BLUE, zorder=3, edgecolor="white", linewidth=.5, label="decoded to recorded resolution")
        ax.axvline(.5, color=INK2, lw=.7, ls=(0, (3, 3)))
        ax.set_yticks(y, [short[v] for v in order])
        ax.set_xlim(.45, 1.03)
        ax.set_title(PANEL(ds, n), loc="left")
        ax.set_xlabel("C2ST AUROC")
        style(ax)
    handles, texts = axes[0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .07, 1, 1), w_pad=.6)
    save(fig, "p3_precision_c2st.png")


def memorization():
    d = pd.read_csv(CSV / "all_jobs.csv")
    d = d[d.status == "complete"] if "status" in d else d
    means = d.assign(membership=d[["membership_nn", "membership_density"]].max(axis=1)).groupby(
        ["dataset", "size", "variant"])[["c2st", "membership"]].mean().reset_index()
    groups = [(ds, n) for ds in r12.it.DATASETS for n in r12.it.SIZES]
    names = {"mhd_corrected": "MedHyperDiff", "tabddpm_upstream": "TabDDPM", "tabdiff": "TabDiff", "bootstrap": "Bootstrap",
             "copula": "Copula", "ctgan": "CTGAN"}
    offsets = {"mhd_corrected": (-6, -12, "right"), "tabddpm_upstream": (6, 6, "left"), "tabdiff": (-6, 10, "right"),
               "bootstrap": (6, -2, "left"), "copula": (8, -12, "left"), "ctgan": (6, 10, "left")}
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.35), sharey=True)
    for ax, (ds, n) in zip(axes, groups):
        g = means[(means.dataset == ds) & (means["size"] == n)]
        ax.scatter(g.c2st, g.membership, s=14, color=[BLUE if v == "mhd_corrected" else GRAY for v in g.variant],
                   edgecolor="white", linewidth=.5, zorder=3)
        for _, row in g.iterrows():
            if row.variant in offsets:
                dx, dy, ha = offsets[row.variant]
                ax.annotate(names[row.variant], (row.c2st, row.membership), textcoords="offset points", xytext=(dx, dy),
                            fontsize=6, color=INK2, ha=ha, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=.4, shrinkA=1, shrinkB=2))
        ax.axhline(.6, color=INK2, lw=.7, ls=(0, (3, 3)))
        ax.set_ylim(.36, 1.04)
        ax.set_xlim(.42, 1.05)
        ax.set_title(PANEL(ds, n), loc="left")
        ax.set_xlabel("C2ST AUROC")
        style(ax, grid="y")
    axes[0].set_ylabel("Membership AUROC")
    fig.tight_layout(w_pad=.6)
    save(fig, "p6_memorization.png")


def pipeline():
    s = pd.read_csv(CSV / "a7_pipeline_summary.csv")
    titles = {"Fidelity: common vs corrected practice": "Realism ranking", "Fidelity, privacy-aware ranking": "Privacy-aware ranking",
              "Rules: violation count vs gap to real": "Rule scoring"}
    groups = [(ds, n) for ds in r12.it.DATASETS for n in r12.it.SIZES]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.75), sharey=True)
    y = np.arange(len(groups))[::-1]
    for ax, comparison in zip(axes, titles):
        g = s[s.comparison == comparison]
        for yi, (ds, n) in zip(y, groups):
            row = g[(g.dataset == ds) & (g["size"] == n)].iloc[0]
            ax.errorbar(row.reversal, yi, xerr=[[row.reversal - row.reversal_low], [row.reversal_high - row.reversal]], fmt="o",
                        color=BLUE, elinewidth=1.1, markersize=3.5, markeredgecolor="white", markeredgewidth=.5)
        ax.set_xlim(0, .6)
        ax.set_yticks(y, [PANEL(ds, n) for ds, n in groups])
        ax.set_title(titles[comparison], loc="left")
        ax.set_xlabel("Fraction of pairs reversed")
        style(ax)
    fig.tight_layout(w_pad=.8)
    save(fig, "pipeline_reversals.png")


def eval_size():
    s = pd.read_csv(CSV / "a6_agreement_by_size.csv")
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    for ds, color in zip(r12.it.DATASETS, (BLUE, ORANGE)):
        g = s[s.dataset == ds]
        ax.plot(g.eval_size, g.agreement, color=color, lw=1.3, marker="o", ms=3.2, markeredgecolor="white", markeredgewidth=.5,
                label="NHANES" if ds == "nhanes" else "Thyroid")
    ax.axhline(.9, color=INK2, lw=.7, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xticks(r12.it.CURVE_SIZES, [str(v) for v in r12.it.CURVE_SIZES])
    ax.minorticks_off()
    ax.set_xlabel("Evaluation records per source (single draw)")
    ax.set_ylabel("Agreement with full reference")
    ax.legend(frameon=False, loc="lower right")
    style(ax, grid="y")
    fig.tight_layout()
    save(fig, "p7_evaluation_size.png")


def relation_fidelity():
    s = pd.read_csv(CSV / "confirm_summary.csv")
    names = {"tabdiff": "TabDiff", "tabddpm_upstream": "TabDDPM", "ctgan": "CTGAN", "tvae": "TVAE", "copula": "Copula",
             "bootstrap": "Bootstrap copy", "tabdiff_reparam": "TabDiff + reparam.", "mhd_unguided": "MedHyperDiff, no relation coding",
             "mhd_reparam": "MedHyperDiff + reparam."}
    processing = {"primary": "as generated", "derived": "re-derived", "resampled": "resampled residuals"}
    keys = s.groupby(["variant", "processing"]).relation_w1.mean().sort_values().index.tolist()
    label = lambda v, p: names.get(v, v) if v in ("tabdiff_reparam", "mhd_reparam") else f"{names.get(v, v)}, {processing[p]}"
    y = np.arange(len(keys))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.3), sharey=True)
    for ax, size in zip(axes, sorted(s["size"].unique())):
        g = s[s["size"] == size].set_index(["variant", "processing"]).reindex(keys)
        ax.scatter(g.relation_w1, y, s=12, color=[BLUE if k == ("mhd_reparam", "primary") else GRAY for k in keys],
                   edgecolor="white", linewidth=.5, zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(y, [label(*k) for k in keys])
        ax.set_title(f"$n$ = {size:,}", loc="left")
        ax.set_xlabel("Relation distance (log scale)")
        style(ax)
    fig.tight_layout(w_pad=.8)
    save(fig, "relation_fidelity.png")


for function in (ablation, precision, memorization, pipeline, eval_size, relation_fidelity):
    function()
print("done")
