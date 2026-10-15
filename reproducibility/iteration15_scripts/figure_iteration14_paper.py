"""Print-size robustness figure for the manuscript (7.16 in wide, 7 pt text)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import iteration14 as it

d = pd.read_csv(it.ROOT / "reports" / "iteration14" / "confirm_all.csv")
d = d[(d.status == "complete") & d["size"].isin([250, 1000, 2000])]
primary = json.loads(it.CHOICE.read_text())["primary"]
titles = {"nhanes_cbc_h": "Blood count 2013-14", "nhanes_cbc_l": "Blood count 2021-23",
          "nhanes_chem_g": "Chemistry 2011-12", "nhanes_chem_l": "Chemistry 2021-23"}
lines = [((f"mhd:{primary}", "primary"), "MedHyperDiff (budget proportional to n)", "#1f5fbf", "-", "o"),
         (("mhd:c0", "primary"), "MedHyperDiff (fixed 5,000 updates)", "#8fb3e6", "--", "o"),
         (("tabdiff", "primary"), "TabDiff, as generated", "#b4410f", ":", "s"),
         (("tabdiff", "derived"), "TabDiff, hard derivation", "#b4410f", "-", "s"),
         (("tabdiff_rp", "primary"), "TabDiff + reparameterization", "#e39b12", "-", "D"),
         (("copula", "resampled"), "Copula, derivation + resampled residuals", "#6e6d69", "-", "^")]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.8, "font.family": "serif"})
fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.75))
for column, dataset in enumerate(it.CONFIRMATION):
    cell = d[d.dataset == dataset]
    sizes = sorted(cell["size"].unique())
    for row, (metric, label) in enumerate((("relation_w1", "Relation distance (log)"), ("c2st", "C2ST AUROC"))):
        ax = axes[row, column]
        for key, text, color, dash, marker in lines:
            if metric == "relation_w1" and key[0] == "mhd:c0":
                continue
            g = cell[(cell.variant == key[0]) & (cell.processing == key[1])].groupby("size")[metric]
            m, s = g.mean(), g.std().fillna(0)
            ax.plot(m.index, m.values, dash, color=color, lw=1.1, marker=marker, ms=2.8, label=text)
            if metric == "c2st":
                ax.fill_between(m.index, (m - s).values, (m + s).values, color=color, alpha=.10, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
        ax.minorticks_off()
        if metric == "relation_w1":
            ax.set_yscale("log")
            ax.set_title(titles[dataset], loc="left")
        else:
            ax.set_xlabel("Nominal sample size $n$")
        if column == 0:
            ax.set_ylabel(label)
        ax.grid(axis="y", color="#e4e3df", lw=.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.01), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=(0, .1, 1, 1), h_pad=.6, w_pad=.5)
out = it.ROOT / "reports" / "iteration14" / "figures" / "robustness_paper.png"
fig.savefig(out, dpi=300, facecolor="white")
print(out)
