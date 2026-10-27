"""Print-size robustness figure (7.16 in, 7 pt) from the final comparator registry, including post hoc controls."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import iteration14 as it

out = sys.argv[1]
d = pd.read_csv(it.ROOT / "reports" / "final" / "registry_iteration14.csv")
titles = {"nhanes_cbc_h": "Blood count 2013-14", "nhanes_cbc_l": "Blood count 2021-23",
          "nhanes_chem_g": "Chemistry 2011-12", "nhanes_chem_l": "Chemistry 2021-23"}
lines = [(("mhd:c1", "primary"), "MedHyperDiff, budget proportional to n", "#1f5fbf", "-", "o"),
         (("mhd:c0", "primary"), "MedHyperDiff, fixed 5,000 updates", "#8fb3e6", "--", "o"),
         (("tabdiff", "primary"), "TabDiff, as generated", "#b4410f", ":", "s"),
         (("tabdiff", "derived"), "TabDiff, rule-direction re-derivation", "#b4410f", "-", "s"),
         (("tabdiff", "textbook"), "TabDiff, textbook re-derivation", "#5e1c06", "-.", "v"),
         (("tabdiff", "joint"), "TabDiff, joint residuals", "#2e8b57", "-", "D"),
         (("copula", "joint"), "Copula, joint residuals", "#6e6d69", "-", "^")]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.6, "font.family": "serif"})
fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.95))
for column, dataset in enumerate(it.CONFIRMATION):
    cell = d[d.dataset == dataset]
    sizes = sorted(cell["size"].unique())
    for row, (metric, ylabel) in enumerate((("relation_w1", "Relation distance (log)"), ("c2st", "C2ST AUROC"))):
        ax = axes[row, column]
        for key, text, color, dash, marker in lines:
            if metric == "relation_w1" and key[0] == "mhd:c0":
                continue
            g = cell[(cell.variant == key[0]) & (cell.processing == key[1])].groupby("size")[metric]
            m = g.mean()
            ax.plot(m.index, m.values, dash, color=color, lw=1.1, marker=marker, ms=2.8, label=text)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
        ax.minorticks_off()
        if metric == "relation_w1":
            ax.set_yscale("log")
            ax.set_title(titles[dataset], loc="left")
        else:
            ax.set_xlabel("Nominal sample size $n$")
        if column == 0:
            ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#e4e3df", lw=.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.01), handlelength=2.4, columnspacing=1.0)
fig.tight_layout(rect=(0, .12, 1, 1), h_pad=.6, w_pad=.5)
fig.savefig(out, dpi=300, facecolor="white")
print(out)
