"""Export a static scientific development figure directly from measured artifacts."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/medhyperdiff-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    table = pd.read_csv("reports/pilot/runs.csv")
    datasets = ["heart_failure", "ilpd", "ckd", "diabetes"]
    labels = ["Heart failure", "ILPD", "CKD", "Diabetes risk"]
    modes = ["copula", "base", "full", "ctgan", "tvae"]
    colors = ["#888888", "#4277b0", "#bd4c35", "#54a18c", "#9a79ab"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    x = np.arange(len(datasets))
    specs = [("association_rmse", "Mixed association error ↓", "Off-diagonal RMSE"),
             ("rule_violation", "Audited rule violations ↓", "Fraction of eligible rows"),
             ("tstr_forest_auroc", "Synthetic-training utility ↑", "Random forest AUROC")]
    width = .145
    for ax, (metric, title, ylabel) in zip(axes, specs):
        for i, (mode, color) in enumerate(zip(modes, colors)):
            rows = table[table.model == mode].set_index("dataset")
            values = [rows.loc[d, metric] for d in datasets]
            ax.bar(x + (i-2)*width, values, width, label=mode, color=color)
        ax.set_xticks(x, labels, rotation=18)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        if metric == "tstr_forest_auroc":
            ax.set_ylim(0, 1.05)
            refs = table[table.model == "base"].set_index("dataset")
            ax.scatter(x, [refs.loc[d, "trtr_forest_auroc"] for d in datasets],
                       marker="_", s=380, color="black", zorder=4, label="real-training reference")
    handles, names = axes[-1].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("Development pilots: 1 fold × 1 seed; inner validation only", fontsize=15, fontweight="bold")
    out = Path("reports/pilot")
    fig.savefig(out / "pilot_comparison.png", dpi=180)
    fig.savefig(out / "pilot_comparison.svg")
    plt.close(fig)
    print(out / "pilot_comparison.png")


if __name__ == "__main__":
    main()
