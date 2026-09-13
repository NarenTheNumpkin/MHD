"""Plot the rejected robust-coordinate tradeoff against retained references."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iteration8 import REPORT


def main():
    df = pd.read_csv(REPORT/"partition_means.csv")
    arms = ["asinh_mad", "full_prior", "copula_repair"]
    labels = ["asinh-MAD\n(rejected)", "full_prior\n(current)", "copula + repair"]
    colors = ["#b0443e", "#456a92", "#238575"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    metrics = [("rf", "Forest TSTR AUROC\n(higher better)"), ("numeric_w", "Numeric W / fit SD\n(lower better)"), ("violation", "Supplied-rule violations\n(lower better)")]
    x = np.arange(3)
    for ax, (metric, title) in zip(axes, metrics):
        for i, arm in enumerate(arms):
            vals = df[df.arm == arm].set_index("inner_seed").loc[[2027, 3027, 4027], metric].to_numpy()
            if metric == "violation": vals = 100*vals
            ax.plot(x, vals, marker="o", lw=1.8, color=colors[i], label=labels[i])
        ax.set_xticks(x, ["2027", "3027", "4027"])
        ax.set_xlabel("Overlapping inner partition")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        if metric == "numeric_w": ax.set_yscale("log")
        if metric == "violation": ax.set_ylim(0, 28)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.suptitle("Iteration 8: fit-only asinh-MAD representation fails tail and rule checks", fontsize=14)
    fig.supxlabel("Means across three training seeds for asinh-MAD/current; three sampling seeds for repaired copula. No confirmatory inference.", fontsize=10)
    out=REPORT/"figures"; out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out/f"asinh_mad_tradeoff.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__": main()
