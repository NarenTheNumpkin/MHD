"""Scientific figure of complete partition-level baseline tradeoffs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iteration7 import REPORT

ORDER = ["full_prior", "dense_prior", "base_prior", "ctgan_compact100", "ctgan_default300",
         "tvae_compact100", "tvae_default300", "copula", "copula_repair", "independent", "bootstrap"]
LABELS = ["MedHyperDiff (current)", "Dense control", "Backbone control", "CTGAN compact / 100", "CTGAN default / 300",
          "TVAE compact / 100", "TVAE default / 300", "Copula", "Copula + repair", "Independent", "Bootstrap (copies)"]


def main():
    data = pd.read_csv(REPORT/"partition_means.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True, layout="constrained")
    colors = ["#c45922" if a == "full_prior" else "#238575" if a == "copula_repair" else "#9b9fa6" if a == "bootstrap" else "#456a92" for a in ORDER]
    metrics = [("rf", "Forest TSTR AUROC (higher better)", (.55, .82)),
               ("numeric_w", "Numeric Wasserstein / fit SD (lower better)", (0, .55)),
               ("violation", "Supplied-rule violations, % (lower better)", (-2, 102))]
    y = np.arange(len(ORDER))
    for col, inner in enumerate((2027, 3027, 4027)):
        cell = data[data.inner_seed == inner].set_index("arm").loc[ORDER]
        for row, (metric, label, limits) in enumerate(metrics):
            ax = axes[row, col]
            values = cell[metric].to_numpy()*(100 if metric == "violation" else 1)
            ax.scatter(values, y, c=colors, s=44, zorder=3)
            ax.set(yticks=y, yticklabels=LABELS, xlim=limits, xlabel=label)
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(f"Inner partition {inner}", fontweight="bold")
                ax.axvline(cell.rf_real.iloc[0], color="#666666", ls="--", lw=1)
            if col == 0:
                for tick in ax.get_yticklabels():
                    if tick.get_text() == LABELS[0]:
                        tick.set_fontweight("bold")
    axes[0, 0].invert_yaxis()
    fig.suptitle("Matched ILPD development comparisons: utility, fidelity and supplied rules", fontsize=16)
    fig.supxlabel("Neural means: 3 training × 3 sampling seeds; statistical means: 3 sampling seeds.\nDashed lines: real-only forest AUROC. Partitions overlap; no confidence intervals or superiority inference.", fontsize=11)
    out = REPORT/"figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out/f"matched_baselines.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
