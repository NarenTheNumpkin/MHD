"""Publication-exportable descriptive size-sensitivity figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from iteration4 import REPORT


means = pd.read_csv(REPORT / "partition_means.csv")
sd = pd.read_csv(REPORT / "training_seed_sd.csv")
arms = {"previous_full": ("Previous full", "#6b7280"),
        "updated_full": ("Updated full", "#146c94"),
        "guided_base": ("Guided backbone", "#bc5a13"),
        "matched_dense": ("Matched dense", "#8b4ca3")}
joined = means.merge(sd, on=["arm", "inner_seed", "multiplier"], suffixes=("_mean", "_sd"))
lower = min((joined[f"{metric}_mean"]-joined[f"{metric}_sd"]).min() for metric in ("rf", "logistic"))-.015
upper = max((joined[f"{metric}_mean"]+joined[f"{metric}_sd"]).max() for metric in ("rf", "logistic"))+.015
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
for col, inner in enumerate((2027, 3027, 4027)):
    for row, (metric, label) in enumerate((("rf", "Forest TSTR AUROC"), ("logistic", "Logistic TSTR AUROC"))):
        ax = axes[row, col]
        for offset, (arm, (display, color)) in zip((-.06, -.02, .02, .06), arms.items()):
            part = means[(means.arm == arm) & (means.inner_seed == inner)].sort_values("multiplier")
            error = sd[(sd.arm == arm) & (sd.inner_seed == inner)].sort_values("multiplier")
            ax.errorbar([0+offset, 1+offset], part[metric], yerr=error[metric], marker="o", capsize=3,
                        color=color, label=display, linewidth=1.6, markersize=4)
        ax.grid(axis="y", alpha=.2)
        ax.set_xticks([0, 1], ["1× fit rows", "5× fit rows"])
        ax.set_xlim(-.2, 1.2)
        ax.set_ylim(lower, upper)
        if row == 0:
            ax.set_title(f"Inner validation seed {inner}", fontsize=11)
        if col == 0:
            ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
fig.suptitle("ILPD: does a larger synthetic table improve prediction utility?", y=.995, fontsize=14)
fig.text(.5, .035, "Means ± SD across 3 training seeds after averaging 3 sampling seeds per checkpoint.\n"
         "Nested 1×/5× tables; fixed evaluator seed. Development partitions overlap; error bars are not confidence intervals.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .09, 1, .90))
out = REPORT / "figures"
out.mkdir(exist_ok=True)
for ext in ("png", "svg", "pdf"):
    fig.savefig(out / f"sample_size_utility.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)
