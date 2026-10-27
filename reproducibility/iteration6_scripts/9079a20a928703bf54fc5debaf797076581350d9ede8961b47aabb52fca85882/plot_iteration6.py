"""Show dispersion/constraint tradeoffs across all three development partitions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from iteration6 import REPORT


df = pd.read_csv(REPORT/"all_evaluations.csv")
ff = pd.read_csv(REPORT/"all_feature_results.csv")
ast = ff[ff.feature == "ast"][["arm", "inner_seed", "training_seed", "sampling_seed", "std_ratio_error"]]
data = df.merge(ast, on=["arm", "inner_seed", "training_seed", "sampling_seed"])
data = data[data.kind != "original"]
checkpoints = data.groupby(["architecture", "variance_mix", "inner_seed", "training_seed"])[["std_ratio_error", "violation"]].mean().reset_index()
means = checkpoints.groupby(["architecture", "variance_mix", "inner_seed"])[["std_ratio_error", "violation"]].mean().reset_index()
sd = checkpoints.groupby(["architecture", "variance_mix", "inner_seed"])[["std_ratio_error", "violation"]].std().reset_index()
fig, axes = plt.subplots(2, 3, figsize=(12, 7.3), sharex=True, sharey="row")
styles = {"full": ("Prior-initialized full", "#146c94"), "dense": ("Matched dense", "#8b4ca3"), "base": ("Guided backbone", "#bc5a13")}
for column, inner in enumerate((2027, 3027, 4027)):
    for row, (metric, label, scale) in enumerate((("std_ratio_error", "AST absolute SD-ratio error", 1), ("violation", "Supplied-rule violation (%)", 100))):
        ax = axes[row, column]
        for architecture, (display, color) in styles.items():
            part = means[(means.architecture == architecture) & (means.inner_seed == inner)].sort_values("variance_mix")
            error = sd[(sd.architecture == architecture) & (sd.inner_seed == inner)].sort_values("variance_mix")
            ax.errorbar(part.variance_mix, scale*part[metric], yerr=scale*error[metric], color=color,
                        marker="o", capsize=3, markersize=4, label=display)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([0, .5, 1], ["0 (current)", "0.5", "1 (beta)"])
        if row == 0:
            ax.set_title(f"Inner validation seed {inner}", fontsize=11)
        else:
            ax.set_xlabel("Fixed log-variance mixture")
        if column == 0:
            ax.set_ylabel(label)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
fig.suptitle("Larger reverse variance: numerical dispersion and rule compliance", fontsize=14, y=.995)
fig.text(.5, .025, "Means ± SD across 3 training seeds after averaging 3 sampling seeds. Both metrics: lower is better.\n"
    "Fixed trained weights, 5× tables and guidance 10. Error bars are descriptive; development partitions overlap.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, .1, 1, .90))
out = REPORT/"figures"
out.mkdir(exist_ok=True)
for extension in ("png", "svg", "pdf"):
    fig.savefig(out/f"variance_tradeoff.{extension}", dpi=180, bbox_inches="tight")
plt.close(fig)
