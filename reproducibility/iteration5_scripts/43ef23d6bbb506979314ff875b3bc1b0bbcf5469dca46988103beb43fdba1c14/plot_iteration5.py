"""Descriptive scientific exports for the frozen observation-head study."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iteration5 import REPORT


df = pd.read_csv(REPORT/"all_evaluations.csv")
ff = pd.read_csv(REPORT/"all_feature_results.csv")
means = pd.read_csv(REPORT/"partition_means.csv")
sd = pd.read_csv(REPORT/"training_seed_sd.csv")
arms = {"reference_updated_full": ("Original full", "#737373"),
        "full_prior": ("Prior initialization", "#146c94"),
        "full_prior_w4": ("Prior + mask loss ×4", "#c16b13")}
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
for i, (arm, (label, color)) in enumerate(arms.items()):
    data = df[(df.arm == arm) & (df.inner_seed == 2027) & (df.training_seed == 1)].sort_values("sampling_seed")
    axes[0].scatter(i+np.array([-.08, 0, .08]), 100*data.ag_missing, color=color, s=45, zorder=4)
    data = means[means.arm == arm].sort_values("inner_seed")
    err = sd[sd.arm == arm].sort_values("inner_seed")
    axes[1].errorbar(np.arange(3)+(i-1)*.055, data.rf, yerr=err.rf, color=color, marker="o", capsize=3, label=label)
fit_rate = ff[(ff.arm == "full_prior") & (ff.inner_seed == 2027) & (ff.feature == "ag_ratio")]["fit"].iloc[0]
axes[0].axhline(100*fit_rate, color="#225c2d", linestyle="--", label=f"Fit data: {100*fit_rate:.2f}%")
axes[0].set_xticks(range(3), ["Original full", "Prior\ninitialization", "Prior +\nmask loss ×4"])
axes[0].set_ylabel("Missing A/G ratios (%)")
axes[0].set_title("Previously failing cell: inner 2027, training seed 1", fontsize=10)
axes[0].legend(loc="upper right", frameon=False, fontsize=9)
axes[0].set_ylim(bottom=0)
axes[1].set_xticks(range(3), ["2027", "3027", "4027"])
axes[1].set_xlabel("Inner validation seed")
axes[1].set_ylabel("Forest TSTR AUROC")
axes[1].set_title("All three validation partitions", fontsize=10)
axes[1].legend(loc="best", frameon=False, fontsize=9)
for ax in axes:
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Observation-head initialization: missingness and prediction utility", fontsize=14)
fig.text(.5, .025, "Left: three sampling seeds per model, using the failure case identified before this study.\n"
    "Right: mean ± training-seed SD after averaging sampling seeds. All tables are 5× fit size; development partitions overlap.",
    ha="center", fontsize=9)
fig.tight_layout(rect=(0, .12, 1, .93))
out = REPORT/"figures"
out.mkdir(exist_ok=True)
for extension in ("png", "pdf", "svg"):
    fig.savefig(out/f"observation_initialization.{extension}", dpi=180, bbox_inches="tight")
plt.close(fig)
