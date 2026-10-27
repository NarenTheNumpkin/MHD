"""Export development partition comparisons; whiskers are generator-seed SD."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/medhyperdiff-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/medhyperdiff-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iteration3 import REPORT

data = pd.read_csv(REPORT / "all_evaluations.csv")
data = data[data.dataset == "ilpd"]
arms = {"full_x0_l0p1_g0p2": "Previous full (x0)",
    "full_v_l0p01_g10": "Updated full (v, weight 0.01, guidance 10)",
    "base_v_l0_g10": "Guided velocity backbone"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
colors = ["#777777", "#0072B2", "#D55E00"]
for i, (arm, label) in enumerate(arms.items()):
    d = data[data.setting == arm].groupby("inner_seed")
    mean = d[["rf", "violation", "w"]].mean().reindex([2027, 3027, 4027])
    sd = d[["rf", "violation", "w"]].std().reindex([2027, 3027, 4027])
    assert (d.size() == 3).all()
    for ax, metric in zip(axes, ("rf", "violation", "w")):
        ax.errorbar(np.arange(3) + (i-1)*.13, mean[metric], yerr=sd[metric],
            fmt=["s", "o", "^"][i], color=colors[i], capsize=3, label=label, markersize=6)
for ax, title in zip(axes, ("Synthetic-trained RF AUROC ↑", "Supplied-rule violation fraction ↓", "Normalized numeric Wasserstein ↓")):
    ax.set(title=title, xlabel="Inner validation partition", xticks=np.arange(3), xticklabels=["2027", "3027", "4027"])
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
fig.suptitle("ILPD development comparisons across overlapping validation partitions\n"
    "Points: three-generator-seed means; whiskers: seed SD, not confidence intervals", fontsize=12)
for extension in ("png", "svg", "pdf"):
    fig.savefig(REPORT / f"replication_comparison.{extension}", dpi=180)
plt.close(fig)
