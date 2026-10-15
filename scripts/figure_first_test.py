"""Print-size relation-fidelity figure for the 2017-18 blood count test, including the post hoc controls."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import iteration14 as it

d = pd.read_csv(it.ROOT / "reports" / "final" / "registry_iteration13.csv")
d = d[d.variant != "bootstrap"]
names = {"tabdiff": "TabDiff", "tabddpm_upstream": "TabDDPM", "ctgan": "CTGAN", "tvae": "TVAE", "copula": "Copula"}
processing = {"primary": "as generated", "derived": "rule-direction re-der.", "textbook": "textbook re-der.",
              "resampled": "independent residuals", "joint": "joint residuals"}
def label(v, p):
    if v == "mhd_reparam":
        return "MedHyperDiff"
    if v == "tabdiff_reparam":
        return "TabDiff + reparameterization"
    if v == "mhd_unguided":
        return f"MedHyperDiff, no relation coding, {processing[p]}"
    return f"{names[v]}, {processing[p]}"
means = d.groupby(["variant", "processing"]).relation_w1.mean().sort_values()
keys = means.index.tolist()
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.2, "font.family": "serif"})
y = np.arange(len(keys))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.3), sharey=True)
for ax, size in zip(axes, (250, 1000)):
    g = d[d["size"] == size].groupby(["variant", "processing"]).relation_w1.mean().reindex(keys)
    colors = ["#1f5fbf" if k == ("mhd_reparam", "primary") else "#2e8b57" if k[1] == "joint" else "#8a8984" for k in keys]
    ax.scatter(g.values, y, s=11, color=colors, edgecolor="white", linewidth=.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, [label(*k) for k in keys])
    ax.set_title(f"$n$ = {size:,}", loc="left")
    ax.set_xlabel("Relation distance (log scale)")
    ax.grid(axis="x", color="#e4e3df", lw=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
fig.tight_layout(w_pad=.8)
fig.savefig(sys.argv[1], dpi=300, facecolor="white")
print(sys.argv[1])
