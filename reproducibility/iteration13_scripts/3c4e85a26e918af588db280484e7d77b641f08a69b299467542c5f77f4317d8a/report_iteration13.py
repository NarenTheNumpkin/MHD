"""Iteration 13 report: pre-registered relation-fidelity claim test (docs/ITERATION_13_PROTOCOL.md)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import iteration13 as it

REPORT = Path(os.environ.get("MHD_REPORT", it.ROOT / "reports" / "iteration13"))
PUBLISHED = ("tabdiff", "tabddpm_upstream", "ctgan", "tvae", "copula")
NAMES = {"tabdiff": "TabDiff", "tabddpm_upstream": "TabDDPM", "ctgan": "CTGAN", "tvae": "TVAE", "copula": "Gaussian copula",
         "bootstrap": "Bootstrap copy control", "tabdiff_reparam": "TabDiff + reparameterization",
         "mhd_unguided": "MedHyperDiff (no relation mechanism)", "mhd_guided": "MedHyperDiff + calibrated guidance",
         "mhd_reparam": "MedHyperDiff + reparameterization", "mhd_reparam_backbone": "MedHyperDiff backbone + reparameterization"}
PROCESSING = {"primary": "as generated", "derived": "hard derivation", "resampled": "derivation + resampled residuals"}


def rows(stage):
    out = []
    for manifest in sorted((it.OUT / stage).rglob("manifest.json")):
        m = json.loads(manifest.read_text())
        if m.get("status") != "complete":
            out.append(dict(id=m.get("id"), status=m.get("status")))
            continue
        metrics = json.loads((manifest.parent / "metrics.json").read_text())
        for table, x in metrics.items():
            row = dict(id=m["id"], status="complete", variant=m["variant"], processing=table, size=m["size"], seed=m["seed"],
                       relation_w1=x["relation_w1"], c2st=x["c2st"], guidance=m.get("guidance"))
            for name, rule in x["realism"]["rules"].items():
                if rule.get("residual_w1_fit_sd") is not None:
                    row[f"w1_{name}"] = rule["residual_w1_fit_sd"]
                    row[f"gap_{name}"] = rule["violation_gap"]
            if "privacy" in x:
                row.update(membership=max(x["privacy"]["nearest_synthetic_membership_auc"], x["density_attack"]["auc"]),
                           membership_nn=x["privacy"]["nearest_synthetic_membership_auc"], membership_density=x["density_attack"]["auc"],
                           numeric_w=x["fidelity"]["mean_numeric_wasserstein_train_sd"], association=x["fidelity"]["association_rmse"],
                           rf=x["utility"]["forest"]["TSTR"]["auroc"], rf_real=x["utility"]["forest"]["TRTR"]["auroc"])
            out.append(row)
    return pd.DataFrame(out)


def claim(d, primary):
    verdict = []
    for size in it.SIZES:
        g = d[d["size"] == size]
        p = g[(g.variant == primary) & (g.processing == "primary")].set_index("seed")
        wins = {}
        for method in PUBLISHED:
            for processing in ("primary", "derived"):
                o = g[(g.variant == method) & (g.processing == processing)].set_index("seed")
                seeds = p.index.intersection(o.index)
                wins[f"{method}/{processing}"] = int((p.loc[seeds, "relation_w1"] < o.loc[seeds, "relation_w1"]).sum())
        published = g[g.variant.isin(PUBLISHED)].groupby(["variant", "processing"])[["c2st", "membership"]].mean()
        candidates = published[published.membership <= .60]
        best_c2st = float(candidates.c2st.min()) if len(candidates) else np.nan
        best_name = "/".join(candidates.c2st.idxmin()) if len(candidates) else None
        c1 = all(v >= 4 for v in wins.values())
        c2 = bool(p.c2st.mean() <= best_c2st + .02)
        c3 = bool(p.membership.mean() <= .60)
        verdict.append(dict(size=size, primary=primary, min_replicate_wins=min(wins.values()), criterion_i=c1,
                            primary_c2st=float(p.c2st.mean()), best_published_nonmemorizing=best_name,
                            best_published_c2st=best_c2st, criterion_ii=c2, primary_membership=float(p.membership.mean()),
                            criterion_iii=c3, supported=c1 and c2 and c3, wins=json.dumps(wins)))
    return pd.DataFrame(verdict)


def figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, blue, gray = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#8a8984"
    fig, axes = plt.subplots(1, len(it.SIZES), figsize=(12, 4.2), sharey=True)
    for ax, size in zip(axes, it.SIZES):
        g = summary[summary["size"] == size].sort_values("relation_w1")
        y = np.arange(len(g))[::-1]
        colors = [blue if v.startswith("mhd") else gray for v in g.variant]
        ax.scatter(g.relation_w1, y, s=40, color=colors, edgecolor="white", linewidth=1, zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(y, [f"{NAMES.get(v, v)} — {PROCESSING.get(p, 'as generated')}" for v, p in zip(g.variant, g.processing)], color=ink, fontsize=7)
        ax.set_title(f"NHANES CBC, n = {size}: relation residual distance (log scale, lower is better)", fontsize=9, color=ink, loc="left")
        ax.grid(axis="x", color=grid, linewidth=.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=ink2, labelsize=7, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(exist_ok=True)
    development = rows("develop")
    development.to_csv(REPORT / "development_all.csv", index=False)
    if not it.CHOICE.exists():
        print(development.groupby("variant")[["relation_w1", "c2st", "membership"]].mean().round(4))
        return
    primary = json.loads(it.CHOICE.read_text())["primary"]
    confirm = rows("confirm")
    confirm.to_csv(REPORT / "confirm_all.csv", index=False)
    complete = confirm[confirm.status == "complete"]
    metrics = ["relation_w1", "c2st", "membership", "numeric_w", "association", "rf"] + sorted(c for c in complete if c.startswith(("w1_", "gap_")))
    summary = complete.groupby(["size", "variant", "processing"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" if b == "std" else a for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(REPORT / "confirm_summary.csv", index=False)
    verdict = claim(complete, primary)
    verdict.to_csv(REPORT / "claim_test.csv", index=False)
    figure(summary, REPORT / "figures" / "relation_fidelity.png")
    print(verdict.drop(columns="wins").to_string(index=False))
    print(summary[["size", "variant", "processing", "relation_w1", "c2st", "membership", "rf"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
