"""Pre-registered analyses for iteration 12 (docs/ITERATION_12_PROTOCOL.md) -> reports/iteration12.

Reads runs/iteration12 (MHD_OUT) and, for analysis 5, the iteration-11 search table. Evaluation-only
extras (raw-output C2ST at 100 per class) are computed from saved synthetic tables; nothing is refit.
"""
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import iteration12 as it

REPORT = Path(os.environ.get("MHD_REPORT", it.ROOT / "reports" / "iteration12"))
CONTINUOUS = ("mhd_corrected", "mhd_earlystop", "mhd_affine", "tabddpm_upstream", "tabddpm_adapter", "tabdiff", "ctgan", "tvae")
# Pipelines for analysis 7: the same eight generator identities under common and corrected practice.
COMMON = {"MedHyperDiff": "mhd_earlystop", "TabDDPM": "tabddpm_adapter", "TabDiff": "tabdiff", "CTGAN": "ctgan",
          "TVAE": "tvae", "Copula": "copula", "Copula+repair": "copula_repair", "Independent": "independent"}
CORRECTED = {**COMMON, "MedHyperDiff": "mhd_corrected", "TabDDPM": "tabddpm_upstream"}
MEMORIZATION = .60


def load_records():
    rows = []
    for path in sorted(it.OUT.rglob("manifest.json")):
        if "smoke" in path.parts:
            continue
        m = json.loads(path.read_text())
        if m.get("status") != "complete":
            rows.append(dict(id=m.get("id"), status=m.get("status"), error=m.get("error")))
            continue
        x = json.loads((path.parent / "metrics.json").read_text())
        realism = x["constraints"]["realism"]
        rules = x["constraints"]["synthetic"]["rules"]
        row = dict(id=m["id"], status="complete", dataset=m["dataset"], size=m["size"], replicate=m["replicate"],
                   variant=m["variant"], selected_epoch=m.get("selected_epoch"), executed=m.get("executed"),
                   fit_seconds=m["fit_seconds"], c2st=x["c2st"]["auc"],
                   numeric_w=x["fidelity"]["mean_numeric_wasserstein_train_sd"], association=x["fidelity"]["association_rmse"],
                   signed_spearman=x["fidelity"]["signed_numeric_spearman_rmse"], rf=x["utility"]["forest"]["TSTR"]["auroc"],
                   logistic=x["utility"]["logistic"]["TSTR"]["auroc"], rf_real=x["utility"]["forest"]["TRTR"]["auroc"],
                   violation=x["constraints"]["synthetic"]["any_rule_violation_rate"],
                   real_violation=x["constraints"]["holdout"]["any_rule_violation_rate"],
                   violation_gap=realism["mean_violation_gap"], relational_w1=realism["mean_relational_residual_w1_fit_sd"],
                   hard_violation=realism["hard_rule_violation_rate"]["synthetic"],
                   membership_nn=x["privacy"]["nearest_synthetic_membership_auc"], membership_density=x["density_attack"]["auc"],
                   exact_copy=x["privacy"]["exact_predictor_match_train"])
        for name, rule in rules.items():
            if not name.endswith("nonnegative"):
                row[f"viol_{name}"] = rule["violation_rate"]
                row[f"w1_{name}"] = realism["rules"][name].get("residual_w1")
        for mode, values in x["decodings"].items():
            row[f"c2st_{mode}"] = values["c2st"]
            for col, w in values["wasserstein_fit_sd"].items():
                row[f"w_{mode}_{col}"] = w
        for r in x["c2st_curve"]:
            row.setdefault(f"curve_{r['size']}", []).append(r["auc"])
        rows.append(row)
    return pd.DataFrame(rows)


def raw_c2st_at(records, size=100, repeats=10):
    """Evaluation-only: raw-output C2ST at `size` per class, mean of `repeats` subsamples (cached)."""
    from medhyperdiff.evaluation import c2st_curve
    cache = REPORT / f"raw_c2st_{size}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    out = []
    plans = json.loads(it.SPLITS.read_text())["datasets"]
    for dataset in it.DATASETS:
        frame, schema = it.load(dataset)
        reference = frame.iloc[plans[dataset]["reference"]]
        for _, row in records[records.dataset == dataset].iterrows():
            syn = pd.read_csv(it.OUT / row["id"] / "synthetic_raw.csv", dtype={c: str for c in (*schema.categorical, schema.target)},
                              float_precision="round_trip")
            values = [r["auc"] for r in c2st_curve(reference, syn, schema, (size,), repeats, seed=it.EVALUATOR_SEED)]
            out.append(dict(id=row["id"], raw_c2st_100=float(np.mean(values)), raw_c2st_100_first=values[0]))
    frame = pd.DataFrame(out)
    frame.to_csv(cache, index=False)
    return frame


def pairwise_reversal(a, b, lower_better=True):
    """Fraction of generator pairs ordered differently by score vectors a and b (ties excluded), and Kendall tau."""
    keys = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
    flips, pairs = 0, 0
    for i, j in itertools.combinations(keys, 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 or db == 0:
            continue
        pairs += 1
        flips += np.sign(da) != np.sign(db)
    tau = kendalltau([a[k] for k in keys], [b[k] for k in keys]).statistic if len(keys) > 1 else np.nan
    best = lambda s: min(keys, key=lambda k: s[k]) if lower_better else max(keys, key=lambda k: s[k])
    return dict(pairs=pairs, reversal=flips / pairs if pairs else np.nan, tau=tau, top_a=best(a), top_b=best(b))


def bootstrap_mean(values, draws=2000, seed=2026):
    values = np.asarray([v for v in values if np.isfinite(v)])
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def analysis_1(d):
    rows = []
    for variant, cap, counterpart in (("mhd_earlystop", 100, "mhd_corrected"), ("tabddpm_adapter", 300, "tabddpm_upstream")):
        x = d[d.variant == variant]
        selected = x.selected_epoch.astype(float)
        rows.append(dict(analysis="selection", variant=variant, fits=len(x), cap=cap,
                         early_fraction=float((selected <= .2 * cap).mean()), cap_fraction=float((x.executed >= cap).mean()),
                         median_selected=float(selected.median()), min_selected=float(selected.min())))
        for metric in ("c2st", "numeric_w", "association"):
            paired = x.merge(d[d.variant == counterpart], on=["dataset", "size", "replicate"], suffixes=("", "_c"))
            for (dataset, size), g in paired.groupby(["dataset", "size"]):
                diff = g[metric] - g[f"{metric}_c"]
                rows.append(dict(analysis="paired", variant=variant, counterpart=counterpart, metric=metric, dataset=dataset,
                                 size=size, mean_difference=float(diff.mean()), sd=float(diff.std()),
                                 replicates_worse=int((diff > 0).sum()), replicates=len(diff)))
    return pd.DataFrame(rows)


def analysis_2(d):
    rows = []
    for dataset in it.DATASETS:
        frame, schema = it.load(dataset)
        skewed = frame[list(schema.numeric)].skew().sort_values(ascending=False).index[:5].tolist()
        pair = d[(d.dataset == dataset) & (d.variant == "mhd_affine")].merge(
            d[(d.dataset == dataset) & (d.variant == "mhd_corrected")], on=["size", "replicate"], suffixes=("_a", "_q"))
        if pair.empty:
            continue
        for col in skewed + ["c2st"]:
            a = pair[f"w_banded_{col}_a"] if col != "c2st" else pair["c2st_a"]
            q = pair[f"w_banded_{col}_q"] if col != "c2st" else pair["c2st_q"]
            for size in it.SIZES:
                m = pair["size"] == size
                rows.append(dict(dataset=dataset, feature=col, size=size, affine=float(a[m].mean()), quantile=float(q[m].mean()),
                                 replicates_quantile_better=int((q[m] < a[m]).sum()), replicates=int(m.sum())))
    return pd.DataFrame(rows)


def analysis_3(d):
    summary = d[d.variant.isin(CONTINUOUS)].groupby(["dataset", "size", "variant"])[["c2st_raw", "c2st_global", "c2st_banded"]].mean().reset_index()
    reversals = []
    for (dataset, size, replicate), g in d.groupby(["dataset", "size", "replicate"]):
        g = g[g.variant != "bootstrap"]
        a, b = dict(zip(g.variant, g.c2st_raw)), dict(zip(g.variant, g.c2st_banded))
        reversals.append(dict(dataset=dataset, size=size, replicate=replicate, **pairwise_reversal(a, b)))
    return summary, pd.DataFrame(reversals)


def analysis_4(d):
    rows = []
    for (dataset, size, replicate), g in d.groupby(["dataset", "size", "replicate"]):
        g = g[g.variant != "bootstrap"]
        violation = dict(zip(g.variant, g.violation))
        gap = dict(zip(g.variant, g.violation_gap))
        w1 = dict(zip(g.variant, g.relational_w1))
        rows.append(dict(dataset=dataset, size=size, replicate=replicate, comparison="violation_vs_gap", **pairwise_reversal(violation, gap)))
        rows.append(dict(dataset=dataset, size=size, replicate=replicate, comparison="violation_vs_residual_w1", **pairwise_reversal(violation, w1)))
    means = d.groupby(["dataset", "variant"])[["violation", "real_violation", "violation_gap", "relational_w1", "hard_violation"]].mean().reset_index()
    return pd.DataFrame(rows), means


def analysis_5(d):
    rows = []
    for (dataset, size, replicate), g in d.groupby(["dataset", "size", "replicate"]):
        for include_copy in (True, False):
            h = g if include_copy else g[g.variant != "bootstrap"]
            best = h.loc[h.c2st.idxmin()]
            rows.append(dict(dataset=dataset, size=size, replicate=replicate, include_copy_control=include_copy,
                             best_variant=best.variant, best_c2st=best.c2st, best_membership_nn=best.membership_nn,
                             best_membership_density=best.membership_density,
                             memorizing=bool(max(best.membership_nn, best.membership_density) > MEMORIZATION)))
    correlation = d.groupby(["dataset", "size"]).apply(lambda g: pd.Series({
        "spearman_c2st_vs_membership_nn": g[["c2st", "membership_nn"]].corr("spearman").iloc[0, 1],
        "spearman_c2st_vs_membership_density": g[["c2st", "membership_density"]].corr("spearman").iloc[0, 1]})).reset_index()
    search_path = it.ROOT / "configs" / "iteration11_selection.json"
    search = pd.DataFrame(json.loads(search_path.read_text())["table"]) if search_path.exists() else pd.DataFrame()
    search_rows = []
    if not search.empty:
        for (dataset, method), g in search.groupby(["dataset", "method"]):
            g = g.dropna(subset=["c2st"])
            best = g.loc[g.c2st.idxmin()]
            search_rows.append(dict(dataset=dataset, method=method, best_c2st=best.c2st, best_membership=best.membership,
                                    memorizing=bool(best.membership > MEMORIZATION),
                                    spearman=g[["c2st", "membership"]].corr("spearman").iloc[0, 1]))
    return pd.DataFrame(rows), correlation, pd.DataFrame(search_rows)


def analysis_6(d):
    rows = []
    for (dataset, size, replicate), g in d.groupby(["dataset", "size", "replicate"]):
        g = g[g.variant != "bootstrap"].set_index("variant")
        for m in it.CURVE_SIZES:
            column = f"curve_{m}"
            if column not in g or g[column].isna().any():
                continue
            for i, j in itertools.combinations(g.index, 2):
                full = g.at[i, "c2st"] - g.at[j, "c2st"]
                if full == 0:
                    continue
                draws = np.array(g.at[i, column]) - np.array(g.at[j, column])
                rows.append(dict(dataset=dataset, size=size, replicate=replicate, eval_size=m, pair=f"{i}|{j}",
                                 full_difference=abs(full), agreement=float((np.sign(draws) == np.sign(full)).mean()),
                                 single_draw_sd_i=float(np.std(g.at[i, column]))))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "eval_size"]).agreement.mean().reset_index()
    frame["difference_bin"] = pd.cut(frame.full_difference, [0, .02, .05, .1, 1], include_lowest=True).astype(str)
    by_bin = frame.groupby(["eval_size", "difference_bin"]).agreement.mean().unstack().reset_index()
    return frame, summary, by_bin


def analysis_7(d, raw100):
    d = d.merge(raw100, on="id", how="left")
    rows = []
    for (dataset, size, replicate), g in d.groupby(["dataset", "size", "replicate"]):
        g = g.set_index("variant")
        common = {name: g.at[v, "raw_c2st_100"] for name, v in COMMON.items() if v in g.index}
        corrected = {name: g.at[v, "c2st"] for name, v in CORRECTED.items() if v in g.index}
        memorizing = {name: max(g.at[v, "membership_nn"], g.at[v, "membership_density"]) > MEMORIZATION
                      for name, v in CORRECTED.items() if v in g.index}
        private = {k: (v if not memorizing[k] else 1.0 + v) for k, v in corrected.items()}  # memorizing ranked last
        rule_common = {name: g.at[v, "violation"] for name, v in COMMON.items() if v in g.index}
        rule_corrected = {name: g.at[v, "violation_gap"] for name, v in CORRECTED.items() if v in g.index}
        for label, a, b in (("Fidelity: common vs corrected practice", common, corrected),
                            ("Fidelity, privacy-aware ranking", common, private),
                            ("Rules: violation count vs gap to real", rule_common, rule_corrected)):
            rows.append(dict(dataset=dataset, size=size, replicate=replicate, comparison=label,
                             memorizing=",".join(k for k, v in memorizing.items() if v), **pairwise_reversal(a, b)))
    frame = pd.DataFrame(rows)
    summary = []
    for (comparison, dataset, size), g in frame.groupby(["comparison", "dataset", "size"]):
        mean, lo, hi = bootstrap_mean(g.reversal)
        tau, tlo, thi = bootstrap_mean(g.tau)
        summary.append(dict(comparison=comparison, dataset=dataset, size=size, reversal=mean, reversal_low=lo, reversal_high=hi,
                            tau=tau, tau_low=tlo, tau_high=thi, top1_agreement=float((g.top_a == g.top_b).mean()),
                            top_common=g.top_a.mode().iat[0], top_corrected=g.top_b.mode().iat[0]))
    return frame, pd.DataFrame(summary)


INK, INK2, GRID, BLUE, ORANGE, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#8a8984"
LABELS = {"mhd_corrected": "MedHyperDiff, corrected", "mhd_earlystop": "MedHyperDiff, early stopping",
          "mhd_affine": "MedHyperDiff, affine coding", "tabddpm_upstream": "TabDDPM, upstream recipe",
          "tabddpm_adapter": "TabDDPM, adapter recipe", "tabdiff": "TabDiff", "ctgan": "CTGAN", "tvae": "TVAE",
          "copula": "Gaussian copula", "copula_repair": "Copula + re-derivation", "independent": "Independent marginals",
          "bootstrap": "Bootstrap copy control"}
PANEL = lambda dataset, size: f"{'NHANES' if dataset == 'nhanes' else 'Thyroid'}, n = {size}"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(grid == "y")
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def figure_selection(d, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    groups = [(ds, n) for ds in it.DATASETS for n in it.SIZES]
    for ax, (variant, cap) in zip(axes, (("mhd_earlystop", 100), ("tabddpm_adapter", 300))):
        for k, (ds, n) in enumerate(groups):
            x = d[(d.variant == variant) & (d.dataset == ds) & (d["size"] == n)].selected_epoch.astype(float) / cap
            ax.scatter(x, np.full(len(x), len(groups) - k) + np.linspace(-.15, .15, len(x)), s=36, color=BLUE,
                       edgecolor="white", linewidth=1, zorder=3)
        ax.axvline(.2, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlim(-.03, 1.03)
        ax.set_yticks(range(len(groups), 0, -1), [PANEL(ds, n) for ds, n in groups], color=INK)
        ax.set_title(f"{LABELS[variant]}: selected epoch / cap ({cap})", fontsize=9, color=INK, loc="left")
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def figure_precision(summary, path):
    plt = _plt()
    groups = [(ds, n) for ds in it.DATASETS for n in it.SIZES]
    order = [v for v in CONTINUOUS]
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 3.6), sharey=True)
    for ax, (ds, n) in zip(axes, groups):
        g = summary[(summary.dataset == ds) & (summary["size"] == n)].set_index("variant").reindex(order)
        y = np.arange(len(order))[::-1]
        for yi, (_, row) in zip(y, g.iterrows()):
            ax.plot([row.c2st_raw, row.c2st_banded], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.scatter(g.c2st_raw, y, s=36, color=MUTED, edgecolor="white", linewidth=1, zorder=3, label="raw output")
        ax.scatter(g.c2st_banded, y, s=36, color=BLUE, edgecolor="white", linewidth=1, zorder=3, label="decoded to recorded resolution")
        ax.axvline(.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_yticks(y, [LABELS[v] for v in order], color=INK)
        ax.set_title(PANEL(ds, n), fontsize=9, color=INK, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.suptitle("C2ST AUROC against the reference population (mean of five training samples)", fontsize=10, color=INK, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def figure_memorization(d, path):
    plt = _plt()
    groups = [(ds, n) for ds in it.DATASETS for n in it.SIZES]
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 3.8), sharey=True)
    means = d.assign(membership=d[["membership_nn", "membership_density"]].max(axis=1)).groupby(
        ["dataset", "size", "variant"])[["c2st", "membership"]].mean().reset_index()
    for ax, (ds, n) in zip(axes, groups):
        g = means[(means.dataset == ds) & (means["size"] == n)]
        colors = [BLUE if v == "mhd_corrected" else MUTED for v in g.variant]
        ax.scatter(g.c2st, g.membership, s=40, color=colors, edgecolor="white", linewidth=1, zorder=3)
        offsets = {"mhd_corrected": (-18, -22, "right"), "tabddpm_upstream": (10, 8, "left"), "tabdiff": (-10, 18, "right"),
                   "bootstrap": (8, 4, "left"), "copula": (14, -26, "left"), "ctgan": (-8, 22, "right")}
        for _, row in g.iterrows():
            if row.variant in offsets:
                dx, dy, ha = offsets[row.variant]
                ax.annotate(LABELS[row.variant].split(",")[0], (row.c2st, row.membership), textcoords="offset points",
                            xytext=(dx, dy), fontsize=7, color=INK2, ha=ha, va="center",
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=.6, shrinkA=1, shrinkB=3))
        ax.axhline(MEMORIZATION, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_ylim(.36, 1.04)
        ax.set_xlabel("C2ST AUROC (lower = more realistic)", fontsize=8, color=INK2)
        ax.set_title(PANEL(ds, n), fontsize=9, color=INK, loc="left")
        _style(ax, grid="y")
    axes[0].set_ylabel("Membership AUROC (max of two attacks)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def figure_eval_size(summary, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for ds, color in zip(it.DATASETS, (BLUE, ORANGE)):
        g = summary[summary.dataset == ds]
        if g.empty:
            continue
        ax.plot(g.eval_size, g.agreement, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor="white")
        ax.annotate("NHANES" if ds == "nhanes" else "Thyroid", (g.eval_size.iloc[-1], g.agreement.iloc[-1]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK2)
    ax.axhline(.9, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xticks(it.CURVE_SIZES, [str(v) for v in it.CURVE_SIZES])
    ax.set_xlabel("Evaluation records per source (single draw)", fontsize=8, color=INK2)
    ax.set_ylabel("Pair order agrees with full reference", fontsize=8, color=INK2)
    _style(ax, grid="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def figure_pipeline(summary, path):
    plt = _plt()
    comparisons = list(dict.fromkeys(summary.comparison))
    groups = [(ds, n) for ds in it.DATASETS for n in it.SIZES]
    fig, axes = plt.subplots(1, len(comparisons), figsize=(13, 3.0), sharey=True)
    for ax, comparison in zip(axes, comparisons):
        g = summary[summary.comparison == comparison]
        y = np.arange(len(groups))[::-1]
        for yi, (ds, n) in zip(y, groups):
            row = g[(g.dataset == ds) & (g["size"] == n)]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(row.reversal, yi, xerr=[[row.reversal - row.reversal_low], [row.reversal_high - row.reversal]], fmt="o",
                        color=BLUE, elinewidth=1.4, markersize=6, markeredgecolor="white")
        ax.set_xlim(0, max(.6, float(summary.reversal_high.max()) + .05))
        ax.set_yticks(y, [PANEL(ds, n) for ds, n in groups], color=INK)
        ax.set_title(comparison, fontsize=9, color=INK, loc="left")
        _style(ax)
    fig.suptitle("Fraction of generator pairs whose order reverses (mean and 95% bootstrap interval over five training samples)",
                 fontsize=10, color=INK, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    d = load_records()
    d.to_csv(REPORT / "all_jobs.csv", index=False)
    complete = d[d.status == "complete"].copy()
    print(f"{len(complete)} complete, {int((d.status != 'complete').sum())} not complete")
    analysis_1(complete).to_csv(REPORT / "a1_checkpoint_selection.csv", index=False)
    analysis_2(complete).to_csv(REPORT / "a2_representation.csv", index=False)
    s3, r3 = analysis_3(complete)
    s3.to_csv(REPORT / "a3_precision_c2st.csv", index=False)
    r3.to_csv(REPORT / "a3_precision_reversals.csv", index=False)
    r4, m4 = analysis_4(complete)
    r4.to_csv(REPORT / "a4_rule_reversals.csv", index=False)
    m4.to_csv(REPORT / "a4_rule_means.csv", index=False)
    b5, c5, s5 = analysis_5(complete)
    b5.to_csv(REPORT / "a5_best_c2st_memorization.csv", index=False)
    c5.to_csv(REPORT / "a5_correlation.csv", index=False)
    s5.to_csv(REPORT / "a5_iteration11_search.csv", index=False)
    f6, s6, bin6 = analysis_6(complete)
    f6.to_csv(REPORT / "a6_pair_agreement.csv", index=False)
    s6.to_csv(REPORT / "a6_agreement_by_size.csv", index=False)
    bin6.to_csv(REPORT / "a6_agreement_by_difference.csv", index=False)
    raw100 = raw_c2st_at(complete)
    f7, s7 = analysis_7(complete, raw100)
    f7.to_csv(REPORT / "a7_pipeline_rankings.csv", index=False)
    s7.to_csv(REPORT / "a7_pipeline_summary.csv", index=False)
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    figure_selection(complete, figures / "p1_selected_epochs.png")
    figure_precision(s3, figures / "p3_precision_c2st.png")
    figure_memorization(complete, figures / "p6_memorization.png")
    figure_eval_size(s6, figures / "p7_evaluation_size.png")
    figure_pipeline(s7, figures / "pipeline_reversals.png")
    summary = complete.groupby(["dataset", "size", "variant"])[["c2st", "numeric_w", "association", "rf", "logistic", "violation",
        "violation_gap", "relational_w1", "membership_nn", "membership_density", "exact_copy", "fit_seconds"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(REPORT / "variant_summary.csv", index=False)


if __name__ == "__main__":
    main()
