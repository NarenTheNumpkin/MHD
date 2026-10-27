"""Aggregate iteration-11 records into reports/iteration11 (tables, figures, bootstrap intervals).

Development (ablation/search/guidance) and confirmation (outer test) results are kept in
separate files. Iteration-5/7 reference tables are re-scored with the current evaluator
(evaluation only, nothing is refit) so that rule-realism metrics exist for them too.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import iteration11 as it

REPORT = Path(os.environ.get("MHD_REPORT", it.ROOT / "reports" / "iteration11"))
LEGACY_RUNS = Path(os.environ.get("MHD_LEGACY_RUNS", it.ROOT / "runs"))
LOWER_BETTER = ("c2st_distance", "numeric_w", "association", "signed_spearman", "categorical_tv", "missing_error",
                "violation_gap", "hard_violation", "relational_w1", "membership")


def endpoints(metrics):
    fidelity, utility, privacy = metrics["fidelity"], metrics["utility"], metrics["privacy"]
    realism = metrics["constraints"].get("realism") or {}
    augment = next((v for k, v in utility["forest"].items() if k.startswith("AUGMENT")), {})
    protein = (realism.get("rules") or {}).get("protein_identity") or {}
    rules = metrics["constraints"]["synthetic"]["rules"]
    c2st = metrics["c2st"]["auc"]
    return dict(c2st=c2st, c2st_distance=None if c2st is None else abs(c2st - .5),
                numeric_w=fidelity["mean_numeric_wasserstein_train_sd"], association=fidelity["association_rmse"],
                signed_spearman=fidelity["signed_numeric_spearman_rmse"], categorical_tv=fidelity["mean_categorical_tv"],
                missing_error=fidelity["mean_missing_rate_error"], rf=utility["forest"]["TSTR"]["auroc"],
                logistic=utility["logistic"]["TSTR"]["auroc"], rf_real=utility["forest"]["TRTR"]["auroc"],
                rf_augment=augment.get("auroc"), violation=metrics["constraints"]["synthetic"]["any_rule_violation_rate"],
                real_violation=metrics["constraints"]["holdout"]["any_rule_violation_rate"],
                violation_gap=realism.get("mean_violation_gap"),
                hard_violation=(realism.get("hard_rule_violation_rate") or {}).get("synthetic"),
                relational_w1=realism.get("mean_relational_residual_w1_fit_sd"), protein_w1=protein.get("residual_w1"),
                parity_violation=(rules.get("parity") or {}).get("violation_rate"),
                protein_violation=(rules.get("protein_identity") or {}).get("violation_rate"),
                membership=privacy["nearest_synthetic_membership_auc"], dcr_member=privacy["dcr_member_median"],
                dcr_holdout=privacy["dcr_holdout_median"], exact_copy=privacy["exact_predictor_match_train"])


def records(stage):
    rows = []
    for manifest_path in sorted((it.OUT / stage).rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            rows.append(dict(id=manifest.get("id"), status=manifest.get("status"), error=manifest.get("error")))
            continue
        metrics = json.loads((manifest_path.parent / "metrics.json").read_text())
        rows.append(dict(id=manifest["id"], status="complete", dataset=manifest["dataset"],
                         arm=manifest.get("arm") or manifest.get("variant") or manifest["method"],
                         method=manifest["method"], config_index=manifest.get("config_index"),
                         inner_seed=manifest["inner_seed"], training_seed=manifest["training_seed"],
                         fit_seconds=manifest["fit_seconds"], sample_seconds=manifest["sample_seconds"],
                         parameters=manifest.get("parameters"), **endpoints(metrics)))
    return pd.DataFrame(rows)


def rescore_references():
    """Re-evaluate saved iteration-5/7 ILPD tables (no refitting) with the current evaluator."""
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.evaluation import evaluate
    cache = REPORT / "reference_rescored.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    sources = [("iteration7", p) for p in sorted((LEGACY_RUNS / "iteration7" / "evaluations").glob("*_n5"))]
    sources += [("iteration5", p) for p in sorted((LEGACY_RUNS / "iteration5" / "evaluations").glob("*_n5"))
                if p.name.startswith(("full_prior_i", "base_prior_i", "dense_prior_i"))]
    splits = {}
    for origin, path in sources:
        name = path.name
        arm, rest = name.split("_i", 1)
        inner = int(rest.split("_")[0])
        if inner not in splits:
            splits[inner] = it.load("ilpd", inner)
        frame, split, schema = splits[inner]
        train, holdout = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        synthetic = pd.read_csv(path / "synthetic.csv", dtype={c: str for c in (*schema.categorical, schema.target)},
                                float_precision="round_trip")
        metrics = evaluate(train, holdout, synthetic, schema, build_constraints(schema, train), it.EVALUATOR_SEED, {})
        rows.append(dict(origin=origin, arm=arm, inner_seed=inner, run=name, **endpoints(metrics)))
    frame = pd.DataFrame(rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


KEY = ["c2st", "numeric_w", "association", "signed_spearman", "rf", "logistic", "violation", "violation_gap",
       "hard_violation", "protein_w1", "membership"]


def fmt_table(frame, columns, digits=3):
    head = "| " + " | ".join(columns) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(columns))) + "|\n"
    body = ""
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            cells.append(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def ablation_report():
    data = records("ablation")
    data.to_csv(REPORT / "ablation_all.csv", index=False)
    complete = data[data.status == "complete"]
    partition = complete.groupby(["arm", "inner_seed"])[KEY].mean().reset_index()
    partition.to_csv(REPORT / "ablation_partition_means.csv", index=False)
    overall = partition.groupby("arm")[KEY].agg(["mean"]).droplevel(1, axis=1).reset_index()
    order = list(it.ABLATION)
    overall["order"] = overall["arm"].map(order.index)
    overall = overall.sort_values("order").drop(columns="order")
    refs = rescore_references()
    ref_partition = refs.groupby(["arm", "inner_seed"])[KEY].mean().reset_index()
    ref_overall = ref_partition.groupby("arm")[KEY].mean().reset_index()
    ref_overall.to_csv(REPORT / "reference_rescored_means.csv", index=False)
    overall.to_csv(REPORT / "ablation_means.csv", index=False)
    return data, partition, overall, ref_partition, ref_overall


def search_report():
    data = records("search")
    data.to_csv(REPORT / "search_all.csv", index=False)
    return data


def paired_bootstrap(labels, a, b, draws=2000, seed=2026):
    rng = np.random.default_rng(seed)
    labels, a, b = map(np.asarray, (labels, a, b))
    deltas = []
    for _ in range(draws):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        deltas.append(roc_auc_score(labels[idx], a[idx]) - roc_auc_score(labels[idx], b[idx]))
    return float(roc_auc_score(labels, a) - roc_auc_score(labels, b)), np.quantile(deltas, [.025, .975]).tolist()


def confirm_report():
    data = records("confirm")
    data.to_csv(REPORT / "confirm_all.csv", index=False)
    complete = data[data.status == "complete"]
    summary = complete.groupby(["dataset", "arm"])[KEY + ["c2st_distance", "rf_augment", "rf_real", "real_violation", "categorical_tv",
                                                          "missing_error", "dcr_member", "dcr_holdout", "exact_copy",
                                                          "fit_seconds"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(REPORT / "confirm_summary.csv", index=False)
    # Seed-paired differences against the proposed model.
    paired = []
    for dataset in it.DATASETS:
        base = complete[(complete.dataset == dataset) & (complete.arm == "mhd")].set_index("training_seed")
        for arm in sorted(set(complete[complete.dataset == dataset].arm) - {"mhd"}):
            other = complete[(complete.dataset == dataset) & (complete.arm == arm)].set_index("training_seed")
            seeds = base.index.intersection(other.index)
            for metric in ("c2st", "c2st_distance", "numeric_w", "association", "rf", "logistic", "violation_gap", "membership"):
                diff = (base.loc[seeds, metric] - other.loc[seeds, metric]).astype(float)
                better = (diff < 0) if metric in LOWER_BETTER or metric == "c2st" else (diff > 0)
                paired.append(dict(dataset=dataset, comparator=arm, metric=metric, mean_difference=float(diff.mean()),
                                   sd_difference=float(diff.std()), seeds=len(seeds), seeds_favoring_mhd=int(better.sum())))
    pd.DataFrame(paired).to_csv(REPORT / "confirm_paired_differences.csv", index=False)
    # Patient-level bootstrap for forest TSTR AUROC, seed-averaged predictions.
    intervals = []
    for dataset in it.DATASETS:
        predictions = {}
        for arm in sorted(set(complete[complete.dataset == dataset].arm)):
            frames = [pd.read_csv(it.OUT / f"confirm/{dataset}/{arm}/t{s}/utility_predictions.csv", float_precision="round_trip")
                      for s in it.CONFIRM_SEEDS if (it.OUT / f"confirm/{dataset}/{arm}/t{s}/utility_predictions.csv").exists()]
            if frames:
                predictions[arm] = (frames[0]["label"].to_numpy(), np.mean([f["forest_TSTR"].to_numpy() for f in frames], 0))
        if "mhd" not in predictions:
            continue
        labels, mhd = predictions["mhd"]
        for arm, (other_labels, scores) in predictions.items():
            if arm == "mhd":
                continue
            assert (other_labels == labels).all()
            delta, ci = paired_bootstrap(labels, mhd, scores)
            intervals.append(dict(dataset=dataset, comparator=arm, auroc_mhd=float(roc_auc_score(labels, mhd)),
                                  auroc_comparator=float(roc_auc_score(labels, scores)), difference=delta,
                                  ci_low=ci[0], ci_high=ci[1]))
    pd.DataFrame(intervals).to_csv(REPORT / "confirm_utility_bootstrap.csv", index=False)
    # Descriptive ranks across datasets (non-memorizing arms only).
    ranks = []
    arms = [a for a in summary.arm.unique() if a != "bootstrap"]
    for metric in ("c2st_distance", "numeric_w", "association", "rf", "violation_gap", "membership"):
        for dataset in it.DATASETS:
            sub = summary[(summary.dataset == dataset) & summary.arm.isin(arms)].dropna(subset=[f"{metric}_mean"])
            if sub.empty:
                continue
            values = sub[f"{metric}_mean"].to_numpy()
            order = values if metric in LOWER_BETTER else -values
            from scipy.stats import rankdata
            for arm, rank in zip(sub.arm, rankdata(order)):
                ranks.append(dict(metric=metric, dataset=dataset, arm=arm, rank=float(rank)))
    pd.DataFrame(ranks).to_csv(REPORT / "confirm_ranks.csv", index=False)
    return data, summary


def mechanism_report():
    data = records("mechanism")
    data.to_csv(REPORT / "mechanism_all.csv", index=False)
    complete = data[data.status == "complete"]
    columns = ["parity_violation", "protein_violation", "c2st", "numeric_w", "association", "rf", "logistic", "membership"]
    partition = complete.groupby(["arm", "inner_seed"])[columns].mean().reset_index()
    partition.to_csv(REPORT / "mechanism_partition_means.csv", index=False)
    overall = partition.groupby("arm")[columns].agg(["mean", "std"])
    overall.columns = [f"{a}_{b}" for a, b in overall.columns]
    overall.reset_index().to_csv(REPORT / "mechanism_means.csv", index=False)
    return partition, overall


# Static paper figures: light surface, one highlight hue for the proposed model, gray comparators.
INK, INK2, GRID, HIGHLIGHT, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#8a8984"


def _style(ax):
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def dot_panel(ax, labels, means, errors, highlight, title, reference=None):
    y = np.arange(len(labels))[::-1]
    for yi, m, e, h in zip(y, means, errors, highlight):
        color = HIGHLIGHT if h else MUTED
        ax.errorbar(m, yi, xerr=e if np.isfinite(e) else None, fmt="o", color=color, ecolor=color,
                    elinewidth=1.4, capsize=0, markersize=6, markeredgecolor="white", markeredgewidth=1.2)
    if reference is not None:
        ax.axvline(reference, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(y, labels, color=INK)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)


ABLATION_LABELS = {"a0_legacy_full_prior": "Legacy full_prior", "a1_fixed_steps_ema": "+ fixed-step EMA training",
    "a2_plus_quantile": "+ quantile coding", "a3_plus_T100": "+ 100 diffusion steps",
    "a4_plus_precision": "+ recorded precision (corrected)", "a4_but_epoch_selection": "corrected, epoch selection",
    "a4_no_guidance": "corrected, no guidance", "a4_backbone": "corrected backbone only",
    "copula_repair": "Copula + repair (ref.)", "copula": "Copula (ref.)", "tvae_compact100": "TVAE compact (ref.)",
    "ctgan_default300": "CTGAN default (ref.)", "bootstrap": "Bootstrap copy control (ref.)"}


def plot_ablation(partition, ref_partition):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    refs = ["copula_repair", "copula", "tvae_compact100", "ctgan_default300", "bootstrap"]
    frame = pd.concat([partition, ref_partition[ref_partition.arm.isin(refs)]])
    order = [a for a in list(it.ABLATION) + refs if a in set(frame.arm)]
    metrics = [("c2st", "C2ST AUROC (0.5 = indistinguishable)", .5), ("numeric_w", "Numeric Wasserstein / fit SD", None),
               ("association", "Association RMSE", None), ("violation_gap", "Rule violation gap vs real", 0.)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    for ax, (metric, title, ref) in zip(axes, metrics):
        stats = frame.groupby("arm")[metric].agg(["mean", "std"]).reindex(order)
        dot_panel(ax, [ABLATION_LABELS.get(a, a) for a in order], stats["mean"].to_numpy(), stats["std"].to_numpy(),
                  [a == "a4_plus_precision" for a in order], title, ref)
    fig.suptitle("ILPD development partitions: one correction at a time (mean ± SD across three partitions)",
                 fontsize=10, color=INK, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(REPORT / "figures" / "ablation.png", dpi=200, facecolor="white")
    plt.close(fig)


ARM_LABELS = {"mhd": "MedHyperDiff (corrected)", "mhd_backbone": "MedHyperDiff backbone", "mhd_legacy": "MedHyperDiff legacy",
              "tabddpm": "TabDDPM", "tabdiff": "TabDiff", "ctgan": "CTGAN", "tvae": "TVAE", "copula": "Gaussian copula",
              "copula_repair": "Copula + repair", "independent": "Independent marginals", "bootstrap": "Bootstrap (copy control)"}
for _m, _name in (("mhd", "MedHyperDiff"), ("tabddpm", "TabDDPM"), ("tabdiff", "TabDiff"), ("ctgan", "CTGAN"), ("tvae", "TVAE")):
    ARM_LABELS[f"{_m}_private"] = f"{_name} (privacy-constrained)"


def plot_confirm(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [a for a in ARM_LABELS if a in set(summary.arm)]
    for metric, title, ref, name in (("c2st", "C2ST AUROC vs reserved test (0.5 = indistinguishable)", .5, "confirm_c2st.png"),
                                     ("rf", "Forest TSTR AUROC on reserved test", None, "confirm_utility.png"),
                                     ("association", "Association RMSE vs reserved test", None, "confirm_association.png")):
        fig, axes = plt.subplots(1, len(it.DATASETS), figsize=(14, 4.2), sharey=True)
        for ax, dataset in zip(axes, it.DATASETS):
            sub = summary[summary.dataset == dataset].set_index("arm").reindex(order)
            real = sub["rf_real_mean"].dropna().mean() if metric == "rf" else ref
            dot_panel(ax, [ARM_LABELS[a] for a in order], sub[f"{metric}_mean"].to_numpy(), sub[f"{metric}_std"].to_numpy(),
                      [a == "mhd" for a in order], dataset.replace("_", " ").upper() if dataset != "heart_failure" else "HEART FAILURE",
                      real)
        fig.suptitle(title + " (mean ± SD over five training seeds" + ("; dashed = real-data TRTR)" if metric == "rf" else ")"),
                     fontsize=10, color=INK, x=.01, ha="left")
        fig.tight_layout()
        fig.savefig(REPORT / "figures" / name, dpi=200, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", choices=("ablation", "search", "confirm", "mechanism"))
    args = parser.parse_args()
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    if "ablation" in args.parts:
        _, partition, overall, ref_partition, ref_overall = ablation_report()
        plot_ablation(partition, ref_partition)
        print(overall.round(4).to_string())
        print(ref_overall.round(4).to_string())
    if "mechanism" in args.parts:
        _, overall = mechanism_report()
        print(overall.round(4).to_string())
    if "search" in args.parts:
        search_report()
    if "confirm" in args.parts:
        _, summary = confirm_report()
        plot_confirm(summary)
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
