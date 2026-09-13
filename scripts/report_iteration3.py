"""Transparent screening selection and descriptive strength-sweep reporting."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/medhyperdiff-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/medhyperdiff-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iteration3 import ROOT, RUNS, REPORT, dump, name, spec


def collect():
    rows, rules, features = [], [], []
    for p in sorted((RUNS / "evaluations").glob("*/manifest.json")):
        m = json.loads(p.read_text())
        if m["status"] != "complete":
            raise ValueError(f"Incomplete experiment: {p}")
        metrics = json.loads((p.parent / "metrics.json").read_text())
        row = dict(run=str(p.parent.relative_to(ROOT)), dataset=m["dataset"], inner_seed=m["inner_seed"],
            seed=m["seed"], setting=name(m["setting"]), **m["setting"],
            parent=m["parent_checkpoint"], parameters=m["parameters"],
            rf=metrics["utility"]["forest"]["TSTR"]["auroc"],
            logistic=metrics["utility"]["logistic"]["TSTR"]["auroc"],
            violation=metrics["constraints"]["synthetic"]["any_rule_violation_rate"],
            w=metrics["fidelity"]["mean_numeric_wasserstein_train_sd"],
            assoc=metrics["fidelity"]["association_rmse"],
            missing=metrics["fidelity"]["mean_missing_rate_error"],
            cat_tv=metrics["fidelity"]["mean_categorical_tv"],
            copy=metrics["privacy"]["exact_predictor_match_train"],
            mia=metrics["privacy"]["nearest_synthetic_membership_auc"],
            sample_seconds=m["sample_seconds"], evaluation_seconds=m["evaluation_seconds"])
        for rule, v in metrics["constraints"]["synthetic"]["rules"].items():
            rules.append(dict(run=row["run"], rule=rule, **v))
            if rule in ("protein_identity", "parity"):
                row[rule] = v["violation_rate"]
        for feature, v in metrics["fidelity"]["per_feature"].items():
            features.append(dict(run=row["run"], feature=feature, **v))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(rules), pd.DataFrame(features)


def select(data):
    registry = json.loads((REPORT / "screen_registry.json").read_text())
    screen = data[data.run.isin(registry)].copy()
    assert len(screen) == len(registry) == 84
    rows, selected = [], {}
    for dataset in ("ilpd", "toy"):
        d = screen[screen.dataset == dataset]
        for _, row in d.iterrows():
            base = d[d.setting == name(spec("base", row.prediction, 0, 0))].iloc[0]
            penalties = [max(0., base.rf - row.rf) / .02,
                max(0., base.logistic - row.logistic) / .02,
                max(0., row.w / base.w - 1) / .1,
                max(0., row.assoc / base.assoc - 1) / .1,
                max(0., row.missing - base.missing) / .01]
            rows.append({**row.to_dict(), "max_penalty": max(penalties),
                "qualifies": max(penalties) <= 1 + 1e-10 and row.violation < base.violation,
                "rf_delta": row.rf - base.rf, "logistic_delta": row.logistic - base.logistic,
                "w_ratio": row.w / base.w, "assoc_ratio": row.assoc / base.assoc,
                "rule_improvement": base.violation - row.violation,
                "mean_auc": (row.rf + row.logistic) / 2})
        ranked = pd.DataFrame([r for r in rows if r["dataset"] == dataset and not r["repair"]])
        qualified = ranked[ranked.qualifies]
        order = ["violation", "w", "mean_auc", "setting"]
        if len(qualified):
            chosen = qualified.sort_values(order, ascending=[True, True, False, True]).head(2)
        else:
            chosen = ranked.sort_values(["max_penalty", *order], ascending=[True, True, True, False, True]).head(2)
        selected[dataset] = [spec(r["mode"], r["prediction"], r["clinical_weight"], r["guidance"]) for _, r in chosen.iterrows()]
    candidates = pd.DataFrame(rows)
    candidates.to_csv(REPORT / "screen_candidates.csv", index=False)
    result = {"selected": selected, "selection_type": "adaptive development screen, not confirmation",
        "protocol": "docs/ITERATION_3_PROTOCOL.md", "screen_evaluations": len(screen),
        "qualification": {dataset: [bool(candidates[(candidates.dataset == dataset) &
            (candidates.setting == name(s))].qualifies.iloc[0]) for s in settings] for dataset, settings in selected.items()}}
    if (REPORT / "selection.json").exists():
        assert json.loads((REPORT / "selection.json").read_text()) == result, "Refusing to change frozen selection"
    else:
        dump(REPORT / "selection.json", result)
    print(json.dumps(result, indent=2))


def table(data, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, r in data.iterrows():
        cells = ["—" if pd.isna(r[c]) else f"{r[c]:.3f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(data, rules, features):
    REPORT.mkdir(parents=True, exist_ok=True)
    data.to_csv(REPORT / "all_evaluations.csv", index=False)
    rules.to_csv(REPORT / "all_rule_results.csv", index=False)
    features.to_csv(REPORT / "all_feature_results.csv", index=False)
    fields = ["rf", "logistic", "violation", "protein_identity", "parity", "w", "assoc", "missing", "cat_tv", "copy", "mia"]
    train = []
    for p in sorted((RUNS / "training").glob("*/manifest.json")):
        m = json.loads(p.read_text())
        train.append(dict(run=str(p.parent.relative_to(ROOT)), status=m["status"], dataset=m["dataset"],
            fit_seconds=m.get("fit_seconds_including_structure"), selected_epoch=m.get("selected_epoch")))
    pd.DataFrame(train).to_csv(REPORT / "new_training_runs.csv", index=False)
    lines = ["# Iteration 3: measured constraint-strength experiments", "",
        f"**{len(train)} new model training runs; {len(data)} distinct sampling/evaluation configurations; {data.parent.nunique()} fitted checkpoints used.** Some checkpoints were reused from iteration 2. Resampling is not new model training.", "",
        "All outcomes are development validation results. Outer fold 0 remains reserved. Seeds and overlapping inner partitions are not independent patient cohorts, and the utility tolerances are engineering screens rather than clinical noninferiority margins.", "",
        "`rf` and `logistic` are synthetic-trained classifier AUROCs (higher better); `violation` is the fraction violating a supplied rule, `w` normalized numeric Wasserstein and `assoc` association RMSE (lower better). Read per-rule coverage and per-feature variance before interpreting compliance.", ""]
    registry_path = REPORT / "replicate_registry.json"
    if registry_path.exists():
        registered = json.loads(registry_path.read_text())
        if (REPORT / "controls_registry.json").exists():
            registered += json.loads((REPORT / "controls_registry.json").read_text())
        replication = data[data.run.isin(registered)]
        grouped = replication.groupby(["dataset", "inner_seed", "setting"])[fields].mean().reset_index()
        grouped.to_csv(REPORT / "partition_means.csv", index=False)
        replication.groupby(["dataset", "inner_seed", "setting"])[fields].agg(["mean", "std", "count"]).to_csv(REPORT / "seed_variation.csv")
        lines += ["## Frozen-settings replication", "",
            "Each row averages three generator seeds on one development partition. Every selected setting is retained, including failures. No setting was retuned after replication.", "",
            table(grouped, ["dataset", "inner_seed", "setting", "rf", "logistic", "violation", "w", "assoc", "parity"]), ""]
        selected = json.loads((REPORT / "selection.json").read_text())["selected"]
        rows = []
        for dataset, settings in selected.items():
            for setting in settings:
                for inner in (2027, 3027, 4027):
                    d = grouped[(grouped.dataset == dataset) & (grouped.inner_seed == inner)]
                    a = d[d.setting == name(setting)].iloc[0]
                    b = d[d.setting == name(spec("base", setting["prediction"], 0, 0))].iloc[0]
                    rows.append(dict(dataset=dataset, inner_seed=inner, setting=name(setting),
                        rf_delta=a.rf-b.rf, logistic_delta=a.logistic-b.logistic,
                        rule_reduction=b.violation-a.violation, w_ratio=a.w/b.w, assoc_ratio=a.assoc/b.assoc,
                        passes_screen_on_means=bool(a.rf >= b.rf-.02 and a.logistic >= b.logistic-.02 and
                            a.w <= 1.1*b.w and a.assoc <= 1.1*b.assoc and a.missing <= b.missing+.01 and a.violation < b.violation)))
        pd.DataFrame(rows).to_csv(REPORT / "replication_deltas.csv", index=False)
        lines += ["## Selected settings against matching backbones", "",
            "Passing the engineering conditions on seed means is descriptive; it is not a confidence interval, statistical test or independent validation.", "",
            table(pd.DataFrame(rows), ["dataset", "inner_seed", "setting", "rf_delta", "logistic_delta", "rule_reduction", "w_ratio", "assoc_ratio", "passes_screen_on_means"]), ""]
    screen_registry = REPORT / "screen_registry.json"
    if screen_registry.exists():
        screen = data[data.run.isin(json.loads(screen_registry.read_text()))]
        lines += ["## Complete seed-0 screen", "", table(screen, ["dataset", "setting", "rf", "logistic", "violation", "w", "assoc"]), ""]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ax, dataset in zip(axes, ("ilpd", "toy")):
            for marker, p in zip(("o", "s"), ("x0", "v")):
                d = screen[(screen.dataset == dataset) & (screen.prediction == p) & ~screen.repair]
                points = ax.scatter(d.violation, d.rf, c=d.w, cmap="viridis", marker=marker, s=45,
                    vmin=0, vmax=max(.5, screen.w.max()), label=p, edgecolors="black", linewidths=.3)
            ax.set(title=dataset, xlabel="Supplied-rule violation fraction ↓", ylabel="RF AUROC ↑")
            ax.legend(title="Numeric prediction")
            ax.grid(alpha=.15)
        fig.colorbar(points, ax=axes, label="Numeric Wasserstein ↓", shrink=.8)
        fig.suptitle("Development strength screen: one generator seed; not a final ranking")
        for ext in ("png", "svg", "pdf"):
            fig.savefig(REPORT / f"strength_tradeoff.{ext}", dpi=170)
        plt.close(fig)
    (REPORT / "RESULTS.md").write_text("\n".join(lines))
    print(REPORT / "RESULTS.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    data, rules, features = collect()
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.select:
        select(data)
    report(data, rules, features)
