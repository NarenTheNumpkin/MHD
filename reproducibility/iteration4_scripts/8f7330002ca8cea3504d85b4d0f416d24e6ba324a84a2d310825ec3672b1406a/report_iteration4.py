"""Report all frozen sample-size comparisons without selecting favorable seeds."""
import json

import numpy as np
import pandas as pd

from iteration4 import ROOT, RUNS, REPORT, PLAN, dump


def markdown(frame, decimals=4):
    def cell(v):
        return f"{v:.{decimals}f}" if isinstance(v, (float, np.floating)) else str(v)
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"]*len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def main():
    rows, features, rules = [], [], []
    registry = json.loads((REPORT / "registry.json").read_text())
    for path in registry:
        out = ROOT / path
        m = json.loads((out / "manifest.json").read_text())
        metric = json.loads((out / "metrics.json").read_text())
        identity = {k: m[k] for k in ("arm", "inner_seed", "training_seed", "sampling_seed", "multiplier")}
        f, u, c = metric["fidelity"], metric["utility"], metric["constraints"]["synthetic"]
        bounds = metric["rule_missingness_bounds"]["synthetic"]
        row = {**identity, "path": path, "rows": m["rows"],
            "rf": u["forest"]["TSTR"]["auroc"], "logistic": u["logistic"]["TSTR"]["auroc"],
            "rf_augment": u["forest"][f"AUGMENT_{m['multiplier']}x"]["auroc"],
            "logistic_augment": u["logistic"][f"AUGMENT_{m['multiplier']}x"]["auroc"],
            "rf_real": u["forest"]["TRTR"]["auroc"], "logistic_real": u["logistic"]["TRTR"]["auroc"],
            "numeric_w": f["mean_numeric_wasserstein_train_sd"], "association_rmse": f["association_rmse"],
            "signed_spearman_rmse": f["signed_numeric_spearman_rmse"],
            "violation": c["any_rule_violation_rate"],
            "violation_lower": bounds["any_rule"]["violation_lower"],
            "violation_upper": bounds["any_rule"]["violation_upper"],
            "protein_coverage": bounds["rules"]["protein_identity"]["coverage"],
            "protein_violation_lower": bounds["rules"]["protein_identity"]["violation_lower"],
            "protein_violation_upper": bounds["rules"]["protein_identity"]["violation_upper"],
            "max_missing_error_fit": max(v["error_vs_fit"] for v in metric["feature_missingness"].values()),
            "max_missing_error_validation": max(v["error_vs_validation"] for v in metric["feature_missingness"].values()),
            "c2st_auc": metric["c2st"]["auc"],
            "exact_train_copy": metric["privacy"]["exact_predictor_match_train"],
            "nearest_membership_auc": metric["privacy"]["nearest_synthetic_membership_auc"],
            "evaluation_seconds": m["evaluation_seconds"]}
        rows.append(row)
        for feature, values in metric["feature_missingness"].items():
            features.append({**identity, "feature": feature, **values, **f["per_feature"][feature]})
        for population, rb in metric["rule_missingness_bounds"].items():
            for rule, values in rb["rules"].items():
                rules.append({**identity, "population": population, "rule": rule, **values})
    df, ff, rr = pd.DataFrame(rows), pd.DataFrame(features), pd.DataFrame(rules)
    assert len(df) == 216
    for filename, table in (("all_evaluations", df), ("all_feature_results", ff), ("all_rule_bounds", rr)):
        table.to_csv(REPORT / f"{filename}.csv", index=False)
    group = ["arm", "inner_seed", "multiplier"]
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in identity]
    # Explicit hierarchy: sampling-seed means within each fitted parent, then training-seed means.
    checkpoint_means = df.groupby(group+["training_seed"])[numeric].mean().reset_index()
    checkpoint_means.to_csv(REPORT / "checkpoint_sampling_means.csv", index=False)
    means = checkpoint_means.groupby(group)[numeric].mean().reset_index()
    means.to_csv(REPORT / "partition_means.csv", index=False)
    checkpoint_means.groupby(group)[numeric].std().reset_index().to_csv(REPORT / "training_seed_sd.csv", index=False)
    sample_sd = df.groupby(group+["training_seed"])[numeric].std().reset_index()
    sample_sd.to_csv(REPORT / "sampling_seed_sd_by_checkpoint.csv", index=False)
    paired = df[df.multiplier == 5].merge(df[df.multiplier == 1],
        on=["arm", "inner_seed", "training_seed", "sampling_seed"], suffixes=("_5x", "_1x"))
    delta_cols = ["rf", "logistic", "rf_augment", "logistic_augment", "numeric_w", "association_rmse", "violation"]
    for col in delta_cols:
        paired[f"delta_{col}"] = paired[f"{col}_5x"] - paired[f"{col}_1x"]
    paired.to_csv(REPORT / "paired_size_deltas.csv", index=False)
    comparison = means[means.multiplier == 5].merge(means[means.multiplier == 1], on=["arm", "inner_seed"], suffixes=("_5x", "_1x"))
    fmeans = ff.groupby(group+["feature"])[["error_vs_fit", "error_vs_validation"]].mean().reset_index()
    rmeans = rr[rr.population == "synthetic"].groupby(group+["rule"])[["coverage"]].mean().reset_index()
    gates = []
    for row in comparison.to_dict("records"):
        arm, inner = row["arm"], row["inner_seed"]
        fm = fmeans[(fmeans.arm == arm) & (fmeans.inner_seed == inner)]
        fd = fm[fm.multiplier == 5].set_index("feature")[["error_vs_fit", "error_vs_validation"]] - fm[fm.multiplier == 1].set_index("feature")[["error_vs_fit", "error_vs_validation"]]
        rm = rmeans[(rmeans.arm == arm) & (rmeans.inner_seed == inner)]
        rd = rm[rm.multiplier == 5].set_index("rule").coverage - rm[rm.multiplier == 1].set_index("rule").coverage
        conditions = dict(rf_non_degradation=row["rf_5x"] >= row["rf_1x"]-.02,
            logistic_non_degradation=row["logistic_5x"] >= row["logistic_1x"]-.02,
            some_utility_increase=(row["rf_5x"] > row["rf_1x"] or row["logistic_5x"] > row["logistic_1x"]),
            numeric_fidelity=row["numeric_w_5x"] <= 1.1*row["numeric_w_1x"],
            association_fidelity=row["association_rmse_5x"] <= 1.1*row["association_rmse_1x"],
            observed_violation=row["violation_5x"] <= row["violation_1x"]+.01,
            missingness_fit=bool((fd.error_vs_fit <= .01).all()),
            missingness_validation=bool((fd.error_vs_validation <= .01).all()),
            rule_coverage=bool((rd >= -.01).all()))
        subgroup = df[(df.arm == arm) & (df.inner_seed == inner) & (df.multiplier == 5)]
        gates.append(dict(arm=arm, inner_seed=inner, **conditions, partition_pass=all(conditions.values()),
            flagged_5x_tables=int((subgroup.max_missing_error_fit > .02).sum()),
            delta_rf=row["rf_5x"]-row["rf_1x"], delta_logistic=row["logistic_5x"]-row["logistic_1x"],
            max_feature_error_increase_fit=float(fd.error_vs_fit.max()),
            min_rule_coverage_delta=float(rd.min())))
    gf = pd.DataFrame(gates)
    gf.to_csv(REPORT / "size_gates.csv", index=False)
    decisions = {}
    for arm, g in gf.groupby("arm"):
        arm_tables = df[df.arm == arm]
        flagged_1x = int(((arm_tables.multiplier == 1) & (arm_tables.max_missing_error_fit > .02)).sum())
        flagged_5x = int(g.flagged_5x_tables.sum())
        decisions[arm] = dict(partitions_passed=int(g.partition_pass.sum()), flagged_1x_tables=flagged_1x,
            flagged_5x_tables=flagged_5x, adopt_5x_default=bool(g.partition_pass.all() and flagged_1x+flagged_5x == 0))
    dump(REPORT / "size_decision.json", decisions)
    pools = [json.loads(p.read_text()) for p in (RUNS / "pools").glob("*/manifest.json")]
    timing = dict(new_training_runs=0, checkpoints=36, generated_pools=len(pools), evaluations=len(df),
        pool_sample_seconds_sum=sum(p["sample_seconds"] for p in pools),
        pool_sample_seconds_min=min(p["sample_seconds"] for p in pools),
        pool_sample_seconds_max=max(p["sample_seconds"] for p in pools),
        evaluation_seconds_sum=float(df.evaluation_seconds.sum()),
        evaluation_seconds_min=float(df.evaluation_seconds.min()),
        evaluation_seconds_max=float(df.evaluation_seconds.max()), device="CPU", gpu_vram_gib=0,
        peak_system_ram="not measured", timing_note="Component times; not total elapsed study wall time.")
    dump(REPORT / "timing.json", timing)
    columns = ["arm", "inner_seed", "multiplier", "rf", "logistic", "numeric_w", "violation", "violation_upper", "protein_coverage"]
    report = "# Iteration 4: all sample-size and independent-sampling results\n\n"
    report += "Completed 216 evaluations of 108 nested synthetic pools from 36 existing fitted checkpoints; **zero new generator training runs**. All outcomes use inner validation. Means average three sampling seeds within each training seed, then three training seeds, separately by partition. Overlapping partitions are not independent cohorts.\n\n"
    report += markdown(means[columns]) + "\n\n## Predeclared 5x versus 1x engineering gates\n\n"
    report += markdown(gf[["arm", "inner_seed", "delta_rf", "delta_logistic", "partition_pass", "flagged_5x_tables"]])
    report += "\n\nThese gates are descriptive development tolerances, not clinical noninferiority tests. Missingness flags mean at least one feature differs from its fit-table missing rate by more than 0.02. Violation upper bounds treat unevaluable rules as possibly violated; they are logical bounds, not confidence intervals.\n\n"
    report += "## Reproducibility\n\nFull records: `all_evaluations.csv`, `all_feature_results.csv`, `all_rule_bounds.csv`, `paired_size_deltas.csv`, `checkpoint_sampling_means.csv`, `sampling_seed_sd_by_checkpoint.csv`, `training_seed_sd.csv`, `size_gates.csv`, `size_decision.json`, `timing.json`. Each evaluation retains its synthetic table and downstream scores. The 5x augmentation outputs are explicitly labelled AUGMENT_5x. No rows were repaired or imputed in released synthetic tables.\n"
    (REPORT / "RESULTS.md").write_text(report)
    print(json.dumps({"decisions": decisions, "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
