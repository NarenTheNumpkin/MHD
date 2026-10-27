"""Descriptive matched comparisons and frozen correction gates for iteration 5."""
import json

import numpy as np
import pandas as pd

from iteration5 import ROOT, RUNS, REPORT, PLAN, VARIANTS, dump
from report_iteration4 import markdown


def main():
    registry = json.loads((REPORT / "registry.json").read_text())
    rows, features, rules = [], [], []
    for path in registry["new"]+registry["references"]:
        out = ROOT/path
        m, metric = [json.loads((out/f).read_text()) for f in ("manifest.json", "metrics.json")]
        arm = m.get("variant", "reference_"+m.get("arm", ""))
        identity = dict(arm=arm, inner_seed=m["inner_seed"], training_seed=m["training_seed"], sampling_seed=m["sampling_seed"])
        f, u = metric["fidelity"], metric["utility"]
        b = metric["rule_missingness_bounds"]["synthetic"]
        rows.append({**identity, "path": path, "new_evaluation": path in registry["new"],
            "rf": u["forest"]["TSTR"]["auroc"], "logistic": u["logistic"]["TSTR"]["auroc"],
            "rf_augment": u["forest"]["AUGMENT_5x"]["auroc"], "logistic_augment": u["logistic"]["AUGMENT_5x"]["auroc"],
            "rf_real": u["forest"]["TRTR"]["auroc"], "logistic_real": u["logistic"]["TRTR"]["auroc"],
            "numeric_w": f["mean_numeric_wasserstein_train_sd"], "association_rmse": f["association_rmse"],
            "violation": metric["constraints"]["synthetic"]["any_rule_violation_rate"],
            "violation_upper": b["any_rule"]["violation_upper"],
            "protein_coverage": b["rules"]["protein_identity"]["coverage"],
            "ag_missing": metric["feature_missingness"]["ag_ratio"]["synthetic"],
            "max_missing_error_fit": max(v["error_vs_fit"] for v in metric["feature_missingness"].values()),
            "c2st_auc": metric["c2st"]["auc"], "exact_train_copy": metric["privacy"]["exact_predictor_match_train"],
            "nearest_membership_auc": metric["privacy"]["nearest_synthetic_membership_auc"]})
        for feature, values in metric["feature_missingness"].items():
            fm = f["per_feature"][feature]
            ratio = fm.get("std_ratio_synthetic_to_real")
            features.append({**identity, "feature": feature, **values, **fm,
                "std_ratio_error": abs(ratio-1) if ratio is not None and np.isfinite(ratio) else None})
        for population, bounds in metric["rule_missingness_bounds"].items():
            for rule, values in bounds["rules"].items():
                rules.append({**identity, "population": population, "rule": rule, **values})
    df, ff, rr = pd.DataFrame(rows), pd.DataFrame(features), pd.DataFrame(rules)
    assert len(df) == 189 and df.new_evaluation.sum() == 108
    for filename, table in (("all_evaluations", df), ("all_feature_results", ff), ("all_rule_bounds", rr)):
        table.to_csv(REPORT/f"{filename}.csv", index=False)
    metrics = [c for c in df.select_dtypes(include="number") if c not in identity]
    checkpoint = df.groupby(["arm", "inner_seed", "training_seed"])[metrics].mean().reset_index()
    means = checkpoint.groupby(["arm", "inner_seed"])[metrics].mean().reset_index()
    checkpoint.to_csv(REPORT/"checkpoint_sampling_means.csv", index=False)
    means.to_csv(REPORT/"partition_means.csv", index=False)
    checkpoint.groupby(["arm", "inner_seed"])[metrics].std().reset_index().to_csv(REPORT/"training_seed_sd.csv", index=False)
    df.groupby(["arm", "inner_seed", "training_seed"])[metrics].std().reset_index().to_csv(REPORT/"sampling_seed_sd.csv", index=False)
    fmeans = ff.groupby(["arm", "inner_seed", "feature"])[["error_vs_fit", "error_vs_validation", "std_ratio_error"]].mean()
    rmeans = rr[rr.population == "synthetic"].groupby(["arm", "inner_seed", "rule"])[["coverage"]].mean()
    gates, paired = [], []
    for variant, (reference_arm, _) in VARIANTS.items():
        reference_arm = "reference_"+reference_arm
        candidate, reference = df[df.arm == variant], df[df.arm == reference_arm]
        pairs = candidate.merge(reference, on=["inner_seed", "training_seed", "sampling_seed"], suffixes=("_new", "_reference"))
        for c in metrics:
            pairs["delta_"+c] = pairs[c+"_new"]-pairs[c+"_reference"]
        paired.append(pairs)
        global_checks = dict(no_missingness_flags=bool((candidate.max_missing_error_fit <= .02).all()),
            worst_missingness_reduced=bool(candidate.max_missing_error_fit.max() < reference.max_missing_error_fit.max()))
        for inner in (2027, 3027, 4027):
            a = means[(means.arm == variant) & (means.inner_seed == inner)].iloc[0]
            b = means[(means.arm == reference_arm) & (means.inner_seed == inner)].iloc[0]
            fd = fmeans.loc[(variant, inner)]-fmeans.loc[(reference_arm, inner)]
            rd = rmeans.loc[(variant, inner)]-rmeans.loc[(reference_arm, inner)]
            conditions = dict(rf=a.rf >= b.rf-.02, logistic=a.logistic >= b.logistic-.02,
                numeric_fidelity=a.numeric_w <= b.numeric_w*1.1, association=a.association_rmse <= b.association_rmse*1.1,
                observed_violation=a.violation <= b.violation+.01, upper_violation=a.violation_upper <= b.violation_upper+.01,
                rule_coverage=bool((rd.coverage >= -.01).all()),
                missingness_fit=bool((fd.error_vs_fit <= .01).all()),
                missingness_validation=bool((fd.error_vs_validation <= .01).all()),
                numerical_variability=bool((fd.std_ratio_error.dropna() <= .10).all()))
            gates.append(dict(variant=variant, inner_seed=inner, **{k: bool(v) for k, v in conditions.items()},
                partition_pass=bool(all(conditions.values())), **global_checks, delta_rf=float(a.rf-b.rf),
                delta_logistic=float(a.logistic-b.logistic), max_sd_error_increase=float(fd.std_ratio_error.max()),
                largest_sd_error_increase_feature=fd.std_ratio_error.idxmax(),
                max_missing_error=float(candidate.max_missing_error_fit.max()),
                reference_max_missing_error=float(reference.max_missing_error_fit.max())))
    gf = pd.DataFrame(gates)
    gf.to_csv(REPORT/"correction_gates.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(REPORT/"paired_deltas.csv", index=False)
    decisions = {}
    for variant, group in gf.groupby("variant"):
        candidate = df[df.arm == variant]
        decisions[variant] = dict(partitions_passed=int(group.partition_pass.sum()),
            flagged_tables=int((candidate.max_missing_error_fit > .02).sum()),
            max_missing_error=float(candidate.max_missing_error_fit.max()),
            meets_correction_gates=bool(group.partition_pass.all() and group.no_missingness_flags.all() and group.worst_missingness_reduced.all()))
    eligible = [v for v in ("full_prior", "full_prior_w4") if decisions[v]["meets_correction_gates"]]
    eligible.sort(key=lambda v: (decisions[v]["max_missing_error"], float(df[df.arm == v].numeric_w.mean())))
    dump(REPORT/"selection.json", dict(decisions=decisions, qualifying_full_variants=eligible,
        selected_development_candidate=eligible[0] if eligible else None,
        interpretation="Engineering development screen, not independent validation or a clinical threshold."))
    trainings = [json.loads(p.read_text()) for p in (RUNS/"training").glob("*/manifest.json")]
    evaluations = [json.loads((ROOT/p/"manifest.json").read_text()) for p in registry["new"]]
    pd.DataFrame([{k: m[k] for k in ("variant", "inner_seed", "training_seed", "fit_seconds", "selected_epoch", "epochs_executed", "parameters")}
                  for m in trainings]).to_csv(REPORT/"new_training_runs.csv", index=False)
    timing = dict(new_fits=len(trainings), new_evaluations=len(evaluations), reused_evaluations=len(registry["references"]),
        fit_seconds_sum=sum(m["fit_seconds"] for m in trainings), fit_seconds_min=min(m["fit_seconds"] for m in trainings),
        fit_seconds_max=max(m["fit_seconds"] for m in trainings),
        sample_seconds_sum=sum(m["sample_seconds"] for m in evaluations),
        sample_seconds_min=min(m["sample_seconds"] for m in evaluations), sample_seconds_max=max(m["sample_seconds"] for m in evaluations),
        evaluation_seconds_sum=sum(m["evaluation_seconds"] for m in evaluations),
        device="CPU", gpu_vram_gib=0, peak_system_ram="not measured")
    dump(REPORT/"timing.json", timing)
    (REPORT/"RESULTS.md").write_text("# Iteration 5: complete observation-head experiment results\n\n"
        "36 new fits, 108 new evaluations and 81 reused 5x references. Three sampling seeds are averaged within each training seed before averaging three training seeds per partition. These overlapping development partitions are not independent cohorts.\n\n"
        + markdown(means[["arm", "inner_seed", "rf", "logistic", "numeric_w", "violation", "ag_missing", "protein_coverage"]])
        + "\n\n## Correction gates\n\n"+markdown(gf[["variant", "inner_seed", "partition_pass", "no_missingness_flags", "delta_rf", "delta_logistic", "max_sd_error_increase"]])
        + "\n\nAll gate components and per-feature/rule results are retained in the accompanying CSVs. The selection JSON applies the predeclared full-variant criteria. Missingness correction alone is insufficient for promotion. Unknown-rule intervals are logical bounds, not confidence intervals.\n")
    print(json.dumps(dict(selection=eligible, decisions=decisions, timing=timing), indent=2))


if __name__ == "__main__":
    main()
