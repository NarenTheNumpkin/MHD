"""Both-reference gates and complete distribution-shape reporting."""
import json

import numpy as np
import pandas as pd

from iteration6 import ROOT, RUNS, REPORT, PLAN, dump
from report_iteration4 import markdown


def main():
    registry = json.loads((REPORT/"registry.json").read_text())
    rows, features, rules = [], [], []
    entries = [dict(path=p, kind="new") for p in registry["new"]]+registry["references"]
    for entry in entries:
        out = ROOT/entry["path"]
        m, v = [json.loads((out/f).read_text()) for f in ("manifest.json", "metrics.json")]
        architecture = entry.get("architecture", m.get("architecture"))
        if entry["kind"] == "new":
            mix = m["variance_mix"]
            shape = v["distribution_shape"]
            arm = f"{architecture}_m{mix:g}"
        else:
            mix = 0.
            shape = json.loads((ROOT/entry["shape"]).read_text())["distribution_shape"]
            arm = f"{architecture}_m0" if entry["kind"] == "prior" else f"{architecture}_original"
        identity = dict(arm=arm, architecture=architecture, kind=entry["kind"], variance_mix=mix,
            inner_seed=m["inner_seed"], training_seed=m["training_seed"], sampling_seed=m["sampling_seed"])
        f, u, b = v["fidelity"], v["utility"], v["rule_missingness_bounds"]["synthetic"]
        rows.append({**identity, "path": entry["path"], "rf": u["forest"]["TSTR"]["auroc"],
            "logistic": u["logistic"]["TSTR"]["auroc"], "rf_augment": u["forest"]["AUGMENT_5x"]["auroc"],
            "logistic_augment": u["logistic"]["AUGMENT_5x"]["auroc"], "rf_real": u["forest"]["TRTR"]["auroc"],
            "logistic_real": u["logistic"]["TRTR"]["auroc"], "numeric_w": f["mean_numeric_wasserstein_train_sd"],
            "association_rmse": f["association_rmse"], "violation": v["constraints"]["synthetic"]["any_rule_violation_rate"],
            "violation_upper": b["any_rule"]["violation_upper"], "protein_coverage": b["rules"]["protein_identity"]["coverage"],
            "max_missing_error_fit": max(x["error_vs_fit"] for x in v["feature_missingness"].values()),
            "ag_missing": v["feature_missingness"]["ag_ratio"]["synthetic"], "c2st_auc": v["c2st"]["auc"],
            "exact_train_copy": v["privacy"]["exact_predictor_match_train"],
            "nearest_membership_auc": v["privacy"]["nearest_synthetic_membership_auc"]})
        for feature, values in v["feature_missingness"].items():
            fm = f["per_feature"][feature]
            ratio = fm.get("std_ratio_synthetic_to_real")
            record = {**identity, "feature": feature, **values, **fm,
                      "std_ratio_error": abs(ratio-1) if ratio is not None and np.isfinite(ratio) else None}
            if feature in shape:
                assert shape[feature]["status"] == "evaluated"
                for key, value in shape[feature].items():
                    record[key] = json.dumps(value) if isinstance(value, list) else value
            features.append(record)
        for population, bounds in v["rule_missingness_bounds"].items():
            for rule, values in bounds["rules"].items():
                rules.append({**identity, "population": population, "rule": rule, **values})
    df, ff, rr = pd.DataFrame(rows), pd.DataFrame(features), pd.DataFrame(rules)
    assert len(df) == 324 and (df.kind == "new").sum() == 162
    for file, table in (("all_evaluations", df), ("all_feature_results", ff), ("all_rule_bounds", rr)):
        table.to_csv(REPORT/f"{file}.csv", index=False)
    metrics = [c for c in df.select_dtypes(include="number") if c not in identity]
    checkmeans = df.groupby(["arm", "inner_seed", "training_seed"])[metrics].mean().reset_index()
    means = checkmeans.groupby(["arm", "inner_seed"])[metrics].mean().reset_index()
    checkmeans.to_csv(REPORT/"checkpoint_sampling_means.csv", index=False)
    means.to_csv(REPORT/"partition_means.csv", index=False)
    checkmeans.groupby(["arm", "inner_seed"])[metrics].std().reset_index().to_csv(REPORT/"training_seed_sd.csv", index=False)
    df.groupby(["arm", "inner_seed", "training_seed"])[metrics].std().reset_index().to_csv(REPORT/"sampling_seed_sd.csv", index=False)
    feature_metrics = ["error_vs_fit", "error_vs_validation", "std_ratio_error", "q05_error", "q95_error", "iqr_ratio"]
    fm = ff.groupby(["arm", "inner_seed", "feature"])[feature_metrics].mean()
    fm.reset_index().to_csv(REPORT/"feature_partition_means.csv", index=False)
    rm = rr[rr.population == "synthetic"].groupby(["arm", "inner_seed", "rule"])[["coverage"]].mean()
    gates, pairs = [], []
    for architecture in ("full", "dense", "base"):
        for mix in (.5, 1.):
            arm = f"{architecture}_m{mix:g}"
            candidate = df[df.arm == arm]
            for reference_kind in ("prior", "original"):
                refarm = f"{architecture}_m0" if reference_kind == "prior" else f"{architecture}_original"
                reference = df[df.arm == refarm]
                pair = candidate.merge(reference, on=["inner_seed", "training_seed", "sampling_seed"], suffixes=("_new", "_reference"))
                for key in metrics:
                    pair["delta_"+key] = pair[key+"_new"]-pair[key+"_reference"]
                pairs.append(pair)
                for inner in (2027, 3027, 4027):
                    a = means[(means.arm == arm) & (means.inner_seed == inner)].iloc[0]
                    b = means[(means.arm == refarm) & (means.inner_seed == inner)].iloc[0]
                    fd = fm.loc[(arm, inner)]-fm.loc[(refarm, inner)]
                    rd = rm.loc[(arm, inner)]-rm.loc[(refarm, inner)]
                    conditions = dict(rf=a.rf >= b.rf-.02, logistic=a.logistic >= b.logistic-.02,
                        numeric_fidelity=a.numeric_w <= b.numeric_w*1.1, association=a.association_rmse <= b.association_rmse*1.1,
                        observed_violation=a.violation <= b.violation+.01, upper_violation=a.violation_upper <= b.violation_upper+.01,
                        rule_coverage=bool((rd.coverage >= -.01).all()), missingness_fit=bool((fd.error_vs_fit <= .01).all()),
                        missingness_validation=bool((fd.error_vs_validation <= .01).all()),
                        numerical_variability=bool((fd.std_ratio_error.dropna() <= .10).all()))
                    if reference_kind == "prior":
                        conditions["quantile_tails"] = bool((fd[["q05_error", "q95_error"]].dropna() <= .10).all().all())
                    gates.append(dict(arm=arm, architecture=architecture, variance_mix=mix, inner_seed=inner, reference_kind=reference_kind,
                        **{k: bool(v) for k, v in conditions.items()}, partition_reference_pass=bool(all(conditions.values())),
                        no_missingness_flags=bool((candidate.max_missing_error_fit <= .02).all()),
                        worst_missingness_reduced=bool(candidate.max_missing_error_fit.max() < reference.max_missing_error_fit.max()),
                        delta_rf=float(a.rf-b.rf), delta_logistic=float(a.logistic-b.logistic),
                        max_sd_error_increase=float(fd.std_ratio_error.max()), largest_sd_error_increase_feature=fd.std_ratio_error.idxmax(),
                        max_q05_error_increase=float(fd.q05_error.max()), max_q95_error_increase=float(fd.q95_error.max())))
    gf = pd.DataFrame(gates)
    gf.to_csv(REPORT/"correction_gates.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(REPORT/"paired_deltas.csv", index=False)
    decisions = {}
    for arm, g in gf.groupby("arm"):
        architecture = g.architecture.iloc[0]
        candidate = df[df.arm == arm]
        ast_gain = float(fm.loc[(architecture+"_m0", 4027, "ast"), "std_ratio_error"]-fm.loc[(arm, 4027, "ast"), "std_ratio_error"])
        reference_checks = bool(g.partition_reference_pass.all() and g.no_missingness_flags.all()
            and g[g.reference_kind == "original"].worst_missingness_reduced.all())
        decisions[arm] = dict(partition_reference_comparisons_passed=int(g.partition_reference_pass.sum()), comparisons=6,
            flagged_tables=int((candidate.max_missing_error_fit > .02).sum()), ast_error_reduction_4027=ast_gain,
            maximum_feature_sd_error=float(fm.loc[arm].std_ratio_error.max()),
            qualifies_full_candidate=bool(architecture == "full" and reference_checks and ast_gain >= .01))
    qualifying = [k for k, v in decisions.items() if v["qualifies_full_candidate"]]
    qualifying.sort(key=lambda k: (decisions[k]["maximum_feature_sd_error"], float(df[df.arm == k].numeric_w.mean()), float(df[df.arm == k].variance_mix.iloc[0])))
    selection = dict(decisions=decisions, qualifying_full_settings=qualifying,
        selected_development_candidate=qualifying[0] if qualifying else None,
        interpretation="Frozen development gates; no independent test, clinical validity or superiority claim.")
    dump(REPORT/"selection.json", selection)
    evaluations = [json.loads((ROOT/p/"manifest.json").read_text()) for p in registry["new"]]
    timing = dict(new_training_runs=0, generation_checkpoints=27, new_evaluations=len(evaluations), reused_evaluations=162)
    for key in ("sample_seconds", "evaluation_seconds"):
        timing[key+"_sum"] = sum(m[key] for m in evaluations)
        timing[key+"_min"] = min(m[key] for m in evaluations)
        timing[key+"_max"] = max(m[key] for m in evaluations)
    timing.update(device="CPU", gpu_vram_gib=0, peak_system_ram="not measured")
    dump(REPORT/"timing.json", timing)
    (REPORT/"RESULTS.md").write_text("# Iteration 6: complete reverse-variance sensitivity results\n\n"
        "162 new sampling/evaluation runs, 162 reused references, zero new fits. Means average three sampling seeds within each fitted checkpoint, then three training seeds separately by partition. Overlapping partitions are not independent cohorts.\n\n"
        + markdown(means[["arm", "inner_seed", "rf", "logistic", "numeric_w", "violation", "ag_missing", "protein_coverage"]])
        + "\n\n## Both-reference comparisons\n\n"
        + markdown(gf[["arm", "inner_seed", "reference_kind", "partition_reference_pass", "delta_rf", "delta_logistic", "max_sd_error_increase", "max_q95_error_increase"]])
        + "\n\nAll per-feature quantiles, IQR ratios, fit-threshold tail frequencies, rule bounds and failed gates are retained in CSVs. `selection.json` additionally applies the absolute missingness screen and the targeted AST improvement requirement.\n")
    print(json.dumps(dict(selection=selection, timing=timing), indent=2))


if __name__ == "__main__":
    main()
