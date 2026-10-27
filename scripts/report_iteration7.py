"""Report every frozen baseline preset, separating fit and sampling variability."""
import json
import numpy as np
import pandas as pd
from iteration7 import ROOT, RUNS, REPORT, PLAN, checked_plan, dump
from iteration6_sampler import distribution_shape
from medhyperdiff.data import SCHEMAS
from report_iteration4 import markdown


def main():
    plan, study = checked_plan()
    registry = json.loads((REPORT/"registry.json").read_text())
    assert len(registry["new"]) == 144 and len(registry["references"]) == 81
    rows, features, rules = [], [], []
    for kind in ("new", "references"):
        for entry in registry[kind]:
            out = ROOT/entry["path"]
            m, v = [json.loads((out/f).read_text()) for f in ("manifest.json", "metrics.json")]
            identity = dict(arm=entry["arm"], inner_seed=m["inner_seed"], training_seed=m["training_seed"], sampling_seed=m["sampling_seed"])
            f, u, b = v["fidelity"], v["utility"], v["rule_missingness_bounds"]["synthetic"]
            rows.append(dict(**identity, kind=kind, path=entry["path"],
                rf=u["forest"]["TSTR"]["auroc"], logistic=u["logistic"]["TSTR"]["auroc"],
                rf_augment=u["forest"]["AUGMENT_5x"]["auroc"], logistic_augment=u["logistic"]["AUGMENT_5x"]["auroc"],
                rf_real=u["forest"]["TRTR"]["auroc"], logistic_real=u["logistic"]["TRTR"]["auroc"],
                numeric_w=f["mean_numeric_wasserstein_train_sd"], association_rmse=f["association_rmse"],
                violation=v["constraints"]["synthetic"]["any_rule_violation_rate"], violation_upper=b["any_rule"]["violation_upper"],
                protein_coverage=b["rules"]["protein_identity"]["coverage"],
                max_missing_error_fit=max(x["error_vs_fit"] for x in v["feature_missingness"].values()),
                c2st_auc=v["c2st"]["auc"], exact_train_copy=v["privacy"]["exact_predictor_match_train"],
                nearest_membership_auc=v["privacy"]["nearest_synthetic_membership_auc"], changed_row_rate=m.get("changed_row_rate", 0.)))
            shape = v.get("distribution_shape")
            if shape is None:
                frame, split = study.data("ilpd", m["inner_seed"])
                synthetic = pd.read_csv(out/"synthetic.csv", dtype={SCHEMAS["ilpd"].target: str}, float_precision="round_trip")
                shape = distribution_shape(frame.iloc[split["train"]], frame.iloc[split["validation"]], synthetic, SCHEMAS["ilpd"])
                dump(REPORT/"reference_shapes"/(out.name+".json"), shape)
            for feature, values in v["feature_missingness"].items():
                fm = f["per_feature"][feature]
                ratio = fm.get("std_ratio_synthetic_to_real")
                record = dict(**identity, feature=feature, **values, **fm)
                record["std_ratio_error"] = abs(ratio-1) if ratio is not None and np.isfinite(ratio) else None
                for key, value in shape.get(feature, {}).items():
                    record[key] = json.dumps(value) if isinstance(value, list) else value
                features.append(record)
            for population, bounds in v["rule_missingness_bounds"].items():
                for rule, values in bounds["rules"].items():
                    rules.append(dict(**identity, population=population, rule=rule, **values))
    df, ff, rr = pd.DataFrame(rows), pd.DataFrame(features), pd.DataFrame(rules)
    for name, table in (("all_evaluations", df), ("all_feature_results", ff), ("all_rule_bounds", rr)):
        table.to_csv(REPORT/f"{name}.csv", index=False)
    metrics = [c for c in df.select_dtypes(include="number") if c not in identity]
    checkpoints = df.groupby(["arm", "inner_seed", "training_seed"], dropna=False)[metrics].mean().reset_index()
    means = checkpoints.groupby(["arm", "inner_seed"])[metrics].mean().reset_index()
    checkpoints.to_csv(REPORT/"checkpoint_sampling_means.csv", index=False)
    means.to_csv(REPORT/"partition_means.csv", index=False)
    checkpoints[checkpoints.training_seed.notna()].groupby(["arm", "inner_seed"])[metrics].std().reset_index().to_csv(REPORT/"training_seed_sd.csv", index=False)
    df.groupby(["arm", "inner_seed", "training_seed"], dropna=False)[metrics].std().reset_index().to_csv(REPORT/"sampling_seed_sd.csv", index=False)
    ff.groupby(["arm", "inner_seed", "feature"])[["error_vs_fit", "error_vs_validation", "std_ratio_error", "q05_error", "q95_error", "iqr_ratio"]].mean().reset_index().to_csv(REPORT/"feature_partition_means.csv", index=False)
    full = means[means.arm == "full_prior"].set_index("inner_seed")
    differences = []
    for _, row in means[means.arm != "full_prior"].iterrows():
        differences.append(dict(baseline=row.arm, inner_seed=row.inner_seed,
            **{"delta_"+key: float(full.loc[row.inner_seed, key]-row[key]) for key in metrics}))
    pd.DataFrame(differences).to_csv(REPORT/"full_minus_baseline.csv", index=False)
    fits = [json.loads(p.read_text()) for p in sorted((RUNS/"training").glob("*/manifest.json"))]
    assert len(fits) == 45
    times = pd.DataFrame([dict(arm=m["arm"], inner_seed=m["inner_seed"], training_seed=m["training_seed"], fit_seconds=m["fit_seconds"]) for m in fits])
    times.to_csv(REPORT/"fit_timings.csv", index=False)
    evals = [json.loads((ROOT/e["path"]/"manifest.json").read_text()) for e in registry["new"]]
    timing = dict(new_medhyperdiff_fits=0, neural_adapter_fits=36, native_class_networks=72, statistical_fits=9,
        new_sampled_tables=135, new_repaired_derivatives=9, new_evaluations=144, reused_evaluations=81,
        fit_seconds_sum=float(times.fit_seconds.sum()),
        sample_or_repair_seconds_sum=sum(m["sample_seconds"] for m in evals), evaluation_seconds_sum=sum(m["evaluation_seconds"] for m in evals),
        device="CPU, torch threads=2", gpu_vram_gib=0, peak_system_ram="not measured")
    dump(REPORT/"timing.json", timing)
    cost = times.groupby("arm").fit_seconds.agg(["count", "mean", "min", "max", "sum"]).reset_index()
    cost.to_csv(REPORT/"fit_costs_by_arm.csv", index=False)
    (REPORT/"RESULTS.md").write_text("# Iteration 7: matched development references\n\n"
        "144 new evaluations + 81 reused = 225 records. Neural cells average three samples per checkpoint, then three training seeds. Statistical cells average three samples from one deterministic fit. Inner partitions overlap; these are descriptive development comparisons. All presets are retained; no confirmatory winner is selected.\n\n"
        "**Measurement caveat:** `exact_train_copy` below is literal numeric-hash equality. It misses even bootstrap copies after float32 round trips. The [supplemental diagnostic](NUMERICAL_DIAGNOSTIC_AUDIT.json) detects 100% training near-copies for bootstrap; see `tolerant_copy_partition_means.csv`. Neither zero literal copies nor zero near-copies establishes privacy. Use `tolerant_boundaries_all.csv` for interpretation of strict boundary ties. Original metrics are retained unchanged.\n\n"
        + markdown(means[["arm", "inner_seed", "rf", "logistic", "numeric_w", "association_rmse", "violation", "protein_coverage", "c2st_auc", "exact_train_copy"]])
        + "\n\n## CPU fitting cost in seconds\n\n"+markdown(cost)
        + "\n\nCTGAN/TVAE adapters each contain two separately trained class networks. Default300 changes capacity, batch size and budget together. Data and evaluation match; compute and development tuning opportunity do not. Bootstrap is a memorization control. Copula repair receives supplied rules and is explicit postprocessing. All utility, shape, missingness and privacy diagnostics and seed variation are retained in companion CSVs.\n")
    print(means[["arm", "inner_seed", "rf", "numeric_w", "violation", "exact_train_copy"]].to_string(index=False))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
