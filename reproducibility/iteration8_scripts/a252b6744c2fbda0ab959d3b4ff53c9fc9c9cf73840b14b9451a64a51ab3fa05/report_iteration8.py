"""Aggregate robust-representation results and matched references."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from iteration8 import ROOT, REPORT, RUNS, PLAN, checked_plan, dump
from iteration6_sampler import distribution_shape
from medhyperdiff.data import SCHEMAS
from report_iteration4 import markdown


def record(path, arm, kind, study):
    m, v = [json.loads((path/f).read_text()) for f in ("manifest.json", "metrics.json")]
    f, u, b = v["fidelity"], v["utility"], v["rule_missingness_bounds"]["synthetic"]
    shape_all = v.get("distribution_shape")
    if shape_all is None:
        frame, split = study.data("ilpd", m["inner_seed"])
        synthetic = pd.read_csv(path/"synthetic.csv", dtype={SCHEMAS["ilpd"].target: str}, float_precision="round_trip")
        shape_all = distribution_shape(frame.iloc[split["train"]], frame.iloc[split["validation"]], synthetic, SCHEMAS["ilpd"])
    row = dict(arm=arm, kind=kind, inner_seed=m["inner_seed"], training_seed=m["training_seed"], sampling_seed=m["sampling_seed"], path=str(path.relative_to(ROOT)),
        rf=u["forest"]["TSTR"]["auroc"], logistic=u["logistic"]["TSTR"]["auroc"], rf_augment=u["forest"]["AUGMENT_5x"]["auroc"],
        logistic_augment=u["logistic"]["AUGMENT_5x"]["auroc"], numeric_w=f["mean_numeric_wasserstein_train_sd"], association_rmse=f["association_rmse"],
        violation=v["constraints"]["synthetic"]["any_rule_violation_rate"], violation_upper=b["any_rule"]["violation_upper"],
        max_missing_error_fit=max(x["error_vs_fit"] for x in v["feature_missingness"].values()), c2st_auc=v["c2st"]["auc"],
        exact_train_copy=v["privacy"]["exact_predictor_match_train"], nearest_membership_auc=v["privacy"]["nearest_synthetic_membership_auc"])
    features=[]
    for feature, values in v["feature_missingness"].items():
        fm=f["per_feature"][feature]; shape=shape_all.get(feature, {})
        ratio=fm.get("std_ratio_synthetic_to_real")
        features.append(dict(arm=arm,kind=kind,inner_seed=m["inner_seed"],training_seed=m["training_seed"],sampling_seed=m["sampling_seed"],feature=feature,
            error_vs_fit=values["error_vs_fit"],error_vs_validation=values["error_vs_validation"],std_ratio_error=abs(ratio-1) if ratio is not None and np.isfinite(ratio) else None,
            q05_error=shape.get("q05_error"),q50_error=shape.get("q50_error"),q95_error=shape.get("q95_error"),iqr_ratio=shape.get("iqr_ratio"),
            synthetic_upper95_rate=shape.get("synthetic_upper95_rate"),synthetic_outside_fit_range=shape.get("synthetic_outside_fit_range")))
    return row, features


def main():
    plan, study = checked_plan(); registry=json.loads((REPORT/"registry.json").read_text())
    assert len(registry["new"]) == 27 and len(registry["references"]) == 36
    rows, features = [], []
    for path in registry["new"]:
        row, ff=record(ROOT/path,"asinh_mad","new",study); rows.append(row); features.extend(ff)
    for ref in registry["references"]:
        path=ROOT/ref["path"]
        arm=ref["arm"]
        row, ff=record(path,arm,"reference",study); rows.append(row); features.extend(ff)
    df, ff = pd.DataFrame(rows), pd.DataFrame(features)
    for filename, table in (("all_evaluations",df),("all_feature_results",ff)):
        table.to_csv(REPORT/f"{filename}.csv",index=False)
    metrics=["rf","logistic","rf_augment","logistic_augment","numeric_w","association_rmse","violation","violation_upper","max_missing_error_fit","c2st_auc","exact_train_copy","nearest_membership_auc"]
    checkpoints=df.groupby(["arm","inner_seed","training_seed"],dropna=False)[metrics].mean().reset_index()
    means=checkpoints.groupby(["arm","inner_seed"],dropna=False)[metrics].mean().reset_index()
    checkpoints.to_csv(REPORT/"checkpoint_sampling_means.csv",index=False); means.to_csv(REPORT/"partition_means.csv",index=False)
    df.groupby(["arm","inner_seed","training_seed"],dropna=False)[metrics].std().reset_index().to_csv(REPORT/"sampling_seed_sd.csv",index=False)
    checkpoints[checkpoints.training_seed.notna()].groupby(["arm","inner_seed"])[metrics].std().reset_index().to_csv(REPORT/"training_seed_sd.csv",index=False)
    feature_metrics=["error_vs_fit","error_vs_validation","std_ratio_error","q05_error","q50_error","q95_error","iqr_ratio"]
    ff.groupby(["arm","inner_seed","feature"],dropna=False)[feature_metrics].mean().reset_index().to_csv(REPORT/"feature_partition_means.csv",index=False)
    pairs=[]
    for baseline in ("full_prior","copula_repair"):
        a=means[means.arm=="asinh_mad"].set_index("inner_seed"); b=means[means.arm==baseline].set_index("inner_seed")
        for inner in (2027,3027,4027):
            pairs.append(dict(baseline=baseline,inner_seed=inner,**{"asinh_minus_baseline_"+m:float(a.loc[inner,m]-b.loc[inner,m]) for m in metrics}))
    pd.DataFrame(pairs).to_csv(REPORT/"asinh_minus_baseline.csv",index=False)
    times=[]
    for p in sorted((RUNS/"training").glob("*/manifest.json")):
        m=json.loads(p.read_text()); times.append(dict(arm="asinh_mad",inner_seed=m["inner_seed"],training_seed=m["training_seed"],fit_seconds=m["fit_seconds"]))
    tdf=pd.DataFrame(times); tdf.to_csv(REPORT/"fit_timings.csv",index=False)
    evals=[json.loads((ROOT/p/"manifest.json").read_text()) for p in registry["new"]]
    timing=dict(new_fits=9,new_evaluations=27,reused_references=36,fit_seconds_sum=float(tdf.fit_seconds.sum()),sample_seconds_sum=sum(x["sample_seconds"] for x in evals),evaluation_seconds_sum=sum(x["evaluation_seconds"] for x in evals),device="CPU, torch threads=2",gpu_vram_gib=0,peak_system_ram="not measured")
    dump(REPORT/"timing.json",timing)
    (REPORT/"RESULTS.md").write_text("# Iteration 8: fit-only asinh-MAD representation\n\n"
        "Nine new transformed fits and 27 new evaluations are compared with 27 full_prior and nine repaired-copula records. No outer test is opened. The transformed representation is retained as a negative development result because inverse decoding produces extreme numerical outliers and worsens rule compliance.\n\n"
        + markdown(means[["arm","inner_seed","rf","logistic","numeric_w","association_rmse","violation","max_missing_error_fit","c2st_auc"]])
        + "\n\n## Component costs\n\n" + markdown(tdf.groupby("arm").fit_seconds.agg(["count","mean","min","max","sum"]).reset_index()) + "\n")
    print(means.to_string(index=False)); print(json.dumps(timing,indent=2))


if __name__ == "__main__": main()
