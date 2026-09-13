"""Aggregate external-cohort portability results."""
import json
import pandas as pd
from iteration10 import ROOT, REPORT, checked_plan
from report_iteration4 import markdown


def main():
    plan,study=checked_plan(); registry=json.loads((REPORT/"registry.json").read_text()); assert len(registry["new"])==27
    rows=[]; features=[]
    for rel in registry["new"]:
        path=ROOT/rel; m,v=[json.loads((path/f).read_text()) for f in ("manifest.json","metrics.json")]; f,u,b=v["fidelity"],v["utility"],v["rule_missingness_bounds"]["synthetic"]
        rows.append(dict(dataset=m["dataset"],training_seed=m["training_seed"],sampling_seed=m["sampling_seed"],rf=u["forest"]["TSTR"]["auroc"],logistic=u["logistic"]["TSTR"]["auroc"],rf_augment=u["forest"]["AUGMENT_5x"]["auroc"],logistic_augment=u["logistic"]["AUGMENT_5x"]["auroc"],rf_real=u["forest"]["TRTR"]["auroc"],logistic_real=u["logistic"]["TRTR"]["auroc"],numeric_w=f["mean_numeric_wasserstein_train_sd"],association_rmse=f["association_rmse"],violation=v["constraints"]["synthetic"]["any_rule_violation_rate"],violation_upper=b["any_rule"]["violation_upper"],max_missing_error_fit=max(x["error_vs_fit"] for x in v["feature_missingness"].values()),c2st_auc=v["c2st"]["auc"],exact_train_copy=v["privacy"]["exact_predictor_match_train"],path=rel))
        for feature,values in v["feature_missingness"].items():
            features.append(dict(dataset=m["dataset"],training_seed=m["training_seed"],sampling_seed=m["sampling_seed"],feature=feature,error_vs_fit=values["error_vs_fit"],error_vs_validation=values["error_vs_validation"]))
    df,ff=pd.DataFrame(rows),pd.DataFrame(features); df.to_csv(REPORT/"all_evaluations.csv",index=False); ff.to_csv(REPORT/"all_feature_results.csv",index=False)
    means=df.groupby("dataset")[[c for c in df.columns if c in ("rf","logistic","rf_augment","logistic_augment","rf_real","logistic_real","numeric_w","association_rmse","violation","violation_upper","max_missing_error_fit","c2st_auc","exact_train_copy")]].mean().reset_index(); means.to_csv(REPORT/"dataset_means.csv",index=False)
    seed_sd=df.groupby(["dataset","training_seed"])[["rf","logistic","numeric_w","violation"]].mean().groupby("dataset").std().reset_index(); seed_sd.to_csv(REPORT/"training_seed_sd.csv",index=False)
    times=[]
    for p in sorted((ROOT/"runs/iteration10/training").glob("*/manifest.json")):
        m=json.loads(p.read_text()); times.append(dict(dataset=m["dataset"],training_seed=m["training_seed"],fit_seconds=m["fit_seconds"]))
    tdf=pd.DataFrame(times); tdf.to_csv(REPORT/"fit_timings.csv",index=False); timing=dict(fits=9,evaluations=27,fit_seconds_sum=float(tdf.fit_seconds.sum()),sample_seconds_sum=sum(json.loads((ROOT/p/"manifest.json").read_text())["sample_seconds"] for p in registry["new"]),evaluation_seconds_sum=sum(json.loads((ROOT/p/"manifest.json").read_text())["evaluation_seconds"] for p in registry["new"]),gpu_vram_gib=0,peak_system_ram="not measured"); (REPORT/"timing.json").write_text(json.dumps(timing,indent=2)+'\n')
    (REPORT/"RESULTS.md").write_text("# Iteration 10: external-cohort portability\n\nNine current full-model fits and 27 tables cover CKD, heart failure and diabetes. These are descriptive inner-validation development results; outer tests remain closed and no external-baseline superiority claim is made.\n\n"+markdown(means)+"\n\nAll per-feature missingness, rule bounds, shape, privacy and seed variation are retained in CSVs.\n")
    print(means.to_string(index=False)); print(json.dumps(timing,indent=2))


if __name__=="__main__": main()
