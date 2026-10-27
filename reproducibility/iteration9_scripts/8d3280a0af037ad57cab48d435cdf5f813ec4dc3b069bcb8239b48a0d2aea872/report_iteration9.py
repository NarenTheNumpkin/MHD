"""Summarize wrong-rule sensitivity without changing saved generators."""
import json
import pandas as pd
from iteration9 import ROOT, REPORT, PLAN, checked_plan, dump
from report_iteration4 import markdown


def main():
    plan, study=checked_plan(); registry=json.loads((REPORT/"registry.json").read_text()); assert len(registry["records"])==1125
    metrics=[]; bounds=[]
    for entry in registry["records"]:
        out=ROOT/entry["path"]; m=json.loads((out/"manifest.json").read_text()); v=json.loads((out/"metrics.json").read_text())
        for population, values in v.items():
            c=values["constraints"]
            metrics.append(dict(arm=entry["arm"],variant=entry["variant"],inner_seed=m["inner_seed"],training_seed=m["training_seed"],sampling_seed=m["sampling_seed"],population=population,
                any_violation_rate=c["any_rule_violation_rate"],any_coverage=c["any_rule_coverage"]))
            for rule, rv in values["rule_bounds"]["rules"].items():
                bounds.append(dict(arm=entry["arm"],variant=entry["variant"],inner_seed=m["inner_seed"],training_seed=m["training_seed"],sampling_seed=m["sampling_seed"],population=population,rule=rule,**rv))
    mf,bf=pd.DataFrame(metrics),pd.DataFrame(bounds); mf.to_csv(REPORT/"rule_sensitivity_metrics.csv",index=False); bf.to_csv(REPORT/"rule_sensitivity_bounds.csv",index=False)
    synth=mf[mf.population=="synthetic"]; means=synth.groupby(["arm","variant","inner_seed"],dropna=False)[["any_violation_rate","any_coverage"]].mean().reset_index(); means.to_csv(REPORT/"partition_means.csv",index=False)
    real=mf[mf.population=="validation"].groupby(["arm","variant","inner_seed"])["any_violation_rate"].mean().reset_index().rename(columns={"any_violation_rate":"validation_any_violation_rate"}); real.to_csv(REPORT/"validation_partition_means.csv",index=False)
    correct=means[means.variant=="correct"].set_index(["arm","inner_seed"]); compare=[]
    for _, row in means[means.variant!="correct"].iterrows():
        base=correct.loc[(row.arm,row.inner_seed)]; compare.append(dict(arm=row.arm,variant=row.variant,inner_seed=row.inner_seed,delta_synthetic_violation=float(row.any_violation_rate-base.any_violation_rate),delta_coverage=float(row.any_coverage-base.any_coverage)))
    pd.DataFrame(compare).to_csv(REPORT/"delta_vs_correct.csv",index=False)
    timing=json.loads((REPORT/"registry.json").read_text()); dump(REPORT/"timing.json",dict(records=len(registry["records"]),fits=0,sampling=0,elapsed_seconds=timing["elapsed_seconds"],gpu_vram_gib=0,peak_system_ram="not measured"))
    (REPORT/"RESULTS.md").write_text("# Iteration 9: wrong-rule sensitivity\n\n"
        "The 225 iteration-7 synthetic tables were re-evaluated under five frozen rule registries (1,125 analyses; no fits or sampling). `correct` is the supplied registry, `omit_protein_identity` removes the exploratory A/G relation, and the three reverse variants intentionally swap biochemical inequality directions. Reversed variants are stress tests, not clinical models.\n\n"
        + markdown(means[means.arm.isin(["full_prior","copula_repair","ctgan_default300","tvae_compact100"])])
        + "\n\nValidation-population rates and all per-rule lower/upper bounds are in the companion CSVs. These descriptive sensitivities do not establish clinical truth, significance, or a model winner.\n")
    print(means.to_string(index=False))


if __name__=="__main__": main()
