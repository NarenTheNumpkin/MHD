"""Audit external-cohort portability outputs and exact first-seed replays."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from iteration10 import ROOT, REPORT, RUNS, checked_plan, job_name, sha
from iteration4 import missingness_audit, rule_bounds
from iteration6_sampler import distribution_shape
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import guidance_schedule


def main():
    plan,study=checked_plan(); registry=json.loads((REPORT/"registry.json").read_text()); assert len(registry["new"])==27; replays=0
    for job in plan["jobs"]:
        parent=RUNS/"training"/job_name(job); pm=json.loads((parent/"manifest.json").read_text()); assert pm["status"]=="complete" and not pm["outer_test_opened"] and sha(parent/"checkpoint.pt")==pm["checkpoint_sha256"]
        model=__import__("iteration5_model").ObservationInitializedDiffusion.load(parent/"checkpoint.pt"); schema=SCHEMAS[job["dataset"]]; frame,split=study.data(job["dataset"],job["inner_seed"]); train,val=frame.iloc[split["train"]],frame.iloc[split["validation"]]; labels=np.tile((train[schema.target]==schema.positive).astype(int),5); model.config.guidance=10.; model.config.schedule="adaptive"; model.schedule=guidance_schedule(model.config.steps,"adaptive"); generated=model.sample(labels,seed=70001,batch_size=256); negative=next(v for v in train[schema.target].unique() if v!=schema.positive); generated[schema.target]=np.where(labels==1,schema.positive,negative); saved=pd.read_csv(RUNS/"evaluations"/(job_name(job)+"_s70001_n5")/"synthetic.csv",dtype={c:str for c in (*schema.categorical,schema.target)},float_precision="round_trip"); pd.testing.assert_frame_equal(generated,saved,check_exact=True); replays+=1
    for rel in registry["new"]:
        path=ROOT/rel; m,v=[json.loads((path/f).read_text()) for f in ("manifest.json","metrics.json")]; schema=SCHEMAS[m["dataset"]]; frame,split=study.data(m["dataset"],m["inner_seed"]); train,val=frame.iloc[split["train"]],frame.iloc[split["validation"]]; synthetic=pd.read_csv(path/"synthetic.csv",dtype={c:str for c in (*schema.categorical,schema.target)},float_precision="round_trip"); assert m["status"]=="complete" and not m["outer_test_opened"] and len(synthetic)==5*len(train) and synthetic[schema.target].tolist()==train[schema.target].tolist()*5; constraints=__import__("medhyperdiff.constraints",fromlist=["build_constraints"]).build_constraints(schema,train); assert v["feature_missingness"]==missingness_audit(train,val,synthetic,schema); assert v["rule_missingness_bounds"]["synthetic"]==rule_bounds(synthetic,constraints); assert v["distribution_shape"]==distribution_shape(train,val,synthetic,schema)
        scores=pd.read_csv(path/"utility_predictions.csv",float_precision="round_trip")
        for kind in ("forest","logistic"):
            for setting in ("TRTR","TSTR","AUGMENT_5x"): np.testing.assert_allclose(roc_auc_score(scores.label,scores[f"{kind}_{setting}"]),v["utility"][kind][setting]["auroc"],atol=1e-12,rtol=0)
        for f,k in (("synthetic.csv","synthetic_sha256"),("metrics.json","metrics_sha256"),("utility_predictions.csv","predictions_sha256")): assert sha(path/f)==m[k]
    result=dict(status="passed",datasets=3,fits=9,evaluations=27,exact_replays=replays,saved_auroc_recalculations=27*6,reserved_test_rows_in_fit_or_validation=0,split_isolation=True,hashes_match=True)
    (REPORT/"ARTIFACT_AUDIT.json").write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
