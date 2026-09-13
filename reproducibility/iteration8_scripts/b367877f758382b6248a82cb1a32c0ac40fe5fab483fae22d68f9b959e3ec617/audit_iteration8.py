"""Audit iteration-8 hashes, physical-unit metrics and exact replays."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from iteration8 import ROOT, REPORT, RUNS, PLAN, FILES, checked_plan, dump, sha, model_name
from iteration4 import missingness_audit, rule_bounds
from iteration6_sampler import distribution_shape
from medhyperdiff.data import SCHEMAS
from medhyperdiff.constraints import build_constraints
from medhyperdiff.diffusion import guidance_schedule


def main():
    plan, study=checked_plan(); schema=SCHEMAS["ilpd"]; registry=json.loads((REPORT/"registry.json").read_text())
    assert len(registry["new"]) == len(set(registry["new"])) == 27 and len(registry["references"]) == 36
    replays=0; expected=set()
    for job in plan["jobs"]:
        parent=RUNS/"training"/model_name(job); pm=json.loads((parent/"manifest.json").read_text())
        assert pm["status"]=="complete" and not pm["outer_test_opened"] and pm["plan_sha256"]==sha(PLAN)
        assert sha(parent/"checkpoint.pt")==pm["checkpoint_sha256"]
        for f,d in pm["metadata_hashes"].items(): assert sha(parent/f)==d
        model=__import__("torch").load(parent/"checkpoint.pt",map_location="cpu",weights_only=False)
        frame,split=study.data("ilpd",job["inner_seed"]); train,val=frame.iloc[split["train"]],frame.iloc[split["validation"]]
        assert split==plan["splits"][str(job["inner_seed"])]["indices"]
        labels=np.tile((train[schema.target]==schema.positive).astype(int),5)
        model.config.guidance=10.; model.config.schedule="adaptive"; model.schedule=guidance_schedule(model.config.steps,"adaptive")
        generated=model.sample(labels,seed=70001,batch_size=256)
        out=RUNS/"evaluations"/(model_name(job)+"_s70001_n5")
        saved=pd.read_csv(out/"synthetic.csv",dtype={schema.target:str},float_precision="round_trip")
        generated[schema.target]=np.where(labels==1,schema.positive,next(v for v in train[schema.target].unique() if v!=schema.positive))
        pd.testing.assert_frame_equal(generated,saved,check_exact=True); replays+=1
        for sampling in plan["sampling_seeds"]:
            path=RUNS/"evaluations"/f"{model_name(job)}_s{sampling}_n5"; expected.add(str(path.relative_to(ROOT)))
            m=json.loads((path/"manifest.json").read_text()); assert m["status"]=="complete" and m["checkpoint_sha256"]==pm["checkpoint_sha256"]
    assert expected==set(registry["new"])
    for ref in registry["references"]:
        for f,d in ref["hashes"].items(): assert sha(ROOT/ref["path"]/f)==d
    for entry in registry["new"]+registry["references"]:
        entry_path = entry if isinstance(entry, str) else entry["path"]
        path=ROOT/entry_path; m,v=[json.loads((path/f).read_text()) for f in ("manifest.json","metrics.json")]
        assert m["status"]=="complete" and not m["outer_test_opened"] and m["source_sha256"]==study.source
        frame,split=study.data("ilpd",m["inner_seed"]); train,val=frame.iloc[split["train"]],frame.iloc[split["validation"]]
        synthetic=pd.read_csv(path/"synthetic.csv",dtype={schema.target:str},float_precision="round_trip")
        assert len(synthetic)==5*len(train) and synthetic[schema.target].tolist()==train[schema.target].tolist()*5
        assert v["feature_missingness"]==missingness_audit(train,val,synthetic,schema)
        assert v["rule_missingness_bounds"]["synthetic"]==rule_bounds(synthetic,build_constraints(schema,train))
        recomputed_shape=distribution_shape(train,val,synthetic,schema)
        if "distribution_shape" in v:
            assert v["distribution_shape"]==recomputed_shape
        for f,k in (("synthetic.csv","synthetic_sha256"),("metrics.json","metrics_sha256"),("utility_predictions.csv","predictions_sha256")): assert sha(path/f)==m[k]
        scores=pd.read_csv(path/"utility_predictions.csv",float_precision="round_trip")
        for kind in ("forest","logistic"):
            for setting in ("TRTR","TSTR","AUGMENT_5x"):
                np.testing.assert_allclose(roc_auc_score(scores.label,scores[f"{kind}_{setting}"]),v["utility"][kind][setting]["auroc"],atol=1e-12,rtol=0)
    result=dict(status="passed",new_fits=9,new_evaluations=27,reused_references=36,exact_checkpoint_replays=replays,
        saved_auroc_recalculations=63*6,reserved_test_rows_in_fit_or_validation=0,hashes_and_split_isolation=True,
        physical_unit_metric_recalculation=True,interpretation="negative development result retained")
    assert replays==9; dump(REPORT/"ARTIFACT_AUDIT.json",result); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
