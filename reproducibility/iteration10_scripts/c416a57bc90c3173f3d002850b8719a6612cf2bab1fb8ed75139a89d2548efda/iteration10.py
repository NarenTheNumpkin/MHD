"""Frozen external-cohort portability study for the current full model."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch

from iteration3 import ROOT, Study, dump, sha
from iteration4 import missingness_audit, rule_bounds
from iteration6_sampler import distribution_shape
from iteration5_model import ObservationInitializedDiffusion
from medhyperdiff.constraints import build_constraints
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import DiffusionConfig, guidance_schedule
from medhyperdiff.evaluation import evaluate
from medhyperdiff.preprocessing import Preprocessor
from medhyperdiff.structure import dependency_graph

PLAN, PROTOCOL = ROOT/"configs/iteration10.json", ROOT/"docs/ITERATION_10_PROTOCOL.md"
RUNS, REPORT = ROOT/"runs/iteration10", ROOT/"reports/iteration10"
FILES=[Path(__file__),PROTOCOL,ROOT/"scripts/iteration3.py",ROOT/"scripts/iteration4.py",ROOT/"scripts/iteration6_sampler.py",ROOT/"scripts/iteration5_model.py"]
DATASETS=("ckd","heart_failure","diabetes")


def archive(path):
    target=ROOT/"reproducibility/iteration10_scripts"/sha(path)/path.name; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): shutil.copy2(path,target)
    assert sha(target)==sha(path)


def prepare():
    assert not PLAN.exists(),"Plan already frozen"
    study=Study(); jobs=[]
    for dataset in DATASETS:
        frame,split=study.data(dataset,2027); dh=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        for training in (0,1,2):
            jobs.append(dict(dataset=dataset,inner_seed=2027,training_seed=training,data_sha256=dh,split=split,
                config=dict(steps=30,epochs=100,batch_size=64,width=32,layers=2,heads=4,learning_rate=.001,patience=20,device="cpu",seed=training,clinical_weight=.01,guidance=0.,schedule="none",graph_enabled=True,hypergraph_enabled=True,unit_confidence=False,numeric_prediction="v",gated_structure=False,dense_structure=False,paired_initialization=True)))
    dump(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256=study.source,runner_sha256=sha(Path(__file__)),protocol_sha256=sha(PROTOCOL),helper_hashes={str(p.relative_to(ROOT)):sha(p) for p in FILES[2:]},datasets=list(DATASETS),jobs=jobs,sampling_seeds=[70001,70002,70003],evaluator_seed=2026,multiplier=5,expected_new_fits=9,expected_new_evaluations=27))
    for p in FILES+[PLAN]: archive(p)
    print("Frozen 9 external-cohort fits and 27 evaluations.")


def checked_plan():
    plan,study=json.loads(PLAN.read_text()),Study(); assert plan["source_sha256"]==study.source and sha(Path(__file__))==plan["runner_sha256"] and sha(PROTOCOL)==plan["protocol_sha256"]
    for p,d in plan["helper_hashes"].items(): assert sha(ROOT/p)==d
    return plan,study


def job_name(job): return f"full_{job['dataset']}_i{job['inner_seed']}_t{job['training_seed']}"


def run():
    plan,study=checked_plan(); registry=[]
    for job in plan["jobs"]:
        schema=SCHEMAS[job["dataset"]]; frame,split=study.data(job["dataset"],job["inner_seed"]); train,val=frame.iloc[split["train"]],frame.iloc[split["validation"]]
        assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()==job["data_sha256"] and split==job["split"]
        parent=RUNS/"training"/job_name(job); identity=dict(dataset=job["dataset"],inner_seed=job["inner_seed"],training_seed=job["training_seed"],source_sha256=study.source,data_sha256=job["data_sha256"],outer_test_opened=False,evaluation_partition="inner_validation",plan_sha256=sha(PLAN),runner_sha256=sha(Path(__file__)),protocol_sha256=sha(PROTOCOL))
        if (parent/"manifest.json").exists():
            pm=json.loads((parent/"manifest.json").read_text()); assert pm["status"]=="complete" and all(pm[k]==v for k,v in identity.items()) and sha(parent/"checkpoint.pt")==pm["checkpoint_sha256"]; model=ObservationInitializedDiffusion.load(parent/"checkpoint.pt")
        else:
            parent.mkdir(parents=True,exist_ok=True); dump(parent/"attempt.json",dict(**identity,started_at=datetime.now(timezone.utc).isoformat())); start=time.perf_counter(); prep=Preprocessor(schema).fit(train); constraints=build_constraints(schema,train); graph,graph_report=dependency_graph(train,prep,seed=2026,bootstraps=10); cfg=DiffusionConfig(**job["config"]); model=ObservationInitializedDiffusion(prep,graph,constraints,cfg); model.fit(train,val); fit_seconds=time.perf_counter()-start; model.save(parent/"checkpoint.pt"); dump(parent/"graph.json",graph_report); dump(parent/"split.json",split); dump(parent/"history.json",model.history); pm=dict(**identity,status="complete",fit_seconds=fit_seconds,selected_epoch=model.best_epoch,epochs_executed=len(model.history),training_config=asdict(model.config),checkpoint_sha256=sha(parent/"checkpoint.pt"),metadata_hashes={f:sha(parent/f) for f in ("graph.json","split.json","history.json")},parameters=sum(p.numel() for p in model.model.parameters()),python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,threads=torch.get_num_threads()); dump(parent/"manifest.json",pm); print(f"FIT COMPLETE {job_name(job)} {fit_seconds:.1f}s",flush=True)
        model.config.guidance=10.; model.config.schedule="adaptive"; model.schedule=guidance_schedule(model.config.steps,"adaptive"); labels=np.tile((train[schema.target].to_numpy()==schema.positive).astype(int),5)
        for sampling in plan["sampling_seeds"]:
            out=RUNS/"evaluations"/f"{job_name(job)}_s{sampling}_n5"; registry.append(str(out.relative_to(ROOT))); eid=dict(**identity,sampling_seed=sampling,evaluator_seed=2026,multiplier=5,rows=len(labels),checkpoint=str((parent/"checkpoint.pt").relative_to(ROOT)),checkpoint_sha256=pm["checkpoint_sha256"])
            if (out/"manifest.json").exists(): em=json.loads((out/"manifest.json").read_text()); assert em["status"]=="complete" and all(em[k]==v for k,v in eid.items()); continue
            out.mkdir(parents=True,exist_ok=True); start=time.perf_counter(); synthetic=model.sample(labels,seed=sampling,batch_size=256); negative=next(v for v in train[schema.target].unique() if v!=schema.positive); synthetic[schema.target]=np.where(labels==1,schema.positive,negative); synthetic.to_csv(out/"synthetic.csv",index=False); sample_seconds=time.perf_counter()-start; synthetic=pd.read_csv(out/"synthetic.csv",dtype={schema.target:str},float_precision="round_trip"); preds={}; start=time.perf_counter(); metrics=evaluate(train,val,synthetic,schema,model.constraints,2026,preds)
            for kind in ("forest","logistic"): metrics["utility"][kind]["AUGMENT_5x"]=metrics["utility"][kind].pop("AUGMENT_1x"); preds[f"{kind}_AUGMENT_5x"]=preds.pop(f"{kind}_AUGMENT_1x")
            metrics["feature_missingness"]=missingness_audit(train,val,synthetic,schema); metrics["rule_missingness_bounds"]={k:rule_bounds(v,model.constraints) for k,v in (("fit",train),("validation",val),("synthetic",synthetic))}; metrics["distribution_shape"]=distribution_shape(train,val,synthetic,schema); dump(out/"metrics.json",metrics); pd.DataFrame(preds).to_csv(out/"utility_predictions.csv",index=False); dump(out/"split.json",split); dump(out/"manifest.json",dict(**eid,status="complete",sample_seconds=sample_seconds,evaluation_seconds=time.perf_counter()-start,synthetic_sha256=sha(out/"synthetic.csv"),metrics_sha256=sha(out/"metrics.json"),predictions_sha256=sha(out/"utility_predictions.csv"))); print(json.dumps(dict(run=out.name,rf=metrics["utility"]["forest"]["TSTR"]["auroc"])),flush=True)
    dump(REPORT/"registry.json",dict(new=registry)); print(f"Completed {len(registry)} external-cohort evaluations.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("prepare","run")); args=parser.parse_args(); prepare() if args.stage=="prepare" else run()
