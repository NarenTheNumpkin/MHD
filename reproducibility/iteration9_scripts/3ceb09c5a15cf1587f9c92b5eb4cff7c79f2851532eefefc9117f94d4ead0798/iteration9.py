"""Frozen evaluation-only sensitivity to omitted and intentionally reversed rules."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import pandas as pd

from iteration3 import ROOT, Study, dump, sha
from iteration4 import rule_bounds
from medhyperdiff.constraints import ConstraintSet, build_constraints
from medhyperdiff.data import SCHEMAS

PLAN, PROTOCOL = ROOT/"configs/iteration9.json", ROOT/"docs/ITERATION_9_PROTOCOL.md"
RUNS, REPORT = ROOT/"runs/iteration9", ROOT/"reports/iteration9"
FILES = [Path(__file__), PROTOCOL, ROOT/"scripts/iteration3.py", ROOT/"scripts/iteration4.py"]
VARIANTS = ("correct", "omit_protein_identity", "reverse_bilirubin_order", "reverse_albumin_order", "reverse_both")


def archive(path):
    target=ROOT/"reproducibility/iteration9_scripts"/sha(path)/path.name
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists(): shutil.copy2(path,target)
    assert sha(target)==sha(path)


def variant_constraints(schema, train, variant):
    base=build_constraints(schema,train)
    rules=[]
    for rule in base.rules:
        if variant in ("omit_protein_identity",) and rule.name=="protein_identity": continue
        if variant in ("reverse_bilirubin_order","reverse_both") and rule.name=="direct_le_total_bilirubin":
            rule=replace(rule,name="direct_le_total_bilirubin_REVERSED",columns=("total_bilirubin","direct_bilirubin"),status="sensitivity_intentionally_wrong")
        if variant in ("reverse_albumin_order","reverse_both") and rule.name=="albumin_le_total":
            rule=replace(rule,name="albumin_le_total_REVERSED",columns=("total_protein","albumin"),status="sensitivity_intentionally_wrong")
        rules.append(rule)
    return ConstraintSet(rules)


def entries(plan):
    out=[]
    for path in plan["source_tables"]:
        out.append(dict(arm=path["arm"],path=path["path"],hashes=path["hashes"]))
    return out


def prepare():
    assert not PLAN.exists(),"Plan already frozen"
    old=json.loads((ROOT/"configs/iteration7.json").read_text()); source=[]
    for path in old["references"]:
        source.append(dict(arm=path["arm"],path=path["path"],hashes=path["hashes"]))
    for path in old["jobs"] if "jobs" in old else []: pass
    # Iteration-7 registry contains the 144 new evaluation paths as strings.
    registry=json.loads((ROOT/"reports/iteration7/registry.json").read_text())
    for entry in registry["new"]:
        path = entry if isinstance(entry, str) else entry["path"]
        p=ROOT/path; m=json.loads((p/"manifest.json").read_text())
        source.append(dict(arm=m["arm"],path=path,hashes={f:sha(p/f) for f in ("manifest.json","metrics.json","synthetic.csv","utility_predictions.csv")}))
    assert len(source)==225
    study=Study(); dump(PLAN,dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256=study.source,protocol_sha256=sha(PROTOCOL),runner_sha256=sha(Path(__file__)),helper_hashes={str(p.relative_to(ROOT)):sha(p) for p in FILES[2:]},source_tables=source,variants=list(VARIANTS),expected_evaluations=225*len(VARIANTS)))
    for p in FILES+[PLAN]: archive(p)
    print("Frozen 225 tables and five rule variants: 1,125 evaluations, zero fits.")


def checked_plan():
    plan,study=json.loads(PLAN.read_text()),Study(); assert plan["source_sha256"]==study.source
    assert sha(Path(__file__))==plan["runner_sha256"] and sha(PROTOCOL)==plan["protocol_sha256"]
    for p,d in plan["helper_hashes"].items(): assert sha(ROOT/p)==d
    return plan,study


def run():
    plan,study=checked_plan(); schema=SCHEMAS["ilpd"]; out_registry=[]; start=time.perf_counter()
    for source in plan["source_tables"]:
        path=ROOT/source["path"]; m=json.loads((path/"manifest.json").read_text()); frame,split=study.data("ilpd",m["inner_seed"]); train,val=frame.iloc[split["train"]],frame.iloc[split["validation"]]
        synthetic=pd.read_csv(path/"synthetic.csv",dtype={schema.target:str},float_precision="round_trip")
        assert source["hashes"]["synthetic.csv"]==sha(path/"synthetic.csv") and synthetic[schema.target].tolist()==train[schema.target].tolist()*5
        for variant in plan["variants"]:
            constraints=variant_constraints(schema,train,variant)
            out=RUNS/"evaluations"/f"{Path(source['path']).name}_{variant}"
            out.mkdir(parents=True,exist_ok=True)
            populations={"fit":train,"validation":val,"synthetic":synthetic}
            metrics={p:{"constraints":constraints.evaluate(t),"rule_bounds":rule_bounds(t,constraints)} for p,t in populations.items()}
            dump(out/"metrics.json",metrics); dump(out/"manifest.json",dict(status="complete",arm=source["arm"],source_path=source["path"],variant=variant,inner_seed=m["inner_seed"],training_seed=m["training_seed"],sampling_seed=m["sampling_seed"],source_synthetic_sha256=sha(path/"synthetic.csv"),metrics_sha256=sha(out/"metrics.json"),source_sha256=study.source,outer_test_opened=False,evaluation_partition="inner_validation"))
            out_registry.append(dict(arm=source["arm"],source_path=source["path"],variant=variant,path=str(out.relative_to(ROOT))))
    dump(REPORT/"registry.json",dict(records=out_registry,elapsed_seconds=time.perf_counter()-start))
    print(f"Completed {len(out_registry)} rule-variant evaluations.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("prepare","run")); args=parser.parse_args(); prepare() if args.stage=="prepare" else run()
