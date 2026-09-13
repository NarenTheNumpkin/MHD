"""Audit every wrong-rule re-evaluation against the exact iteration-7 tables."""
import json
import pandas as pd
from iteration9 import ROOT, REPORT, PLAN, FILES, checked_plan, dump, sha, variant_constraints
from iteration4 import rule_bounds
from medhyperdiff.data import SCHEMAS


def main():
    plan,study=checked_plan(); schema=SCHEMAS["ilpd"]; registry=json.loads((REPORT/"registry.json").read_text()); assert len(registry["records"])==1125
    source_by={(x["arm"],x["path"]):x for x in plan["source_tables"]}; seen=set()
    for entry in registry["records"]:
        path=ROOT/entry["path"]; m=json.loads((path/"manifest.json").read_text()); source=source_by[(entry["arm"],entry["source_path"])]
        source_path=ROOT/source["path"]; sm=json.loads((source_path/"manifest.json").read_text()); assert m["status"]=="complete" and not m["outer_test_opened"] and m["source_synthetic_sha256"]==sha(source_path/"synthetic.csv")
        for f,d in source["hashes"].items(): assert sha(source_path/f)==d
        frame,split=study.data("ilpd",m["inner_seed"]); train,val=frame.iloc[split["train"]],frame.iloc[split["validation"]]; synthetic=pd.read_csv(source_path/"synthetic.csv",dtype={schema.target:str},float_precision="round_trip")
        assert synthetic[schema.target].tolist()==train[schema.target].tolist()*5
        constraints=variant_constraints(schema,train,m["variant"]); values=json.loads((path/"metrics.json").read_text())
        for population,table in (("fit",train),("validation",val),("synthetic",synthetic)):
            assert values[population]["constraints"]==constraints.evaluate(table)
            assert values[population]["rule_bounds"]==rule_bounds(table,constraints)
        assert sha(path/"metrics.json")==m["metrics_sha256"]; seen.add(entry["path"])
    assert len(seen)==1125
    result=dict(status="passed",source_tables=225,variants=5,evaluations=1125,fits=0,sampling=0,source_hashes_match=True,split_isolation=True,rule_metrics_recalculated=True,outer_test_rows=0)
    dump(REPORT/"ARTIFACT_AUDIT.json",result); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
