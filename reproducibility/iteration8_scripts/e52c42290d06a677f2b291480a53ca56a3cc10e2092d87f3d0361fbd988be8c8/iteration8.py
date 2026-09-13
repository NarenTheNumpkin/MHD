"""Frozen fit-only asinh-MAD representation experiment."""
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
from iteration8_model import AsinhMADPreprocessor, RobustObservationDiffusion, TransformedConstraintSet
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import DiffusionConfig, guidance_schedule
from medhyperdiff.evaluation import evaluate
from medhyperdiff.structure import dependency_graph
from medhyperdiff.constraints import build_constraints

RUNS, REPORT = ROOT/"runs/iteration8", ROOT/"reports/iteration8"
PLAN, PROTOCOL = ROOT/"configs/iteration8.json", ROOT/"docs/ITERATION_8_PROTOCOL.md"
MODEL_SCRIPT = ROOT/"scripts/iteration8_model.py"
FILES = [Path(__file__), MODEL_SCRIPT, PROTOCOL, ROOT/"scripts/iteration3.py", ROOT/"scripts/iteration4.py", ROOT/"scripts/iteration6_sampler.py"]


def archive(path):
    target = ROOT/"reproducibility/iteration8_scripts"/sha(path)/path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    assert sha(target) == sha(path)


def prepare():
    assert not PLAN.exists(), "Plan already frozen"
    study = Study()
    jobs, refs, splits = [], [], {}
    for inner in (2027, 3027, 4027):
        frame, split = study.data("ilpd", inner)
        data_hash = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        splits[str(inner)] = dict(indices=split, data_sha256=data_hash)
        for training in (0, 1, 2):
            full = ROOT/"runs/iteration5/training"/f"full_prior_i{inner}_t{training}"/"manifest.json"
            fm = json.loads(full.read_text())
            jobs.append(dict(inner_seed=inner, training_seed=training,
                reference_config=fm["training_config"], reference_checkpoint=str(full.parent.relative_to(ROOT))))
        for training in (0, 1, 2):
            for sampling in (70001, 70002, 70003):
                path = ROOT/"runs/iteration5/evaluations"/f"full_prior_i{inner}_t{training}_s{sampling}_n5"
                m = json.loads((path/"manifest.json").read_text())
                assert m["status"] == "complete" and not m["outer_test_opened"] and m["data_sha256"] == data_hash
                refs.append(dict(arm="full_prior", path=str(path.relative_to(ROOT)),
                    hashes={f: sha(path/f) for f in ("manifest.json", "metrics.json", "synthetic.csv", "utility_predictions.csv")}))
        for sampling in (70001, 70002, 70003):
            path = ROOT/"runs/iteration7/evaluations"/f"copula_repair_i{inner}_tNone_s{sampling}_n5"
            m = json.loads((path/"manifest.json").read_text())
            assert m["status"] == "complete" and not m["outer_test_opened"] and m["data_sha256"] == data_hash
            refs.append(dict(arm="copula_repair", path=str(path.relative_to(ROOT)),
                hashes={f: sha(path/f) for f in ("manifest.json", "metrics.json", "synthetic.csv", "utility_predictions.csv")}))
    dump(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), source_sha256=study.source,
        runner_sha256=sha(Path(__file__)), model_script_sha256=sha(MODEL_SCRIPT), protocol_sha256=sha(PROTOCOL),
        helper_hashes={str(p.relative_to(ROOT)): sha(p) for p in FILES[3:]}, splits=splits, jobs=jobs, references=refs,
        training_seeds=[0, 1, 2], sampling_seeds=[70001, 70002, 70003], multiplier=5, evaluator_seed=2026,
        expected_new_fits=9, expected_new_evaluations=27, expected_reused_references=36,
        representation="asinh_mad_v1"))
    for path in FILES+[PLAN]:
        archive(path)
    print("Frozen 9 transformed fits, 27 new evaluations and 36 reused references.")


def checked_plan():
    plan, study = json.loads(PLAN.read_text()), Study()
    assert plan["source_sha256"] == study.source
    assert sha(Path(__file__)) == plan["runner_sha256"] and sha(MODEL_SCRIPT) == plan["model_script_sha256"] and sha(PROTOCOL) == plan["protocol_sha256"]
    for path, digest in plan["helper_hashes"].items():
        assert sha(ROOT/path) == digest
    return plan, study


def model_name(job):
    return f"asinh_mad_i{job['inner_seed']}_t{job['training_seed']}"


def fit_model(job, train, validation, study, pm):
    schema = SCHEMAS["ilpd"]
    prep = AsinhMADPreprocessor(schema).fit(train)
    base = build_constraints(schema, train)
    constraints = TransformedConstraintSet(base.rules)
    graph, graph_report = dependency_graph(train, prep, seed=2026, bootstraps=10)
    cfg = DiffusionConfig(**pm["reference_config"])
    model = RobustObservationDiffusion(prep, graph, constraints, cfg, observation_weight=1.)
    model.fit(train, validation)
    return model, graph_report


def run():
    plan, study = checked_plan()
    schema, new, refs = SCHEMAS["ilpd"], [], []
    ref_by_path = {r["path"]: r for r in plan["references"]}
    for job in plan["jobs"]:
        frame, split = study.data("ilpd", job["inner_seed"])
        train, validation = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        data_hash = plan["splits"][str(job["inner_seed"])]["data_sha256"]
        parent = RUNS/"training"/model_name(job)
        identity = dict(inner_seed=job["inner_seed"], training_seed=job["training_seed"], representation=plan["representation"],
            source_sha256=study.source, data_sha256=data_hash, outer_test_opened=False, evaluation_partition="inner_validation",
            protocol_sha256=sha(PROTOCOL), runner_sha256=sha(Path(__file__)), model_script_sha256=sha(MODEL_SCRIPT), plan_sha256=sha(PLAN))
        if (parent/"manifest.json").exists():
            pm = json.loads((parent/"manifest.json").read_text())
            assert pm["status"] == "complete" and all(pm[k] == v for k, v in identity.items()) and sha(parent/"checkpoint.pt") == pm["checkpoint_sha256"]
            model = torch.load(parent/"checkpoint.pt", map_location="cpu", weights_only=False)
            graph_report = json.loads((parent/"graph.json").read_text())
        else:
            parent.mkdir(parents=True, exist_ok=True)
            dump(parent/"attempt.json", dict(**identity, started_at=datetime.now(timezone.utc).isoformat()))
            print("FIT "+model_name(job), flush=True)
            start = time.perf_counter()
            model, graph_report = fit_model(job, train, validation, study, job)
            fit_seconds = time.perf_counter()-start
            torch.save(model, parent/"checkpoint.pt")
            dump(parent/"graph.json", graph_report)
            dump(parent/"preprocessing.json", model.preprocessor.to_dict())
            dump(parent/"constraints.json", model.constraints.to_dict())
            dump(parent/"history.json", model.history)
            pm = dict(**identity, status="complete", fit_seconds=fit_seconds, selected_epoch=model.best_epoch,
                epochs_executed=len(model.history), parameters=sum(p.numel() for p in model.model.parameters()),
                training_config=asdict(model.config), checkpoint_sha256=sha(parent/"checkpoint.pt"),
                metadata_hashes={f: sha(parent/f) for f in ("graph.json", "preprocessing.json", "constraints.json", "history.json")},
                python=platform.python_version(), platform=platform.platform(), torch=torch.__version__, threads=torch.get_num_threads())
            dump(parent/"manifest.json", pm)
            print(f"FIT COMPLETE {model_name(job)} {fit_seconds:.1f}s", flush=True)
        model.config.guidance = 10.
        model.config.schedule = "adaptive"
        model.schedule = guidance_schedule(model.config.steps, "adaptive")
        labels = np.tile((train[schema.target].to_numpy() == schema.positive).astype(int), 5)
        for sampling in plan["sampling_seeds"]:
            out = RUNS/"evaluations"/f"{model_name(job)}_s{sampling}_n5"
            new.append(str(out.relative_to(ROOT)))
            eid = dict(**identity, sampling_seed=sampling, evaluator_seed=2026, multiplier=5, rows=len(labels),
                checkpoint=str((parent/"checkpoint.pt").relative_to(ROOT)), checkpoint_sha256=pm["checkpoint_sha256"],
                sampling_config=asdict(model.config), sampling_batch_size=256)
            if (out/"manifest.json").exists():
                em=json.loads((out/"manifest.json").read_text()); assert em["status"] == "complete" and all(em[k] == v for k,v in eid.items())
                continue
            out.mkdir(parents=True, exist_ok=True)
            start=time.perf_counter(); synthetic=model.sample(labels, seed=sampling, batch_size=256)
            negative=next(v for v in train[schema.target].unique() if v != schema.positive)
            synthetic[schema.target]=np.where(labels==1,schema.positive,negative)
            synthetic.to_csv(out/"synthetic.csv",index=False); sample_seconds=time.perf_counter()-start
            synthetic=pd.read_csv(out/"synthetic.csv",dtype={schema.target:str},float_precision="round_trip")
            preds={}; start=time.perf_counter(); metrics=evaluate(train,validation,synthetic,schema,model.constraints,2026,preds)
            for kind in ("forest","logistic"):
                metrics["utility"][kind]["AUGMENT_5x"]=metrics["utility"][kind].pop("AUGMENT_1x")
                preds[f"{kind}_AUGMENT_5x"]=preds.pop(f"{kind}_AUGMENT_1x")
            metrics["feature_missingness"]=missingness_audit(train,validation,synthetic,schema)
            metrics["rule_missingness_bounds"]={k:rule_bounds(v,model.constraints) for k,v in (("fit",train),("validation",validation),("synthetic",synthetic))}
            metrics["distribution_shape"]=distribution_shape(train,validation,synthetic,schema)
            dump(out/"metrics.json",metrics); pd.DataFrame(preds).to_csv(out/"utility_predictions.csv",index=False); dump(out/"split.json",split)
            dump(out/"manifest.json",dict(**eid,status="complete",sample_seconds=sample_seconds,evaluation_seconds=time.perf_counter()-start,
                synthetic_sha256=sha(out/"synthetic.csv"),metrics_sha256=sha(out/"metrics.json"),predictions_sha256=sha(out/"utility_predictions.csv")))
            print(json.dumps(dict(completed=len(new),run=out.name,rf=metrics["utility"]["forest"]["TSTR"]["auroc"])),flush=True)
    for r in plan["references"]:
        assert all(sha(ROOT/r["path"]/f)==d for f,d in r["hashes"].items())
    dump(REPORT/"registry.json",dict(new=new,references=plan["references"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("prepare","run")); args=parser.parse_args(); prepare() if args.stage=="prepare" else run()
