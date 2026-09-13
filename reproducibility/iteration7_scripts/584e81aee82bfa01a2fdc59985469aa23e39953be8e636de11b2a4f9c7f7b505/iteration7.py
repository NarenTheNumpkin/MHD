"""Matched baseline development study; frozen plan, resumable checked outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import time

import ctgan
import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN, TVAE

from iteration3 import ROOT, Study, dump, sha
from iteration4 import missingness_audit, rule_bounds
from iteration6_sampler import distribution_shape
from iteration7_model import NativeAdapter, settings
from medhyperdiff.baselines import ReferenceGenerator
from medhyperdiff.constraints import build_constraints, repair_simple
from medhyperdiff.data import SCHEMAS
from medhyperdiff.evaluation import evaluate

RUNS, REPORT = ROOT/"runs/iteration7", ROOT/"reports/iteration7"
PLAN, PROTOCOL = ROOT/"configs/iteration7.json", ROOT/"docs/ITERATION_7_PROTOCOL.md"
FILES = [Path(__file__), ROOT/"scripts/iteration7_model.py", PROTOCOL,
         ROOT/"scripts/iteration3.py", ROOT/"scripts/iteration4.py", ROOT/"scripts/iteration6_sampler.py"]


def archive(path):
    target = ROOT/"reproducibility/iteration7_scripts"/sha(path)/path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    assert sha(target) == sha(path)


def environment():
    package_root = Path(ctgan.__file__).parent
    sources = {str(p.relative_to(package_root)): sha(p) for p in sorted(package_root.rglob("*.py"))}
    return dict(versions={p: importlib.metadata.version(p) for p in
        ("ctgan", "torch", "numpy", "pandas", "scikit-learn", "scipy", "rdt")}, ctgan_sources=sources)


def prepare():
    assert not PLAN.exists(), "Plan already frozen"
    study, references, jobs, splits = Study(), [], [], {}
    for inner in (2027, 3027, 4027):
        frame, split = study.data("ilpd", inner)
        splits[str(inner)] = dict(indices=split, data_sha256=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest())
        for kind in ("copula", "independent", "bootstrap"):
            jobs.append(dict(arm=kind, kind=kind, preset=None, inner_seed=inner, training_seed=None))
        for kind in ("ctgan", "tvae"):
            for preset in ("compact100", "default300"):
                for training in (0, 1, 2):
                    jobs.append(dict(arm=kind+"_"+preset, kind=kind, preset=preset, inner_seed=inner, training_seed=training))
        for arm in ("full_prior", "dense_prior", "base_prior"):
            for training in (0, 1, 2):
                for sampling in (70001, 70002, 70003):
                    out = ROOT/"runs/iteration5/evaluations"/f"{arm}_i{inner}_t{training}_s{sampling}_n5"
                    m = json.loads((out/"manifest.json").read_text())
                    assert m["status"] == "complete" and not m["outer_test_opened"]
                    parent_split = ROOT/m["trained_checkpoint"]
                    assert json.loads((parent_split.parent/"split.json").read_text()) == split
                    assert m["source_sha256"] == study.source and m["multiplier"] == 5 and m["evaluator_seed"] == 2026
                    references.append(dict(arm=arm, path=str(out.relative_to(ROOT)),
                        split_path=str((parent_split.parent/"split.json").relative_to(ROOT)), split_sha256=sha(parent_split.parent/"split.json"),
                        hashes={f: sha(out/f) for f in ("manifest.json", "metrics.json", "synthetic.csv", "utility_predictions.csv")}))
    defaults = {k: {n: p.default for n, p in inspect.signature(cls).parameters.items()} for k, cls in (("ctgan", CTGAN), ("tvae", TVAE))}
    dump(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), source_sha256=study.source,
        files={str(p.relative_to(ROOT)): sha(p) for p in FILES}, environment=environment(), native_defaults=defaults,
        splits=splits, jobs=jobs, references=references, sampling_seeds=[70001, 70002, 70003], multiplier=5,
        evaluator_seed=2026, new_neural_adapter_fits=36, new_native_networks=72, new_statistical_fits=9,
        expected_new_evaluations=144, expected_reference_evaluations=81))
    for path in FILES+[PLAN]:
        archive(path)
    print("Frozen 45 fits (36 neural adapters / 72 native networks + 9 statistical), 144 new evaluations, 81 reused.")


def checked_plan():
    plan, study = json.loads(PLAN.read_text()), Study()
    assert plan["source_sha256"] == study.source and plan["environment"] == environment()
    for path, digest in plan["files"].items():
        assert sha(ROOT/path) == digest, path
    return plan, study


def job_name(job):
    return f"{job['arm']}_i{job['inner_seed']}_t{job['training_seed']}"


def load_checkpoint(path, expected_sha):
    assert sha(path) == expected_sha
    return torch.load(path, map_location="cpu", weights_only=False)


def make_table(model, labels, train, seed):
    schema = SCHEMAS["ilpd"]
    generated = model.sample(labels, seed=seed)
    negative = next(v for v in train[schema.target].unique() if v != schema.positive)
    generated[schema.target] = np.where(labels == 1, schema.positive, negative)
    return generated


def run():
    plan, study = checked_plan()
    schema, outputs = SCHEMAS["ilpd"], []
    for job in plan["jobs"]:
        frame, split = study.data("ilpd", job["inner_seed"])
        assert split == plan["splits"][str(job["inner_seed"])]["indices"]
        assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == plan["splits"][str(job["inner_seed"])]["data_sha256"]
        train, validation = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        constraints = build_constraints(schema, train)
        labels = np.tile((train[schema.target].to_numpy() == schema.positive).astype(int), 5)
        parent = RUNS/"training"/job_name(job)
        identity = dict(**job, plan_sha256=sha(PLAN), source_sha256=study.source,
            data_sha256=plan["splits"][str(job["inner_seed"])]["data_sha256"], outer_test_opened=False)
        if (parent/"manifest.json").exists():
            pm = json.loads((parent/"manifest.json").read_text())
            assert pm["status"] == "complete" and all(pm[k] == v for k, v in identity.items())
            model = load_checkpoint(parent/"checkpoint.pt", pm["checkpoint_sha256"])
        else:
            parent.mkdir(parents=True, exist_ok=True)
            dump(parent/"attempt.json", dict(**identity, started_at=datetime.now(timezone.utc).isoformat()))
            print("FIT "+job_name(job), flush=True)
            start = time.perf_counter()
            model = (NativeAdapter(schema, job["kind"], job["preset"], job["training_seed"])
                if job["preset"] else ReferenceGenerator(schema, job["kind"]))
            model.fit(train)
            fit_seconds = time.perf_counter()-start
            torch.save(model, parent/"checkpoint.pt")
            losses = {}
            for label, native in getattr(model, "models", {}).items():
                path = parent/f"loss_label{label}.csv"
                native.loss_values.to_csv(path, index=False)
                losses[path.name] = sha(path)
            dump(parent/"split.json", split)
            dump(parent/"preprocessing.json", model.preprocessor.to_dict())
            dump(parent/"constraints.json", constraints.to_dict())
            pm = dict(**identity, status="complete", fit_seconds=fit_seconds,
                checkpoint_sha256=sha(parent/"checkpoint.pt"), loss_hashes=losses,
                native_constructor_overrides=settings(job["kind"], job["preset"]) if job["preset"] else None,
                metadata_hashes={f: sha(parent/f) for f in ("split.json", "preprocessing.json", "constraints.json")})
            dump(parent/"manifest.json", pm)
            print(f"FIT COMPLETE {job_name(job)} {fit_seconds:.1f}s", flush=True)
        for sampling in plan["sampling_seeds"]:
            raw = None
            for repaired in ([False, True] if job["kind"] == "copula" else [False]):
                arm = "copula_repair" if repaired else job["arm"]
                out = RUNS/"evaluations"/f"{arm}_i{job['inner_seed']}_t{job['training_seed']}_s{sampling}_n5"
                outputs.append(dict(path=str(out.relative_to(ROOT)), arm=arm))
                eid = {**identity, "arm": arm, "sampling_seed": sampling, "multiplier": 5, "evaluator_seed": 2026,
                    "checkpoint": str((parent/"checkpoint.pt").relative_to(ROOT)), "checkpoint_sha256": pm["checkpoint_sha256"],
                    "evaluation_partition": "inner_validation", "evaluation_version": 2, "repaired": repaired}
                if (out/"manifest.json").exists():
                    m = json.loads((out/"manifest.json").read_text())
                    assert m["status"] == "complete" and all(m[k] == v for k, v in eid.items())
                    for filename, key in (("synthetic.csv", "synthetic_sha256"), ("metrics.json", "metrics_sha256"), ("utility_predictions.csv", "predictions_sha256")):
                        assert sha(out/filename) == m[key]
                    if not repaired:
                        raw = pd.read_csv(out/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
                    continue
                out.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                synthetic, changed = repair_simple(raw, constraints) if repaired else (make_table(model, labels, train, sampling), 0.)
                synthetic.to_csv(out/"synthetic.csv", index=False)
                sample_seconds = time.perf_counter()-start
                synthetic = pd.read_csv(out/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
                if not repaired:
                    raw = synthetic
                start, predictions = time.perf_counter(), {}
                metrics = evaluate(train, validation, synthetic, schema, constraints, 2026, predictions)
                for kind in ("forest", "logistic"):
                    metrics["utility"][kind]["AUGMENT_5x"] = metrics["utility"][kind].pop("AUGMENT_1x")
                    predictions[f"{kind}_AUGMENT_5x"] = predictions.pop(f"{kind}_AUGMENT_1x")
                metrics["feature_missingness"] = missingness_audit(train, validation, synthetic, schema)
                metrics["rule_missingness_bounds"] = {k: rule_bounds(v, constraints) for k, v in (("fit", train), ("validation", validation), ("synthetic", synthetic))}
                metrics["distribution_shape"] = distribution_shape(train, validation, synthetic, schema)
                dump(out/"metrics.json", metrics)
                pd.DataFrame(predictions).to_csv(out/"utility_predictions.csv", index=False)
                dump(out/"split.json", split)
                dump(out/"manifest.json", dict(**eid, status="complete", rows=len(synthetic), changed_row_rate=changed,
                    sample_seconds=sample_seconds, evaluation_seconds=time.perf_counter()-start,
                    synthetic_sha256=sha(out/"synthetic.csv"), metrics_sha256=sha(out/"metrics.json"), predictions_sha256=sha(out/"utility_predictions.csv")))
                print(json.dumps(dict(completed=len(outputs), run=out.name, rf=metrics["utility"]["forest"]["TSTR"]["auroc"])), flush=True)
    for ref in plan["references"]:
        for filename, digest in ref["hashes"].items():
            assert sha(ROOT/ref["path"]/filename) == digest
    dump(REPORT/"registry.json", dict(new=outputs, references=plan["references"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run()
