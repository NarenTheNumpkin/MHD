"""Frozen fixed-variance sampler experiment with two generations of matched references."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from iteration3 import ROOT, Study, dump, sha
from iteration4 import missingness_audit, rule_bounds
from iteration6_sampler import VarianceSampler, distribution_shape
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import guidance_schedule
from medhyperdiff.evaluation import evaluate

RUNS, REPORT = ROOT/"runs/iteration6", ROOT/"reports/iteration6"
PLAN, PROTOCOL = ROOT/"configs/iteration6.json", ROOT/"docs/ITERATION_6_PROTOCOL.md"
SAMPLER = ROOT/"scripts/iteration6_sampler.py"


def archive(path):
    target = ROOT/"reproducibility/iteration6_scripts"/sha(path)/path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == path.read_bytes()
    else:
        shutil.copy2(path, target)


def name(architecture, mix, inner, training, sampling):
    return f"{architecture}_m{mix:g}_i{inner}_t{training}_s{sampling}".replace(".", "p")


def prepare():
    assert not PLAN.exists(), "Plan already frozen."
    study, parents = Study(), []
    for architecture in ("full", "dense", "base"):
        for inner in (2027, 3027, 4027):
            frame, split = study.data("ilpd", inner)
            for training in (0, 1, 2):
                parent = ROOT/"runs/iteration5/training"/f"{architecture}_prior_i{inner}_t{training}"/"checkpoint.pt"
                pm = json.loads((parent.parent/"manifest.json").read_text())
                assert pm["source_sha256"] == study.source and pm["status"] == "complete" and not pm["outer_test_opened"]
                assert sha(parent) == pm["checkpoint_sha256"] and pm["observation_weight"] == 1
                assert json.loads((parent.parent/"split.json").read_text()) == split
                references = []
                for sampling in (70001, 70002, 70003):
                    prior = ROOT/"runs/iteration5/evaluations"/f"{architecture}_prior_i{inner}_t{training}_s{sampling}_n5"
                    m = json.loads((prior/"manifest.json").read_text())
                    original = ROOT/m["reference_evaluation"]
                    old = json.loads((original/"manifest.json").read_text())
                    assert m["status"] == old["status"] == "complete"
                    assert not m["outer_test_opened"] and not old["outer_test_opened"]
                    for key in ("sampling_seed", "training_seed", "inner_seed", "rows", "evaluator_seed", "sampling_batch_size", "multiplier"):
                        assert m[key] == old[key]
                    references.append(dict(sampling_seed=sampling,
                        prior=str(prior.relative_to(ROOT)), prior_metrics_sha256=sha(prior/"metrics.json"),
                        original=str(original.relative_to(ROOT)), original_metrics_sha256=sha(original/"metrics.json")))
                parents.append(dict(architecture=architecture, inner_seed=inner, training_seed=training,
                    checkpoint=str(parent.relative_to(ROOT)), checkpoint_sha256=sha(parent), references=references))
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), parents=parents, variance_mixtures=[.5, 1.],
        source_sha256=study.source, runner_sha256=sha(Path(__file__)), sampler_sha256=sha(SAMPLER),
        protocol_sha256=sha(PROTOCOL), study_helpers_sha256=sha(ROOT/"scripts/iteration3.py"),
        audit_helpers_sha256=sha(ROOT/"scripts/iteration4.py"), sampling_seeds=[70001, 70002, 70003],
        multiplier=5, sampling_batch_size=256, evaluator_seed=2026, guidance=10.,
        new_training_runs=0, expected_new_evaluations=162, expected_reference_evaluations=162)
    dump(PLAN, plan)
    for path in (PLAN, PROTOCOL, SAMPLER, Path(__file__)):
        archive(path)
    print("Frozen 27 fitted parents, 162 new evaluations, 162 reused references; zero new fits.")


def run():
    plan, study, schema = json.loads(PLAN.read_text()), Study(), SCHEMAS["ilpd"]
    assert plan["source_sha256"] == study.source
    for path, key in ((Path(__file__), "runner_sha256"), (SAMPLER, "sampler_sha256"), (PROTOCOL, "protocol_sha256"),
                      (ROOT/"scripts/iteration3.py", "study_helpers_sha256"), (ROOT/"scripts/iteration4.py", "audit_helpers_sha256")):
        assert sha(path) == plan[key]
    outputs, references = [], {}
    for p in plan["parents"]:
        parent = ROOT/p["checkpoint"]
        assert sha(parent) == p["checkpoint_sha256"]
        pm = json.loads((parent.parent/"manifest.json").read_text())
        model = VarianceSampler.load(parent)
        model.config.guidance = 10.
        model.config.schedule = "adaptive"
        model.schedule = guidance_schedule(model.config.steps, "adaptive")
        frame, split = study.data("ilpd", p["inner_seed"])
        assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == pm["data_sha256"]
        train, validation = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        labels = np.tile((train[schema.target].to_numpy() == schema.positive).astype(int), 5)
        for ref in p["references"]:
            for kind in ("prior", "original"):
                path = ROOT/ref[kind]
                assert sha(path/"metrics.json") == ref[kind+"_metrics_sha256"]
                rm = json.loads((path/"manifest.json").read_text())
                assert sha(path/"synthetic.csv") == rm["synthetic_sha256"]
                output = REPORT/"reference_shapes"/(kind+"_"+path.name+".json")
                generated = pd.read_csv(path/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
                shape = distribution_shape(train, validation, generated, schema)
                dump(output, dict(reference=ref[kind], reference_synthetic_sha256=sha(path/"synthetic.csv"),
                    sampler_sha256=sha(SAMPLER), distribution_shape=shape))
                references[ref[kind]] = dict(kind=kind, architecture=p["architecture"],
                    path=ref[kind], shape=str(output.relative_to(ROOT)))
            for mix in plan["variance_mixtures"]:
                model.variance_mix = mix
                out = RUNS/"evaluations"/name(p["architecture"], mix, p["inner_seed"], p["training_seed"], ref["sampling_seed"])
                outputs.append(str(out.relative_to(ROOT)))
                identity = dict(architecture=p["architecture"], inner_seed=p["inner_seed"], training_seed=p["training_seed"],
                    sampling_seed=ref["sampling_seed"], variance_mix=mix, checkpoint=p["checkpoint"], checkpoint_sha256=p["checkpoint_sha256"],
                    prior_reference=ref["prior"], original_reference=ref["original"], source_sha256=study.source,
                    runner_sha256=plan["runner_sha256"], sampler_sha256=plan["sampler_sha256"], plan_sha256=sha(PLAN),
                    data_sha256=pm["data_sha256"], training_config=pm["training_config"], sampling_config=asdict(model.config),
                    sampling_batch_size=256, evaluator_seed=2026, multiplier=5, parameters=pm["parameters"],
                    new_training_performed=False, outer_test_opened=False, evaluation_partition="inner_validation", evaluation_version=2)
                if (out/"manifest.json").exists():
                    m = json.loads((out/"manifest.json").read_text())
                    assert m["status"] == "complete" and all(m[k] == v for k, v in identity.items())
                    for filename, key in (("synthetic.csv", "synthetic_sha256"), ("metrics.json", "metrics_sha256"), ("utility_predictions.csv", "predictions_sha256")):
                        assert sha(out/filename) == m[key]
                    continue
                out.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                synthetic = model.sample(labels, seed=ref["sampling_seed"], batch_size=256)
                negative = next(v for v in train[schema.target].unique() if v != schema.positive)
                synthetic[schema.target] = np.where(labels == 1, schema.positive, negative)
                synthetic.to_csv(out/"synthetic.csv", index=False)
                sample_seconds = time.perf_counter()-start
                synthetic = pd.read_csv(out/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
                predictions = {}
                start = time.perf_counter()
                metrics = evaluate(train, validation, synthetic, schema, model.constraints, 2026, predictions)
                for kind in ("forest", "logistic"):
                    metrics["utility"][kind]["AUGMENT_5x"] = metrics["utility"][kind].pop("AUGMENT_1x")
                    predictions[f"{kind}_AUGMENT_5x"] = predictions.pop(f"{kind}_AUGMENT_1x")
                metrics["feature_missingness"] = missingness_audit(train, validation, synthetic, schema)
                metrics["rule_missingness_bounds"] = {k: rule_bounds(v, model.constraints)
                    for k, v in (("fit", train), ("validation", validation), ("synthetic", synthetic))}
                metrics["distribution_shape"] = distribution_shape(train, validation, synthetic, schema)
                dump(out/"metrics.json", metrics)
                pd.DataFrame(predictions).to_csv(out/"utility_predictions.csv", index=False)
                dump(out/"split.json", split)
                dump(out/"manifest.json", {**identity, "status": "complete", "rows": len(synthetic),
                    "sample_seconds": sample_seconds, "evaluation_seconds": time.perf_counter()-start,
                    "synthetic_sha256": sha(out/"synthetic.csv"), "metrics_sha256": sha(out/"metrics.json"),
                    "predictions_sha256": sha(out/"utility_predictions.csv")})
                print(json.dumps(dict(completed=len(outputs), run=out.name, rf=metrics["utility"]["forest"]["TSTR"]["auroc"])), flush=True)
    dump(REPORT/"registry.json", dict(new=outputs, references=list(references.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run()
