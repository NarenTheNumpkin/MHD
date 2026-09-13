"""Frozen observation-head training experiment with matched historical references."""
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
from iteration5_model import ObservationInitializedDiffusion
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import DiffusionConfig, MixedDiffusion, guidance_schedule
from medhyperdiff.evaluation import evaluate

RUNS, REPORT = ROOT / "runs/iteration5", ROOT / "reports/iteration5"
PLAN, PROTOCOL = ROOT / "configs/iteration5.json", ROOT / "docs/ITERATION_5_PROTOCOL.md"
VARIANTS = {"full_prior": ("updated_full", 1.), "full_prior_w4": ("updated_full", 4.),
            "dense_prior": ("matched_dense", 1.), "base_prior": ("guided_base", 1.)}
MODEL_SCRIPT = ROOT / "scripts/iteration5_model.py"


def archive(path):
    target = ROOT / "reproducibility/iteration5_scripts" / sha(path) / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == path.read_bytes()
    else:
        shutil.copy2(path, target)


def prepare():
    assert not PLAN.exists(), "Frozen plan already exists."
    study = Study()
    previous = json.loads((ROOT / "configs/iteration4.json").read_text())
    jobs = []
    for variant, (reference_arm, weight) in VARIANTS.items():
        for p in previous["parents"]:
            if p["arm"] != reference_arm:
                continue
            parent = ROOT / p["parent_checkpoint"]
            m = json.loads((parent.parent / "manifest.json").read_text())
            frame, split = study.data("ilpd", p["inner_seed"])
            assert sha(parent) == p["parent_checkpoint_sha256"] and m["source_sha256"] == study.source
            assert not m["outer_test_opened"] and m["status"] == "complete"
            assert json.loads((parent.parent / "splits.json").read_text())["folds"][0] == split
            refs = []
            for s in previous["sampling_seeds"]:
                ref = ROOT / "runs/iteration4/evaluations" / f"{p['arm']}_i{p['inner_seed']}_t{p['training_seed']}_s{s}_n5"
                rm = json.loads((ref / "manifest.json").read_text())
                assert rm["status"] == "complete" and not rm["outer_test_opened"]
                assert sha(ref / "metrics.json") == rm["metrics_sha256"]
                refs.append(dict(path=str(ref.relative_to(ROOT)), metrics_sha256=rm["metrics_sha256"]))
            jobs.append(dict(variant=variant, observation_weight=weight, reference_arm=reference_arm,
                inner_seed=p["inner_seed"], training_seed=p["training_seed"], references=refs,
                parent_checkpoint=p["parent_checkpoint"], parent_checkpoint_sha256=p["parent_checkpoint_sha256"]))
    dump(PLAN, dict(created_at=datetime.now(timezone.utc).isoformat(), jobs=jobs, source_sha256=study.source,
        runner_sha256=sha(Path(__file__)), model_script_sha256=sha(MODEL_SCRIPT), protocol_sha256=sha(PROTOCOL),
        audit_helpers_sha256=sha(ROOT / "scripts/iteration4.py"), study_helpers_sha256=sha(ROOT / "scripts/iteration3.py"),
        sampling_seeds=[70001, 70002, 70003], evaluator_seed=2026, multiplier=5,
        expected_new_fits=36, expected_new_evaluations=108, expected_reference_evaluations=81))
    for path in (Path(__file__), MODEL_SCRIPT, PROTOCOL, PLAN):
        archive(path)
    print("Frozen 36 new fits, 108 new evaluations and 81 reused reference evaluations.")


def run():
    plan, study, schema = json.loads(PLAN.read_text()), Study(), SCHEMAS["ilpd"]
    assert plan["source_sha256"] == study.source
    for path, key in ((Path(__file__), "runner_sha256"), (MODEL_SCRIPT, "model_script_sha256"), (PROTOCOL, "protocol_sha256"),
                      (ROOT / "scripts/iteration4.py", "audit_helpers_sha256"), (ROOT / "scripts/iteration3.py", "study_helpers_sha256")):
        assert sha(path) == plan[key]
    registry, references = [], set()
    for job in plan["jobs"]:
        parent = ROOT / job["parent_checkpoint"]
        assert sha(parent) == job["parent_checkpoint_sha256"]
        pm = json.loads((parent.parent / "manifest.json").read_text())
        frame, split = study.data("ilpd", job["inner_seed"])
        assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == pm["data_sha256"]
        train, val = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        identifier = f"{job['variant']}_i{job['inner_seed']}_t{job['training_seed']}"
        out = RUNS / "training" / identifier
        identity = {**job, "source_sha256": study.source, "runner_sha256": plan["runner_sha256"],
            "model_script_sha256": plan["model_script_sha256"], "plan_sha256": sha(PLAN),
            "data_sha256": pm["data_sha256"], "outer_test_opened": False,
            "evaluation_partition": "inner_validation", "warm_started": False}
        if (out / "manifest.json").exists():
            manifest = json.loads((out / "manifest.json").read_text())
            assert manifest["status"] == "complete" and all(manifest[k] == v for k, v in identity.items())
            assert sha(out / "checkpoint.pt") == manifest["checkpoint_sha256"]
            model = ObservationInitializedDiffusion.load(out / "checkpoint.pt")
        else:
            reference = MixedDiffusion.load(parent)
            # Only architecture, preprocessing and fit-derived graph/rules are reused, not learned weights.
            model = ObservationInitializedDiffusion(reference.preprocessor, reference.model.graph.cpu().numpy(),
                reference.constraints, DiffusionConfig(**pm["config"]), observation_weight=job["observation_weight"])
            out.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            model.fit(train, val)
            fit_seconds = time.perf_counter()-start
            model.save(out / "checkpoint.pt")
            dump(out / "history.json", model.history)
            dump(out / "split.json", split)
            manifest = {**identity, "status": "complete", "fit_seconds": fit_seconds,
                "training_config": asdict(model.config), "observation_weight": model.observation_weight,
                "initialization": model.observation_initialization, "selected_epoch": model.best_epoch,
                "epochs_executed": len(model.history), "parameters": sum(p.numel() for p in model.model.parameters()),
                "checkpoint_sha256": sha(out / "checkpoint.pt"), "history_sha256": sha(out / "history.json"),
                "python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "threads": torch.get_num_threads(),
                "completed_at": datetime.now(timezone.utc).isoformat()}
            dump(out / "manifest.json", manifest)
            print(json.dumps(dict(trained=identifier, selected_epoch=model.best_epoch, fit_seconds=fit_seconds)), flush=True)
        model.config.guidance = 10.
        model.config.schedule = "adaptive"
        model.schedule = guidance_schedule(model.config.steps, model.config.schedule)
        labels = np.tile((train[schema.target].to_numpy() == schema.positive).astype(int), 5)
        for sampling_seed, ref in zip(plan["sampling_seeds"], job["references"]):
            assert sha(ROOT / ref["path"] / "metrics.json") == ref["metrics_sha256"]
            references.add(ref["path"])
            evaluation = RUNS / "evaluations" / f"{identifier}_s{sampling_seed}_n5"
            registry.append(str(evaluation.relative_to(ROOT)))
            if (evaluation / "manifest.json").exists():
                em = json.loads((evaluation / "manifest.json").read_text())
                assert em["status"] == "complete" and em["plan_sha256"] == sha(PLAN)
                continue
            evaluation.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            synthetic = model.sample(labels, seed=sampling_seed, batch_size=256)
            negative = next(v for v in train[schema.target].unique() if v != schema.positive)
            synthetic[schema.target] = np.where(labels == 1, schema.positive, negative)
            synthetic.to_csv(evaluation / "synthetic.csv", index=False)
            sample_seconds = time.perf_counter()-start
            synthetic = pd.read_csv(evaluation / "synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
            preds = {}
            start = time.perf_counter()
            metrics = evaluate(train, val, synthetic, schema, model.constraints, plan["evaluator_seed"], preds)
            for kind in ("forest", "logistic"):
                metrics["utility"][kind]["AUGMENT_5x"] = metrics["utility"][kind].pop("AUGMENT_1x")
                preds[f"{kind}_AUGMENT_5x"] = preds.pop(f"{kind}_AUGMENT_1x")
            metrics["feature_missingness"] = missingness_audit(train, val, synthetic, schema)
            metrics["rule_missingness_bounds"] = {k: rule_bounds(v, model.constraints)
                for k, v in (("fit", train), ("validation", val), ("synthetic", synthetic))}
            dump(evaluation / "metrics.json", metrics)
            pd.DataFrame(preds).to_csv(evaluation / "utility_predictions.csv", index=False)
            dump(evaluation / "manifest.json", {**identity, "status": "complete", "sampling_seed": sampling_seed,
                "evaluator_seed": plan["evaluator_seed"], "sampling_config": asdict(model.config),
                "sampling_batch_size": 256, "multiplier": 5, "rows": len(synthetic),
                "trained_checkpoint": str((out / "checkpoint.pt").relative_to(ROOT)),
                "checkpoint_sha256": manifest["checkpoint_sha256"], "reference_evaluation": ref["path"],
                "sample_seconds": sample_seconds, "evaluation_seconds": time.perf_counter()-start,
                "metrics_sha256": sha(evaluation / "metrics.json"), "synthetic_sha256": sha(evaluation / "synthetic.csv"),
                "predictions_sha256": sha(evaluation / "utility_predictions.csv")})
            print(json.dumps(dict(evaluated=len(registry), run=evaluation.name,
                rf=metrics["utility"]["forest"]["TSTR"]["auroc"],
                missing=metrics["feature_missingness"]["ag_ratio"]["synthetic"])), flush=True)
    dump(REPORT / "registry.json", dict(new=registry, references=sorted(references)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run()
