"""Frozen checkpoint study: independent sampling seeds and nested 1x/5x pools."""
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

from iteration3 import ROOT, Study, dump, name, sha, spec
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import MixedDiffusion, guidance_schedule
from medhyperdiff.evaluation import evaluate

RUNS = ROOT / "runs/iteration4"
REPORT = ROOT / "reports/iteration4"
PLAN = ROOT / "configs/iteration4.json"
PROTOCOL = ROOT / "docs/ITERATION_4_PROTOCOL.md"
ARMS = {
    "previous_full": spec("full", "x0", .1, .2),
    "updated_full": spec("full", "v", .01, 10),
    "guided_base": spec("base", "v", 0, 10),
    "matched_dense": spec("dense", "v", .01, 10),
}


def missingness_audit(train, validation, synthetic, schema):
    return {c: dict(fit=float(train[c].isna().mean()), validation=float(validation[c].isna().mean()),
                    synthetic=float(synthetic[c].isna().mean()),
                    error_vs_fit=abs(float(synthetic[c].isna().mean()-train[c].isna().mean())),
                    error_vs_validation=abs(float(synthetic[c].isna().mean()-validation[c].isna().mean())))
            for c in schema.columns}


def rule_bounds(frame, constraints):
    n = len(frame)
    observed_union, unknown_union = np.zeros(n, bool), np.zeros(n, bool)
    result = {}
    for rule in constraints.rules:
        values = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy() for c in rule.columns}
        raw = np.asarray(constraints.residual(rule, values), float)
        valid = frame[list(rule.columns)].notna().all(axis=1).to_numpy() & np.isfinite(raw)
        violated = valid & (np.maximum(raw-rule.tolerance, 0)/rule.scale > 1e-6)
        observed_union |= violated
        unknown_union |= ~valid
        result[rule.name] = dict(eligible=int(valid.sum()), unknown=int((~valid).sum()),
            observed_violations=int(violated.sum()), coverage=float(valid.mean()),
            violation_lower=float(violated.mean()), violation_upper=float((violated | ~valid).mean()))
    return dict(rows=n, rules=result, any_rule=dict(observed_violations=int(observed_union.sum()),
        rows_with_unknown_rule=int(unknown_union.sum()), violation_lower=float(observed_union.mean()),
        violation_upper=float((observed_union | unknown_union).mean())))


def prepare():
    assert not PLAN.exists(), "Frozen plan already exists; run/resume it instead."
    study = Study()
    parents = []
    for inner in (2027, 3027, 4027):
        frame, split = study.data("ilpd", inner)
        for seed in (0, 1, 2):
            for arm, setting in ARMS.items():
                reference = ROOT / "runs/iteration3/evaluations" / f"ilpd_{name(setting)}_i{inner}_s{seed}"
                m = json.loads((reference / "manifest.json").read_text())
                parent = ROOT / m["parent_checkpoint"]
                pm = json.loads((parent.parent / "manifest.json").read_text())
                assert m["status"] == pm["status"] == "complete"
                assert m["source_sha256"] == pm["source_sha256"] == study.source
                assert not m["outer_test_opened"] and not pm["outer_test_opened"]
                assert m["setting"] == setting and m["seed"] == seed and m["inner_seed"] == inner
                assert sha(parent) == m["parent_checkpoint_sha256"]
                assert json.loads((parent.parent / "splits.json").read_text())["folds"][0] == split
                assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == m["data_sha256"]
                parents.append(dict(arm=arm, setting=setting, inner_seed=inner, training_seed=seed,
                    parent_checkpoint=m["parent_checkpoint"], parent_checkpoint_sha256=sha(parent),
                    reference_evaluation=str(reference.relative_to(ROOT))))
    snapshot = ROOT / "reproducibility/iteration4_scripts" / sha(Path(__file__))
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, snapshot / "iteration4.py")
    shutil.copy2(PROTOCOL, snapshot / PROTOCOL.name)
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), dataset="ilpd", parents=parents,
        sampling_seeds=[70001, 70002, 70003], multipliers=[1, 5], evaluator_seed=2026,
        source_sha256=study.source, runner_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
        new_training_runs=0, expected_pools=108, expected_evaluations=216)
    dump(PLAN, plan)
    shutil.copy2(PLAN, snapshot / PLAN.name)
    print(f"Frozen {len(parents)} parents, 108 pools, 216 evaluations.")


def run():
    plan = json.loads(PLAN.read_text())
    study, schema = Study(), SCHEMAS["ilpd"]
    assert study.source == plan["source_sha256"]
    assert sha(Path(__file__)) == plan["runner_sha256"] and sha(PROTOCOL) == plan["protocol_sha256"]
    outputs = []
    for p in plan["parents"]:
        parent = ROOT / p["parent_checkpoint"]
        assert sha(parent) == p["parent_checkpoint_sha256"]
        pm = json.loads((parent.parent / "manifest.json").read_text())
        model = MixedDiffusion.load(parent)
        model.config.guidance = p["setting"]["guidance"]
        model.config.schedule = "adaptive"
        model.schedule = guidance_schedule(model.config.steps, model.config.schedule)
        frame, split = study.data("ilpd", p["inner_seed"])
        train, validation = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        labels = np.tile((train[schema.target].to_numpy() == schema.positive).astype(int), 5)
        for sampling_seed in plan["sampling_seeds"]:
            pool_id = f"{p['arm']}_i{p['inner_seed']}_t{p['training_seed']}_s{sampling_seed}"
            pool_path = RUNS / "pools" / pool_id
            pool_csv = pool_path / "synthetic_5x.csv"
            identity = {**p, "sampling_seed": sampling_seed, "source_sha256": study.source,
                "runner_sha256": plan["runner_sha256"], "plan_sha256": sha(PLAN),
                "data_sha256": pm["data_sha256"], "training_config": pm["config"],
                "sampling_config": asdict(model.config), "sampling_batch_size": 256,
                "new_training_performed": False, "outer_test_opened": False,
                "evaluation_partition": "inner_validation", "parameters": pm["parameters"],
                "evaluator_seed": plan["evaluator_seed"]}
            if (pool_path / "manifest.json").exists():
                pool_m = json.loads((pool_path / "manifest.json").read_text())
                assert pool_m["status"] == "complete" and all(pool_m[k] == v for k, v in identity.items())
                assert sha(pool_csv) == pool_m["synthetic_sha256"]
            else:
                start = time.perf_counter()
                generated = model.sample(labels, seed=sampling_seed, batch_size=256)
                negative = next(v for v in train[schema.target].unique() if v != schema.positive)
                generated[schema.target] = np.where(labels == 1, schema.positive, negative)
                pool_path.mkdir(parents=True, exist_ok=True)
                generated.to_csv(pool_csv, index=False)
                pool_m = {**identity, "status": "complete", "sample_seconds": time.perf_counter()-start,
                    "synthetic_sha256": sha(pool_csv), "rows": len(generated),
                    "completed_at": datetime.now(timezone.utc).isoformat()}
                dump(pool_path / "manifest.json", pool_m)
                dump(pool_path / "split.json", split)
            # Evaluate exactly the saved numerical values, also on resumed runs.
            pool = pd.read_csv(pool_csv, dtype={schema.target: str}, float_precision="round_trip")
            for multiplier in plan["multipliers"]:
                out = RUNS / "evaluations" / f"{pool_id}_n{multiplier}"
                outputs.append(str(out.relative_to(ROOT)))
                if (out / "manifest.json").exists():
                    existing = json.loads((out / "manifest.json").read_text())
                    assert existing["status"] == "complete" and existing["plan_sha256"] == sha(PLAN)
                    continue
                generated = pool.iloc[:len(train)*multiplier].copy()
                out.mkdir(parents=True, exist_ok=True)
                generated.to_csv(out / "synthetic.csv", index=False)
                predictions = {}
                start = time.perf_counter()
                metrics = evaluate(train, validation, generated, schema, model.constraints,
                    seed=plan["evaluator_seed"], predictions=predictions)
                if multiplier != 1:
                    for kind in ("forest", "logistic"):
                        metrics["utility"][kind][f"AUGMENT_{multiplier}x"] = metrics["utility"][kind].pop("AUGMENT_1x")
                        predictions[f"{kind}_AUGMENT_{multiplier}x"] = predictions.pop(f"{kind}_AUGMENT_1x")
                metrics["feature_missingness"] = missingness_audit(train, validation, generated, schema)
                metrics["rule_missingness_bounds"] = {key: rule_bounds(table, model.constraints)
                    for key, table in (("fit", train), ("validation", validation), ("synthetic", generated))}
                dump(out / "metrics.json", metrics)
                pd.DataFrame(predictions).to_csv(out / "utility_predictions.csv", index=False)
                dump(out / "manifest.json", {**identity, "status": "complete", "multiplier": multiplier,
                    "pool": str(pool_path.relative_to(ROOT)), "pool_sha256": sha(pool_csv),
                    "rows": len(generated), "synthetic_sha256": sha(out / "synthetic.csv"),
                    "metrics_sha256": sha(out / "metrics.json"),
                    "predictions_sha256": sha(out / "utility_predictions.csv"),
                    "evaluation_seconds": time.perf_counter()-start})
                print(json.dumps(dict(completed=len(outputs), run=out.name,
                    rf=metrics["utility"]["forest"]["TSTR"]["auroc"])), flush=True)
    dump(REPORT / "registry.json", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    prepare() if args.stage == "prepare" else run()
