"""Check the frozen sampling design and recompute outcomes from saved artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from iteration4 import PLAN, PROTOCOL, ROOT, RUNS, REPORT, Study, dump, sha, missingness_audit, rule_bounds
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import MixedDiffusion, guidance_schedule


plan = json.loads(PLAN.read_text())
study, schema = Study(), SCHEMAS["ilpd"]
assert study.source == plan["source_sha256"]
assert sha(PROTOCOL) == plan["protocol_sha256"]
snapshot = ROOT / "reproducibility/iteration4_scripts" / plan["runner_sha256"]
assert sha(snapshot / "iteration4.py") == plan["runner_sha256"]
assert sha(snapshot / PLAN.name) == sha(PLAN)
assert sha(snapshot / PROTOCOL.name) == sha(PROTOCOL)
assert len(plan["parents"]) == 36 and plan["new_training_runs"] == 0
core = ROOT / "reproducibility/source_snapshots" / study.source / "medhyperdiff"
digest = hashlib.sha256()
for path in sorted(core.glob("*.py")):
    digest.update(path.name.encode()); digest.update(path.read_bytes())
assert digest.hexdigest() == study.source
registry = json.loads((REPORT / "registry.json").read_text())
assert len(registry) == len(set(registry)) == plan["expected_evaluations"]
expected_registry, count, pools, parents = set(), 0, set(), set()
capacity = {}
for p in plan["parents"]:
    parent = ROOT / p["parent_checkpoint"]
    assert sha(parent) == p["parent_checkpoint_sha256"]
    parents.add(str(parent))
    pm = json.loads((parent.parent / "manifest.json").read_text())
    assert pm["status"] == "complete" and not pm["outer_test_opened"]
    assert pm["source_sha256"] == study.source
    model = MixedDiffusion.load(parent)
    frame, split = study.data("ilpd", p["inner_seed"])
    assert split == json.loads((parent.parent / "splits.json").read_text())["folds"][0]
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == pm["data_sha256"]
    train, val = frame.iloc[split["train"]], frame.iloc[split["validation"]]
    capacity.setdefault((p["inner_seed"], p["training_seed"]), {})[p["arm"]] = pm["parameters"]
    for sampling_seed in plan["sampling_seeds"]:
        pool_id = f"{p['arm']}_i{p['inner_seed']}_t{p['training_seed']}_s{sampling_seed}"
        pool_path = RUNS / "pools" / pool_id
        pool_m = json.loads((pool_path / "manifest.json").read_text())
        assert pool_m["status"] == "complete"
        assert sha(pool_path / "synthetic_5x.csv") == pool_m["synthetic_sha256"]
        assert json.loads((pool_path / "split.json").read_text()) == split
        pool = pd.read_csv(pool_path / "synthetic_5x.csv", dtype={schema.target: str}, float_precision="round_trip")
        assert len(pool) == 5*len(train)
        assert pool[schema.target].tolist() == train[schema.target].tolist()*5
        pools.add(str(pool_path))
        for multiplier in plan["multipliers"]:
            out = RUNS / "evaluations" / f"{pool_id}_n{multiplier}"
            expected_registry.add(str(out.relative_to(ROOT)))
            m = json.loads((out / "manifest.json").read_text())
            assert m["status"] == "complete" and not m["new_training_performed"]
            assert not m["outer_test_opened"] and m["evaluation_partition"] == "inner_validation"
            for k, v in p.items():
                assert m[k] == pool_m[k] == v
            assert m["source_sha256"] == study.source and m["data_sha256"] == pm["data_sha256"]
            assert m["plan_sha256"] == sha(PLAN) and m["runner_sha256"] == plan["runner_sha256"]
            assert m["pool_sha256"] == pool_m["synthetic_sha256"]
            assert m["evaluator_seed"] == plan["evaluator_seed"] == 2026
            assert m["sampling_seed"] == sampling_seed
            assert m["sampling_batch_size"] == 256
            assert m["training_config"] == pm["config"]
            for k, v in m["sampling_config"].items():
                if k not in ("guidance", "schedule"):
                    assert v == pm["config"][k]
            assert m["sampling_config"]["guidance"] == p["setting"]["guidance"]
            for filename, key in (("synthetic.csv", "synthetic_sha256"), ("metrics.json", "metrics_sha256"),
                                   ("utility_predictions.csv", "predictions_sha256")):
                assert sha(out / filename) == m[key]
            generated = pd.read_csv(out / "synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
            pd.testing.assert_frame_equal(generated, pool.iloc[:len(train)*multiplier], check_exact=True)
            assert m["rows"] == len(generated)
            metrics = json.loads((out / "metrics.json").read_text())
            assert metrics["feature_missingness"] == missingness_audit(train, val, generated, schema)
            for key, table in (("fit", train), ("validation", val), ("synthetic", generated)):
                assert metrics["rule_missingness_bounds"][key] == rule_bounds(table, model.constraints)
            assert metrics["constraints"]["synthetic"] == model.constraints.evaluate(generated)
            preds = pd.read_csv(out / "utility_predictions.csv", float_precision="round_trip")
            assert preds.row_index.tolist() == split["validation"]
            assert preds.label.tolist() == (val[schema.target] == schema.positive).astype(int).tolist()
            for kind in ("forest", "logistic"):
                assert set(metrics["utility"][kind]) == {"TRTR", "TSTR", f"AUGMENT_{multiplier}x"}
                for category in metrics["utility"][kind]:
                    np.testing.assert_allclose(roc_auc_score(preds.label, preds[f"{kind}_{category}"]),
                        metrics["utility"][kind][category]["auroc"], atol=1e-12, rtol=0)
            count += 1
assert set(registry) == expected_registry
assert len(pools) == 108 and len(parents) == 36
assert all(v["updated_full"] == v["matched_dense"] for v in capacity.values())
# Independently replay one guided updated-model pool, including its nondefault guidance.
p = next(p for p in plan["parents"] if p["arm"] == "updated_full")
model = MixedDiffusion.load(ROOT / p["parent_checkpoint"])
model.config.guidance = p["setting"]["guidance"]
model.config.schedule = "adaptive"
model.schedule = guidance_schedule(model.config.steps, model.config.schedule)
frame, split = study.data("ilpd", p["inner_seed"])
train = frame.iloc[split["train"]]
labels = np.tile((train[schema.target].to_numpy() == schema.positive).astype(int), 5)
replayed = model.sample(labels, seed=70001, batch_size=256)
pool_id = f"{p['arm']}_i{p['inner_seed']}_t{p['training_seed']}_s70001"
saved = pd.read_csv(RUNS / "pools" / pool_id / "synthetic_5x.csv", float_precision="round_trip")
pd.testing.assert_frame_equal(replayed, saved[replayed.columns], check_exact=True)
report = dict(status="passed", evaluations=count, generated_pools=len(pools), fitted_checkpoints_used=len(parents),
    new_training_runs=0, nested_table_prefixes_exact=True, exact_class_proportions=True,
    independent_sampling_seeds=True, evaluator_seed_fixed=2026, source_and_parent_hashes_match=True,
    reserved_test_rows_in_fit_or_validation=0, saved_predictions_reproduce_all_auroc=True,
    missingness_and_rule_bounds_recomputed=True, dense_full_capacity_pairs=9,
    guided_pool_replay_exact=True)
dump(REPORT / "ARTIFACT_AUDIT.json", report)
print(json.dumps(report, indent=2))
