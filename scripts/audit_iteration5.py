"""Verify new fits, matched references, data isolation and saved measurements."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from iteration5 import ROOT, RUNS, REPORT, PLAN, PROTOCOL, MODEL_SCRIPT, Study, dump, sha, missingness_audit, rule_bounds
from iteration5_model import ObservationInitializedDiffusion
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import MixedDiffusion, guidance_schedule


plan, registry = [json.loads(p.read_text()) for p in (PLAN, REPORT/"registry.json")]
study, schema = Study(), SCHEMAS["ilpd"]
assert plan["source_sha256"] == study.source
for path, key in ((ROOT/"scripts/iteration5.py", "runner_sha256"), (MODEL_SCRIPT, "model_script_sha256"), (PROTOCOL, "protocol_sha256"),
                  (ROOT/"scripts/iteration4.py", "audit_helpers_sha256"), (ROOT/"scripts/iteration3.py", "study_helpers_sha256")):
    assert sha(path) == plan[key]
for path in (PLAN, PROTOCOL, MODEL_SCRIPT, ROOT/"scripts/iteration5.py"):
    snapshot = ROOT/"reproducibility/iteration5_scripts"/sha(path)/path.name
    assert sha(snapshot) == sha(path)
digest = hashlib.sha256()
for path in sorted((ROOT/"reproducibility/source_snapshots"/study.source/"medhyperdiff").glob("*.py")):
    digest.update(path.name.encode()); digest.update(path.read_bytes())
assert digest.hexdigest() == study.source
assert len(registry["new"]) == len(set(registry["new"])) == 108
assert len(registry["references"]) == len(set(registry["references"])) == 81
assert not set(registry["new"]) & set(registry["references"])
models, expected_new, expected_references = {}, set(), set()
for job in plan["jobs"]:
    parent = ROOT/job["parent_checkpoint"]
    assert sha(parent) == job["parent_checkpoint_sha256"]
    pm = json.loads((parent.parent/"manifest.json").read_text())
    out = RUNS/"training"/f"{job['variant']}_i{job['inner_seed']}_t{job['training_seed']}"
    m = json.loads((out/"manifest.json").read_text())
    assert all(m[k] == v for k, v in job.items())
    assert m["status"] == "complete" and not m["warm_started"] and not m["outer_test_opened"]
    assert m["training_config"] == pm["config"]
    assert m["parameters"] == pm["parameters"]
    assert m["plan_sha256"] == sha(PLAN) and m["model_script_sha256"] == sha(MODEL_SCRIPT)
    assert m["source_sha256"] == pm["source_sha256"] == study.source
    assert sha(out/"checkpoint.pt") == m["checkpoint_sha256"]
    assert sha(out/"history.json") == m["history_sha256"]
    history = json.loads((out/"history.json").read_text())
    assert m["selected_epoch"] == min(history, key=lambda h: h["validation_loss"])["epoch"]
    assert len(history) == m["epochs_executed"] <= pm["config"]["epochs"]
    frame, split = study.data("ilpd", job["inner_seed"])
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == m["data_sha256"] == pm["data_sha256"]
    assert json.loads((out/"split.json").read_text()) == split
    assert json.loads((parent.parent/"splits.json").read_text())["folds"][0] == split
    train = frame.iloc[split["train"]]
    expected = {}
    for col in schema.numeric:
        counts = [int(train[col].isna().sum()), int(train[col].notna().sum())]
        if min(counts) > 0:
            expected[col] = dict(counts=counts, probabilities=(np.array(counts)/len(train)).tolist())
    assert m["initialization"] == expected
    model = ObservationInitializedDiffusion.load(out/"checkpoint.pt")
    original = MixedDiffusion.load(parent)
    assert model.observation_initialization == expected
    assert model.observation_weight == job["observation_weight"]
    assert model.preprocessor.to_dict() == original.preprocessor.to_dict()
    assert model.constraints.to_dict() == original.constraints.to_dict()
    np.testing.assert_array_equal(model.model.graph.numpy(), original.model.graph.numpy())
    models[str((out/"checkpoint.pt").relative_to(ROOT))] = model
    models[job["parent_checkpoint"]] = original
    for s, ref in zip(plan["sampling_seeds"], job["references"]):
        path = str((RUNS/"evaluations"/f"{out.name}_s{s}_n5").relative_to(ROOT))
        expected_new.add(path)
        expected_references.add(ref["path"])
        assert sha(ROOT/ref["path"]/"metrics.json") == ref["metrics_sha256"]
assert set(registry["new"]) == expected_new and set(registry["references"]) == expected_references
assert len(models) == 63
for path in registry["new"]+registry["references"]:
    out = ROOT/path
    m = json.loads((out/"manifest.json").read_text())
    new = path in registry["new"]
    assert m["status"] == "complete" and not m["outer_test_opened"]
    assert m["evaluation_partition"] == "inner_validation" and m["multiplier"] == 5
    assert m["source_sha256"] == study.source and m["evaluator_seed"] == 2026
    assert m["sampling_batch_size"] == 256 and m["sampling_seed"] in plan["sampling_seeds"]
    model = models[m["trained_checkpoint"] if new else m["parent_checkpoint"]]
    if new:
        assert m["plan_sha256"] == sha(PLAN)
        assert sha(ROOT/m["trained_checkpoint"]) == m["checkpoint_sha256"]
        ref = json.loads((ROOT/m["reference_evaluation"]/"manifest.json").read_text())
        for key in ("inner_seed", "training_seed", "sampling_seed", "evaluator_seed", "sampling_batch_size", "multiplier", "rows"):
            assert m[key] == ref[key]
        assert m["sampling_config"] == ref["sampling_config"]
    frame, split = study.data("ilpd", m["inner_seed"])
    train, val = frame.iloc[split["train"]], frame.iloc[split["validation"]]
    for filename, key in (("synthetic.csv", "synthetic_sha256"), ("metrics.json", "metrics_sha256"), ("utility_predictions.csv", "predictions_sha256")):
        assert sha(out/filename) == m[key]
    synthetic = pd.read_csv(out/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
    assert len(synthetic) == 5*len(train) == m["rows"]
    assert synthetic[schema.target].tolist() == train[schema.target].tolist()*5
    metrics = json.loads((out/"metrics.json").read_text())
    assert metrics["feature_missingness"] == missingness_audit(train, val, synthetic, schema)
    assert metrics["constraints"]["synthetic"] == model.constraints.evaluate(synthetic)
    for population, table in (("fit", train), ("validation", val), ("synthetic", synthetic)):
        assert metrics["rule_missingness_bounds"][population] == rule_bounds(table, model.constraints)
    for col in schema.numeric:
        np.testing.assert_allclose(metrics["fidelity"]["per_feature"][col]["std_ratio_synthetic_to_real"],
                                   synthetic[col].std()/val[col].std(), atol=1e-12, rtol=0)
    preds = pd.read_csv(out/"utility_predictions.csv", float_precision="round_trip")
    assert preds.row_index.tolist() == split["validation"]
    assert preds.label.tolist() == (val[schema.target] == schema.positive).astype(int).tolist()
    for kind in ("forest", "logistic"):
        assert set(metrics["utility"][kind]) == {"TRTR", "TSTR", "AUGMENT_5x"}
        for category in metrics["utility"][kind]:
            np.testing.assert_allclose(roc_auc_score(preds.label, preds[f"{kind}_{category}"]),
                metrics["utility"][kind][category]["auroc"], atol=1e-12, rtol=0)
# Replay the previously problematic reference and both new full variants at that cell.
replays = ["runs/iteration4/evaluations/updated_full_i2027_t1_s70001_n5",
           "runs/iteration5/evaluations/full_prior_i2027_t1_s70001_n5",
           "runs/iteration5/evaluations/full_prior_w4_i2027_t1_s70001_n5"]
for path in replays:
    m = json.loads((ROOT/path/"manifest.json").read_text())
    model = MixedDiffusion.load(ROOT/m.get("trained_checkpoint", m["parent_checkpoint"]))
    model.config.guidance = 10.
    model.config.schedule = "adaptive"
    model.schedule = guidance_schedule(model.config.steps, "adaptive")
    frame, split = study.data("ilpd", 2027)
    labels = np.tile((frame.iloc[split["train"]][schema.target] == schema.positive).astype(int), 5)
    actual = model.sample(labels, seed=70001, batch_size=256)
    saved = pd.read_csv(ROOT/path/"synthetic.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(actual, saved[actual.columns], check_exact=True)
result = dict(status="passed", new_training_runs=36, new_evaluations=108, reused_reference_evaluations=81,
    fitted_models=63, fit_only_initialization_counts=True, unchanged_core_source=study.source,
    matched_preprocessing_graph_rules_capacity=True, matching_sampling_configs_and_label_order=True,
    original_reserved_test_rows_in_fit_or_validation=0, all_saved_auroc_reproduced=True,
    all_missingness_bounds_and_variance_ratios_recomputed=True, exact_sampler_replays=3)
dump(REPORT/"ARTIFACT_AUDIT.json", result)
print(json.dumps(result, indent=2))
