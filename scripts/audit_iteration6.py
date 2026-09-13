"""Audit reused/new records and replay fixed-variance samples independently."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from iteration6 import ROOT, RUNS, REPORT, PLAN, PROTOCOL, SAMPLER, Study, dump, sha, name, missingness_audit, rule_bounds
from iteration6_sampler import VarianceSampler, distribution_shape
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import guidance_schedule


plan, registry = [json.loads(p.read_text()) for p in (PLAN, REPORT/"registry.json")]
study, schema = Study(), SCHEMAS["ilpd"]
assert plan["source_sha256"] == study.source and plan["new_training_runs"] == 0
assert not list(RUNS.rglob("checkpoint.pt"))
for path, key in ((ROOT/"scripts/iteration6.py", "runner_sha256"), (SAMPLER, "sampler_sha256"), (PROTOCOL, "protocol_sha256"),
                  (ROOT/"scripts/iteration3.py", "study_helpers_sha256"), (ROOT/"scripts/iteration4.py", "audit_helpers_sha256")):
    assert sha(path) == plan[key]
for path in (PLAN, PROTOCOL, SAMPLER, ROOT/"scripts/iteration6.py"):
    assert sha(ROOT/"reproducibility/iteration6_scripts"/sha(path)/path.name) == sha(path)
digest = hashlib.sha256()
for file in sorted((ROOT/"reproducibility/source_snapshots"/study.source/"medhyperdiff").glob("*.py")):
    digest.update(file.name.encode()); digest.update(file.read_bytes())
assert digest.hexdigest() == study.source
assert len(registry["new"]) == len(set(registry["new"])) == 162
assert len(registry["references"]) == len({r["path"] for r in registry["references"]}) == 162
models, expected_new, expected_references, capacity = {}, set(), set(), {}
for p in plan["parents"]:
    parent = ROOT/p["checkpoint"]
    assert sha(parent) == p["checkpoint_sha256"]
    pm = json.loads((parent.parent/"manifest.json").read_text())
    assert pm["status"] == "complete" and not pm["outer_test_opened"] and pm["observation_weight"] == 1
    assert pm["source_sha256"] == study.source
    models[p["checkpoint"]] = VarianceSampler.load(parent)
    capacity.setdefault((p["inner_seed"], p["training_seed"]), {})[p["architecture"]] = pm["parameters"]
    frame, split = study.data("ilpd", p["inner_seed"])
    assert json.loads((parent.parent/"split.json").read_text()) == split
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == pm["data_sha256"]
    for ref in p["references"]:
        for kind in ("prior", "original"):
            expected_references.add(ref[kind])
            assert sha(ROOT/ref[kind]/"metrics.json") == ref[kind+"_metrics_sha256"]
        old = json.loads((ROOT/ref["original"]/"manifest.json").read_text())
        assert sha(ROOT/old["parent_checkpoint"]) == old["parent_checkpoint_sha256"]
        models.setdefault(old["parent_checkpoint"], VarianceSampler.load(ROOT/old["parent_checkpoint"]))
        for mix in plan["variance_mixtures"]:
            path = str((RUNS/"evaluations"/name(p["architecture"], mix, p["inner_seed"], p["training_seed"], ref["sampling_seed"])).relative_to(ROOT))
            expected_new.add(path)
            m = json.loads((ROOT/path/"manifest.json").read_text())
            assert m["checkpoint"] == p["checkpoint"] and m["checkpoint_sha256"] == p["checkpoint_sha256"]
            assert m["variance_mix"] == mix and m["sampling_seed"] == ref["sampling_seed"]
            assert m["architecture"] == p["architecture"] and m["inner_seed"] == p["inner_seed"] and m["training_seed"] == p["training_seed"]
            assert m["prior_reference"] == ref["prior"] and m["original_reference"] == ref["original"]
            assert m["training_config"] == pm["training_config"] and not m["new_training_performed"]
            assert m["plan_sha256"] == sha(PLAN) and m["sampler_sha256"] == sha(SAMPLER)
            prior = json.loads((ROOT/ref["prior"]/"manifest.json").read_text())
            assert m["sampling_config"] == prior["sampling_config"]
            assert json.loads((ROOT/path/"split.json").read_text()) == split
assert expected_new == set(registry["new"])
assert expected_references == {r["path"] for r in registry["references"]}
assert len(models) == 54
assert all(v["full"] == v["dense"] for v in capacity.values())
entries = [dict(path=p, kind="new") for p in registry["new"]]+registry["references"]
for entry in entries:
    out = ROOT/entry["path"]
    m = json.loads((out/"manifest.json").read_text())
    assert m["status"] == "complete" and not m["outer_test_opened"]
    assert m["evaluation_partition"] == "inner_validation" and m["source_sha256"] == study.source
    assert m["evaluator_seed"] == 2026 and m["sampling_batch_size"] == 256 and m["multiplier"] == 5
    parent = m["checkpoint"] if entry["kind"] == "new" else m["trained_checkpoint"] if entry["kind"] == "prior" else m["parent_checkpoint"]
    model = models[parent]
    frame, split = study.data("ilpd", m["inner_seed"])
    train, val = frame.iloc[split["train"]], frame.iloc[split["validation"]]
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == m["data_sha256"]
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
    if entry["kind"] == "new":
        shape = metrics["distribution_shape"]
    else:
        saved = json.loads((ROOT/entry["shape"]).read_text())
        assert saved["reference_synthetic_sha256"] == sha(out/"synthetic.csv")
        assert saved["sampler_sha256"] == sha(SAMPLER) and saved["reference"] == entry["path"]
        shape = saved["distribution_shape"]
    assert shape == distribution_shape(train, val, synthetic, schema)
    for col in schema.numeric:
        np.testing.assert_allclose(metrics["fidelity"]["per_feature"][col]["std_ratio_synthetic_to_real"],
            synthetic[col].std()/val[col].std(), atol=1e-12, rtol=0)
    predictions = pd.read_csv(out/"utility_predictions.csv", float_precision="round_trip")
    assert predictions.row_index.tolist() == split["validation"]
    assert predictions.label.tolist() == (val[schema.target] == schema.positive).astype(int).tolist()
    for kind in ("forest", "logistic"):
        assert set(metrics["utility"][kind]) == {"TRTR", "TSTR", "AUGMENT_5x"}
        for category in metrics["utility"][kind]:
            np.testing.assert_allclose(roc_auc_score(predictions.label, predictions[f"{kind}_{category}"]),
                metrics["utility"][kind][category]["auroc"], atol=1e-12, rtol=0)
# Exact full-model replays at the earlier missingness cell and the AST-failure partition.
replays = 0
for inner, training in ((2027, 1), (4027, 0)):
    checkpoint = f"runs/iteration5/training/full_prior_i{inner}_t{training}/checkpoint.pt"
    model = VarianceSampler.load(ROOT/checkpoint)
    model.config.guidance = 10.
    model.config.schedule = "adaptive"
    model.schedule = guidance_schedule(model.config.steps, "adaptive")
    frame, split = study.data("ilpd", inner)
    labels = np.tile((frame.iloc[split["train"]][schema.target] == schema.positive).astype(int), 5)
    for mix in (0., .5, 1.):
        model.variance_mix = mix
        generated = model.sample(labels, seed=70001, batch_size=256)
        path = (ROOT/"runs/iteration5/evaluations"/f"full_prior_i{inner}_t{training}_s70001_n5" if mix == 0 else
                RUNS/"evaluations"/name("full", mix, inner, training, 70001))
        saved = pd.read_csv(path/"synthetic.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(generated, saved[generated.columns], check_exact=True)
        replays += 1
result = dict(status="passed", new_training_runs=0, generation_checkpoints=27, reference_models=54,
    new_evaluations=162, reused_evaluations=162, comparison_records=324,
    source_parent_plan_hashes_match=True, fixed_weights_and_matched_sampling_settings=True,
    reserved_test_rows_in_fit_or_validation=0, all_saved_auroc_reproduced=True,
    all_missingness_variance_quantile_and_tail_metrics_recomputed=True, matched_full_dense_pairs=9,
    exact_sampler_replays=replays, zero_mixture_matches_original=True)
dump(REPORT/"ARTIFACT_AUDIT.json", result)
print(json.dumps(result, indent=2))
