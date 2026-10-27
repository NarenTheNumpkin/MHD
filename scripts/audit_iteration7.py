"""Verify artifacts, independent saved-score AUROCs and actual checkpoint replay."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from iteration7 import ROOT, RUNS, REPORT, PLAN, FILES, checked_plan, dump, sha, job_name, load_checkpoint, make_table
from iteration4 import missingness_audit, rule_bounds
from iteration6_sampler import distribution_shape
from medhyperdiff.constraints import build_constraints, repair_simple
from medhyperdiff.data import SCHEMAS


def main():
    plan, study = checked_plan()
    schema = SCHEMAS["ilpd"]
    registry = json.loads((REPORT/"registry.json").read_text())
    assert len(registry["new"]) == len({e["path"] for e in registry["new"]}) == 144
    assert len(registry["references"]) == 81
    for p in FILES+[PLAN]:
        assert sha(ROOT/"reproducibility/iteration7_scripts"/sha(p)/p.name) == sha(p)
    replay_count, expected, native_networks = 0, set(), 0
    for job in plan["jobs"]:
        parent = RUNS/"training"/job_name(job)
        pm = json.loads((parent/"manifest.json").read_text())
        assert pm["status"] == "complete" and not pm["outer_test_opened"] and pm["plan_sha256"] == sha(PLAN)
        assert all(pm[k] == v for k, v in job.items())
        for f, digest in {**pm["loss_hashes"], **pm["metadata_hashes"]}.items():
            assert sha(parent/f) == digest
        model = load_checkpoint(parent/"checkpoint.pt", pm["checkpoint_sha256"])
        native_networks += len(getattr(model, "models", {}))
        frame, split = study.data("ilpd", job["inner_seed"])
        assert json.loads((parent/"split.json").read_text()) == split
        train = frame.iloc[split["train"]]
        constraints = build_constraints(schema, train)
        assert json.loads((parent/"preprocessing.json").read_text()) == model.preprocessor.to_dict()
        assert json.loads((parent/"constraints.json").read_text()) == json.loads(json.dumps(constraints.to_dict()))
        labels = np.tile((train[schema.target] == schema.positive).astype(int), 5)
        replay = make_table(model, labels, train, 70001)
        path = RUNS/"evaluations"/(job_name(job)+"_s70001_n5")
        saved = pd.read_csv(path/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
        pd.testing.assert_frame_equal(replay, saved, check_exact=True)
        replay_count += 1
        for sampling in plan["sampling_seeds"]:
            for arm in ([job["arm"], "copula_repair"] if job["kind"] == "copula" else [job["arm"]]):
                out = RUNS/"evaluations"/f"{arm}_i{job['inner_seed']}_t{job['training_seed']}_s{sampling}_n5"
                expected.add(str(out.relative_to(ROOT)))
                em = json.loads((out/"manifest.json").read_text())
                assert em["checkpoint_sha256"] == pm["checkpoint_sha256"] and em["plan_sha256"] == sha(PLAN)
                assert em["arm"] == arm and em["sampling_seed"] == sampling
                assert em["inner_seed"] == job["inner_seed"] and em["training_seed"] == job["training_seed"]
                assert json.loads((out/"split.json").read_text()) == split
                if arm == "copula_repair":
                    original = RUNS/"evaluations"/(job_name(job)+f"_s{sampling}_n5")
                    raw = pd.read_csv(original/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
                    repaired, changed = repair_simple(raw, constraints)
                    actual = pd.read_csv(out/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
                    pd.testing.assert_frame_equal(repaired, actual, check_exact=True)
                    assert changed == em["changed_row_rate"]
                    replay_count += 1
        del model
    assert native_networks == 72 and expected == {e["path"] for e in registry["new"]}
    for ref in registry["references"]:
        for filename, digest in ref["hashes"].items():
            assert sha(ROOT/ref["path"]/filename) == digest
        assert sha(ROOT/ref["split_path"]) == ref["split_sha256"]
        m = json.loads((ROOT/ref["path"]/"manifest.json").read_text())
        assert sha(ROOT/m["trained_checkpoint"]) == m["checkpoint_sha256"]
    for entry in registry["new"]+registry["references"]:
        out = ROOT/entry["path"]
        m, v = [json.loads((out/f).read_text()) for f in ("manifest.json", "metrics.json")]
        assert m["status"] == "complete" and not m["outer_test_opened"] and m["source_sha256"] == study.source
        assert m["evaluator_seed"] == 2026 and m["multiplier"] == 5 and m["evaluation_partition"] == "inner_validation"
        frame, split = study.data("ilpd", m["inner_seed"])
        assert split == plan["splits"][str(m["inner_seed"])]["indices"]
        assert not set(split["test"]) & set(split["train"]+split["validation"])
        assert m["data_sha256"] == plan["splits"][str(m["inner_seed"])]["data_sha256"]
        train, val = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        constraints = build_constraints(schema, train)
        for f, key in (("synthetic.csv", "synthetic_sha256"), ("metrics.json", "metrics_sha256"), ("utility_predictions.csv", "predictions_sha256")):
            assert sha(out/f) == m[key]
        synthetic = pd.read_csv(out/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
        assert len(synthetic) == len(train)*5 == m["rows"]
        assert synthetic[schema.target].tolist() == train[schema.target].tolist()*5
        assert v["feature_missingness"] == missingness_audit(train, val, synthetic, schema)
        assert v["constraints"]["synthetic"] == constraints.evaluate(synthetic)
        for population, table in (("fit", train), ("validation", val), ("synthetic", synthetic)):
            assert v["rule_missingness_bounds"][population] == rule_bounds(table, constraints)
        shape = v.get("distribution_shape")
        if shape is None:
            shape = json.loads((REPORT/"reference_shapes"/(out.name+".json")).read_text())
        assert shape == distribution_shape(train, val, synthetic, schema)
        scores = pd.read_csv(out/"utility_predictions.csv", float_precision="round_trip")
        assert scores.row_index.tolist() == split["validation"]
        assert scores.label.tolist() == (val[schema.target] == schema.positive).astype(int).tolist()
        for model in ("forest", "logistic"):
            assert set(v["utility"][model]) == {"TRTR", "TSTR", "AUGMENT_5x"}
            for setting in v["utility"][model]:
                np.testing.assert_allclose(roc_auc_score(scores.label, scores[f"{model}_{setting}"]), v["utility"][model][setting]["auroc"], atol=1e-12, rtol=0)
    result = dict(status="passed", records=225, new_evaluations=144, reused_evaluations=81,
        newly_fitted_adapters_or_statistical_models=45, native_class_networks=72,
        exact_first_seed_checkpoint_replays=45, exact_repair_replays=9,
        saved_auroc_recalculations=225*6, reserved_test_rows_in_fit_or_validation=0,
        source_plan_data_checkpoint_sample_prediction_hashes_match=True,
        missingness_rule_coverage_and_shape_metrics_recalculated=True)
    assert replay_count == 54
    dump(REPORT/"ARTIFACT_AUDIT.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
