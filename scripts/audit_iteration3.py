"""Verify each evaluation's source, fitted parent, split isolation and metric scores."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from iteration3 import ROOT, RUNS, REPORT, Study, dump, sha


study = Study()
parents, counts, parameters = set(), 0, {}
for path in sorted((RUNS / "evaluations").glob("*/manifest.json")):
    m = json.loads(path.read_text())
    assert m["status"] == "complete", str(path)
    assert not m["outer_test_opened"] and m["evaluation_partition"] == "inner_validation"
    parent = ROOT / m["parent_checkpoint"]
    assert sha(parent) == m["parent_checkpoint_sha256"]
    parents.add(m["parent_checkpoint"])
    pm = json.loads((parent.parent / "manifest.json").read_text())
    assert pm["status"] == "complete" and not pm["outer_test_opened"]
    assert pm["source_sha256"] == m["source_sha256"] == study.source
    assert m["training_config"] == pm["config"]
    # Only runtime sampling guidance/schedule may differ from the fitted configuration.
    for key, value in m["sampling_config"].items():
        if key not in ("guidance", "schedule"):
            assert value == pm["config"][key], (path, key)
    assert m["sampling_config"]["guidance"] == m["setting"]["guidance"]
    assert m["sampling_config"]["numeric_prediction"] == m["setting"]["prediction"]
    assert m["sampling_config"]["clinical_weight"] == m["setting"]["clinical_weight"]
    frame, expected = study.data(m["dataset"], m["inner_seed"])
    split = json.loads((path.parent / "split.json").read_text())
    assert split == expected
    assert split == json.loads((parent.parent / "splits.json").read_text())["folds"][0]
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == m["data_sha256"]
    preds = pd.read_csv(path.parent / "utility_predictions.csv", float_precision="round_trip")
    assert preds.row_index.tolist() == split["validation"]
    from medhyperdiff.data import SCHEMAS
    schema = SCHEMAS[m["dataset"]]
    expected_y = (frame.iloc[split["validation"]][schema.target] == schema.positive).astype(int)
    assert preds.label.tolist() == expected_y.tolist()
    metrics = json.loads((path.parent / "metrics.json").read_text())
    for kind in ("forest", "logistic"):
        assert np.isclose(roc_auc_score(preds.label, preds[f"{kind}_TSTR"]),
                          metrics["utility"][kind]["TSTR"]["auroc"], rtol=0, atol=1e-12)
    synthetic = pd.read_csv(path.parent / "synthetic.csv", dtype={schema.target: str})
    assert len(synthetic) == len(split["train"])
    assert (synthetic[schema.target] == schema.positive).sum() == (frame.iloc[split["train"]][schema.target] == schema.positive).sum()
    runner = ROOT / "reproducibility" / "iteration3_scripts" / m["runner_sha256"] / "iteration3.py"
    assert sha(runner) == m["runner_sha256"]
    snapshot = ROOT / "reproducibility" / "source_snapshots" / study.source / "medhyperdiff"
    digest = hashlib.sha256()
    for file in sorted(snapshot.glob("*.py")):
        digest.update(file.name.encode()); digest.update(file.read_bytes())
    assert digest.hexdigest() == study.source
    if m["setting"]["mode"] in ("full", "dense"):
        key = (m["dataset"], m["inner_seed"], m["seed"], m["setting"]["prediction"], m["setting"]["clinical_weight"])
        parameters.setdefault(key, {})[m["setting"]["mode"]] = m["parameters"]
    counts += 1
matched = 0
for key, modes in parameters.items():
    if set(modes) == {"full", "dense"}:
        assert modes["full"] == modes["dense"]
        matched += 1
new_train = list((RUNS / "training").glob("*/manifest.json"))
assert all(json.loads(p.read_text())["status"] == "complete" for p in new_train)
for registry in ("screen", "replicate", "controls"):
    paths = json.loads((REPORT / f"{registry}_registry.json").read_text())
    assert len(paths) == len(set(paths))
    assert all((ROOT / path / "metrics.json").exists() for path in paths)
report = dict(status="passed", evaluations=counts, new_training_runs=len(new_train),
    fitted_checkpoints_used=len(parents), dense_full_training_pairs_checked=matched,
    parent_checkpoint_hashes_match=True, source_snapshots_match=True,
    reserved_outer_test_records_in_fit_or_validation=0,
    saved_predictions_reproduce_auroc=True, class_counts_match=True,
    parent_training_config_unchanged=True, package_source=study.source)
dump(REPORT / "ARTIFACT_AUDIT.json", report)
print(json.dumps(report, indent=2))
