"""Audit stored artifacts and reserved-test isolation without opening test metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from medhyperdiff.data import SCHEMAS, normalize, toy_data, nested_splits


root = Path(__file__).resolve().parents[1]
frames, reserved = {}, {}
for name in ("ilpd", "toy"):
    frame = toy_data(512, 7301) if name == "toy" else normalize(pd.read_csv(root / "data" / "processed" / f"{name}.csv"), SCHEMAS[name])
    frames[name] = frame
    reserved[name] = set(next(nested_splits(frame, SCHEMAS[name]))["test"])
counts, hashes, params = {}, set(), {}
for path in sorted((root / "runs" / "iteration2").glob("*/*/manifest.json")):
    m = json.loads(path.read_text())
    assert m["status"] == "complete", str(path)
    assert not m["outer_test_opened"] and m["evaluation_partition"] == "inner_validation"
    assert m["fold"] == 0 and m["split_seed"] == 2026
    split = json.loads((path.parent / "splits.json").read_text())["folds"][0]
    assert set(split["test"]) == reserved[m["dataset"]]
    assert not (set(split["train"] + split["validation"]) & reserved[m["dataset"]])
    schema, frame = SCHEMAS[m["dataset"]], frames[m["dataset"]]
    synth = pd.read_csv(path.parent / "synthetic.csv", dtype={schema.target: str})
    assert len(synth) == m["train_rows"] == len(split["train"])
    assert (synth[schema.target] == schema.positive).sum() == (frame.iloc[split["train"]][schema.target] == schema.positive).sum()
    # Preserve floating-point ties exactly when reconstructing rank metrics.
    preds = pd.read_csv(path.parent / "utility_predictions.csv", float_precision="round_trip")
    assert preds.row_index.tolist() == split["validation"]
    assert preds.label.tolist() == (frame.iloc[split["validation"]][schema.target] == schema.positive).astype(int).tolist()
    metrics = json.loads((path.parent / "metrics.json").read_text())
    assert np.isclose(roc_auc_score(preds.label, preds.forest_TSTR), metrics["utility"]["forest"]["TSTR"]["auroc"])
    snapshot = root / "reproducibility" / "source_snapshots" / m["source_sha256"] / "medhyperdiff"
    digest = hashlib.sha256()
    for file in sorted(snapshot.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    assert digest.hexdigest() == m["source_sha256"]
    hashes.add(m["source_sha256"])
    counts[path.parent.parent.name] = counts.get(path.parent.parent.name, 0) + 1
    if m["mode"] in ("dense", "full") and m["config"]["paired_initialization"]:
        cfg = m["config"]
        key = (m["dataset"], cfg["width"], cfg["layers"], cfg["gated_structure"])
        params.setdefault(key, {}).setdefault(m["mode"], set()).add(m["parameters"])
for key, values in params.items():
    if "dense" in values and "full" in values:
        assert values["dense"] == values["full"], (key, values)
report = {"status": "passed", "completed_runs": sum(counts.values()), "stages": counts,
    "source_hashes": sorted(hashes), "all_reserved_tests_identical_to_original": True,
    "reserved_test_rows_used_for_fit_or_validation": 0, "saved_predictions_match_reported_auroc": True,
    "class_counts_match_fit_partition": True, "dense_full_parameter_counts_match": True}
out = root / "reports" / "iteration2" / "ARTIFACT_AUDIT.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
