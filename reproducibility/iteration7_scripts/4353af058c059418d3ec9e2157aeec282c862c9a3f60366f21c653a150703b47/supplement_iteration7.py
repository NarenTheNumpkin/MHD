"""Audit copy and strict-boundary roundoff using a documented post-protocol check."""
import json
import pandas as pd
from iteration7 import ROOT, REPORT, checked_plan, dump, sha, archive
from iteration7_diagnostics import near_copy_rate, boundary_rates
from medhyperdiff.data import SCHEMAS


def main():
    plan, study = checked_plan()
    registry = json.loads((REPORT/"registry.json").read_text())
    rows, boundaries = [], []
    for entry in registry["new"]+registry["references"]:
        path = ROOT/entry["path"]
        m = json.loads((path/"manifest.json").read_text())
        assert sha(path/"synthetic.csv") == m["synthetic_sha256"]
        frame, split = study.data("ilpd", m["inner_seed"])
        train, validation = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        schema = SCHEMAS["ilpd"]
        generated = pd.read_csv(path/"synthetic.csv", dtype={schema.target: str}, float_precision="round_trip")
        identity = dict(arm=entry["arm"], inner_seed=m["inner_seed"], training_seed=m["training_seed"], sampling_seed=m["sampling_seed"])
        row = dict(**identity, path=entry["path"], synthetic_sha256=m["synthetic_sha256"],
            near_copy_train=near_copy_rate(generated, train, train, schema),
            near_copy_validation=near_copy_rate(generated, validation, train, schema))
        shape = boundary_rates(train, validation, generated, schema)
        if entry["arm"] == "bootstrap":
            assert row["near_copy_train"] == 1.
            assert all(v["synthetic_outside_fit_range_tolerant"] == 0 for v in shape.values())
        rows.append(row)
        for feature, values in shape.items():
            boundaries.append(dict(**identity, feature=feature, **values))
    df, bf = pd.DataFrame(rows), pd.DataFrame(boundaries)
    df.to_csv(REPORT/"tolerant_copy_all.csv", index=False)
    bf.to_csv(REPORT/"tolerant_boundaries_all.csv", index=False)
    means = df.groupby(["arm", "inner_seed"])[["near_copy_train", "near_copy_validation"]].mean().reset_index()
    means.to_csv(REPORT/"tolerant_copy_partition_means.csv", index=False)
    files = [ROOT/"scripts/iteration7_diagnostics.py", ROOT/"scripts/supplement_iteration7.py", ROOT/"docs/ITERATION_7_DIAGNOSTIC_ADDENDUM.md", ROOT/"tests/test_iteration7_diagnostics.py"]
    for file in files:
        archive(file)
    dump(REPORT/"NUMERICAL_DIAGNOSTIC_AUDIT.json", dict(status="passed", records=len(df), bootstrap_tables=9,
        bootstrap_near_copy_rate=1., bootstrap_tolerant_outside_fit_rate=0.,
        prespecified_primary_metric=False, original_metrics_modified=False,
        files={str(f.relative_to(ROOT)): sha(f) for f in files},
        outputs={f: sha(REPORT/f) for f in ("tolerant_copy_all.csv", "tolerant_boundaries_all.csv", "tolerant_copy_partition_means.csv")}))
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
