"""Write an auditable data/constraint report without evaluating outer test utility."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from medhyperdiff.constraints import build_constraints
from medhyperdiff.data import SCHEMAS, normalize, nested_splits


def main():
    rows, reports = [], {}
    lines = ["# Dataset audit", "", "Public UCI source tables; phenotype prediction benchmarks, not clinically deployed models.", "",
             "Constraint residual analysis below uses only the fixed inner-fit partition of outer fold 0. Global counts are descriptive ingestion checks.", "",
             "| Dataset | Rows | Predictors | Missing cells | Duplicate predictor profiles | Fit / validation / test |",
             "|---|---:|---:|---:|---:|---|"]
    for name, schema in SCHEMAS.items():
        if name == "toy":
            continue
        path = Path("data/processed") / f"{name}.csv"
        if not path.exists():
            continue
        frame = normalize(pd.read_csv(path), schema)
        split = next(nested_splits(frame, schema))
        train = frame.iloc[split["train"]]
        rules = build_constraints(schema, train)
        report = {"rows": len(frame), "predictors": len(schema.features), "target_counts": frame[schema.target].value_counts().to_dict(),
            "missing_by_column": frame.isna().sum().to_dict(), "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_predictors": int(frame.duplicated(schema.features).sum()),
            "splits": {k: len(split[k]) for k in ("train", "validation", "test")},
            "fit_rule_audit": rules.evaluate(train), "rules": rules.to_dict()}
        if name == "ilpd":
            residual = abs(train.ag_ratio * (train.total_protein - train.albumin) - train.albumin).dropna()
            report["protein_absolute_residual_fit_quantiles"] = {str(q): float(residual.quantile(q)) for q in (0, .5, .9, .95, .99, 1)}
            report["protein_residual_rows_observed_fit"] = len(residual)
        reports[name] = report
        lines.append(f"| {name} | {len(frame)} | {len(schema.features)} | {int(frame.isna().sum().sum())} | {report['duplicate_predictors']} | " + " / ".join(str(report["splits"][k]) for k in ("train", "validation", "test")) + " |")
    for name, report in reports.items():
        lines += ["", f"## {name}", "", f"Target counts: `{json.dumps(report['target_counts'])}`.", "",
                  f"Missing cells by column: `{json.dumps({k:v for k,v in report['missing_by_column'].items() if v})}`."]
        if "protein_absolute_residual_fit_quantiles" in report:
            lines += ["", "Protein identity residual quantiles in the fit partition (source protein units):", "",
                      "```json", json.dumps(report["protein_absolute_residual_fit_quantiles"], indent=2), "```", "",
                      "This is an empirical inconsistency/rounding audit. A fit-only quantile is not a clinically validated tolerance."]
        lines += ["", "| Rule | Eligible fit rows | Violation rate | Status |", "|---|---:|---:|---|"]
        for key, metric in report["fit_rule_audit"]["rules"].items():
            rate = "undefined" if metric["violation_rate"] is None else f"{metric['violation_rate']:.4f}"
            lines.append(f"| {key} | {metric['eligible']} | {rate} | {metric['status']} |")
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "data_audit.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    (out / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(out / "DATA_AUDIT.md")


if __name__ == "__main__":
    main()
