"""Aggregate measured runs only, without treating fold×seed scores as independent."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(root, output):
    records, failures = [], []
    for file in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(file.read_text())
        if manifest["status"] != "complete":
            failures.append({"run": str(file.parent), "status": manifest["status"], "error": manifest.get("error")})
            continue
        metric = json.loads((file.parent / "metrics.json").read_text())
        config = {k: v for k, v in manifest["config"].items() if k != "seed"}
        graph = json.loads((file.parent / "graph.json").read_text())
        identity = {"config": config, "split_seed": manifest["split_seed"], "data": manifest["data_sha256"],
                    "inner_seed": manifest.get("inner_seed"), "evaluation_version": manifest.get("evaluation_version", 1),
                    "upstream_commit": manifest.get("upstream_commit"),
                    "withheld_rules": manifest.get("withheld_rules", []),
                    "graph_settings": {k: graph[k] for k in ("bootstraps", "threshold", "stability")}}
        config_signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        records.append({"run": str(file.parent), "dataset": manifest["dataset"], "model": manifest["mode"],
            "config_signature": config_signature,
            "partition": manifest["evaluation_partition"], "fold": manifest["fold"], "seed": manifest["config"]["seed"],
            "epochs_budget": manifest["config"]["epochs"], "steps": manifest["config"]["steps"],
            "source_sha256": manifest["source_sha256"],
            "wasserstein": metric["fidelity"]["mean_numeric_wasserstein_train_sd"],
            "association_rmse": metric["fidelity"]["association_rmse"],
            "rule_violation": metric["constraints"]["synthetic"]["any_rule_violation_rate"],
            "missingness_error": metric["fidelity"]["mean_missing_rate_error"],
            "tstr_forest_auroc": metric["utility"]["forest"]["TSTR"].get("auroc"),
            "trtr_forest_auroc": metric["utility"]["forest"]["TRTR"].get("auroc"),
            "membership_auc": metric["privacy"].get("nearest_synthetic_membership_auc"),
            "fit_seconds": manifest["fit_seconds_including_structure"],
            "sample_seconds": manifest["sample_seconds"]})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No completed runs")
    table = pd.DataFrame(records)
    table.to_csv(out / "runs.csv", index=False)
    fields = ["wasserstein", "association_rmse", "rule_violation", "missingness_error", "tstr_forest_auroc", "trtr_forest_auroc", "membership_auc", "fit_seconds", "sample_seconds"]
    # Never mix development/test partitions, code versions, or training budgets.
    groups = ["dataset", "model", "partition", "epochs_budget", "steps", "source_sha256", "config_signature"]
    folded = table.groupby(groups + ["fold"], dropna=False)[fields].mean().reset_index()
    folded.to_csv(out / "seed_averaged_folds.csv", index=False)
    grouped = folded.groupby(groups, dropna=False)[fields].agg(["mean", "std", "count"])
    grouped.to_csv(out / "descriptive_summary.csv")
    lines = ["# Measured experiment summary", "", "Development runs evaluate inner validation after checkpoint selection and cannot support confirmatory performance claims. A single fold or seed is not evidence of statistical superiority. All values below come from saved run artifacts.", "",
             "This report averages seeds within folds, then summarizes folds descriptively. It performs no significance or noninferiority tests. Budgets and source versions remain separate in the CSV summaries.", "",
             "`Rule violation` combines all registered rules, including exploratory rules. Consult each run's per-rule status and coverage before interpretation. Membership AUC is a weak exploratory attack, not a privacy score to optimize.", "",
             "| Dataset | Model | Partition | Fold | Seed | Numeric W↓ | Assoc. RMSE↓ | Rule violation↓ | TSTR RF AUROC↑ | TRTR RF AUROC |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "undefined" if v is None or not np.isfinite(v) else f"{v:.3f}"
    for r in records:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['partition']} | {r['fold']} | {r['seed']} | " + " | ".join(fmt(r[k]) for k in ("wasserstein", "association_rmse", "rule_violation", "tstr_forest_auroc", "trtr_forest_auroc")) + " |")
    lines += ["", f"Completed runs: {len(records)}. Incomplete/failed runs: {len(failures)}.", ""]
    if failures:
        lines += ["```json", json.dumps(failures, indent=2), "```"]
    (out / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(out / "EXPERIMENT_SUMMARY.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/pilot")
    parser.add_argument("--output", default="reports/pilot")
    args = parser.parse_args()
    summarize(args.root, args.output)
