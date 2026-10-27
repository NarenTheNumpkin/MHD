"""Report every retained iteration-2 experiment; no inferential tests on seeds/folds."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/medhyperdiff-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/medhyperdiff-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "iteration2"
LABELS = {"paired_base": "Base", "paired_full": "Full, paired", "gated_full": "Gated full",
    "velocity_full": "Velocity full", "gated_dense": "Gated dense", "paired_repair": "Base + repair",
    "tabddpm_larger": "TabDDPM, larger budget", "paired_dense": "Dense, paired"}


def collect():
    records, rules, features, failed = [], [], [], []
    for path in sorted((ROOT / "runs" / "iteration2").glob("*/*/manifest.json")):
        m = json.loads(path.read_text())
        if m["status"] != "complete":
            failed.append({"run": str(path.parent.relative_to(ROOT)), "status": m["status"], "error": m.get("error")})
            continue
        metrics = json.loads((path.parent / "metrics.json").read_text())
        cfg = m["config"]
        arm = path.parent.name[len(m["dataset"]) + 1:].rsplit("_s", 1)[0]
        if "_i" in arm:
            arm = arm.rsplit("_i", 1)[0]
        r = {"run": str(path.parent.relative_to(ROOT)), "stage": path.parent.parent.name,
            "dataset": m["dataset"], "arm": arm, "seed": cfg["seed"],
            "inner_seed": m.get("inner_seed") if m.get("inner_seed") is not None else 2027,
            "outer_fold": m["fold"], "outer_test_opened": m["outer_test_opened"],
            "source_sha256": m["source_sha256"], "evaluation_version": m["evaluation_version"],
            "parameters": m["parameters"], "epoch_budget": cfg["epochs"], "selected_epoch": m["selected_epoch"],
            "epochs_executed": len(json.loads((path.parent / "history.json").read_text())),
            "train_rows": m["train_rows"], "validation_rows": m["validation_rows"],
            "fit_seconds": m["fit_seconds_including_structure"], "sample_seconds": m["sample_seconds"],
            "rule_violation": metrics["constraints"]["synthetic"]["any_rule_violation_rate"],
            "numeric_wasserstein": metrics["fidelity"]["mean_numeric_wasserstein_train_sd"],
            "association_rmse": metrics["fidelity"]["association_rmse"],
            "signed_numeric_correlation_rmse": metrics["fidelity"]["signed_numeric_spearman_rmse"],
            "missing_rate_error": metrics["fidelity"]["mean_missing_rate_error"],
            "rf_auc": metrics["utility"]["forest"]["TSTR"]["auroc"],
            "logistic_auc": metrics["utility"]["logistic"]["TSTR"]["auroc"],
            "real_rf_auc": metrics["utility"]["forest"]["TRTR"]["auroc"],
            "copy_fraction": metrics["privacy"]["exact_predictor_match_train"],
            "membership_auc": metrics["privacy"]["nearest_synthetic_membership_auc"],
            "c2st_auc": metrics["c2st"]["auc"]}
        if m["mode"] != "tabddpm":
            import torch
            state = torch.load(path.parent / "checkpoint.pt", weights_only=True, map_location="cpu")["state_dict"]
            r["graph_coefficient"] = float(state["graph_strength"])
            r["hypergraph_gate_coefficients"] = json.dumps([float(v.tanh()) for k, v in state.items() if k.endswith("residual_gate")])
        for name, v in metrics["constraints"]["synthetic"]["rules"].items():
            rules.append({"run": r["run"], "dataset": r["dataset"], "arm": arm, "seed": cfg["seed"],
                "inner_seed": r["inner_seed"], "rule": name, **v})
            if name in ("protein_identity", "parity"):
                r[name + "_violation"] = v["violation_rate"]
        for name, v in metrics["fidelity"]["per_feature"].items():
            features.append({"run": r["run"], "dataset": r["dataset"], "arm": arm, "seed": cfg["seed"],
                "inner_seed": r["inner_seed"], "feature": name, **v})
        records.append(r)
    return pd.DataFrame(records), pd.DataFrame(rules), pd.DataFrame(features), failed


def markdown_table(frame, columns, headings):
    lines = ["| " + " | ".join(headings) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            v = row[col]
            cells.append("—" if pd.isna(v) else f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, rules, features, failed = collect()
    if data.empty:
        raise ValueError("No completed iteration-2 runs")
    data.to_csv(OUT / "all_runs.csv", index=False)
    rules.to_csv(OUT / "all_rule_results.csv", index=False)
    features.to_csv(OUT / "all_feature_results.csv", index=False)
    (OUT / "incomplete_runs.json").write_text(json.dumps(failed, indent=2) + "\n")
    fields = ["rf_auc", "logistic_auc", "rule_violation", "protein_identity_violation", "parity_violation",
        "numeric_wasserstein", "association_rmse", "fit_seconds", "sample_seconds", "membership_auc"]
    primary = data[(data.inner_seed == 2027) & data.arm.isin(LABELS)]
    summary = primary.groupby(["dataset", "arm"])[fields].agg(["mean", "std", "count"])
    summary.to_csv(OUT / "seed_summary.csv")
    mean = primary.groupby(["dataset", "arm"])[fields].mean().reset_index()
    mean["seeds"] = primary.groupby(["dataset", "arm"]).seed.nunique().to_numpy()
    mean.to_csv(OUT / "seed_means.csv", index=False)
    inner = data[data.stage == "inner_sensitivity"]
    if len(inner):
        inner.groupby(["dataset", "inner_seed", "arm"])[fields].mean().to_csv(OUT / "inner_partition_means.csv")
    lines = ["# Iteration 2 — measured development results", "",
        f"Completed: **{len(data)}**. Incomplete/failed: **{len(failed)}**. Outer test opened in these runs: **{bool(data.outer_test_opened.any())}**.", "",
        "These are validation results used for method development and checkpoint selection. The mean and standard deviation across generator seeds measure optimization/sampling variation conditional on one cohort and split; they are not confidence intervals. Inner partitions overlap and are not independent replications of the clinical population.", "",
        "## Repeated-seed comparison on the original inner split", "",
        "All internal arms use a 100-epoch maximum, width 32, two layers, batch 64 and 30 diffusion steps. TabDDPM's larger-budget sensitivity uses 500 maximum epochs, width 128 and 1,000 steps; its quantile preprocessing and optimization also differ. It is an official-source adaptation, not a published benchmark replication.", "",
        "Rule violations below mean at least one supplied rule violated among eligible rows. Lower is better. A lower rate can reflect distorted data, so assess utility and fidelity alongside it. RF and logistic AUROC are downstream classifiers trained on synthetic records and evaluated on real validation records.", "",
        markdown_table(mean, ["dataset", "arm", "seeds", "rf_auc", "logistic_auc", "rule_violation", "numeric_wasserstein", "association_rmse"],
            ["Dataset", "Arm", "Seeds", "RF AUROC ↑", "Logistic AUROC ↑", "Any-rule violation ↓", "Numeric W ↓", "Association RMSE ↓"]), "",
        "## Simulator-specific mechanism", "",
        markdown_table(mean[mean.dataset == "toy"], ["arm", "protein_identity_violation", "parity_violation", "rf_auc"],
            ["Arm", "Protein violation ↓", "Parity violation ↓", "RF AUROC ↑"]), "",
        "## All screening candidates", "",
        "Rejected candidates remain visible. These single-seed results must not be compared as definitive rankings.", "",
        markdown_table(data[data.stage == "screen"], ["dataset", "arm", "rf_auc", "rule_violation", "numeric_wasserstein", "association_rmse"],
            ["Dataset", "Arm", "RF AUROC ↑", "Rule violation ↓", "Numeric W ↓", "Association RMSE ↓"]), "",
        "## Costs and reproducibility", "",
        "All runs executed locally on CPU with two threads per training process; GPU VRAM consumption was zero. Fit times include graph estimation and sampling times exclude downstream evaluation. Concurrent execution affects wall time comparisons. Inspect all_runs.csv for parameters, actual epochs, source hashes, split identities and costs.", "",
        f"Sum of recorded fit time: {data.fit_seconds.sum():.1f} seconds. Sum of sampling time: {data.sample_seconds.sum():.1f} seconds. These sums are not elapsed wall time.", "",
        "Prediction scores, checkpoints and generated tables are retained under runs/iteration2. Every package hash resolves to a source snapshot. Raw single-run values, all per-rule severity/coverage results, and seed standard deviations accompany this report.", ""]
    if len(inner):
        im = inner.groupby(["dataset", "inner_seed", "arm"])[fields].mean().reset_index()
        lines += ["## Additional inner partitions", "", "The reserved outer test set is identical across these experiments. Values below average generator seeds within each partition; do not interpret the overlapping partitions as independent test cohorts.", "",
            markdown_table(im, ["dataset", "inner_seed", "arm", "rf_auc", "rule_violation", "numeric_wasserstein", "association_rmse"],
                ["Dataset", "Inner seed", "Arm", "RF AUROC ↑", "Rule violation ↓", "Numeric W ↓", "Association RMSE ↓"]), ""]
    diagnostic = data[data.stage == "external_diagnostic"]
    if len(diagnostic):
        lines += ["## Extended TabDDPM training diagnostic", "",
            "This single-seed check uses 8,500 epochs and is separate from the repeated-seed comparisons. It diagnoses an undertraining hypothesis; it is not a tuned final baseline.", "",
            markdown_table(diagnostic, ["dataset", "arm", "selected_epoch", "rf_auc", "rule_violation", "numeric_wasserstein", "association_rmse"],
                ["Dataset", "Arm", "Selected epoch", "RF AUROC ↑", "Rule violation ↓", "Numeric W ↓", "Association RMSE ↓"]), ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    colors = plt.get_cmap("tab10")
    for i, (arm, label) in enumerate(LABELS.items()):
        for ax, dataset in zip(axes, ("ilpd", "toy")):
            subset = primary[(primary.dataset == dataset) & (primary.arm == arm)]
            if subset.empty:
                continue
            ax.scatter(subset.rule_violation, subset.rf_auc, s=25, alpha=.25, color=colors(i))
            ax.scatter(subset.rule_violation.mean(), subset.rf_auc.mean(), s=85, color=colors(i), label=label,
                edgecolors="white", linewidth=.7)
    for ax, title in zip(axes, ("ILPD: real clinical development cohort", "Simulator: supplied protein and parity rules")):
        ax.set(title=title, xlabel="Fraction violating a supplied rule (lower is better)",
               ylabel="Synthetic-trained RF AUROC (higher is better)", xlim=(-.04, 1.04))
        ax.grid(alpha=.15)
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Development validity–utility tradeoff | large points: seed means; small: individual seeds\n"
        "TabDDPM adaptation has unresolved sampling instability; no final baseline ranking", fontsize=11)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"validity_utility.{ext}", dpi=180)
    plt.close(fig)
    print(OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
