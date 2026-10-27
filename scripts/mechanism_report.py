"""Create a rule-specific mechanism report from actual development runs."""
import json
from pathlib import Path

import pandas as pd


def main():
    records = []
    for root in ("runs/mechanism", "runs/withheld"):
        for path in sorted(Path(root).glob("*/metrics.json")):
            manifest = json.loads((path.parent / "manifest.json").read_text())
            if manifest["status"] != "complete":
                continue
            metric = json.loads(path.read_text())
            rules = metric["constraints"]["synthetic"]["rules"]
            records.append({"mode": manifest["mode"] if root == "runs/mechanism" else "full_withheld_parity",
                "seed": manifest["config"]["seed"], "protein_violation": rules["protein_identity"]["violation_rate"],
                "protein_severity": rules["protein_identity"]["mean_severity"], "parity_violation": rules["parity"]["violation_rate"],
                "association_rmse": metric["fidelity"]["association_rmse"],
                "tstr_auroc": metric["utility"]["forest"]["TSTR"]["auroc"]})
    table = pd.DataFrame(records)
    out = Path("reports/mechanism")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "rule_specific_runs.csv", index=False)
    grouped = table.groupby("mode").agg({key: ["mean", "std", "count"] for key in table.columns if key not in ("mode", "seed")})
    grouped.to_csv(out / "rule_specific_summary.csv")
    lines = ["# Controlled higher-order mechanism experiments", "",
        "These are simulated records, not clinical patient data. Every configuration uses the same single development fold with three generator seeds, 100-epoch neural budgets and 30 reverse steps. Values are mean ± sample SD across training/sampling seeds; they are not confidence intervals over patient populations. No significance test is performed.", "",
        "The simulator includes a protein-like identity and a binary XOR triple. Protein violation uses a prespecified 0.02 residual tolerance; parity violation is the fraction of incorrect XOR triples. Separate held-out-rule runs remove parity from both hypergraph and energy while still evaluating it.", "",
        "| Arm | Runs | Protein violation ↓ | Parity violation ↓ | Association RMSE ↓ | TSTR RF AUROC ↑ |",
        "|---|---:|---:|---:|---:|---:|"]
    for mode, row in grouped.iterrows():
        values = [f"{row[(k, 'mean')]:.3f} ± {row[(k, 'std')]:.3f}" for k in ("protein_violation", "parity_violation", "association_rmse", "tstr_auroc")]
        lines.append(f"| {mode} | {int(row[('tstr_auroc','count')])} | " + " | ".join(values) + " |")
    lines += ["", "## Interpretation", "",
        "The full model reduces parity violations relative to base diffusion and the clique control in this simulator, but loses downstream utility relative to base diffusion. Its protein constraint satisfaction remains incomplete. Shuffled incidence and withheld parity weaken the parity result, consistent with the imposed factor structure contributing to the effect. This does not establish generalization to new clinical relationships.", "",
        "Adaptive guidance improves the protein residual behavior compared with equal-integrated-strength constant guidance; parity behavior need not change because absorbing categories may be fixed before the strongest late guidance. Equal schedule sums do not imply equal effective changes in samples.", "",
        "Simple repair satisfies both registered relationships exactly and retains useful predictive signal, while changing the generated distribution and some associations. Therefore the evidence favors reporting a trade-off and method limitations rather than claiming hypergraph guidance is necessary for validity.", "",
        "The simulator, one fold, modest training budget, unbalanced parameter counts and small number of seeds limit inference. The performance of withheld parity also cautions against claiming preservation of relationships that the model has not been given. Wrong-rule sensitivity, parameter-matched dense controls and larger independent cohorts remain necessary.", "",
        "All source hashes, configurations, original samples and per-rule metrics are retained. See the general summary for complete run-level values."]
    (out / "MECHANISM_INTERPRETATION.md").write_text("\n".join(lines) + "\n")
    print(out / "MECHANISM_INTERPRETATION.md")


if __name__ == "__main__":
    main()
