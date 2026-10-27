# Measured experiment summary

Development runs evaluate inner validation after checkpoint selection and cannot support confirmatory performance claims. A single fold or seed is not evidence of statistical superiority. All values below come from saved run artifacts.

This report averages seeds within folds, then summarizes folds descriptively. It performs no significance or noninferiority tests. Budgets and source versions remain separate in the CSV summaries.

`Rule violation` combines all registered rules, including exploratory rules. Consult each run's per-rule status and coverage before interpretation. Membership AUC is a weak exploratory attack, not a privacy score to optimize.

| Dataset | Model | Partition | Fold | Seed | Numeric W↓ | Assoc. RMSE↓ | Rule violation↓ | TSTR RF AUROC↑ | TRTR RF AUROC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ilpd | repair | inner_validation | 0 | 0 | 1.134 | 0.182 | 0.000 | 0.795 | 0.755 |

Completed runs: 1. Incomplete/failed runs: 0.

