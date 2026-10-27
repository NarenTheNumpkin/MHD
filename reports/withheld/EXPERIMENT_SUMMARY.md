# Measured experiment summary

Development runs evaluate inner validation after checkpoint selection and cannot support confirmatory performance claims. A single fold or seed is not evidence of statistical superiority. All values below come from saved run artifacts.

This report averages seeds within folds, then summarizes folds descriptively. It performs no significance or noninferiority tests. Budgets and source versions remain separate in the CSV summaries.

`Rule violation` combines all registered rules, including exploratory rules. Consult each run's per-rule status and coverage before interpretation. Membership AUC is a weak exploratory attack, not a privacy score to optimize.

| Dataset | Model | Partition | Fold | Seed | Numeric W↓ | Assoc. RMSE↓ | Rule violation↓ | TSTR RF AUROC↑ | TRTR RF AUROC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| toy | full | inner_validation | 0 | 0 | 0.250 | 0.057 | 0.827 | 0.603 | 0.803 |
| toy | full | inner_validation | 0 | 1 | 0.434 | 0.092 | 0.895 | 0.762 | 0.795 |
| toy | full | inner_validation | 0 | 2 | 0.250 | 0.049 | 0.892 | 0.791 | 0.804 |

Completed runs: 3. Incomplete/failed runs: 0.

