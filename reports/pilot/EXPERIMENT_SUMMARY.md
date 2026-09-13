# Measured experiment summary

Development runs evaluate inner validation after checkpoint selection and cannot support confirmatory performance claims. A single fold or seed is not evidence of statistical superiority. All values below come from saved run artifacts.

This report averages seeds within folds, then summarizes folds descriptively. It performs no significance or noninferiority tests. Budgets and source versions remain separate in the CSV summaries.

`Rule violation` combines all registered rules, including exploratory rules. Consult each run's per-rule status and coverage before interpretation. Membership AUC is a weak exploratory attack, not a privacy score to optimize.

| Dataset | Model | Partition | Fold | Seed | Numeric W↓ | Assoc. RMSE↓ | Rule violation↓ | TSTR RF AUROC↑ | TRTR RF AUROC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ckd | base | inner_validation | 0 | 0 | 0.339 | 0.186 | 0.008 | 0.990 | 1.000 |
| ckd | copula | inner_validation | 0 | 0 | 0.199 | 0.088 | 0.000 | 1.000 | 1.000 |
| ckd | ctgan | inner_validation | 0 | 0 | 0.432 | 0.134 | 0.142 | 1.000 | 1.000 |
| ckd | full | inner_validation | 0 | 0 | 0.341 | 0.179 | 0.017 | 0.991 | 1.000 |
| ckd | tvae | inner_validation | 0 | 0 | 0.360 | 0.166 | 0.025 | 0.988 | 1.000 |
| diabetes | base | inner_validation | 0 | 0 | 0.181 | 0.063 | 0.000 | 0.946 | 0.951 |
| diabetes | copula | inner_validation | 0 | 0 | 0.169 | 0.062 | 0.000 | 0.931 | 0.951 |
| diabetes | ctgan | inner_validation | 0 | 0 | 0.511 | 0.074 | 0.035 | 0.920 | 0.951 |
| diabetes | full | inner_validation | 0 | 0 | 0.179 | 0.063 | 0.000 | 0.946 | 0.951 |
| diabetes | tvae | inner_validation | 0 | 0 | 0.275 | 0.231 | 0.000 | 0.904 | 0.951 |
| heart_failure | base | inner_validation | 0 | 0 | 0.315 | 0.074 | 0.022 | 0.748 | 0.836 |
| heart_failure | copula | inner_validation | 0 | 0 | 0.218 | 0.080 | 0.000 | 0.715 | 0.836 |
| heart_failure | ctgan | inner_validation | 0 | 0 | 0.652 | 0.081 | 0.162 | 0.689 | 0.836 |
| heart_failure | full | inner_validation | 0 | 0 | 0.316 | 0.074 | 0.028 | 0.739 | 0.836 |
| heart_failure | tvae | inner_validation | 0 | 0 | 0.393 | 0.090 | 0.039 | 0.770 | 0.836 |
| ilpd | base | inner_validation | 0 | 0 | 0.430 | 0.199 | 0.917 | 0.794 | 0.755 |
| ilpd | copula | inner_validation | 0 | 0 | 0.204 | 0.089 | 0.605 | 0.726 | 0.755 |
| ilpd | ctgan | inner_validation | 0 | 0 | 0.438 | 0.259 | 0.903 | 0.704 | 0.755 |
| ilpd | full | inner_validation | 0 | 0 | 0.342 | 0.147 | 0.453 | 0.752 | 0.755 |
| ilpd | tvae | inner_validation | 0 | 0 | 0.426 | 0.194 | 0.857 | 0.780 | 0.755 |

Completed runs: 20. Incomplete/failed runs: 0.

