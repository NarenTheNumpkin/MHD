# Measured experiment summary

Development runs evaluate inner validation after checkpoint selection and cannot support confirmatory performance claims. A single fold or seed is not evidence of statistical superiority. All values below come from saved run artifacts.

This report averages seeds within folds, then summarizes folds descriptively. It performs no significance or noninferiority tests. Budgets and source versions remain separate in the CSV summaries.

`Rule violation` combines all registered rules, including exploratory rules. Consult each run's per-rule status and coverage before interpretation. Membership AUC is a weak exploratory attack, not a privacy score to optimize.

| Dataset | Model | Partition | Fold | Seed | Numeric W↓ | Assoc. RMSE↓ | Rule violation↓ | TSTR RF AUROC↑ | TRTR RF AUROC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| toy | base | inner_validation | 0 | 0 | 0.205 | 0.099 | 0.974 | 0.787 | 0.803 |
| toy | base | inner_validation | 0 | 1 | 0.260 | 0.150 | 0.964 | 0.814 | 0.795 |
| toy | base | inner_validation | 0 | 2 | 0.183 | 0.087 | 1.000 | 0.806 | 0.804 |
| toy | clique | inner_validation | 0 | 0 | 0.260 | 0.094 | 0.879 | 0.768 | 0.803 |
| toy | clique | inner_validation | 0 | 1 | 0.207 | 0.096 | 0.755 | 0.751 | 0.795 |
| toy | clique | inner_validation | 0 | 2 | 0.332 | 0.096 | 0.905 | 0.769 | 0.804 |
| toy | constant | inner_validation | 0 | 0 | 0.371 | 0.053 | 0.958 | 0.652 | 0.803 |
| toy | constant | inner_validation | 0 | 1 | 0.342 | 0.101 | 0.977 | 0.758 | 0.795 |
| toy | constant | inner_validation | 0 | 2 | 0.274 | 0.049 | 0.948 | 0.746 | 0.804 |
| toy | copula | inner_validation | 0 | 0 | 0.144 | 0.030 | 0.993 | 0.777 | 0.803 |
| toy | copula | inner_validation | 0 | 1 | 0.120 | 0.037 | 0.987 | 0.798 | 0.795 |
| toy | copula | inner_validation | 0 | 2 | 0.128 | 0.039 | 0.990 | 0.712 | 0.804 |
| toy | energy | inner_validation | 0 | 0 | 0.199 | 0.091 | 0.967 | 0.787 | 0.803 |
| toy | energy | inner_validation | 0 | 1 | 0.289 | 0.161 | 0.980 | 0.823 | 0.795 |
| toy | energy | inner_validation | 0 | 2 | 0.164 | 0.085 | 1.000 | 0.815 | 0.804 |
| toy | full | inner_validation | 0 | 0 | 0.376 | 0.061 | 0.768 | 0.708 | 0.803 |
| toy | full | inner_validation | 0 | 1 | 0.338 | 0.108 | 0.814 | 0.777 | 0.795 |
| toy | full | inner_validation | 0 | 2 | 0.274 | 0.045 | 0.667 | 0.750 | 0.804 |
| toy | graph | inner_validation | 0 | 0 | 0.175 | 0.070 | 0.977 | 0.774 | 0.803 |
| toy | graph | inner_validation | 0 | 1 | 0.271 | 0.086 | 0.980 | 0.776 | 0.795 |
| toy | graph | inner_validation | 0 | 2 | 0.191 | 0.054 | 0.997 | 0.819 | 0.804 |
| toy | guidance | inner_validation | 0 | 0 | 0.208 | 0.102 | 0.905 | 0.787 | 0.803 |
| toy | guidance | inner_validation | 0 | 1 | 0.258 | 0.156 | 0.745 | 0.821 | 0.795 |
| toy | guidance | inner_validation | 0 | 2 | 0.181 | 0.072 | 0.912 | 0.821 | 0.804 |
| toy | hypergraph | inner_validation | 0 | 0 | 0.208 | 0.044 | 0.990 | 0.710 | 0.803 |
| toy | hypergraph | inner_validation | 0 | 1 | 0.405 | 0.107 | 0.974 | 0.764 | 0.795 |
| toy | hypergraph | inner_validation | 0 | 2 | 0.270 | 0.052 | 0.964 | 0.756 | 0.804 |
| toy | independent | inner_validation | 0 | 0 | 0.122 | 0.159 | 0.997 | 0.769 | 0.803 |
| toy | independent | inner_validation | 0 | 1 | 0.108 | 0.152 | 0.993 | 0.731 | 0.795 |
| toy | independent | inner_validation | 0 | 2 | 0.114 | 0.155 | 1.000 | 0.797 | 0.804 |
| toy | repair | inner_validation | 0 | 0 | 0.193 | 0.098 | 0.000 | 0.775 | 0.803 |
| toy | repair | inner_validation | 0 | 1 | 0.270 | 0.170 | 0.000 | 0.770 | 0.795 |
| toy | repair | inner_validation | 0 | 2 | 0.713 | 0.107 | 0.000 | 0.775 | 0.804 |
| toy | shuffled | inner_validation | 0 | 0 | 0.188 | 0.080 | 0.889 | 0.786 | 0.803 |
| toy | shuffled | inner_validation | 0 | 1 | 0.428 | 0.058 | 0.905 | 0.808 | 0.795 |
| toy | shuffled | inner_validation | 0 | 2 | 0.324 | 0.068 | 0.902 | 0.728 | 0.804 |

Completed runs: 36. Incomplete/failed runs: 0.

