# Controlled higher-order mechanism experiments

These are simulated records, not clinical patient data. Every configuration uses the same single development fold with three generator seeds, 100-epoch neural budgets and 30 reverse steps. Values are mean ± sample SD across training/sampling seeds; they are not confidence intervals over patient populations. No significance test is performed.

The simulator includes a protein-like identity and a binary XOR triple. Protein violation uses a prespecified 0.02 residual tolerance; parity violation is the fraction of incorrect XOR triples. Separate held-out-rule runs remove parity from both hypergraph and energy while still evaluating it.

| Arm | Runs | Protein violation ↓ | Parity violation ↓ | Association RMSE ↓ | TSTR RF AUROC ↑ |
|---|---:|---:|---:|---:|---:|
| base | 3 | 0.971 ± 0.024 | 0.477 ± 0.024 | 0.112 ± 0.034 | 0.802 ± 0.014 |
| clique | 3 | 0.734 ± 0.129 | 0.462 ± 0.014 | 0.095 ± 0.001 | 0.763 ± 0.010 |
| constant | 3 | 0.954 ± 0.015 | 0.244 ± 0.157 | 0.068 ± 0.029 | 0.719 ± 0.058 |
| copula | 3 | 0.983 ± 0.007 | 0.483 ± 0.026 | 0.035 ± 0.005 | 0.762 ± 0.045 |
| energy | 3 | 0.971 ± 0.027 | 0.475 ± 0.023 | 0.113 ± 0.042 | 0.808 ± 0.019 |
| full | 3 | 0.664 ± 0.054 | 0.244 ± 0.157 | 0.071 ± 0.033 | 0.745 ± 0.035 |
| full_withheld_parity | 3 | 0.742 ± 0.080 | 0.479 ± 0.025 | 0.066 ± 0.022 | 0.719 ± 0.101 |
| graph | 3 | 0.975 ± 0.019 | 0.483 ± 0.018 | 0.070 ± 0.016 | 0.790 ± 0.026 |
| guidance | 3 | 0.717 ± 0.154 | 0.475 ± 0.025 | 0.110 ± 0.043 | 0.810 ± 0.020 |
| hypergraph | 3 | 0.961 ± 0.003 | 0.279 ± 0.137 | 0.067 ± 0.034 | 0.743 ± 0.029 |
| independent | 3 | 0.997 ± 0.003 | 0.483 ± 0.005 | 0.155 ± 0.004 | 0.765 ± 0.033 |
| repair | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.125 ± 0.039 | 0.773 ± 0.003 |
| shuffled | 3 | 0.805 ± 0.035 | 0.476 ± 0.008 | 0.068 ± 0.011 | 0.774 ± 0.042 |

## Interpretation

The full model reduces parity violations relative to base diffusion and the clique control in this simulator, but loses downstream utility relative to base diffusion. Its protein constraint satisfaction remains incomplete. Shuffled incidence and withheld parity weaken the parity result, consistent with the imposed factor structure contributing to the effect. This does not establish generalization to new clinical relationships.

Adaptive guidance improves the protein residual behavior compared with equal-integrated-strength constant guidance; parity behavior need not change because absorbing categories may be fixed before the strongest late guidance. Equal schedule sums do not imply equal effective changes in samples.

Simple repair satisfies both registered relationships exactly and retains useful predictive signal, while changing the generated distribution and some associations. Therefore the evidence favors reporting a trade-off and method limitations rather than claiming hypergraph guidance is necessary for validity.

The simulator, one fold, modest training budget, unbalanced parameter counts and small number of seeds limit inference. The performance of withheld parity also cautions against claiming preservation of relationships that the model has not been given. Wrong-rule sensitivity, parameter-matched dense controls and larger independent cohorts remain necessary.

All source hashes, configurations, original samples and per-rule metrics are retained. See the general summary for complete run-level values.
