# Iteration 2 — measured development results

Completed: **110**. Incomplete/failed: **0**. Outer test opened in these runs: **False**.

These are validation results used for method development and checkpoint selection. The mean and standard deviation across generator seeds measure optimization/sampling variation conditional on one cohort and split; they are not confidence intervals. Inner partitions overlap and are not independent replications of the clinical population.

## Repeated-seed comparison on the original inner split

All internal arms use a 100-epoch maximum, width 32, two layers, batch 64 and 30 diffusion steps. TabDDPM's larger-budget sensitivity uses 500 maximum epochs, width 128 and 1,000 steps; its quantile preprocessing and optimization also differ. It is an official-source adaptation, not a published benchmark replication.

Rule violations below mean at least one supplied rule violated among eligible rows. Lower is better. A lower rate can reflect distorted data, so assess utility and fidelity alongside it. RF and logistic AUROC are downstream classifiers trained on synthetic records and evaluated on real validation records.

| Dataset | Arm | Seeds | RF AUROC ↑ | Logistic AUROC ↑ | Any-rule violation ↓ | Numeric W ↓ | Association RMSE ↓ |
|---|---|---|---|---|---|---|---|
| ilpd | gated_dense | 3 | 0.688 | 0.718 | 0.619 | 0.472 | 0.204 |
| ilpd | gated_full | 3 | 0.713 | 0.726 | 0.387 | 0.386 | 0.185 |
| ilpd | paired_base | 3 | 0.701 | 0.726 | 0.727 | 0.410 | 0.202 |
| ilpd | paired_dense | 3 | 0.654 | 0.731 | 0.376 | 0.414 | 0.188 |
| ilpd | paired_full | 3 | 0.726 | 0.753 | 0.153 | 0.391 | 0.203 |
| ilpd | paired_repair | 3 | 0.699 | 0.722 | 0.000 | 0.401 | 0.200 |
| ilpd | tabddpm_larger | 3 | 0.738 | 0.736 | 0.976 | 3.673 | 0.240 |
| ilpd | velocity_full | 3 | 0.720 | 0.732 | 0.719 | 0.309 | 0.190 |
| toy | gated_dense | 3 | 0.803 | 0.819 | 0.880 | 0.202 | 0.079 |
| toy | gated_full | 3 | 0.757 | 0.816 | 0.812 | 0.379 | 0.063 |
| toy | paired_base | 3 | 0.790 | 0.833 | 0.991 | 0.270 | 0.105 |
| toy | paired_dense | 3 | 0.790 | 0.826 | 0.846 | 0.395 | 0.090 |
| toy | paired_full | 3 | 0.691 | 0.812 | 0.728 | 0.288 | 0.078 |
| toy | paired_repair | 3 | 0.772 | 0.816 | 0.000 | 0.495 | 0.103 |
| toy | tabddpm_larger | 3 | 0.658 | 0.709 | 0.997 | 1.260 | 0.171 |
| toy | velocity_full | 3 | 0.820 | 0.838 | 0.814 | 0.210 | 0.063 |

## Simulator-specific mechanism

| Arm | Protein violation ↓ | Parity violation ↓ | RF AUROC ↑ |
|---|---|---|---|
| gated_dense | 0.767 | 0.467 | 0.803 |
| gated_full | 0.694 | 0.401 | 0.757 |
| paired_base | 0.976 | 0.474 | 0.790 |
| paired_dense | 0.722 | 0.463 | 0.790 |
| paired_full | 0.702 | 0.122 | 0.691 |
| paired_repair | 0.000 | 0.000 | 0.772 |
| tabddpm_larger | 0.997 | 0.243 | 0.658 |
| velocity_full | 0.649 | 0.447 | 0.820 |

## All screening candidates

Rejected candidates remain visible. These single-seed results must not be compared as definitive rankings.

| Dataset | Arm | RF AUROC ↑ | Rule violation ↓ | Numeric W ↓ | Association RMSE ↓ |
|---|---|---|---|---|---|
| ilpd | gated_dense | 0.671 | 0.696 | 0.383 | 0.210 |
| ilpd | gated_full | 0.747 | 0.456 | 0.375 | 0.197 |
| ilpd | gated_no_guidance | 0.748 | 0.456 | 0.375 | 0.197 |
| ilpd | gated_velocity_full | 0.690 | 0.911 | 0.408 | 0.238 |
| ilpd | legacy_base | 0.658 | 0.779 | 0.421 | 0.227 |
| ilpd | legacy_full | 0.723 | 0.307 | 0.379 | 0.199 |
| ilpd | paired_base | 0.665 | 0.751 | 0.385 | 0.196 |
| ilpd | paired_full | 0.768 | 0.020 | 0.339 | 0.223 |
| ilpd | paired_repair | 0.655 | 0.000 | 0.354 | 0.171 |
| ilpd | tabddpm_small | 0.496 | 1.000 | 3.300 | 0.218 |
| ilpd | velocity_base | 0.671 | 0.963 | 0.418 | 0.259 |
| ilpd | velocity_full | 0.729 | 0.504 | 0.284 | 0.156 |
| toy | gated_dense | 0.825 | 0.856 | 0.185 | 0.084 |
| toy | gated_full | 0.781 | 0.859 | 0.366 | 0.077 |
| toy | gated_no_guidance | 0.780 | 0.984 | 0.371 | 0.061 |
| toy | gated_velocity_full | 0.833 | 0.837 | 0.189 | 0.061 |
| toy | legacy_base | 0.787 | 0.974 | 0.205 | 0.099 |
| toy | legacy_full | 0.708 | 0.768 | 0.376 | 0.061 |
| toy | paired_base | 0.800 | 0.980 | 0.210 | 0.093 |
| toy | paired_full | 0.611 | 0.693 | 0.276 | 0.061 |
| toy | paired_repair | 0.784 | 0.000 | 0.191 | 0.094 |
| toy | tabddpm_small | 0.531 | 1.000 | 1.353 | 0.189 |
| toy | velocity_base | 0.856 | 1.000 | 0.181 | 0.157 |
| toy | velocity_full | 0.781 | 0.748 | 0.200 | 0.040 |

## Costs and reproducibility

All runs executed locally on CPU with two threads per training process; GPU VRAM consumption was zero. Fit times include graph estimation and sampling times exclude downstream evaluation. Concurrent execution affects wall time comparisons. Inspect all_runs.csv for parameters, actual epochs, source hashes, split identities and costs.

Sum of recorded fit time: 684.4 seconds. Sum of sampling time: 32.7 seconds. These sums are not elapsed wall time.

Prediction scores, checkpoints and generated tables are retained under runs/iteration2. Every package hash resolves to a source snapshot. Raw single-run values, all per-rule severity/coverage results, and seed standard deviations accompany this report.

## Additional inner partitions

The reserved outer test set is identical across these experiments. Values below average generator seeds within each partition; do not interpret the overlapping partitions as independent test cohorts.

| Dataset | Inner seed | Arm | RF AUROC ↑ | Rule violation ↓ | Numeric W ↓ | Association RMSE ↓ |
|---|---|---|---|---|---|---|
| ilpd | 2027 | paired_dense | 0.654 | 0.376 | 0.414 | 0.188 |
| ilpd | 3027 | paired_base | 0.694 | 0.761 | 0.306 | 0.242 |
| ilpd | 3027 | paired_dense | 0.692 | 0.570 | 0.341 | 0.191 |
| ilpd | 3027 | paired_full | 0.683 | 0.191 | 0.408 | 0.200 |
| ilpd | 3027 | velocity_full | 0.693 | 0.633 | 0.220 | 0.137 |
| ilpd | 4027 | paired_base | 0.692 | 0.779 | 0.282 | 0.228 |
| ilpd | 4027 | paired_dense | 0.666 | 0.417 | 0.328 | 0.172 |
| ilpd | 4027 | paired_full | 0.661 | 0.277 | 0.303 | 0.237 |
| ilpd | 4027 | velocity_full | 0.713 | 0.824 | 0.227 | 0.184 |
| toy | 2027 | paired_dense | 0.790 | 0.846 | 0.395 | 0.090 |
| toy | 3027 | paired_base | 0.731 | 0.987 | 0.302 | 0.096 |
| toy | 3027 | paired_dense | 0.765 | 0.908 | 0.326 | 0.099 |
| toy | 3027 | paired_full | 0.667 | 0.714 | 0.355 | 0.055 |
| toy | 3027 | velocity_full | 0.763 | 0.843 | 0.235 | 0.076 |
| toy | 4027 | paired_base | 0.795 | 0.992 | 0.343 | 0.059 |
| toy | 4027 | paired_dense | 0.756 | 0.866 | 0.291 | 0.071 |
| toy | 4027 | paired_full | 0.778 | 0.839 | 0.304 | 0.084 |
| toy | 4027 | velocity_full | 0.789 | 0.772 | 0.221 | 0.063 |

## Extended TabDDPM training diagnostic

This single-seed check uses 8,500 epochs and is separate from the repeated-seed comparisons. It diagnoses an undertraining hypothesis; it is not a tuned final baseline.

| Dataset | Arm | Selected epoch | RF AUROC ↑ | Rule violation ↓ | Numeric W ↓ | Association RMSE ↓ |
|---|---|---|---|---|---|---|
| ilpd | tabddpm_long | 1020 | 0.708 | 0.986 | 3.260 | 0.245 |
| toy | tabddpm_long | 3128 | 0.755 | 0.974 | 0.558 | 0.212 |
