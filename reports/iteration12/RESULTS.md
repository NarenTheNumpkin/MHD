# Iteration 12 results: controlled study of pitfalls

Generated from `runs/iteration12` (240 jobs, all complete) by `scripts/report_iteration12.py`; protocol `docs/ITERATION_12_PROTOCOL.md`. Means over five independent training samples per setting; the reference half of each population is the evaluation set. Lower is better for C2ST, distances, gaps and membership AUROC.

## Generators: nhanes, n = 250

| variant | c2st_mean | c2st_std | numeric_w_mean | association_mean | rf_mean | violation_mean | violation_gap_mean | relational_w1_mean | membership_nn_mean | membership_density_mean | exact_copy_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MedHyperDiff corrected | 0.731 | 0.024 | 0.105 | 0.058 | 0.879 | 0.962 | 0.057 | 16.304 | 0.512 | 0.572 | 0.000 |
| MedHyperDiff early stopping | 0.951 | 0.010 | 0.162 | 0.153 | 0.836 | 0.998 | 0.081 | 213.079 | 0.461 | 0.512 | 0.000 |
| MedHyperDiff affine | 0.701 | 0.013 | 0.106 | 0.061 | 0.881 | 0.917 | 0.051 | 9.484 | 0.520 | 0.588 | 0.000 |
| TabDDPM upstream | 0.911 | 0.010 | 0.379 | 0.131 | 0.874 | 0.975 | 0.063 | 46.903 | 0.891 | 0.853 | 0.000 |
| TabDDPM adapter | 1.000 | 0.000 | 2.829 | 0.148 | 0.545 | 1.000 | 0.082 | 587.718 | 0.470 | 0.506 | 0.000 |
| TabDiff | 0.810 | 0.068 | 0.123 | 0.079 | 0.807 | 0.978 | 0.066 | 59.087 | 0.548 | 0.614 | 0.000 |
| CTGAN | 0.990 | 0.004 | 0.506 | 0.148 | 0.803 | 0.994 | 0.094 | 276.085 | 0.480 | 0.491 | 0.000 |
| TVAE | 0.915 | 0.011 | 0.243 | 0.108 | 0.858 | 1.000 | 0.080 | 112.614 | 0.498 | 0.552 | 0.000 |
| Gaussian copula | 0.779 | 0.009 | 0.100 | 0.057 | 0.873 | 0.992 | 0.074 | 55.420 | 0.509 | 0.548 | 0.000 |
| Copula + re-derivation | 0.759 | 0.013 | 0.101 | 0.058 | 0.874 | 0.001 | 0.000 | 0.064 | 0.502 | 0.549 | 0.000 |
| Independent marginals | 0.956 | 0.004 | 0.106 | 0.157 | 0.868 | 0.999 | 0.081 | 203.193 | 0.482 | 0.524 | 0.000 |
| Bootstrap copy control | 0.482 | 0.019 | 0.107 | 0.058 | 0.869 | 0.000 | 0.000 | 0.070 | 0.998 | 0.979 | 1.000 |

## Generators: nhanes, n = 1000

| variant | c2st_mean | c2st_std | numeric_w_mean | association_mean | rf_mean | violation_mean | violation_gap_mean | relational_w1_mean | membership_nn_mean | membership_density_mean | exact_copy_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MedHyperDiff corrected | 0.692 | 0.018 | 0.073 | 0.053 | 0.895 | 0.935 | 0.052 | 10.593 | 0.483 | 0.517 | 0.000 |
| MedHyperDiff early stopping | 0.857 | 0.059 | 0.092 | 0.117 | 0.885 | 0.968 | 0.065 | 56.615 | 0.481 | 0.504 | 0.000 |
| MedHyperDiff affine | 0.716 | 0.010 | 0.076 | 0.054 | 0.893 | 0.910 | 0.050 | 7.619 | 0.488 | 0.516 | 0.000 |
| TabDDPM upstream | 0.738 | 0.024 | 0.283 | 0.071 | 0.897 | 0.963 | 0.058 | 27.908 | 0.583 | 0.612 | 0.000 |
| TabDDPM adapter | 1.000 | 0.001 | 3.674 | 0.162 | 0.704 | 0.995 | 0.079 | 402.903 | 0.489 | 0.497 | 0.000 |
| TabDiff | 0.703 | 0.039 | 0.070 | 0.040 | 0.896 | 0.896 | 0.049 | 7.095 | 0.574 | 0.595 | 0.000 |
| CTGAN | 0.988 | 0.004 | 0.422 | 0.151 | 0.776 | 0.989 | 0.086 | 290.711 | 0.489 | 0.503 | 0.000 |
| TVAE | 0.913 | 0.015 | 0.224 | 0.085 | 0.873 | 0.999 | 0.080 | 119.584 | 0.492 | 0.525 | 0.000 |
| Gaussian copula | 0.711 | 0.004 | 0.062 | 0.039 | 0.892 | 0.992 | 0.073 | 49.503 | 0.488 | 0.512 | 0.000 |
| Copula + re-derivation | 0.705 | 0.005 | 0.062 | 0.039 | 0.893 | 0.002 | 0.000 | 0.047 | 0.491 | 0.511 | 0.000 |
| Independent marginals | 0.965 | 0.004 | 0.062 | 0.162 | 0.885 | 0.998 | 0.081 | 203.520 | 0.483 | 0.507 | 0.000 |
| Bootstrap copy control | 0.504 | 0.010 | 0.063 | 0.036 | 0.888 | 0.000 | 0.000 | 0.046 | 0.996 | 0.968 | 1.000 |

## Generators: thyroid, n = 250

| variant | c2st_mean | c2st_std | numeric_w_mean | association_mean | rf_mean | violation_mean | violation_gap_mean | relational_w1_mean | membership_nn_mean | membership_density_mean | exact_copy_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MedHyperDiff corrected | 0.707 | 0.019 | 0.142 | 0.022 | 0.916 | 0.701 | 0.104 | 2.851 | 0.550 | 0.622 | 0.001 |
| MedHyperDiff early stopping | 0.866 | 0.004 | 0.145 | 0.070 | 0.686 | 0.742 | 0.133 | 24.114 | 0.531 | 0.519 | 0.000 |
| MedHyperDiff affine | 0.738 | 0.013 | 0.141 | 0.022 | 0.914 | 0.621 | 0.102 | 1.307 | 0.546 | 0.596 | 0.001 |
| TabDDPM upstream | 0.832 | 0.014 | 0.336 | 0.034 | 0.922 | 0.760 | 0.109 | 6.822 | 0.875 | 0.883 | 0.005 |
| TabDDPM adapter | 1.000 | 0.000 | 2.892 | 0.071 | 0.486 | 0.934 | 0.133 | 132.764 | 0.478 | 0.497 | 0.000 |
| TabDiff | 0.742 | 0.016 | 0.156 | 0.022 | 0.889 | 0.745 | 0.112 | 4.298 | 0.591 | 0.686 | 0.002 |
| CTGAN | 0.954 | 0.025 | 0.522 | 0.064 | 0.805 | 0.776 | 0.197 | 161.068 | 0.503 | 0.506 | 0.000 |
| TVAE | 0.929 | 0.005 | 0.275 | 0.057 | 0.828 | 0.957 | 0.140 | 13.843 | 0.498 | 0.528 | 0.000 |
| Gaussian copula | 0.768 | 0.012 | 0.161 | 0.025 | 0.911 | 0.810 | 0.127 | 8.304 | 0.537 | 0.573 | 0.000 |
| Copula + re-derivation | 0.763 | 0.021 | 0.156 | 0.025 | 0.908 | 0.001 | 0.007 | 0.202 | 0.535 | 0.569 | 0.000 |
| Independent marginals | 0.871 | 0.008 | 0.158 | 0.071 | 0.903 | 0.750 | 0.133 | 24.994 | 0.532 | 0.518 | 0.000 |
| Bootstrap copy control | 0.494 | 0.050 | 0.133 | 0.030 | 0.938 | 0.042 | 0.001 | 0.147 | 0.995 | 0.941 | 0.999 |

## Generators: thyroid, n = 1000

| variant | c2st_mean | c2st_std | numeric_w_mean | association_mean | rf_mean | violation_mean | violation_gap_mean | relational_w1_mean | membership_nn_mean | membership_density_mean | exact_copy_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MedHyperDiff corrected | 0.685 | 0.011 | 0.058 | 0.014 | 0.929 | 0.619 | 0.091 | 2.097 | 0.506 | 0.528 | 0.001 |
| MedHyperDiff early stopping | 0.757 | 0.012 | 0.081 | 0.026 | 0.909 | 0.758 | 0.118 | 5.022 | 0.502 | 0.513 | 0.000 |
| MedHyperDiff affine | 0.758 | 0.014 | 0.079 | 0.017 | 0.928 | 0.647 | 0.108 | 1.581 | 0.499 | 0.519 | 0.001 |
| TabDDPM upstream | 0.745 | 0.014 | 0.160 | 0.017 | 0.944 | 0.709 | 0.103 | 7.391 | 0.653 | 0.713 | 0.003 |
| TabDDPM adapter | 0.995 | 0.008 | 3.827 | 0.086 | 0.545 | 0.920 | 0.126 | 128.715 | 0.515 | 0.513 | 0.000 |
| TabDiff | 0.692 | 0.017 | 0.059 | 0.012 | 0.938 | 0.533 | 0.077 | 1.572 | 0.547 | 0.613 | 0.002 |
| CTGAN | 0.961 | 0.012 | 0.505 | 0.061 | 0.837 | 0.763 | 0.168 | 248.113 | 0.497 | 0.496 | 0.000 |
| TVAE | 0.934 | 0.008 | 0.203 | 0.041 | 0.861 | 0.953 | 0.144 | 16.420 | 0.490 | 0.504 | 0.000 |
| Gaussian copula | 0.776 | 0.013 | 0.083 | 0.017 | 0.940 | 0.812 | 0.126 | 8.123 | 0.497 | 0.514 | 0.000 |
| Copula + re-derivation | 0.760 | 0.006 | 0.082 | 0.017 | 0.938 | 0.000 | 0.007 | 0.239 | 0.493 | 0.512 | 0.000 |
| Independent marginals | 0.903 | 0.008 | 0.080 | 0.074 | 0.935 | 0.775 | 0.133 | 26.297 | 0.507 | 0.501 | 0.000 |
| Bootstrap copy control | 0.504 | 0.017 | 0.054 | 0.014 | 0.964 | 0.047 | 0.001 | 0.131 | 0.997 | 0.908 | 0.998 |

## Analysis 1: checkpoint selection (P1, P5)

| variant | fits | cap | early_fraction | cap_fraction | median_selected | min_selected |
|---|---:|---:|---:|---:|---:|---:|
| mhd_earlystop | 20.000 | 100.000 | 0.300 | 0.450 | 46.500 | 3.000 |
| tabddpm_adapter | 20.000 | 300.000 | 0.250 | 0.200 | 149.500 | 19.000 |

| variant | counterpart | metric | dataset | size | mean_difference | sd | replicates_worse | replicates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mhd_earlystop | mhd_corrected | c2st | nhanes | 250.000 | 0.220 | 0.028 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | c2st | nhanes | 1000.000 | 0.165 | 0.071 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | c2st | thyroid | 250.000 | 0.159 | 0.018 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | c2st | thyroid | 1000.000 | 0.072 | 0.012 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | numeric_w | nhanes | 250.000 | 0.056 | 0.050 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | numeric_w | nhanes | 1000.000 | 0.019 | 0.013 | 4.000 | 5.000 |
| mhd_earlystop | mhd_corrected | numeric_w | thyroid | 250.000 | 0.003 | 0.007 | 4.000 | 5.000 |
| mhd_earlystop | mhd_corrected | numeric_w | thyroid | 1000.000 | 0.024 | 0.014 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | association | nhanes | 250.000 | 0.095 | 0.004 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | association | nhanes | 1000.000 | 0.064 | 0.029 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | association | thyroid | 250.000 | 0.048 | 0.006 | 5.000 | 5.000 |
| mhd_earlystop | mhd_corrected | association | thyroid | 1000.000 | 0.012 | 0.004 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | c2st | nhanes | 250.000 | 0.089 | 0.010 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | c2st | nhanes | 1000.000 | 0.261 | 0.025 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | c2st | thyroid | 250.000 | 0.168 | 0.014 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | c2st | thyroid | 1000.000 | 0.250 | 0.014 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | numeric_w | nhanes | 250.000 | 2.449 | 0.213 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | numeric_w | nhanes | 1000.000 | 3.391 | 0.551 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | numeric_w | thyroid | 250.000 | 2.555 | 0.309 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | numeric_w | thyroid | 1000.000 | 3.667 | 0.305 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | association | nhanes | 250.000 | 0.017 | 0.031 | 3.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | association | nhanes | 1000.000 | 0.091 | 0.035 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | association | thyroid | 250.000 | 0.037 | 0.012 | 5.000 | 5.000 |
| tabddpm_adapter | tabddpm_upstream | association | thyroid | 1000.000 | 0.069 | 0.027 | 5.000 | 5.000 |

## Analysis 2: affine vs quantile coding (P2)

| dataset | feature | size | affine | quantile | replicates_quantile_better | replicates |
|---|---:|---:|---:|---:|---:|---:|
| nhanes | creatinine | 250 | 0.241 | 0.244 | 1 | 5 |
| nhanes | creatinine | 1000 | 0.114 | 0.111 | 3 | 5 |
| nhanes | ggt | 250 | 0.155 | 0.111 | 4 | 5 |
| nhanes | ggt | 1000 | 0.133 | 0.102 | 5 | 5 |
| nhanes | triglycerides | 250 | 0.080 | 0.069 | 2 | 5 |
| nhanes | triglycerides | 1000 | 0.078 | 0.062 | 4 | 5 |
| nhanes | ast | 250 | 0.092 | 0.098 | 2 | 5 |
| nhanes | ast | 1000 | 0.115 | 0.091 | 5 | 5 |
| nhanes | glucose | 250 | 0.076 | 0.076 | 3 | 5 |
| nhanes | glucose | 1000 | 0.036 | 0.065 | 0 | 5 |
| nhanes | c2st | 250 | 0.701 | 0.731 | 0 | 5 |
| nhanes | c2st | 1000 | 0.716 | 0.692 | 5 | 5 |
| thyroid | tsh | 250 | 0.207 | 0.179 | 5 | 5 |
| thyroid | tsh | 1000 | 0.077 | 0.038 | 5 | 5 |
| thyroid | fti | 250 | 0.129 | 0.135 | 1 | 5 |
| thyroid | fti | 1000 | 0.074 | 0.070 | 4 | 5 |
| thyroid | t3 | 250 | 0.188 | 0.199 | 0 | 5 |
| thyroid | t3 | 1000 | 0.095 | 0.082 | 4 | 5 |
| thyroid | tt4 | 250 | 0.129 | 0.150 | 1 | 5 |
| thyroid | tt4 | 1000 | 0.069 | 0.067 | 2 | 5 |
| thyroid | t4u | 250 | 0.092 | 0.099 | 1 | 5 |
| thyroid | t4u | 1000 | 0.063 | 0.053 | 3 | 5 |
| thyroid | c2st | 250 | 0.738 | 0.707 | 5 | 5 |
| thyroid | c2st | 1000 | 0.758 | 0.685 | 5 | 5 |

## Analysis 3: recording resolution (P3)

C2ST against the full reference by decoding:

| dataset | size | variant | c2st_raw | c2st_global | c2st_banded |
|---|---:|---:|---:|---:|---:|
| nhanes | 250 | CTGAN | 0.992 | 0.990 | 0.990 |
| nhanes | 250 | MedHyperDiff affine | 0.984 | 0.701 | 0.701 |
| nhanes | 250 | MedHyperDiff corrected | 0.979 | 0.731 | 0.731 |
| nhanes | 250 | MedHyperDiff early stopping | 0.991 | 0.951 | 0.951 |
| nhanes | 250 | TabDDPM adapter | 1.000 | 1.000 | 1.000 |
| nhanes | 250 | TabDDPM upstream | 0.912 | 0.911 | 0.911 |
| nhanes | 250 | TabDiff | 0.810 | 0.810 | 0.810 |
| nhanes | 250 | TVAE | 0.969 | 0.915 | 0.915 |
| nhanes | 1000 | CTGAN | 0.992 | 0.988 | 0.988 |
| nhanes | 1000 | MedHyperDiff affine | 0.999 | 0.716 | 0.716 |
| nhanes | 1000 | MedHyperDiff corrected | 0.999 | 0.692 | 0.692 |
| nhanes | 1000 | MedHyperDiff early stopping | 0.998 | 0.857 | 0.857 |
| nhanes | 1000 | TabDDPM adapter | 1.000 | 1.000 | 1.000 |
| nhanes | 1000 | TabDDPM upstream | 0.744 | 0.738 | 0.738 |
| nhanes | 1000 | TabDiff | 0.703 | 0.703 | 0.703 |
| nhanes | 1000 | TVAE | 0.991 | 0.913 | 0.913 |
| thyroid | 250 | CTGAN | 0.956 | 0.956 | 0.954 |
| thyroid | 250 | MedHyperDiff affine | 0.772 | 0.743 | 0.738 |
| thyroid | 250 | MedHyperDiff corrected | 0.749 | 0.721 | 0.707 |
| thyroid | 250 | MedHyperDiff early stopping | 0.886 | 0.872 | 0.866 |
| thyroid | 250 | TabDDPM adapter | 1.000 | 1.000 | 1.000 |
| thyroid | 250 | TabDDPM upstream | 0.840 | 0.833 | 0.832 |
| thyroid | 250 | TabDiff | 0.752 | 0.752 | 0.742 |
| thyroid | 250 | TVAE | 0.941 | 0.933 | 0.929 |
| thyroid | 1000 | CTGAN | 0.969 | 0.966 | 0.961 |
| thyroid | 1000 | MedHyperDiff affine | 0.831 | 0.802 | 0.758 |
| thyroid | 1000 | MedHyperDiff corrected | 0.808 | 0.763 | 0.685 |
| thyroid | 1000 | MedHyperDiff early stopping | 0.836 | 0.806 | 0.757 |
| thyroid | 1000 | TabDDPM adapter | 0.995 | 0.995 | 0.995 |
| thyroid | 1000 | TabDDPM upstream | 0.742 | 0.747 | 0.745 |
| thyroid | 1000 | TabDiff | 0.698 | 0.696 | 0.692 |
| thyroid | 1000 | TVAE | 0.950 | 0.947 | 0.934 |

Ranking by raw vs banded C2ST (11 generators, copy control excluded):

| dataset | size | reversal | tau | top_raw | top_banded |
|---|---:|---:|---:|---:|---:|
| nhanes | 250 | 0.291 | 0.418 | {'copula_repair': 4, 'tabdiff': 1} | {'mhd_affine': 5} |
| nhanes | 1000 | 0.324 | 0.353 | {'tabdiff': 3, 'copula_repair': 2} | {'mhd_corrected': 3, 'tabdiff': 2} |
| thyroid | 250 | 0.065 | 0.869 | {'tabdiff': 2, 'copula_repair': 1, 'mhd_corrected': 1, 'mhd_affine': 1} | {'mhd_corrected': 5} |
| thyroid | 1000 | 0.142 | 0.716 | {'tabdiff': 5} | {'mhd_corrected': 3, 'tabdiff': 2} |

## Analysis 4: rule scoring (P4)

| comparison | dataset | size | reversal | tau | top_violation | top_realism |
|---|---:|---:|---:|---:|---:|---:|
| violation_vs_gap | nhanes | 250 | 0.137 | 0.708 | copula_repair | copula_repair |
| violation_vs_gap | nhanes | 1000 | 0.114 | 0.770 | copula_repair | copula_repair |
| violation_vs_gap | thyroid | 250 | 0.215 | 0.571 | copula_repair | copula_repair |
| violation_vs_gap | thyroid | 1000 | 0.153 | 0.695 | copula_repair | copula_repair |
| violation_vs_residual_w1 | nhanes | 250 | 0.107 | 0.768 | copula_repair | copula_repair |
| violation_vs_residual_w1 | nhanes | 1000 | 0.114 | 0.770 | copula_repair | copula_repair |
| violation_vs_residual_w1 | thyroid | 250 | 0.215 | 0.571 | copula_repair | copula_repair |
| violation_vs_residual_w1 | thyroid | 1000 | 0.167 | 0.665 | copula_repair | copula_repair |

## Analysis 5: memorization (P6)

Lowest-C2ST generator per training sample and whether either attack exceeds 0.60:

| include_copy_control | dataset | size | memorizing_fraction | best |
|---|---:|---:|---:|---:|
| False | nhanes | 250 | 0.400 | {'mhd_affine': 5} |
| False | nhanes | 1000 | 0.000 | {'mhd_corrected': 3, 'tabdiff': 2} |
| False | thyroid | 250 | 0.800 | {'mhd_corrected': 5} |
| False | thyroid | 1000 | 0.200 | {'mhd_corrected': 3, 'tabdiff': 2} |
| True | nhanes | 250 | 1.000 | {'bootstrap': 5} |
| True | nhanes | 1000 | 1.000 | {'bootstrap': 5} |
| True | thyroid | 250 | 1.000 | {'bootstrap': 5} |
| True | thyroid | 1000 | 1.000 | {'bootstrap': 5} |

| dataset | size | spearman_c2st_vs_membership_nn | spearman_c2st_vs_membership_density |
|---|---:|---:|---:|
| nhanes | 250 | -0.604 | -0.643 |
| nhanes | 1000 | -0.326 | -0.533 |
| thyroid | 250 | -0.592 | -0.766 |
| thyroid | 1000 | -0.474 | -0.710 |

Iteration-11 search (C2ST-best configuration per cohort and method):

| dataset | method | best_c2st | best_membership | memorizing | spearman |
|---|---:|---:|---:|---:|---:|
| ckd | ctgan | 0.852 | 0.498 | False | -0.349 |
| ckd | mhd | 0.595 | 0.468 | False | -0.048 |
| ckd | tabddpm | 0.717 | 0.933 | True | -0.663 |
| ckd | tabdiff | 0.596 | 0.691 | True | 0.357 |
| ckd | tvae | 0.629 | 0.553 | False | -0.595 |
| diabetes | ctgan | 0.636 | 0.585 | False | 0.253 |
| diabetes | mhd | 0.471 | 0.642 | True | -0.262 |
| diabetes | tabddpm | 0.474 | 0.964 | True | -0.833 |
| diabetes | tabdiff | 0.482 | 0.862 | True | -0.143 |
| diabetes | tvae | 0.569 | 0.808 | True | -0.929 |
| heart_failure | ctgan | 0.830 | 0.532 | False | -0.904 |
| heart_failure | mhd | 0.553 | 0.544 | False | 0.643 |
| heart_failure | tabddpm | 0.507 | 0.986 | True | -0.946 |
| heart_failure | tabdiff | 0.523 | 0.743 | True | -0.810 |
| heart_failure | tvae | 0.686 | 0.600 | True | -0.952 |
| ilpd | ctgan | 0.922 | 0.528 | False | 0.012 |
| ilpd | mhd | 0.537 | 0.554 | False | -0.024 |
| ilpd | tabddpm | 0.539 | 0.687 | True | -0.635 |
| ilpd | tabdiff | 0.481 | 0.550 | False | -0.024 |
| ilpd | tvae | 0.680 | 0.556 | False | -0.048 |

## Analysis 6: evaluation size (P7)

Probability that one C2ST draw at m records per source orders a generator pair as the full reference does:

| dataset | eval_size | agreement |
|---|---:|---:|
| nhanes | 50 | 0.771 |
| nhanes | 100 | 0.850 |
| nhanes | 200 | 0.879 |
| nhanes | 500 | 0.930 |
| nhanes | 1000 | 0.963 |
| thyroid | 50 | 0.784 |
| thyroid | 100 | 0.842 |
| thyroid | 200 | 0.896 |
| thyroid | 500 | 0.940 |
| thyroid | 1000 | 0.951 |

By true C2ST difference between the pair:

| eval_size | (-0.001, 0.02] | (0.02, 0.05] | (0.05, 0.1] | (0.1, 1.0] |
|---|---:|---:|---:|---:|
| 50.000 | 0.575 | 0.620 | 0.730 | 0.883 |
| 100.000 | 0.591 | 0.671 | 0.798 | 0.968 |
| 200.000 | 0.580 | 0.736 | 0.894 | 0.996 |
| 500.000 | 0.647 | 0.873 | 0.981 | 1.000 |
| 1000.000 | 0.706 | 0.945 | 0.998 | 1.000 |

Median single-draw C2ST SD by m: {"50": 0.062, "100": 0.045, "200": 0.031, "500": 0.015, "1000": 0.008}

## Analysis 7: common vs corrected practice

| comparison | dataset | size | reversal | reversal_low | reversal_high | tau | tau_low | tau_high | top1_agreement | top_common | top_corrected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fidelity, privacy-aware ranking | nhanes | 250 | 0.271 | 0.171 | 0.350 | 0.457 | 0.300 | 0.657 | 0.200 | Copula+repair | MedHyperDiff |
| Fidelity, privacy-aware ranking | nhanes | 1000 | 0.336 | 0.250 | 0.436 | 0.329 | 0.129 | 0.500 | 0.000 | Copula | MedHyperDiff |
| Fidelity, privacy-aware ranking | thyroid | 250 | 0.364 | 0.307 | 0.450 | 0.271 | 0.100 | 0.386 | 0.000 | TabDiff | Copula+repair |
| Fidelity, privacy-aware ranking | thyroid | 1000 | 0.200 | 0.107 | 0.286 | 0.600 | 0.429 | 0.786 | 0.200 | TabDiff | MedHyperDiff |
| Fidelity: common vs corrected practice | nhanes | 250 | 0.321 | 0.293 | 0.350 | 0.357 | 0.300 | 0.414 | 0.000 | Copula+repair | MedHyperDiff |
| Fidelity: common vs corrected practice | nhanes | 1000 | 0.336 | 0.293 | 0.379 | 0.329 | 0.243 | 0.414 | 0.000 | Copula | MedHyperDiff |
| Fidelity: common vs corrected practice | thyroid | 250 | 0.250 | 0.221 | 0.279 | 0.500 | 0.443 | 0.557 | 0.000 | TabDiff | MedHyperDiff |
| Fidelity: common vs corrected practice | thyroid | 1000 | 0.264 | 0.221 | 0.307 | 0.471 | 0.386 | 0.557 | 0.400 | TabDiff | MedHyperDiff |
| Rules: violation count vs gap to real | nhanes | 250 | 0.386 | 0.326 | 0.453 | 0.217 | 0.089 | 0.330 | 1.000 | Copula+repair | Copula+repair |
| Rules: violation count vs gap to real | nhanes | 1000 | 0.240 | 0.188 | 0.278 | 0.517 | 0.440 | 0.623 | 1.000 | Copula+repair | Copula+repair |
| Rules: violation count vs gap to real | thyroid | 250 | 0.329 | 0.279 | 0.386 | 0.343 | 0.229 | 0.443 | 1.000 | Copula+repair | Copula+repair |
| Rules: violation count vs gap to real | thyroid | 1000 | 0.271 | 0.221 | 0.321 | 0.457 | 0.357 | 0.557 | 1.000 | Copula+repair | Copula+repair |

