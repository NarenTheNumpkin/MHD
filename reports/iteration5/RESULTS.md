# Iteration 5: complete observation-head experiment results

36 new fits, 108 new evaluations and 81 reused 5x references. Three sampling seeds are averaged within each training seed before averaging three training seeds per partition. These overlapping development partitions are not independent cohorts.

| arm | inner_seed | rf | logistic | numeric_w | violation | ag_missing | protein_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_prior | 2027 | 0.7211 | 0.6982 | 0.3924 | 0.0902 | 0.0080 | 0.9920 |
| base_prior | 3027 | 0.7102 | 0.6992 | 0.2701 | 0.0789 | 0.0082 | 0.9918 |
| base_prior | 4027 | 0.6947 | 0.6893 | 0.2777 | 0.0766 | 0.0044 | 0.9956 |
| dense_prior | 2027 | 0.7700 | 0.7595 | 0.2192 | 0.0669 | 0.0074 | 0.9926 |
| dense_prior | 3027 | 0.6507 | 0.7053 | 0.1985 | 0.0627 | 0.0086 | 0.9914 |
| dense_prior | 4027 | 0.6826 | 0.6893 | 0.2083 | 0.0789 | 0.0046 | 0.9954 |
| full_prior | 2027 | 0.7383 | 0.7450 | 0.3049 | 0.0550 | 0.0085 | 0.9915 |
| full_prior | 3027 | 0.6991 | 0.7126 | 0.2037 | 0.0497 | 0.0082 | 0.9918 |
| full_prior | 4027 | 0.7076 | 0.6886 | 0.2223 | 0.0596 | 0.0045 | 0.9955 |
| full_prior_w4 | 2027 | 0.7324 | 0.7425 | 0.3039 | 0.0546 | 0.0082 | 0.9918 |
| full_prior_w4 | 3027 | 0.7198 | 0.7193 | 0.2419 | 0.0500 | 0.0069 | 0.9931 |
| full_prior_w4 | 4027 | 0.6920 | 0.6754 | 0.2784 | 0.0648 | 0.0048 | 0.9952 |
| reference_guided_base | 2027 | 0.7030 | 0.6871 | 0.3983 | 0.0879 | 0.0179 | 0.9821 |
| reference_guided_base | 3027 | 0.7185 | 0.7145 | 0.2625 | 0.0653 | 0.0086 | 0.9914 |
| reference_guided_base | 4027 | 0.6990 | 0.6927 | 0.2800 | 0.0746 | 0.0098 | 0.9902 |
| reference_matched_dense | 2027 | 0.7581 | 0.7497 | 0.2242 | 0.0506 | 0.0080 | 0.9920 |
| reference_matched_dense | 3027 | 0.6695 | 0.6947 | 0.2024 | 0.0593 | 0.0087 | 0.9913 |
| reference_matched_dense | 4027 | 0.6824 | 0.7081 | 0.1770 | 0.0708 | 0.0058 | 0.9942 |
| reference_updated_full | 2027 | 0.7441 | 0.7442 | 0.3015 | 0.0518 | 0.0239 | 0.9761 |
| reference_updated_full | 3027 | 0.6905 | 0.7060 | 0.2135 | 0.0521 | 0.0082 | 0.9918 |
| reference_updated_full | 4027 | 0.7017 | 0.6947 | 0.2324 | 0.0608 | 0.0086 | 0.9914 |

## Correction gates

| variant | inner_seed | partition_pass | no_missingness_flags | delta_rf | delta_logistic | max_sd_error_increase |
| --- | --- | --- | --- | --- | --- | --- |
| full_prior | 2027 | True | True | -0.0058 | 0.0008 | 0.0404 |
| full_prior | 3027 | True | True | 0.0086 | 0.0066 | 0.0290 |
| full_prior | 4027 | False | True | 0.0060 | -0.0061 | 0.1210 |
| full_prior_w4 | 2027 | True | True | -0.0117 | -0.0016 | 0.0415 |
| full_prior_w4 | 3027 | False | True | 0.0293 | 0.0133 | 0.1452 |
| full_prior_w4 | 4027 | False | True | -0.0097 | -0.0192 | 0.1062 |
| dense_prior | 2027 | False | True | 0.0119 | 0.0098 | 0.0323 |
| dense_prior | 3027 | True | True | -0.0188 | 0.0106 | 0.0419 |
| dense_prior | 4027 | False | True | 0.0002 | -0.0188 | 0.0454 |
| base_prior | 2027 | True | True | 0.0181 | 0.0111 | 0.0196 |
| base_prior | 3027 | False | True | -0.0083 | -0.0152 | 0.0293 |
| base_prior | 4027 | True | True | -0.0043 | -0.0034 | 0.0213 |

All gate components and per-feature/rule results are retained in the accompanying CSVs. The selection JSON applies the predeclared full-variant criteria. Missingness correction alone is insufficient for promotion. Unknown-rule intervals are logical bounds, not confidence intervals.
