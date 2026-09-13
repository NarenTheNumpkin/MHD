# Iteration 4: all sample-size and independent-sampling results

Completed 216 evaluations of 108 nested synthetic pools from 36 existing fitted checkpoints; **zero new generator training runs**. All outcomes use inner validation. Means average three sampling seeds within each training seed, then three training seeds, separately by partition. Overlapping partitions are not independent cohorts.

| arm | inner_seed | multiplier | rf | logistic | numeric_w | violation | violation_upper | protein_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guided_base | 2027 | 1 | 0.7061 | 0.6947 | 0.3989 | 0.0885 | 0.1063 | 0.9822 |
| guided_base | 2027 | 5 | 0.7030 | 0.6871 | 0.3983 | 0.0879 | 0.1058 | 0.9821 |
| guided_base | 3027 | 1 | 0.7125 | 0.7128 | 0.2657 | 0.0592 | 0.0716 | 0.9876 |
| guided_base | 3027 | 5 | 0.7185 | 0.7145 | 0.2625 | 0.0653 | 0.0737 | 0.9914 |
| guided_base | 4027 | 1 | 0.7032 | 0.6950 | 0.2882 | 0.0721 | 0.0851 | 0.9870 |
| guided_base | 4027 | 5 | 0.6990 | 0.6927 | 0.2800 | 0.0746 | 0.0844 | 0.9902 |
| matched_dense | 2027 | 1 | 0.7486 | 0.7508 | 0.2258 | 0.0462 | 0.0573 | 0.9889 |
| matched_dense | 2027 | 5 | 0.7581 | 0.7497 | 0.2242 | 0.0506 | 0.0586 | 0.9920 |
| matched_dense | 3027 | 1 | 0.6717 | 0.6863 | 0.2140 | 0.0643 | 0.0764 | 0.9879 |
| matched_dense | 3027 | 5 | 0.6695 | 0.6947 | 0.2024 | 0.0593 | 0.0679 | 0.9913 |
| matched_dense | 4027 | 1 | 0.6830 | 0.7055 | 0.1904 | 0.0692 | 0.0771 | 0.9921 |
| matched_dense | 4027 | 5 | 0.6824 | 0.7081 | 0.1770 | 0.0708 | 0.0766 | 0.9942 |
| previous_full | 2027 | 1 | 0.7287 | 0.7618 | 0.3962 | 0.1500 | 0.1592 | 0.9895 |
| previous_full | 2027 | 5 | 0.7048 | 0.7598 | 0.3881 | 0.1503 | 0.1568 | 0.9926 |
| previous_full | 3027 | 1 | 0.6904 | 0.6782 | 0.4202 | 0.2213 | 0.2334 | 0.9879 |
| previous_full | 3027 | 5 | 0.6711 | 0.6861 | 0.4098 | 0.2134 | 0.2221 | 0.9913 |
| previous_full | 4027 | 1 | 0.6550 | 0.6234 | 0.3135 | 0.2629 | 0.2708 | 0.9914 |
| previous_full | 4027 | 5 | 0.6762 | 0.6094 | 0.3017 | 0.2604 | 0.2659 | 0.9942 |
| updated_full | 2027 | 1 | 0.7389 | 0.7476 | 0.3044 | 0.0544 | 0.0770 | 0.9774 |
| updated_full | 2027 | 5 | 0.7441 | 0.7442 | 0.3015 | 0.0518 | 0.0757 | 0.9761 |
| updated_full | 3027 | 1 | 0.6941 | 0.7055 | 0.2232 | 0.0468 | 0.0589 | 0.9879 |
| updated_full | 3027 | 5 | 0.6905 | 0.7060 | 0.2135 | 0.0521 | 0.0602 | 0.9918 |
| updated_full | 4027 | 1 | 0.7047 | 0.6951 | 0.2410 | 0.0575 | 0.0692 | 0.9883 |
| updated_full | 4027 | 5 | 0.7017 | 0.6947 | 0.2324 | 0.0608 | 0.0692 | 0.9914 |

## Predeclared 5x versus 1x engineering gates

| arm | inner_seed | delta_rf | delta_logistic | partition_pass | flagged_5x_tables |
| --- | --- | --- | --- | --- | --- |
| guided_base | 2027 | -0.0031 | -0.0076 | False | 0 |
| guided_base | 3027 | 0.0060 | 0.0017 | True | 0 |
| guided_base | 4027 | -0.0041 | -0.0024 | False | 0 |
| matched_dense | 2027 | 0.0094 | -0.0011 | True | 0 |
| matched_dense | 3027 | -0.0022 | 0.0084 | True | 0 |
| matched_dense | 4027 | -0.0005 | 0.0026 | True | 0 |
| previous_full | 2027 | -0.0239 | -0.0020 | False | 0 |
| previous_full | 3027 | -0.0193 | 0.0080 | True | 0 |
| previous_full | 4027 | 0.0212 | -0.0140 | True | 0 |
| updated_full | 2027 | 0.0052 | -0.0035 | True | 3 |
| updated_full | 3027 | -0.0036 | 0.0005 | True | 0 |
| updated_full | 4027 | -0.0030 | -0.0004 | False | 0 |

These gates are descriptive development tolerances, not clinical noninferiority tests. Missingness flags mean at least one feature differs from its fit-table missing rate by more than 0.02. Violation upper bounds treat unevaluable rules as possibly violated; they are logical bounds, not confidence intervals.

## Reproducibility

Full records: `all_evaluations.csv`, `all_feature_results.csv`, `all_rule_bounds.csv`, `paired_size_deltas.csv`, `checkpoint_sampling_means.csv`, `sampling_seed_sd_by_checkpoint.csv`, `training_seed_sd.csv`, `size_gates.csv`, `size_decision.json`, `timing.json`. Each evaluation retains its synthetic table and downstream scores. The 5x augmentation outputs are explicitly labelled AUGMENT_5x. No rows were repaired or imputed in released synthetic tables.
