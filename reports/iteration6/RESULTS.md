# Iteration 6: complete reverse-variance sensitivity results

162 new sampling/evaluation runs, 162 reused references, zero new fits. Means average three sampling seeds within each fitted checkpoint, then three training seeds separately by partition. Overlapping partitions are not independent cohorts.

| arm | inner_seed | rf | logistic | numeric_w | violation | ag_missing | protein_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base_m0 | 2027 | 0.7211 | 0.6982 | 0.3924 | 0.0902 | 0.0080 | 0.9920 |
| base_m0 | 3027 | 0.7102 | 0.6992 | 0.2701 | 0.0789 | 0.0082 | 0.9918 |
| base_m0 | 4027 | 0.6947 | 0.6893 | 0.2777 | 0.0766 | 0.0044 | 0.9956 |
| base_m0.5 | 2027 | 0.7157 | 0.6973 | 0.3906 | 0.1519 | 0.0080 | 0.9920 |
| base_m0.5 | 3027 | 0.7012 | 0.7060 | 0.2756 | 0.1358 | 0.0081 | 0.9919 |
| base_m0.5 | 4027 | 0.6928 | 0.6891 | 0.2851 | 0.1379 | 0.0044 | 0.9956 |
| base_m1 | 2027 | 0.7111 | 0.6966 | 0.3907 | 0.2451 | 0.0080 | 0.9920 |
| base_m1 | 3027 | 0.7072 | 0.7092 | 0.2891 | 0.2238 | 0.0081 | 0.9919 |
| base_m1 | 4027 | 0.6972 | 0.6867 | 0.3000 | 0.2241 | 0.0044 | 0.9956 |
| base_original | 2027 | 0.7030 | 0.6871 | 0.3983 | 0.0879 | 0.0179 | 0.9821 |
| base_original | 3027 | 0.7185 | 0.7145 | 0.2625 | 0.0653 | 0.0086 | 0.9914 |
| base_original | 4027 | 0.6990 | 0.6927 | 0.2800 | 0.0746 | 0.0098 | 0.9902 |
| dense_m0 | 2027 | 0.7700 | 0.7595 | 0.2192 | 0.0669 | 0.0074 | 0.9926 |
| dense_m0 | 3027 | 0.6507 | 0.7053 | 0.1985 | 0.0627 | 0.0086 | 0.9914 |
| dense_m0 | 4027 | 0.6826 | 0.6893 | 0.2083 | 0.0789 | 0.0046 | 0.9954 |
| dense_m0.5 | 2027 | 0.7607 | 0.7567 | 0.2169 | 0.1247 | 0.0074 | 0.9926 |
| dense_m0.5 | 3027 | 0.6485 | 0.7086 | 0.2044 | 0.1172 | 0.0086 | 0.9914 |
| dense_m0.5 | 4027 | 0.6903 | 0.6866 | 0.2203 | 0.1362 | 0.0046 | 0.9954 |
| dense_m1 | 2027 | 0.7629 | 0.7504 | 0.2221 | 0.2059 | 0.0074 | 0.9926 |
| dense_m1 | 3027 | 0.6538 | 0.7091 | 0.2190 | 0.1988 | 0.0085 | 0.9915 |
| dense_m1 | 4027 | 0.6869 | 0.6861 | 0.2416 | 0.2203 | 0.0046 | 0.9954 |
| dense_original | 2027 | 0.7581 | 0.7497 | 0.2242 | 0.0506 | 0.0080 | 0.9920 |
| dense_original | 3027 | 0.6695 | 0.6947 | 0.2024 | 0.0593 | 0.0087 | 0.9913 |
| dense_original | 4027 | 0.6824 | 0.7081 | 0.1770 | 0.0708 | 0.0058 | 0.9942 |
| full_m0 | 2027 | 0.7383 | 0.7450 | 0.3049 | 0.0550 | 0.0085 | 0.9915 |
| full_m0 | 3027 | 0.6991 | 0.7126 | 0.2037 | 0.0497 | 0.0082 | 0.9918 |
| full_m0 | 4027 | 0.7076 | 0.6886 | 0.2223 | 0.0596 | 0.0045 | 0.9955 |
| full_m0.5 | 2027 | 0.7374 | 0.7398 | 0.3043 | 0.1009 | 0.0085 | 0.9915 |
| full_m0.5 | 3027 | 0.7042 | 0.7144 | 0.2144 | 0.0901 | 0.0082 | 0.9918 |
| full_m0.5 | 4027 | 0.7025 | 0.6888 | 0.2381 | 0.1103 | 0.0045 | 0.9955 |
| full_m1 | 2027 | 0.7393 | 0.7394 | 0.3085 | 0.1793 | 0.0085 | 0.9915 |
| full_m1 | 3027 | 0.7017 | 0.7164 | 0.2355 | 0.1592 | 0.0082 | 0.9918 |
| full_m1 | 4027 | 0.6908 | 0.6887 | 0.2631 | 0.1876 | 0.0045 | 0.9955 |
| full_original | 2027 | 0.7441 | 0.7442 | 0.3015 | 0.0518 | 0.0239 | 0.9761 |
| full_original | 3027 | 0.6905 | 0.7060 | 0.2135 | 0.0521 | 0.0082 | 0.9918 |
| full_original | 4027 | 0.7017 | 0.6947 | 0.2324 | 0.0608 | 0.0086 | 0.9914 |

## Both-reference comparisons

| arm | inner_seed | reference_kind | partition_reference_pass | delta_rf | delta_logistic | max_sd_error_increase | max_q95_error_increase |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_m0.5 | 2027 | prior | False | -0.0009 | -0.0052 | 0.0162 | 0.0564 |
| full_m0.5 | 3027 | prior | False | 0.0052 | 0.0019 | 0.0256 | 0.0731 |
| full_m0.5 | 4027 | prior | False | -0.0052 | 0.0002 | 0.2955 | 0.4494 |
| full_m0.5 | 2027 | original | False | -0.0067 | -0.0043 | 0.0305 | 0.2176 |
| full_m0.5 | 3027 | original | False | 0.0137 | 0.0085 | 0.0354 | 0.1811 |
| full_m0.5 | 4027 | original | False | 0.0008 | -0.0058 | 0.3268 | 0.4642 |
| full_m1 | 2027 | prior | False | 0.0010 | -0.0056 | 0.0638 | 0.1337 |
| full_m1 | 3027 | prior | False | 0.0026 | 0.0039 | 0.0728 | 0.3225 |
| full_m1 | 4027 | prior | False | -0.0169 | 0.0001 | 0.6995 | 1.1579 |
| full_m1 | 2027 | original | False | -0.0048 | -0.0047 | 0.0542 | 0.1332 |
| full_m1 | 3027 | original | False | 0.0112 | 0.0105 | 0.0825 | 0.4306 |
| full_m1 | 4027 | original | False | -0.0109 | -0.0060 | 0.7308 | 1.1727 |
| dense_m0.5 | 2027 | prior | False | -0.0093 | -0.0029 | 0.0084 | 0.0116 |
| dense_m0.5 | 3027 | prior | False | -0.0022 | 0.0033 | 0.0190 | 0.1113 |
| dense_m0.5 | 4027 | prior | False | 0.0076 | -0.0027 | 0.3074 | 0.4723 |
| dense_m0.5 | 2027 | original | False | 0.0026 | 0.0070 | 0.0251 | 0.2256 |
| dense_m0.5 | 3027 | original | False | -0.0211 | 0.0139 | 0.0285 | 0.1971 |
| dense_m0.5 | 4027 | original | False | 0.0079 | -0.0215 | 0.3194 | 0.7631 |
| dense_m1 | 2027 | prior | False | -0.0071 | -0.0092 | 0.0455 | 0.0501 |
| dense_m1 | 3027 | prior | False | 0.0031 | 0.0038 | 0.0545 | 0.2266 |
| dense_m1 | 4027 | prior | False | 0.0042 | -0.0032 | 0.6679 | 1.1188 |
| dense_m1 | 2027 | original | False | 0.0049 | 0.0007 | 0.0412 | 0.1776 |
| dense_m1 | 3027 | original | False | -0.0157 | 0.0144 | 0.0431 | 0.3123 |
| dense_m1 | 4027 | original | False | 0.0045 | -0.0220 | 0.6799 | 1.4096 |
| base_m0.5 | 2027 | prior | False | -0.0054 | -0.0009 | -0.0067 | 0.0609 |
| base_m0.5 | 3027 | prior | False | -0.0090 | 0.0068 | 0.0179 | 0.1807 |
| base_m0.5 | 4027 | prior | False | -0.0019 | -0.0002 | 0.2327 | 0.3260 |
| base_m0.5 | 2027 | original | False | 0.0127 | 0.0102 | 0.0044 | 0.0468 |
| base_m0.5 | 3027 | original | False | -0.0173 | -0.0084 | 0.0229 | 0.1508 |
| base_m0.5 | 4027 | original | False | -0.0063 | -0.0036 | 0.2539 | 0.3270 |
| base_m1 | 2027 | prior | False | -0.0100 | -0.0016 | 0.0380 | 0.1216 |
| base_m1 | 3027 | prior | False | -0.0030 | 0.0100 | 0.0520 | 0.6134 |
| base_m1 | 4027 | prior | False | 0.0025 | -0.0026 | 0.5367 | 0.8115 |
| base_m1 | 2027 | original | False | 0.0081 | 0.0096 | 0.0481 | 0.0933 |
| base_m1 | 3027 | original | False | -0.0113 | -0.0052 | 0.0571 | 0.4329 |
| base_m1 | 4027 | original | False | -0.0019 | -0.0059 | 0.5579 | 0.8125 |

All per-feature quantiles, IQR ratios, fit-threshold tail frequencies, rule bounds and failed gates are retained in CSVs. `selection.json` additionally applies the absolute missingness screen and the targeted AST improvement requirement.
