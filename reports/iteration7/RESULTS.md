# Iteration 7: matched development references

144 new evaluations + 81 reused = 225 records. Neural cells average three samples per checkpoint, then three training seeds. Statistical cells average three samples from one deterministic fit. Inner partitions overlap; these are descriptive development comparisons. All presets are retained; no confirmatory winner is selected.

**Measurement caveat:** `exact_train_copy` below is literal numeric-hash equality. It misses even bootstrap copies after float32 round trips. The [supplemental diagnostic](NUMERICAL_DIAGNOSTIC_AUDIT.json) detects 100% training near-copies for bootstrap; see `tolerant_copy_partition_means.csv`. Neither zero literal copies nor zero near-copies establishes privacy. Use `tolerant_boundaries_all.csv` for interpretation of strict boundary ties. Original metrics are retained unchanged.

| arm | inner_seed | rf | logistic | numeric_w | association_rmse | violation | protein_coverage | c2st_auc | exact_train_copy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_prior | 2027 | 0.7211 | 0.6982 | 0.3924 | 0.2333 | 0.0902 | 0.9920 | 0.9400 | 0.0000 |
| base_prior | 3027 | 0.7102 | 0.6992 | 0.2701 | 0.1911 | 0.0789 | 0.9918 | 0.9088 | 0.0000 |
| base_prior | 4027 | 0.6947 | 0.6893 | 0.2777 | 0.2108 | 0.0766 | 0.9956 | 0.9169 | 0.0000 |
| bootstrap | 2027 | 0.7374 | 0.6969 | 0.1829 | 0.0757 | 0.0508 | 0.9893 | 0.5149 | 0.0000 |
| bootstrap | 3027 | 0.7110 | 0.6707 | 0.1321 | 0.0532 | 0.0441 | 0.9912 | 0.4596 | 0.0000 |
| bootstrap | 4027 | 0.7093 | 0.7055 | 0.1363 | 0.0699 | 0.0558 | 0.9949 | 0.5031 | 0.0000 |
| copula | 2027 | 0.7355 | 0.7195 | 0.1842 | 0.0849 | 0.6481 | 0.9937 | 0.5722 | 0.0000 |
| copula | 3027 | 0.7138 | 0.6901 | 0.1285 | 0.0608 | 0.6464 | 0.9920 | 0.5617 | 0.0000 |
| copula | 4027 | 0.6656 | 0.7056 | 0.1292 | 0.0830 | 0.6501 | 0.9950 | 0.5099 | 0.0000 |
| copula_repair | 2027 | 0.7380 | 0.7284 | 0.1964 | 0.0898 | 0.0000 | 0.9937 | 0.5884 | 0.0000 |
| copula_repair | 3027 | 0.7035 | 0.6996 | 0.1299 | 0.0615 | 0.0000 | 0.9920 | 0.5666 | 0.0000 |
| copula_repair | 4027 | 0.6880 | 0.6986 | 0.1338 | 0.0869 | 0.0000 | 0.9950 | 0.5627 | 0.0000 |
| ctgan_compact100 | 2027 | 0.7346 | 0.6796 | 0.4356 | 0.2609 | 0.9402 | 0.9298 | 0.9494 | 0.0000 |
| ctgan_compact100 | 3027 | 0.6180 | 0.6063 | 0.3073 | 0.2632 | 0.9403 | 0.9282 | 0.9636 | 0.0000 |
| ctgan_compact100 | 4027 | 0.6425 | 0.5558 | 0.3509 | 0.2682 | 0.9658 | 0.9730 | 0.9709 | 0.0000 |
| ctgan_default300 | 2027 | 0.7649 | 0.7468 | 0.5044 | 0.2701 | 0.9721 | 0.9769 | 0.9653 | 0.0000 |
| ctgan_default300 | 3027 | 0.6908 | 0.6497 | 0.3599 | 0.2726 | 0.9598 | 0.9768 | 0.9647 | 0.0000 |
| ctgan_default300 | 4027 | 0.6781 | 0.6315 | 0.4168 | 0.2874 | 0.9719 | 0.9823 | 0.9729 | 0.0000 |
| dense_prior | 2027 | 0.7700 | 0.7595 | 0.2192 | 0.1235 | 0.0669 | 0.9926 | 0.8378 | 0.0000 |
| dense_prior | 3027 | 0.6507 | 0.7053 | 0.1985 | 0.1228 | 0.0627 | 0.9914 | 0.8378 | 0.0000 |
| dense_prior | 4027 | 0.6826 | 0.6893 | 0.2083 | 0.1789 | 0.0789 | 0.9954 | 0.8658 | 0.0000 |
| full_prior | 2027 | 0.7383 | 0.7450 | 0.3049 | 0.1843 | 0.0550 | 0.9915 | 0.8722 | 0.0000 |
| full_prior | 3027 | 0.6991 | 0.7126 | 0.2037 | 0.1489 | 0.0497 | 0.9918 | 0.8475 | 0.0000 |
| full_prior | 4027 | 0.7076 | 0.6886 | 0.2223 | 0.1932 | 0.0596 | 0.9955 | 0.8788 | 0.0000 |
| independent | 2027 | 0.7674 | 0.7545 | 0.1892 | 0.2861 | 0.9056 | 0.9947 | 0.7992 | 0.0000 |
| independent | 3027 | 0.7260 | 0.7250 | 0.1292 | 0.2760 | 0.9028 | 0.9929 | 0.7669 | 0.0000 |
| independent | 4027 | 0.6804 | 0.6989 | 0.1320 | 0.2778 | 0.8996 | 0.9949 | 0.7745 | 0.0000 |
| tvae_compact100 | 2027 | 0.7862 | 0.7469 | 0.4145 | 0.1893 | 0.8572 | 1.0000 | 0.9363 | 0.0000 |
| tvae_compact100 | 3027 | 0.6969 | 0.7408 | 0.3023 | 0.1643 | 0.8540 | 1.0000 | 0.9238 | 0.0000 |
| tvae_compact100 | 4027 | 0.7166 | 0.7148 | 0.2912 | 0.1945 | 0.8552 | 1.0000 | 0.9027 | 0.0000 |
| tvae_default300 | 2027 | 0.7859 | 0.7284 | 0.3993 | 0.1908 | 0.8499 | 1.0000 | 0.9204 | 0.0000 |
| tvae_default300 | 3027 | 0.6828 | 0.7175 | 0.2522 | 0.1578 | 0.8396 | 1.0000 | 0.8870 | 0.0000 |
| tvae_default300 | 4027 | 0.6838 | 0.7041 | 0.2776 | 0.1870 | 0.8453 | 1.0000 | 0.8821 | 0.0000 |

## CPU fitting cost in seconds

| arm | count | mean | min | max | sum |
| --- | --- | --- | --- | --- | --- |
| bootstrap | 3 | 0.0016 | 0.0015 | 0.0017 | 0.0048 |
| copula | 3 | 0.0018 | 0.0018 | 0.0019 | 0.0054 |
| ctgan_compact100 | 9 | 1.9012 | 1.8064 | 2.4409 | 17.1111 |
| ctgan_default300 | 9 | 7.5916 | 7.4236 | 8.2672 | 68.3240 |
| independent | 3 | 0.0016 | 0.0015 | 0.0016 | 0.0047 |
| tvae_compact100 | 9 | 1.4212 | 1.3883 | 1.4453 | 12.7910 |
| tvae_default300 | 9 | 1.8587 | 1.8246 | 1.9063 | 16.7281 |

CTGAN/TVAE adapters each contain two separately trained class networks. Default300 changes capacity, batch size and budget together. Data and evaluation match; compute and development tuning opportunity do not. Bootstrap is a memorization control. Copula repair receives supplied rules and is explicit postprocessing. All utility, shape, missingness and privacy diagnostics and seed variation are retained in companion CSVs.
