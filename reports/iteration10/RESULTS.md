# Iteration 10: external-cohort portability

Nine current full-model fits and 27 tables cover CKD, heart failure and diabetes. These are descriptive inner-validation development results; outer tests remain closed and no external-baseline superiority claim is made.

| dataset | rf | logistic | rf_augment | logistic_augment | rf_real | logistic_real | numeric_w | association_rmse | violation | violation_upper | max_missing_error_fit | c2st_auc | exact_train_copy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ckd | 0.9991 | 0.9985 | 0.9999 | 0.9996 | 1.0000 | 1.0000 | 0.2121 | 0.1105 | 0.0000 | 0.7479 | 0.0402 | 0.7884 | 0.0000 |
| diabetes | 0.9286 | 0.9258 | 0.9411 | 0.9382 | 0.9578 | 0.9348 | 0.1759 | 0.0668 | 0.0000 | 0.0000 | 0.0000 | 0.5041 | 0.0000 |
| heart_failure | 0.7826 | 0.7290 | 0.8136 | 0.7443 | 0.8614 | 0.7805 | 0.3060 | 0.0749 | 0.0000 | 0.0000 | 0.0000 | 0.7576 | 0.0000 |

All per-feature missingness, rule bounds, shape, privacy and seed variation are retained in CSVs.
