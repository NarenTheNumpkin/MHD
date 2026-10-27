# Dataset audit

Public UCI source tables; phenotype prediction benchmarks, not clinically deployed models.

Constraint residual analysis below uses only the fixed inner-fit partition of outer fold 0. Global counts are descriptive ingestion checks.

| Dataset | Rows | Predictors | Missing cells | Duplicate predictor profiles | Fit / validation / test |
|---|---:|---:|---:|---:|---|
| heart_failure | 299 | 11 | 0 | 0 | 179 / 60 / 60 |
| ilpd | 583 | 10 | 4 | 13 | 349 / 117 / 117 |
| ckd | 400 | 24 | 1012 | 0 | 240 / 80 / 80 |
| diabetes | 520 | 16 | 0 | 269 | 312 / 104 / 104 |

## heart_failure

Target counts: `{"0": 203, "1": 96}`.

Missing cells by column: `{}`.

| Rule | Eligible fit rows | Violation rate | Status |
|---|---:|---:|---|
| age_nonnegative | 179 | 0.0000 | schema |
| creatinine_phosphokinase_nonnegative | 179 | 0.0000 | schema |
| ejection_fraction_nonnegative | 179 | 0.0000 | schema |
| platelets_nonnegative | 179 | 0.0000 | schema |
| serum_creatinine_nonnegative | 179 | 0.0000 | schema |
| serum_sodium_nonnegative | 179 | 0.0000 | schema |
| ejection_fraction_percentage | 179 | 0.0000 | schema |

## ilpd

Target counts: `{"1": 416, "2": 167}`.

Missing cells by column: `{"ag_ratio": 4}`.

Protein identity residual quantiles in the fit partition (source protein units):

```json
{
  "0": 0.0,
  "0.5": 0.09999999999999964,
  "0.9": 0.2799999999999998,
  "0.95": 0.3000000000000007,
  "0.99": 0.5488000000000004,
  "1": 8.68
}
```

This is an empirical inconsistency/rounding audit. A fit-only quantile is not a clinically validated tolerance.

| Rule | Eligible fit rows | Violation rate | Status |
|---|---:|---:|---|
| age_nonnegative | 349 | 0.0000 | schema |
| total_bilirubin_nonnegative | 349 | 0.0000 | schema |
| direct_bilirubin_nonnegative | 349 | 0.0000 | schema |
| alkaline_phosphatase_nonnegative | 349 | 0.0000 | schema |
| alt_nonnegative | 349 | 0.0000 | schema |
| ast_nonnegative | 349 | 0.0000 | schema |
| total_protein_nonnegative | 349 | 0.0000 | schema |
| albumin_nonnegative | 349 | 0.0000 | schema |
| ag_ratio_nonnegative | 346 | 0.0000 | schema |
| albumin_le_total | 349 | 0.0000 | soft_biochemical |
| protein_identity | 346 | 0.0491 | exploratory_train_calibrated |
| direct_le_total_bilirubin | 349 | 0.0057 | soft_biochemical |

## ckd

Target counts: `{"ckd": 250, "notckd": 150}`.

Missing cells by column: `{"age": 9, "bp": 12, "bgr": 44, "bu": 19, "sc": 17, "sod": 87, "pot": 88, "hemo": 52, "pcv": 71, "wc": 106, "rc": 131, "sg": 47, "al": 46, "su": 49, "rbc": 152, "pc": 65, "pcc": 4, "ba": 4, "htn": 2, "dm": 2, "cad": 2, "appet": 1, "pe": 1, "ane": 1}`.

| Rule | Eligible fit rows | Violation rate | Status |
|---|---:|---:|---|
| age_nonnegative | 235 | 0.0000 | schema |
| bp_nonnegative | 232 | 0.0000 | schema |
| bgr_nonnegative | 214 | 0.0000 | schema |
| bu_nonnegative | 229 | 0.0000 | schema |
| sc_nonnegative | 230 | 0.0000 | schema |
| sod_nonnegative | 187 | 0.0000 | schema |
| pot_nonnegative | 186 | 0.0000 | schema |
| hemo_nonnegative | 208 | 0.0000 | schema |
| pcv_nonnegative | 197 | 0.0000 | schema |
| wc_nonnegative | 181 | 0.0000 | schema |
| rc_nonnegative | 164 | 0.0000 | schema |

## diabetes

Target counts: `{"Positive": 320, "Negative": 200}`.

Missing cells by column: `{}`.

| Rule | Eligible fit rows | Violation rate | Status |
|---|---:|---:|---|
| Age_nonnegative | 312 | 0.0000 | schema |
