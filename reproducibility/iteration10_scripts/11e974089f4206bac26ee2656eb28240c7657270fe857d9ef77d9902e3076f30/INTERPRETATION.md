# Iteration 10: external-cohort portability is mixed

The current observation-initialized full model was fit on CKD, heart-failure and diabetes tables using the same nominal training settings as the ILPD full_prior model. Nine models (three training seeds per dataset) produced 27 fivefold-size evaluations. Each dataset used its own locked outer fold 0 and inner seed 2027; no outer test records were opened.

| Dataset | Forest TSTR AUROC | Logistic TSTR AUROC | Numeric W | Association RMSE | Observed rule violations | Maximum fit missingness error | C2ST AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CKD | 0.9991 | 0.9985 | 0.2121 | 0.1105 | 0.0% | 0.0402 | 0.7884 |
| Diabetes | 0.9286 | 0.9258 | 0.1759 | 0.0668 | 0.0% | 0.0000 | 0.5041 |
| Heart failure | 0.7826 | 0.7290 | 0.3060 | 0.0749 | 0.0% | 0.0000 | 0.7576 |

These are portability results, not clinical validation. The cohorts have different labels, sampling, feature semantics and missingness patterns, and no matched external baseline was fit in this stage. CKD's near-perfect downstream utility is accompanied by a missingness error above the development screen (0.0402), so it cannot be presented as an unqualified success. Heart failure shows materially weaker utility and substantial train-versus-synthetic discrimination. Diabetes has more moderate utility and C2ST near chance, but one cohort cannot establish generalization.

The run caught a preflight serialization issue: CKD categorical levels that look numeric were inferred as floats after CSV round-trip, while fit data retained strings. The incomplete first table is preserved under `runs/iteration10/preflight_failed/`; all accepted tables force every categorical column to string before evaluation. The corrected run's 27 records, nine exact first-seed replays and 162 saved AUROC recalculations pass the artifact audit. This correction changes no fitted weights or real data.

All rule violations are zero because these cohorts currently use mostly nonnegativity constraints; this is not evidence of clinical consistency. A reviewed external cohort, task-specific utility definitions, matched baselines and rule validation remain required before publication claims.

[Results](RESULTS.md), [artifact audit](ARTIFACT_AUDIT.json), [protocol](../../docs/ITERATION_10_PROTOCOL.md), [runbook](../../docs/ITERATION_10_RUNBOOK.md).
