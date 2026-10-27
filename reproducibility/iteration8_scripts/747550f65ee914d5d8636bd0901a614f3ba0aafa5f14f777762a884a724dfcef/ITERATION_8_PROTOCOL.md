# Iteration 8: fit-only robust numerical representation

This protocol freezes one representation change aimed at the central-versus-tail mismatch found in iteration 6. It preserves the observation-frequency initialization from iteration 5 and the same diffusion architecture, training budget, guidance, labels, evaluator, partitions and output multiplier. It does not open the outer test or modify the core package.

## Representation

For each numeric feature, fit the median `m` and `s = max(1.4826 * MAD, 1e-6)` on the fit partition only. Replace missing numeric values by the fit median, transform observed/imputed values with `asinh((x-m)/s)`, then standardize that transformed column using its fit mean and standard deviation. The inverse is `x = m + s*sinh(z*sd + mean)` before missingness masks are applied. Categorical levels, mask tokens and allowed states are unchanged. The transform is monotone, fit-only and source-unit invertible; it is a representation change, not clipping or post-generation repair.

The constraint energy receives physical-unit values by differentiably applying the inverse transform before evaluating the existing fit-derived rules. This is required because the unmodified constraint code assumes an affine inverse. Metrics are always computed after source-unit decoding. Any nonfinite inverse, rule-energy or generated table fails the run.

## Matched experiment

- Dataset ILPD; outer fold 0 and split seed 2026 remain locked. Fit/validation partitions are inner seeds 2027, 3027 and 4027. They overlap and are not independent cohorts.
- Nine new fits: three training seeds (0,1,2) per partition. Each uses the current full_prior training configuration: 30 diffusion steps, 100 epochs maximum, width 32, two layers, four heads, batch 64, velocity prediction, clinical weight 0.01, paired initialization, CPU, fixed validation corruption and early stopping. Training guidance is zero; sampling guidance is 10 with the adaptive schedule, matching full_prior evaluation.
- Recompute the association graph from transformed fit-partition values with 10 graph bootstraps. Hyperedges and rule definitions are unchanged, and all rule residuals are evaluated in physical units.
- Generate three fivefold-size tables per checkpoint with sampling seeds 70001, 70002 and 70003 and the repeated fit-label vector. Evaluate with seed 2026. This creates 27 new evaluations. Reuse the 27 full_prior records and nine repaired-copula records from iteration 7; no new baseline fitting occurs.
- Primary descriptive checks: numeric Wasserstein, per-feature SD/IQR/5th/50th/95th-quantile errors, fit-tail/range frequencies, associations, C2ST, utility, missingness, observed and whole-table rule bounds, near-copy and nearest-distance diagnostics. The repaired copula remains a required challenge; compliance alone cannot select the transformed model.

## Decision rule

No automatic winner is exported. The representation is a promising development candidate only if, relative to full_prior, it reduces AST and bilirubin central/tail errors on at least two partitions, has no feature missingness error above 0.02, does not increase mean observed rule violation by more than 0.01, does not reduce either classifier's TSTR AUROC by more than 0.02 on a partition, and does not worsen association RMSE or numeric W by more than 10% on more than one partition. These are development gates, not clinical margins, statistical tests or population claims. Compare the retained repaired copula separately; do not hide a failure behind a favorable gate.

Any successful candidate still requires another cohort and a frozen confirmatory protocol. If the transformed representation fails, retain the result and proceed to reviewed rule sensitivity and external-cohort design rather than tuning transforms until a gate passes.

## Provenance

Freeze the runner, model helper, protocol, source snapshot, installed versions, normalized data, splits, fit-derived preprocessing, graph, rules, checkpoints, samples, scores and hashes before fitting. Whole-object experimental checkpoints are trusted local artifacts and require the same module/dependency versions; verify the checksum before loading. Confirm exact first-seed replay for all nine fitted models, recalculate all saved AUROCs and shape/rule/missingness metrics, and verify zero outer-test overlap.
