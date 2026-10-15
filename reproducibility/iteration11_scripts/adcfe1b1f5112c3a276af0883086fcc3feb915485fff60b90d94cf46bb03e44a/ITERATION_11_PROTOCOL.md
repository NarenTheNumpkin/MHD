# Iteration 11 protocol: corrected training and representation, equal-budget tuning, frozen confirmation

Written 5 October 2026 before any iteration-11 search, guidance-calibration or outer-test result existed. The only results available when it was written are the single-partition diagnostics listed under "Pre-protocol diagnostics"; they are development evidence and are reported, not hidden.

## Why this iteration exists: defects found in the audit

1. **Checkpoint selection selects untrained or truncated models.** The original loop keeps the epoch with the lowest validation diffusion loss under one fixed noise draw, with patience 20 and a 100-epoch cap (about 600 optimizer updates on 349 ILPD fit rows). In iteration 5, 25 of 36 fits selected epoch 86–100 while validation loss was still falling, and 11 stopped during the initial loss plateau and selected epochs 3–28 (for example `full_prior_i2027_t1` selected epoch 5 at validation loss 1.618 versus 1.627 at epoch 2). Model comparisons in iterations 2–10 are therefore confounded with optimization luck. Validation records also chose the checkpoint and then scored it.
2. **Affine standardization of heavy-tailed laboratory values.** ILPD enzymes and bilirubin span two to three orders of magnitude; the bulk of each feature occupies a few hundredths of a standard deviation after affine scaling, which the Gaussian diffusion smears (iteration 6 measured an AST IQR ratio of 2.35 with an SD ratio below one).
3. **Continuous outputs for values recorded on a fixed grid.** Every numerical source column is recorded at a fixed resolution (for example bilirubin 0.1 mg/dL, enzymes in whole U/L, A/G ratio 0.01). Off-grid synthetic values are detectable by a tree discriminator. Statistical references (copula, bootstrap, independent marginals) emit observed values and were implicitly advantaged on C2ST.
4. **Rule compliance was scored as "lower is better" for a fit-quantile envelope.** The protein identity tolerance is the 95th percentile of fit residuals, so about 5% of fit and 6–9% of validation records violate it by construction. Zero violations (repaired copula) is over-correction, not extra validity.
5. **The TabDDPM adapter did not use the upstream training recipe.** Upstream `scripts/train.py` trains a fixed number of steps with linear learning-rate annealing, no clipping and no validation selection, and `pipeline.py` samples the final `model.pt`. The adapter used constant learning rate with validation early stopping; its "instability" cannot be attributed to TabDDPM.

## Corrections implemented (opt-in; legacy defaults unchanged)

- `training="steps"`: fixed optimizer-step budget, linear learning-rate decay to zero, EMA weights (0.999); validation loss logged with a separate generator and never used for selection. Tests prove weights are identical with or without validation records.
- `numeric_transform="quantile"`: fit-only mid-rank empirical normal scores; piecewise-linear inverse bounded by the fit range (explicit bounded-support rationale required by `PUBLICATION_READINESS.md`); differentiable decode used by the physical-unit constraint energy.
- `decode_precision=True`: round decoded values to the fit partition's detected recording resolution. Applied identically to every continuous-output generator (MedHyperDiff, TabDDPM, CTGAN, TVAE) and harmlessly to statistical references.
- `rule_realism` metrics: per-rule violation-rate gap versus real held-out records, Wasserstein distance between synthetic and real rule-residual distributions (scaled by fit residual SD), and a pooled hard-rule (schema/biochemical) violation rate. Original metrics are retained.
- TabDDPM `training="steps"` reproduces the upstream recipe at the pinned commit, including the upstream default 128-dimensional time embedding.

The modified core reproduces the archived iteration-5 `full_prior_i2027_t0_s70001_n5` synthetic table byte-for-byte under legacy settings, and all 38 earlier tests pass with 9 new tests.

## Pre-protocol diagnostics (ILPD, inner seed 2027, training seed 0, sampling seed 70001)

| Variant | C2ST | Numeric W | Association RMSE | Any-rule violation (validation 9.4%) | RF TSTR |
|---|---:|---:|---:|---:|---:|
| Legacy full_prior (exact replay) | 0.853 | 0.282 | 0.184 | 5.2% | 0.764 |
| + fixed steps/EMA | 0.731 | 0.245 | 0.082 | 0.7% | 0.786 |
| + quantile, T=100 | 0.680 | 0.201 | 0.081 | 0.4% | 0.750 |
| + recorded precision (guidance 10) | 0.496–0.517 | 0.200 | 0.081 | 6.7% | 0.730–0.734 |
| same, backbone without structure/energy/guidance | 0.519 | 0.203 | 0.081 | 44.4% | 0.754 |
| width 64, 3 layers, 15k steps, recorded precision, no guidance | 0.655 | 0.216 | 0.083 | 10.3% | 0.747 |

The last row had nearest-synthetic membership AUC 0.693, so capacity and budget must be judged with privacy diagnostics.

## Stage A: correction ablation (ILPD development)

Eight fixed variants (`ABLATION` in `scripts/iteration11.py`): legacy full_prior; cumulative fixed-step EMA training, quantile coding, T=100 and recorded precision; and three leave-one-out/structure variants (epoch selection instead of fixed steps, no guidance, backbone without graph/hypergraph/energy/guidance). Three inner partitions (2027, 3027, 4027) × three training seeds (0, 1, 2), sampling seed 70001 + training seed, fivefold fit-label vector, evaluator seed 2026. The original outer fold 0 remains closed.

## Stage B: equal-budget search (all four datasets, inner validation only)

Each neural generator receives exactly eight configurations: its recommended default plus seven random draws from a predeclared space, generated by a fixed hash-derived seed (`search_configs`). Defaults: corrected MedHyperDiff recipe; upstream TabDDPM `exp/diabetes/ddpm_cb_best` (Pima diabetes, the closest small medical table in the TabDDPM benchmark); native ctgan-package defaults for CTGAN and TVAE. The TabDDPM space reproduces upstream `scripts/tune_ddpm.py` (learning rate 1e-5–3e-3 log-uniform, 2–8 layer MLP of 128–1024 units, batch 256/4096, 5k/20k/30k steps, 100/1000 diffusion steps, zero weight decay). Each configuration is fit on the three inner partitions with training seed 0 and evaluated on inner validation. MedHyperDiff uses guidance 10 during the search.

**Selection rule (fixed now).** Per dataset and method, average each metric across the three partitions; rank configurations by C2ST AUROC (ascending), mean numeric Wasserstein/fit-SD (ascending) and forest TSTR AUROC (descending); choose the lowest mean rank, breaking ties by lower C2ST and then lower configuration index. A configuration that fails on any partition is excluded and reported.

## Stage C: guidance calibration (MedHyperDiff only, sampling-time)

Refit the selected MedHyperDiff configuration on the three partitions (training seed 0) and sample with guidance 0, 2, 5, 10, 20. Choose the value minimizing the partition-mean `mean_violation_gap` (synthetic versus real validation violation rates); ties within 1e-9 choose the smallest guidance. This targets real rule behaviour rather than zero violations.

## Stage D: frozen confirmation on the reserved outer test (opened once)

For each dataset, fit on the full development pool (inner fit ∪ inner validation of outer fold 0, split seed 2026) and evaluate on the untouched outer test fold 0. Methods: MedHyperDiff (selected configuration, calibrated guidance); MedHyperDiff backbone ablation (same configuration without graph, hypergraph, energy and guidance); legacy full_prior (original recipe, fit/validation selection on inner seed 2027, as originally specified); TabDDPM, CTGAN and TVAE (selected configurations); Gaussian copula; copula with explicit repair; independent marginals; bootstrap (memorization positive control). Training seeds 0–4, sampling seed 70001 + training seed, fivefold synthetic rows, evaluator seed 2026. The same development pool is the `train` reference for TRTR utility, scale normalization and privacy members for every method.

**Primary endpoints:** C2ST AUROC against the outer test (fidelity; 0.5 is chance) and forest TSTR AUROC on the outer test (utility). **Secondary:** numeric Wasserstein, association RMSE, signed Spearman RMSE, categorical TV, missingness error, rule realism (mean violation gap, relational residual W1, hard-rule violation), logistic TSTR, fivefold augmentation, and empirical privacy (nearest-synthetic membership AUC, DCR to members versus holdout, exact predictor copies).

**Analysis:** per dataset, mean and SD over the five training seeds; paired seed-wise differences between MedHyperDiff and each comparator with the number of seeds favouring each; across datasets, descriptive mean ranks. For the primary utility endpoint, a paired test-patient bootstrap (2,000 resamples of outer-test patients, seed-averaged predictions) gives a 95% interval for MedHyperDiff minus the strongest competing method, disclosed as conditional on the fitted generators. No significance claim is made from overlapping development partitions. A superiority statement requires lower C2ST than every non-memorizing comparator on at least three of four datasets with membership AUC no worse than the copula's by more than 0.05; utility is described against the strongest comparator with its interval. All outcomes, including failures and comparator wins, are reported.
