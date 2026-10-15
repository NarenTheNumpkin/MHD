# Iteration 11: corrected training and evaluation, and a frozen external comparison

**Bottom line.** An audit found five defects that had hidden MedHyperDiff's behaviour. Correcting them transforms the model: on the reserved test sets its classifier two-sample (C2ST) AUROC falls from 0.944 to 0.631 on ILPD, from 0.761 to 0.615 on heart failure and from 0.782 to 0.586 on CKD, and ILPD association error falls from 0.199 to 0.081. It is the only generator that reproduces the real behaviour of the supplied ILPD clinical relations without post-hoc repair (violation-rate gap 0.002, protein-residual distance 0.062 versus 0.104–0.145 for the strongest alternatives), with zero hard-rule violations and no measured memorization on ILPD, heart failure or CKD. **It does not beat the strongest external baseline on overall realism:** a properly trained TabDiff, and on most cohorts the simple Gaussian copula, reach lower C2ST. The predeclared superiority criterion is not met, and utility differences are within patient-bootstrap uncertainty. All figures below come from [RESULTS.md](RESULTS.md); the protocol and its two amendments are in [ITERATION_11_PROTOCOL.md](../../docs/ITERATION_11_PROTOCOL.md).

## 1. Defects found in the audit

| Defect | Evidence | Correction |
|---|---|---|
| Checkpoint selection by a noisy validation diffusion loss (patience 20, 100-epoch cap) | In iteration 5, 13/36 fits stopped on the initial loss plateau and kept epochs 3–28 (≈20–170 updates); the other 23 hit the cap and selected epochs 86–100 while still improving. Every earlier ablation mixes architecture with optimization luck, and validation rows both selected and scored the model. | Fixed optimizer-step budget, linear learning-rate decay, EMA weights; validation never selects. Training is bitwise identical with or without validation rows (tested). |
| Affine scaling of heavy-tailed laboratory values | Enzyme and bilirubin bulks occupy hundredths of an SD; iteration 6 measured AST IQR ratio 2.35. | Fit-only mid-rank quantile coding with a bounded, differentiable piecewise-linear inverse used inside the physical-unit rule energy. |
| Continuous outputs for values recorded on fixed grids | All numeric source columns have a fixed resolution; off-grid values are detectable. Statistical references emit observed values and were implicitly favoured. | Decode to the fit partition's recorded resolution, applied to every continuous-output generator. |
| "Fewer violations is better" for a 95%-quantile envelope | 5% of fit and 6–9% of validation records violate the protein envelope by construction; repaired copula's 0% is over-correction. | Rule-realism metrics: violation-rate gap and residual-distribution distance versus real held-out records; hard-rule violations reported separately. |
| TabDDPM adapter did not use the upstream recipe | Upstream trains fixed steps with annealed learning rate and samples the final weights; the adapter used constant learning rate with early stopping. | Upstream recipe reproduced; TabDiff added with its upstream trainer. |

The corrected core reproduces the archived iteration-5 `full_prior` table byte-for-byte under legacy settings; 48 tests pass.

## 2. Development evidence (ILPD inner partitions; outer test closed)

Applying one correction at a time to the iteration-5 recipe: C2ST 0.855 → 0.730 (fixed-step EMA training) → 0.714 (quantile coding, 100 steps) → 0.571 (recorded precision); association RMSE 0.176 → 0.066, equal to the training-resampling floor (0.066). Restoring only the old epoch selection on top of every other correction raises association RMSE back to 0.158 and C2ST to 0.697, so selection was the largest single defect. Without the rule machinery the same backbone violates at least one supplied rule in 40% of rows (real validation ≈8%); hypergraph structure plus energy training and calibrated guidance bring the violation gap to 0.001.

In the controlled simulator, matched hypergraph structure preserves the supplied XOR parity better than a parameter-matched dense-incidence control and the unstructured backbone (0.9% versus 3.5% and 6.4% violations; legacy training 34–36%). The benefit comes from structure and energy training, not guidance: unguided full gives the same 0.9% with C2ST 0.541. Strong guidance (10) designed for ILPD destroys simulator fidelity (C2ST 0.949) because the simulator's protein identity is exact with a very strict tolerance, which is why the confirmation calibrated guidance per dataset.

## 3. Frozen confirmation (outer test fold 0, five training seeds)

| Dataset | MedHyperDiff C2ST | Best non-memorizing comparator | MedHyperDiff vs legacy | Notes |
|---|---:|---|---:|---|
| ILPD | 0.631 ± 0.037 | TabDiff 0.533 ± 0.043; copula 0.566 | 0.944 → 0.631 | best rule realism; forest TSTR 0.730 (real 0.720), equal to TabDiff |
| Heart failure | 0.615 ± 0.068 | TabDiff (privacy-constrained) 0.531; copula 0.538 | 0.761 → 0.615 | MedHyperDiff backbone 0.576; TabDDPM 0.542 with membership AUROC 0.99 |
| CKD | 0.586 ± 0.035 | copula 0.571; TabDiff 0.594 (membership 0.72) | 0.782 → 0.586 | MedHyperDiff and its backbone share the lowest association error of non-copying arms (0.086) |
| Diabetes | 0.491 ± 0.065 | TabDiff (privacy-constrained) 0.481; copula 0.500 | 0.590 → 0.491 | **memorization**: membership 0.894, 35.5% exact predictor copies |

Against the pre-specified criterion (lower C2ST than every non-memorizing comparator on at least three of four cohorts), MedHyperDiff wins on none. TabDiff is the strongest general-purpose generator in this study; the Gaussian copula remains a hard reference on these small tables. Patient-bootstrap 95% intervals for forest TSTR AUROC differences include zero for every strong comparator; the only clear utility difference is against the diverging privacy-constrained TabDDPM on CKD (+0.195, 0.109 to 0.297).

**Privacy.** Membership AUROC exposes near-copying that C2ST rewards. TabDDPM's selected configurations memorized heart failure (0.990), CKD (0.906) and diabetes (0.932; 83% exact copies), and on diabetes every diffusion model did, including MedHyperDiff (0.894). Diabetes has only 251 distinct profiles of 15 binary symptoms plus age; no MedHyperDiff search configuration stayed below membership AUROC 0.60 there, so that cohort's MedHyperDiff result must not be released or presented as privacy-preserving. On ILPD, heart failure and CKD MedHyperDiff's membership AUROC was 0.518, 0.599 and 0.528.

## 4. What this supports

1. A methodological contribution: on small clinical tables, validation-loss checkpointing, affine scaling of heavy-tailed laboratory values, continuous decoding of fixed-resolution measurements, zero-violation scoring of empirical envelopes and non-upstream baseline recipes each materially distort conclusions. Correcting them changed every major ranking in iterations 2–10.
2. A domain contribution with a clear scope: supplied nonlinear hyperedge structure with physical-unit energy and calibrated guidance reproduces real clinical-relation behaviour (simulator parity; ILPD protein and bilirubin relations) better than unstructured diffusion, TabDiff, TabDDPM and explicit repair, at a C2ST cost of about 0.05–0.10 relative to TabDiff.
3. Not supported: superiority in overall realism or utility, privacy guarantees, clinical validity of the supplied rules (still unreviewed), or generalization beyond these four public cohorts and fold.

## 5. Limitations specific to this iteration

One reserved outer fold per cohort; small test sets (60–117 patients) make C2ST noisy (seed SD up to 0.07). Hyperparameters were selected on overlapping inner partitions. MedHyperDiff additionally had a sampling-only guidance calibration. The development diagnostics used ILPD partition 2027 before the protocol was written. TabDiff and TabDDPM are documented adaptations (joint label column for TabDiff; fit-only quantile coding with explicit missingness flags for both), not reproductions of published scores. All runs used CPU under variable background load; reported fit times are wall-clock and not comparable across jobs.
