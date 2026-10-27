# What iteration 2 establishes

**We trained the proposed generator and new variants. We have not established a publication-ready superior medical generator.** This iteration completed 110 additional training/evaluation experiments, separate from the original 60 development runs. A run fits one configuration on one fit partition with one seed, generates a table and evaluates it. It is not a new architecture or an independent clinical trial.

The full results, rejected screen variants, per-feature and per-rule outputs, seed variation and costs are in [RESULTS.md](RESULTS.md) and the accompanying CSVs. All 110 package hashes match the archived source `1a9e99d67a8131df426c2292c785cfe2b7c0f9d99d465fe4f476278d69a5645c`.

## Clinical findings that survived the added checks

Each value below averages three generator seeds on one ILPD validation partition. Inner partitions overlap; these are descriptive development results, not independent confirmation.

| Inner partition seed | Base RF AUROC | Full RF AUROC | Base supplied-rule violation | Full supplied-rule violation | Base numeric W | Velocity-full numeric W |
|---|---:|---:|---:|---:|---:|---:|
| 2027, original | 0.701 | 0.726 | 72.7% | 15.3% | 0.410 | 0.309 |
| 3027 | 0.694 | 0.683 | 76.1% | 19.1% | 0.306 | 0.220 |
| 4027 | 0.692 | 0.661 | 77.9% | 27.7% | 0.282 | 0.227 |

The original apparent utility gain does **not** replicate consistently. The full model decreases the supplied-rule violation rate in all three partitions, but has lower forest utility in two. Its numerical fidelity also does not consistently improve. We must not select only the favorable first row for the paper.

Velocity-full improves mean normalized numerical Wasserstein distance on all three ILPD partitions. Its forest AUROC is 0.720, 0.693 and 0.713, respectively. However, logistic utility is worse than base on the two additional partitions, and rule violations remain 71.9%, 63.3% and 82.4%. It is a useful fidelity candidate, not a complete solution to clinical consistency. These conclusions depend on the specified downstream classifiers and metrics.

The zero-initialized gated variant does not consistently dominate the ungated full model. It remains optional rather than becoming the new default. Matched initialization is retained as a comparison option because it removes an avoidable source of randomness; it is not claimed as a novel generative mechanism.

## Evidence about higher-order structure

On the controlled simulator, the ungated full and dense-incidence models have exactly **36,633 parameters** each. The dense control keeps the same edge slots and networks but removes the specific clinical/simulator memberships. Both retain the same semantic training energy and sampling guidance.

| Inner partition seed | Full parity violation | Matched dense parity violation | Base RF AUROC | Full RF AUROC |
|---|---:|---:|---:|---:|
| 2027 | 12.2% | 46.3% | 0.790 | 0.691 |
| 3027 | 14.0% | 46.0% | 0.731 | 0.667 |
| 4027 | 13.1% | 46.9% | 0.795 | 0.778 |

This supports a narrow mechanism finding: appropriate incidence structure helps preserve a **supplied** nonlinear relationship beyond the particular matched dense control. It does not establish preservation of unknown medical relationships. Forest utility is lower than base in all three partitions. Earlier withheld-parity results also showed that preservation depends on giving the model that relationship.

On ILPD the ungated full and dense controls each have **37,600 parameters**. Full has fewer supplied-rule violations than dense on all three partitions, while its forest utility advantage over dense appears only on the original partition. The gated full/dense pair has 37,602 parameters each.

## External baseline limitation, investigated rather than hidden

The official TabDDPM diffusion and MLP are now integrated from pinned, unmodified source. We ran the tiny screen, a larger three-seed budget and a longer single-seed diagnostic. This is not yet a successful final benchmark reproduction.

The longer diagnostic trained 8,500 epochs: 51,000 minibatch updates on ILPD and 42,500 on the simulator. Checkpoint selection still used validation loss. ILPD numerical Wasserstein remained poor at **3.260**, with forest AUROC **0.708**. On the simulator numerical Wasserstein improved to **0.558**, with AUROC **0.755**. These longer diagnostics are reported separately from the repeated-seed table.

An analytic standard-Gaussian denoiser passed an upstream-sampler sanity check: sampled coordinate means were between −0.009 and 0.013 and standard deviations between 0.966 and 1.012 for 4,000 samples. Quantile preprocessing approximately round-trips the fitted data. In contrast, the learned ILPD checkpoint generated transformed-coordinate standard deviations between roughly 683 and 8,724, before empirical quantile inversion maps tails to observed feature limits. This localizes the observed symptom to unstable learned sampling, rather than a gross inverse-transform or oracle-sampler failure; it does not prove the optimizer or architecture is the sole cause. See [diagnostics](TABDDPM_DIAGNOSTIC.json).

The baseline's instability must be resolved with native-capacity/settings validation and a fair tuning protocol before using it to support superiority claims. Increasing epochs alone did not resolve ILPD. TabDiff source was inspected but **TabDiff was not trained**. Knowledge-equivalent constrained baselines remain outstanding.

## Reproducibility and decisions

- **20 tests passed**, including exact common initial predictions, matched capacity, velocity reconstruction, reserved-test invariance, grouped discrimination, training and checkpoint reuse.
- The [artifact audit](ARTIFACT_AUDIT.json) checks every run's status, source snapshot, reserved-test exclusion, class counts, saved prediction scores and matched capacity. All 110 passed.
- No reserved outer-test record was used in fit or validation partitions. No outer-test performance was calculated. New validation partitions stay within the original outer development pool.
- Read `utility_predictions.csv` with Pandas `float_precision="round_trip"` when reconstructing rank metrics: tiny parser rounding changes can break probability ties and slightly change AUROC.
- Training ran on CPU; GPU VRAM use by these training runs was zero. No peak system RAM measurement was obtained.
- All core source changes remain optional/backward-compatible. Do not replace the final-study method with a selected single-seed checkpoint.

## Next scientific milestone

Develop and validate a constraint-strength/coverage tradeoff that preserves numerical variability and utility, using the same fixed development pool. The mechanism evidence justifies studying that question; it does not justify opening the final test set yet. A submission also needs stronger external baseline validation, a clinically reviewed rule registry, suitable external clinical data, wrong-rule and missingness sensitivity, stronger privacy attacks, a frozen inferential protocol and the researchers' authorship/ethics declarations. A defensible paper may report when structure helps and when it hurts; universal superiority is not supported by the current results.
