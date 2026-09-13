# Iteration 7: matched development baseline comparison

Freeze this protocol and the machine-readable plan before training. This stage compares the existing observation-initialized full model against successful native CTGAN/TVAE baselines, statistical generators and an explicit repair control. It does not train a new MedHyperDiff variant or open the reserved test set.

## Design

- Dataset ILPD; original outer fold 0, split seed 2026 remains reserved. Use only fit and inner validation indices for inner seeds 2027, 3027, 4027. These development partitions overlap; they are not independent cohorts.
- Neural references: CTGAN and TVAE from installed `ctgan==0.11.1`. Two presets each: existing compact dimensions at 100 epochs, and native package defaults at 300 epochs (CPU explicitly selected). Default CTGAN embedding 128, generator/discriminator (256,256), batch 500; default TVAE embedding 128, encoder/decoder (128,128), batch 500. Preserve other native optimizer defaults. Report every preset; do not select a favorable budget after inspecting results.
- Each adapter fits separate class-0 and class-1 native networks, with seeds training_seed + label; training seeds 0,1,2. Shared fit-only preprocessing carries explicit observation tokens and empirical category support. This is a class-conditional adaptation, not an exact replication of the original published benchmark or the SDV wrapper.
- Statistical references: empirical Gaussian copula with ordinal category encoding, independent marginals, and bootstrap. These deterministic fitting procedures each fit once per partition; do not manufacture three training replicates. Bootstrap is a memorization positive control, not a privacy-preserving candidate.
- Generate each fitted reference with sampling seeds 70001,70002,70003, using the fit label vector repeated five times, exactly as iteration 5. No new real information is created by this row count. Evaluate a fourth statistical arm by applying the existing `repair_simple` to each saved copula table. Retain raw and repaired tables and changed-row rates; repair receives the same fit-derived constraints and may alter distribution shape.
- Reuse all 81 iteration-5 `full_prior`, `dense_prior`, `base_prior` evaluations, matching 3 inner partitions × 3 training seeds × 3 sampling seeds per arm. The full_prior candidate remains experimental and has not passed every previous fidelity gate.
- Expected new work: 36 neural adapter fits = 72 class-specific native networks, plus 9 deterministic reference fits. Generate 135 new tables and 9 repaired derivatives = 144 new evaluations, alongside 81 reused evaluations (225 records total). No new MedHyperDiff training.

## Evaluation and interpretation

Identical evaluator seed 2026, 5x synthetic counts and prevalence, downstream RF/logistic models, fit-derived rule tolerances, missingness bounds, numerical Wasserstein distance, association RMSE, C2ST, copy diagnostics, per-feature SD/IQR/quantiles and tail checks. Repaired and bootstrap results remain visible even if they outperform our method. Record both utility and compliance: compliance with supplied rules alone is not clinical validity or a novel learned contribution.

Average sampling seeds within each neural checkpoint, then training seeds within each partition. For deterministic references average the three samples directly; no training-seed SD or artificial sample size of nine. Retain all per-seed outputs. Report partition differences descriptively, without significance claims or population confidence intervals. Real-only utility is a reference, not a synthetic generator.

The matching controls data, labels, sample count and evaluation, **not equal compute or tuning opportunity**. MedHyperDiff was selected through repeated prior development and validation checkpointing; CTGAN/TVAE use fixed training endpoints and two prespecified capacity/budget combinations. Changing capacity, batch size and epochs together is a reference-strength sensitivity, not an isolated epoch ablation. A future confirmatory benchmark needs a common tuning protocol and additional cohorts. Unstable TabDDPM attempts remain unsuccessful reproductions; TabDiff is not trained here.

## Retention and checks

Freeze source, adapter, runner, helper, protocol, normalized data and installed-package source hashes, environment versions, split indices and existing-reference hashes. Retain CPU adapter checkpoints, native training loss histories, fitted preprocessors, samples, scores, timings and hashes. These whole-object local checkpoints are for this trusted research environment and require the same Python/module versions; load only after checking their recorded hash. Fail rather than silently skipping incomplete or corrupt artifacts. Verify replay for every newly fitted generator's first sampling seed and repair derivatives, plus recalculated utility from all saved predictions, split separation, matching labels, rules and hashes. Keep failed attempts visible if any occur.

Implementation provenance: [official CTGAN repository](https://github.com/sdv-dev/CTGAN) and locally inspected version-0.11.1 constructor defaults. Core MedHyperDiff source remains unchanged.
