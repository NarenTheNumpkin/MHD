# Iteration 6: fixed reverse-noise variance and distribution-shape sensitivity

Frozen before generating this stage's samples. This is an ILPD development experiment, not an independent test evaluation. It follows iteration 5's missingness improvement and AST dispersion failure. No new generator fitting is planned: changing reverse variance tests a sampler modification while keeping learned weights fixed.

## Scientific basis and scope

The current sampler uses posterior variance beta_tilde = beta_t (1−abar_previous)/(1−abar_current). Test sigma² = exp(m log(beta) + (1−m) log(beta_tilde)) for fixed m = 0.5 and 1 at reverse steps t > 1. At t = 1, retain deterministic clean decoding with no added noise. The m = 0 implementation must exactly reproduce the existing sampler. Initial noise, random draws, labels, guidance, reverse mean, categorical transitions and batch order remain unchanged.

Fixed beta/beta_tilde choices and log-variance interpolation are established diffusion design choices; see [Nichol and Dhariwal, ICML 2021, §3.1 and Eq. 14](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf). That work learns an interpolation parameter on images. Here we test two fixed scalar values on an existing compact medical-table generator. This is not its learned-variance method, an exact posterior claim, or a novel architecture. Whether it improves the observed dispersion failure is a hypothesis.

## Frozen matrix

Use iteration-5 prior-initialized full, dense and backbone checkpoints with observation-loss weight 1. Three architectures × three inner partitions (2027, 3027, 4027) × three training seeds (0, 1, 2) = 27 fixed fitted checkpoints. Two variance settings × three independent sampling seeds (70001, 70002, 70003) produce **162 new generated tables/evaluations**. Use five repetitions of the fit-label vector, sampling batches of 256, guidance 10 with the same adaptive schedule and evaluator seed 2026. Table sizes are 1,745 or 1,750 rows.

Reuse the 81 iteration-5 weight-1 prior-initialized evaluations at m = 0, and the 81 corresponding iteration-4 evaluations before prior initialization. There are **324 comparison records total, 162 new and 162 reused**. New generation uses 27 models; references span 54 previously fitted models. No reference is counted as new training or resampling. Freeze every referenced checkpoint and metric hash before execution. Outer fold 0 / split seed 2026 and the original reserved test remain fixed; evaluate inner validation only.

The entire grid runs without selecting a favorable partition, seed or architecture. Comparisons average sampling seeds within each fitted model, then training seeds separately by partition. Those seeds and overlapping partitions are not independent clinical cohorts. Retain all failed settings. Do not pool earlier outcomes from different sample sizes/evaluator seeds.

## Measurements and selection

Retain all existing fidelity, utility, missingness, rule coverage/unknown-rule bounds and empirical privacy measurements. Add per-numerical-feature IQR ratios, normalized 5th/50th/95th quantile errors, upper-tail frequency above the fit-only 95th percentile, and frequency outside the observed fit range. Report fit/validation/synthetic quantiles. No validation statistic changes generated values. These diagnostics distinguish broader central distributions from added tail errors and are descriptive, not clinical limits.

For a full-model sampler candidate, require the iteration-5 correction gates relative to its original iteration-4 architecture reference: no table-level feature missingness error >0.02 versus fit, lower worst missingness error, ≤0.02 decline in either AUROC on every partition mean, numeric W and association RMSE ≤1.1× reference, observed/upper-bound violation increases ≤0.01, rule-coverage decline ≤0.01, per-feature missing-error increases ≤0.01, and per-feature mean absolute SD-ratio-error increases ≤0.10. These are engineering tolerances, not clinical noninferiority margins.

Also require the same relative utility/fidelity/coverage/variability tolerances against the immediately preceding iteration-5 prior-initialized model, and no >0.10 increase in any feature's mean normalized 5th/95th quantile error against it. The previously failing AST SD-ratio error on partition 4027 must improve by at least 0.01 against that prior-initialized reference. Missingness must remain under the absolute table-level threshold. A full setting qualifies only if all partitions and both reference comparisons pass. Rank qualifying settings by the maximum per-feature mean SD-ratio error across partitions, then mean numeric W, then smaller m. Export only a recipe pointing to existing weights if a full setting qualifies; otherwise retain the current recipes and report the failure.

Controls use exactly the same variance settings and are reported regardless of outcome. No claim of hypergraph superiority follows from an improvement shared by dense/backbone controls. Newly added tail diagnostics, failed gates, and discriminator results must remain in the manuscript interpretation. The changes do not resolve external-baseline reproduction, clinical rule review, incorrect-rule sensitivity, external-cohort evaluation or stronger privacy attacks.
