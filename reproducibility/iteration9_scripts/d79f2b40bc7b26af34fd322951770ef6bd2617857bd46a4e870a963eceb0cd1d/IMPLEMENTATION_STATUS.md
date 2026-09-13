# Implementation and publication boundary

## Implemented

- Explicit ingestion/schema adapters for all four public UCI datasets and reproducible source manifests.
- Patient-profile grouping, five outer folds and group-disjoint inner validation.
- Train-fitted numerical transforms, categorical support and explicit joint missingness generation.
- Bootstrap-stable empirical graph; nonlinear incidence message passing; clique/permutation controls.
- Conditional Gaussian/absorbing-mask diffusion with an explicit reverse kernel.
- Physical-unit rule residuals, confidence weights, observed coverage and approximate numerical/logit guidance.
- Separated ablations, equal-integrated-strength schedules, rule withholding, simple repair, baseline adapters.
- Training/checkpoint/sample manifests, CLI, batch runner, descriptive aggregation and manuscript scaffold.
- Meaningful numerical, split-isolation, gradient, missingness, memorization-control and reload tests.
- Paired common-weight initialization/training randomness, optional zero-initialized structural gates and Gaussian velocity prediction.
- Gated and ungated dense-incidence controls with matched full-model parameter counts.
- Pinned official-source TabDDPM adapter, checkpoint reuse, provenance and budget sensitivity.
- Inner-partition sensitivity with a fixed reserved outer test set; grouped discriminator folds; numerical variance ratios and saved downstream predictions.
- Independent training/sampling seed axes, fixed downstream seeds, and nested synthetic table-size comparisons using archived checkpoints.
- Feature-wise missingness audits and whole-table rule-violation bounds that keep unknown relationships separate from observed compliance.
- Train-only observation-head frequency initialization and observation-loss weighting experiments, isolated from the unchanged core sampler; fitted checkpoints are compatible with ordinary sampling.
- Fixed reverse-variance sensitivity with exact zero-setting equivalence, immutable parent weights and matched historical references.
- Distribution-shape diagnostics covering numerical quantiles, IQR ratios, fit-threshold upper tails and observed fit-range exceedance.
- A fit-only asinh-MAD numerical representation with differentiable physical-unit constraint energy; nine fits and 27 evaluations were retained as a negative result after extreme inverse-decoded tails.
- Evaluation-only wrong-rule sensitivity over all iteration-7 tables: omission of the exploratory A/G identity and intentionally reversed biochemical inequalities, with fit/validation/synthetic unknown-rule bounds retained.

- Matched current ILPD CTGAN/TVAE compact/default-capacity comparisons, statistical controls and retained explicit copula repair.
- Supplemental tolerance-aware predictor-copy and boundary diagnostics with a bootstrap positive control; literal exact hashes alone miss round-trip copies.

## Deliberate differences from the original proposal

The initial backbone uses Gaussian and absorbing categorical diffusion; it is not TabDiff. Confidence is fixed, not calibrated uncertainty. No minibatch dependency loss or outcome-classifier auxiliary loss is used. All primary arms share label conditioning. Numerical observation masks are generated explicitly. CKD ordinal codes are treated categorically. Sampling uses approximate clean-prediction/logit descent followed by the actual reverse kernels; it is not an exact energy-conditioned process.

## Remaining evidence required for a paper

The supplied proposal is now technically concrete enough for testing, but neither that correction nor a working model establishes novelty or publishability. Outstanding requirements are specified in the revised protocol and review: clinical rule validation, meaningful higher-order clinical cohorts, fair and successful strong-baseline reproduction, wrong-rule/sample-size/missingness sensitivity, a frozen confirmatory evaluation, stronger privacy attacks and statistically defensible inference. Authors and clinical collaborators must provide appropriate authorship, domain review and institutional declarations. The initial parameter-matched controls are now implemented and measured; their existence does not itself establish an advantage of clinical structure.

The actual measured results are in `reports/`. Iteration 4 completes the first ILPD sample-size and feature-wise missingness sensitivity study: 216 evaluations, 108 pools, 36 reused checkpoints and no new training. It identifies a repeated missing A/G-ratio failure and no consistent utility benefit from 5x synthesis for the proposed model. Broader sensitivity and fixes remain required; see `reports/iteration4/INTERPRETATION.md`.

Iteration 5 trains 36 new models and adds 108 evaluations against 81 reused references. Fit-frequency observation initialization removes the measured missingness flags, but AST dispersion worsens beyond a frozen guardrail for full-prior on one partition. The weight-4 variant adds fidelity/association losses. None passes all correction criteria; see `reports/iteration5/INTERPRETATION.md`. Twenty-seven tests and all provenance/metric checks pass, which does not establish scientific superiority.

Iteration 6 adds 162 sampler evaluations against 162 reused references, with no new fitting. Larger reverse variance partly corrects one AST SD metric but worsens supplied-rule compliance and bilirubin tails. New quantile diagnostics reveal that excessive central spread can coexist with low SD. All 324 artifact checks and 31 tests pass, but no recipe qualifies. `docs/PUBLICATION_READINESS.md` maps the scientific evidence and remaining requirements.

Iteration 8 fits nine asinh-MAD representation variants and adds 27 evaluations against 36 reused references. The differentiable physical-unit energy is tested and all 63 records audit, but inverse decoding produces extreme bilirubin outliers, numeric Wasserstein errors up to 1e14 and 21–25% supplied-rule violations. The representation is rejected; no core source or sampler changes. See `reports/iteration8/INTERPRETATION.md`.

Iteration 9 re-evaluates all 225 iteration-7 tables under five supplied/omitted/reversed rule registries, for 1,125 analyses with zero fitting or sampling. Omitting the exploratory protein identity lowers current full synthetic violations from about 5–6% to 1–2%, while real validation violates that relation at 6–9%; reversed relations produce near-universal violations. This closes the engineering sensitivity check but demonstrates the need for clinical/laboratory review before any compliance claim. See `reports/iteration9/INTERPRETATION.md`.

Iteration 7 adds 36 native neural adapter fits (72 class networks), nine statistical fits and 144 new evaluations, reusing 81 prior records. It provides successful executable CTGAN/TVAE comparisons with mixed utility rankings, while repaired copula improves numeric/association fidelity and observed compliance across all partitions. It also identifies and supplements literal-copy/boundary roundoff diagnostics. All 225 artifacts and 34 tests pass; no new MedHyperDiff model or default recipe is produced. See `reports/iteration7/INTERPRETATION.md`.

Runs using `inner_validation` are development artifacts, including any metrics labeled `holdout` inside the common evaluator. The run manifest identifies the partition; inner validation has influenced checkpoint selection and is not an untouched outer test cohort.
