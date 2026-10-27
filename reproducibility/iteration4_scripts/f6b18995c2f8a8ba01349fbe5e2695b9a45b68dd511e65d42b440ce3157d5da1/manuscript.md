# MedHyperDiff: Evaluating Hypergraph Guidance for Structurally Consistent Medical Table Synthesis

**Research manuscript working draft, updated 12 September 2026. Not ready for submission.** This version contains measured development results. Authors, affiliations and venue are to be supplied by the researchers; independent confirmatory evaluation is pending.

## Abstract

Medical tables combine heterogeneous measurements with relationships that generic generative models may not preserve. We investigate mixed-type diffusion combining empirical feature associations, nonlinear constraint hyperedges, fixed confidence-weighted penalties and approximate sampling guidance. Numerical values follow Gaussian diffusion; categories and observation indicators follow absorbing-mask diffusion. Sequential development experiments use the Indian Liver Patient Dataset and a controlled simulator while reserving the original outer test patients. A strength sweep adds 46 fitted models and 174 sampling/evaluation configurations, reusing matching checkpoints. Across three overlapping ILPD validation partitions with three generator seeds each, a full velocity model with training weight 0.01 and guidance strength 10 reduces supplied-rule violations to 5.8–6.6%, versus 15.3–27.7% for the previous full x0 configuration. Numerical Wasserstein improves by approximately 20–44%, but utility superiority does not hold consistently against matching backbones or dense controls. Another 216 evaluations separate sampling and training seeds and compare nested 1x/5x synthetic tables using 36 existing checkpoints. Fivefold synthesis does not consistently improve the proposed model's utility; one checkpoint produces 5.33–5.67% missing A/G ratios versus 0.86% in its fit data across three sampling seeds. In the simulator, the original hypergraph model preserves supplied parity better than a parameter-matched dense control, but no structured strength-sweep candidate passes all screening criteria. These findings identify fidelity–utility and missingness tradeoffs, not clinical validity, general superiority or independent test performance.

## 1. Introduction

Matching marginal feature distributions does not establish that synthetic records preserve inter-attribute structure. In clinical data, a discrepancy may reflect an invalid generated combination, but it may also expose measurement noise, recording conventions, genuine patient heterogeneity or a poorly specified constraint. A useful evaluation therefore needs to distinguish mathematical/schema constraints from approximate biochemical relations and learned associations.

The study asks whether nonlinear factor structure improves a matched diffusion generator beyond what can be obtained from a pairwise representation, energy guidance alone, or straightforward constraint-aware correction. It also asks when imposed structure harms coverage or utility. The current contribution is an auditable experimental formulation and a controlled analysis of that tradeoff. Combining existing graphs, diffusion and constraints does not by itself establish a novel method.

## 2. Related work

CTGAN and TVAE provide established mixed-table baselines. TabDDPM and TabDiff establish diffusion for heterogeneous tabular variables. GOGGLE demonstrates feature-relational generation, while CausalDiffTab studies structural regularization with adaptive weighting. HFGF reconstructs dependent features using functional/logical structure. C-DGM and DRL integrate constraints into generative models, and Harpoon provides tabular inference-time guidance. TabGraphSyn uses patient-neighborhood representations for clinical synthesis. These precedents rule out broad novelty claims based on graphs, constraints or adaptive guidance alone.

The verified source ledger and links appear in [the research review](../docs/RESEARCH_REVIEW.md). Bibliographic metadata must be exported from the final authoritative paper versions for the selected venue. The targeted audit is not an exhaustive literature review.

## 3. Method

### 3.1 Representation and denoiser

Let a record contain numerical values, categorical values, numerical observation indicators and a binary conditioning label. Fit-partition medians fill numerical placeholders; observed numerical means and standard deviations define an affine transform. Numerical placeholders contribute no numerical reconstruction loss when unobserved. Categorical levels are fitted on the fit partition, with explicit missing/unknown states. Generation excludes states not observed there. Observation indicators are generated jointly and restore numerical missingness during decoding.

Numerical features receive learned scalar-to-token embeddings, while categories receive embedding lookups including an absorbing mask token. Feature identities, sinusoidal-time features and a label embedding condition a small feature transformer. Fit-only bootstrap-stable associations add a weighted pairwise attention bias. The clinical incidence matrix maps variables to rule-specific hyperedges. Each incidence block applies nonlinear feature transforms, mean aggregation to hyperedges, learned edge identities and nonlinear edge updates, then distributes messages back to incident features. Isolated features receive no hypergraph residual.

The optional gated residual is `H_next = H + tanh(g) * LayerNorm(U(H))` on incident features, with `g = 0` initially. The graph-bias coefficient also starts at zero in this variant. Paired initialization constructs structural modules without advancing the RNG used for common output heads and resets training randomness independently of architecture size. With gating, the full model therefore starts with exactly the backbone's predictions. The dense-incidence control replaces memberships by all ones while retaining the same edge slots and learnable modules. It controls parameter count for this factor-mixing architecture, not every possible dense architecture.

### 3.2 Mixed diffusion

With a cosine noise schedule, numerical corruption is

`x_t = sqrt(a_bar_t) x_0 + sqrt(1 − a_bar_t) epsilon`, with `epsilon ~ N(0, I)`.

The original parameterization predicts clean standardized values and minimizes observed-feature MSE. The optional velocity parameterization instead predicts

`v_t = sqrt(a_bar_t) epsilon − sqrt(1 − a_bar_t) x_0`,

and reconstructs `x_0_hat = sqrt(a_bar_t) x_t − sqrt(1 − a_bar_t) v_hat`. Velocity prediction is an established diffusion parameterization, including in [Salimans and Ho (ICLR 2022)](https://arxiv.org/abs/2202.00512); it is not claimed as novel here. Its loss changes the effective weighting of clean reconstruction across time.

At discrete time `t` in `1,…,T`, each categorical token is masked with probability `t/T`. Training uses clean-category cross-entropy weighted by `(T/t)` for masked tokens. In the reverse process an already revealed token is copied; a masked token reveals a predicted category with probability `1/t` and otherwise remains masked. All remaining masks reveal at the final step. Numerical generation uses the Gaussian reverse posterior with the reconstructed clean prediction.

### 3.3 Rules, energy and approximate guidance

Each rule has a residual in original feature units, tolerance, scale and fixed confidence weight. Energy is the confidence-weighted sum of smooth-L1 penalties on positive normalized residuals, multiplied by observation eligibility, and divided by the total rule weight. Training adds this energy with default coefficient 0.1 and larger weight at low noise. Sampling takes a bounded per-record gradient step on clean numerical predictions and categorical logits before applying reverse transitions. The default guidance strength is 0.2 with a late-emphasized quadratic schedule normalized to the same integrated strength as constant/late controls. Revealed categorical states are fixed, and missingness probabilities are detached from the energy gradient.

ILPD rules comprise nonnegative values, albumin no greater than total protein, direct bilirubin no greater than total bilirubin with tolerance 0.1, and the approximate protein relation `abs(AG * (total_protein − albumin) − albumin)`. Its tolerance is the fit-only 95th percentile of observed residuals, and confidence is 0.5. This is a descriptive envelope, not a clinically validated acceptance criterion. The bilirubin relation has confidence 0.75. The simulator uses a known protein relation and binary XOR parity, with fixed tolerances and unit confidence. Full rule registries retain provenance and review status.

The method is not an exact sampler from a constrained distribution and does not guarantee feasibility. Fixed confidence values are sensitivity parameters, not calibrated clinical uncertainty.

## 4. Experimental protocol

### 4.1 Data and development isolation

The earlier pilot includes four public UCI cohorts: Heart Failure, ILPD, CKD and symptom-based diabetes risk. Heart Failure excludes post-baseline follow-up duration. The current mechanism-focused iteration uses ILPD (583 records, nine numerical and one categorical predictor) and a separately generated simulator (512 records, four numerical and three binary predictors). ILPD has four missing A/G measurements and 13 repeated predictor profiles beyond their first occurrences. The simulator generates albumin and globulin independently, derives total protein and A/G ratio, and sets a third symptom to the XOR of two independent binary symptoms. Outcome probability depends on age and the third symptom. Simulator parameters and the fixed data-generation seed are recorded in code. It is not an additional clinical cohort.

Identical predictor profiles are grouped, including conflicting labels. Five outer grouped stratified folds are constructed with split seed 2026, and outer fold 0 is fixed for development. Its 117 ILPD and 102 simulator records remain reserved. The original fit/validation sizes are 349/117 for ILPD and 306/104 for the simulator. Inner seeds 3027 and 4027 repartition only the same outer development pool; group constraints can slightly change counts. Fit-only transformations, graph estimation, categorical support and rule tolerances are recomputed within each partition. No outer-test performance is calculated. Dataset versions, raw checksums, ingestion decisions and source acknowledgments are retained in the [data audit](../reports/DATA_AUDIT.md) and run manifests.

### 4.2 Comparisons and budgets

The initial screen evaluates 12 configurations on both datasets using generator seed 0. Six retained arms are repeated with seeds 1 and 2. Paired base/full, velocity-full and ungated dense controls are evaluated on two additional inner partitions with three seeds, with dense controls also added on the original partition. All internal arms use width 32, two layers, batch size 64, learning rate 0.001, up to 100 epochs, patience 20 and 30 reverse steps. Graph estimation uses 10 bootstraps. AdamW uses weight decay 1e-4 and gradient norm clipping at 1. The selected checkpoint minimizes validation loss under fixed corruption. Consequently validation has influenced selection; the reported development estimates are not unbiased test estimates.

Label conditioning, synthesis count and fit-partition class prevalence are shared. Full and dense models have 37,600 parameters on ILPD and 36,633 on the simulator; gated variants add two scalar gates. The smaller backbone is not parameter matched to full. Matching width and epoch limits also does not match elapsed computation or selected update counts.

An adapter imports the unmodified official [TabDDPM diffusion and conditional MLP](https://proceedings.mlr.press/v202/kotelnikov23a.html), pinned to commit `b476257dd460b778ba09eb97f7a51d6490fa17f8`. Its fit-only quantile preprocessing and local optimization differ from the internal backbone. Beyond the tiny screen, a sensitivity uses width 128, 1,000 diffusion steps and up to 500 epochs across three seeds. A separate seed-0 diagnostic runs 8,500 epochs. These are disclosed adaptations; numerical sampling instability prevents treating them as a successful final reproduction. [TabDiff](https://arxiv.org/abs/2410.20626) is inspected but untrained in this project. The [external-baseline ledger](../docs/EXTERNAL_BASELINES.md) lists implementation and license details.

The original 60 development runs additionally cover graph-only, energy-only, guidance-only, clique, shuffled-incidence, withheld-rule, confidence and simple-repair controls, as well as statistical and CTGAN/TVAE references. Their different budgets/source versions remain separately identified. Final comparisons still require matched tuning of successful current diffusion and knowledge-equivalent constrained baselines, wrong-rule sensitivity and independent evaluation after a documented freeze.

### 4.3 Endpoints and analysis

We report per-rule violation, severity and eligibility; numerical Wasserstein distance divided by fit standard deviation; categorical total variation; association and signed numerical correlation errors; numerical standard-deviation ratios; missingness; and downstream utility. Logistic regression and a fixed 150-tree random forest are trained separately on real, synthetic or combined data and evaluated on the same real validation records. Every downstream preprocessor fits only its own training table. AUROC is accompanied by average precision, calibration and threshold metrics in the artifacts. The discriminator groups identical joint records across its internal folds.

Exact-copy rates and nearest-distance membership diagnostics are exploratory privacy checks, not formal guarantees. Repeated generator seeds measure conditional optimization/sampling variation. Overlapping inner partitions are not independent clinical samples. We report descriptive means and retain seed standard deviations; we perform no significance or noninferiority test on these scores. Primary endpoints and inferential units must be frozen before confirmatory evaluation.

## 5. Development results

All 110 iteration-2 experiments completed and passed an artifact audit. The original 60 experiments are retained separately. No confirmatory outer-test results are available. Full single-run values, rejected candidates, per-rule/feature outputs and costs accompany the [measured report](../reports/iteration2/RESULTS.md).

### 5.1 ILPD validity does not imply stable utility improvement

Table 1 reports means over three generator seeds within each validation partition. “Violation” means violating at least one supplied rule among eligible records.

| Inner seed | Base RF AUROC | Full RF AUROC | Base violation | Full violation | Base numeric W | Velocity-full numeric W |
|---|---:|---:|---:|---:|---:|---:|
| 2027 | 0.701 | 0.726 | 72.7% | 15.3% | 0.410 | 0.309 |
| 3027 | 0.694 | 0.683 | 76.1% | 19.1% | 0.306 | 0.220 |
| 4027 | 0.692 | 0.661 | 77.9% | 27.7% | 0.282 | 0.227 |

The full model improves supplied-rule compliance on every partition but improves forest utility only on the original partition. Velocity-full lowers numerical Wasserstein on all three; its forest AUROCs are 0.720, 0.693 and 0.713. Its logistic utility is lower than base on the two additional partitions, and its rule violations are 71.9%, 63.3% and 82.4%. Thus neither parameterization dominates the reported objectives. Optional gating also fails to consistently improve upon the ungated full configuration.

### 5.2 Incidence structure preserves supplied parity at a utility cost

| Inner seed | Full parity violation | Matched dense parity violation | Base RF AUROC | Full RF AUROC |
|---|---:|---:|---:|---:|
| 2027 | 12.2% | 46.3% | 0.790 | 0.691 |
| 3027 | 14.0% | 46.0% | 0.731 | 0.667 |
| 4027 | 13.1% | 46.9% | 0.795 | 0.778 |

Full and dense models have identical parameter counts and semantic energies. The lower full-model parity violations therefore cannot be explained solely by additional parameter count relative to that control. Forest utility nevertheless falls below base on every partition. Earlier withheld-parity experiments approach chance parity violation, limiting any claim that the model discovers an unknown higher-order relation. Repair can enforce the supplied simulator equations directly; it remains a necessary reference rather than proof that a learned structured generator is better.

### 5.3 External baseline and software findings

The larger TabDDPM adaptation has severe numerical fidelity problems despite usable ILPD classification scores. Longer training does not resolve ILPD: numerical Wasserstein is 3.260 and forest AUROC 0.708 at the selected checkpoint after an 8,500-epoch diagnostic. A standard-Gaussian oracle gives plausible samples under the upstream sampler, and numerical preprocessing approximately round-trips the training values. The learned checkpoint instead generates extreme transformed tails before empirical quantile inversion. This diagnoses a sampling failure symptom; it does not establish its sole cause or TabDDPM's attainable performance. We do not use these unstable runs to claim superiority over that method.

Twenty software tests pass. The artifact audit verifies original reserved-test exclusion, exact class counts, source snapshots, saved prediction-score consistency and matched full/dense capacity for all 110 experiments. All runs use CPU. Summed fit and sample time is 684.4 and 32.7 seconds, respectively; concurrent execution and excluded evaluation costs mean these are not elapsed study wall time. GPU VRAM use by training is zero; peak system RAM was not measured.

### 5.4 Follow-up strength sweep and frozen-settings replication

Iteration 3 separates training-energy coefficients (0, 0.01, 0.1, 1) from generation-time guidance (0, 0.2, 2, 10) for x0/velocity full models, and adds guidance-only backbone and repair references. The 84-setting initial screen uses inner seed 2027 and generator seed 0. Matching fitted checkpoints are reused; changing guidance is not counted as new training. The declared screen requires lower supplied-rule violation, RF/logistic AUROC drops no greater than 0.02, numerical Wasserstein and association error increases no greater than 10%, and mean missingness-error increase no greater than 0.01 relative to the matching zero-guidance backbone. These are engineering tolerances, not clinical noninferiority margins. At most two non-repair settings per dataset are selected by lowest violation among qualifying candidates.

Two full velocity configurations qualify on ILPD: training weights 0.1 and 0.01, both with guidance 10. Only the x0 backbone with guidance 0.2 qualifies on the simulator. Settings are then frozen across three generator seeds and inner seeds 2027/3027/4027, together with dense controls, matching zero-guidance and same-guidance backbones, and the previous full configuration. This stage completes 46 new fits and 174 distinct sampling/evaluation configurations using 94 checkpoints, 48 reused from iteration 2. All original test patients remain excluded.

| Inner seed | Previous full rule violation | Weight-0.01 velocity full violation | Previous numeric W | Updated numeric W | Previous RF AUROC | Updated RF AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 2027 | 15.3% | 5.8% | 0.391 | 0.294 | 0.726 | 0.716 |
| 3027 | 19.1% | 6.0% | 0.408 | 0.227 | 0.683 | 0.680 |
| 4027 | 27.7% | 6.6% | 0.303 | 0.242 | 0.661 | 0.712 |

The updated candidate improves numerical fidelity and compliance relative to the previous full model in all partitions, with mixed forest utility changes. Logistic AUROC is 0.753, 0.703 and 0.694, compared with the previous 0.753, 0.676 and 0.639. However, the same-guidance velocity backbone achieves forest AUROC 0.701, 0.722 and 0.715; the full model's utility advantage is not consistent. The guided backbone itself reduces violations to 9.6%, 6.8% and 8.4%, demonstrating that generation-time guidance accounts for a substantial part of the compliance gain. Dense-control utility rankings also vary by partition.

The weight-0.01 candidate meets the engineering conditions on seed means in two of three partitions; weight 0.1 meets them in one. Neither passes everywhere. The simulator's selected guided backbone reduces overall violations modestly but leaves parity near chance and fails the association condition on one partition. Protein-rule coverage for the weight-0.01 ILPD candidate ranges from 93.4% to 100%; average missingness screening can miss a concentrated feature-specific increase. These results are descriptive and retain all screening and replication failures in the [iteration-3 report](../reports/iteration3/RESULTS.md).

### 5.5 Independent sampling variation and synthetic table size

A frozen follow-up protocol crosses four ILPD configurations (previous full, weight-0.01 velocity full, same-guidance velocity backbone and matched dense velocity) with three training seeds, three sampling seeds and three inner partitions. No new generator is fitted: 36 checkpoints generate 108 five-fit-size pools, each evaluated using its first fit-size rows and its entire table, producing 216 evaluations. Repeating the fit-label vector preserves class proportions and exact nesting. The evaluator seed is fixed at 2026; sampling seeds 70001–70003 are independent of training seeds. Means first average sampling seeds within each checkpoint and then average training seeds within each partition.

| Inner seed | Updated full forest AUROC, 1x → 5x | Updated full logistic AUROC, 1x → 5x | Updated full numeric W, 1x → 5x | Rule violation, 1x → 5x |
|---|---:|---:|---:|---:|
| 2027 | 0.7389 → 0.7441 | 0.7476 → 0.7442 | 0.3044 → 0.3015 | 5.44% → 5.18% |
| 3027 | 0.6941 → 0.6905 | 0.7055 → 0.7060 | 0.2232 → 0.2135 | 4.68% → 5.21% |
| 4027 | 0.7047 → 0.7017 | 0.6951 → 0.6947 | 0.2410 → 0.2324 | 5.75% → 6.08% |

Larger synthetic tables reduce numerical Wasserstein modestly without consistently improving utility. The declared size screen permits at most 0.02 AUROC loss for either classifier, requires an increase in at least one, permits at most 10% increases in numerical/association error and 0.01 increases in observed violation or each feature's missingness error, and permits at most 0.01 loss in each rule's coverage. Separately, any table with a feature missingness error above 0.02 versus fit is flagged. These are development tolerances, not clinical noninferiority margins. The updated model passes relative gates on two of three partitions and has three flagged 5x tables; fivefold synthesis is not adopted as its default. The dense comparator passes all three relative gates without a 5x missingness flag, supporting its use as a stronger development reference without establishing superiority over full models.

For updated full, inner seed 2027 and training seed 1, missing A/G ratios are 5.33–5.67% in the three 5x pools, versus 0.86% in fit and zero in validation. The corresponding 1x prefixes also fail the missingness safeguard. This repeated deviation is not resolved by additional synthetic rows; its cause remains undetermined. We now report each rule's whole-table violation bounds: observed violations divided by all rows as the lower bound, and observed violations plus unevaluable rows divided by all rows as the upper bound. For any-rule bounds, overlapping unknowns and observed violations are counted once. These are logical missing-data intervals, not confidence intervals. At 5x, updated full's partition-2027 mean any-rule interval is [5.18%, 7.57%], with mean protein coverage 97.61%.

Augmentation also remains mixed: updated-full forest AUROC at 5x is 0.7431, 0.6984 and 0.7009 versus real-only 0.7473, 0.7098 and 0.7006. Its discriminator AUROC remains approximately 0.85–0.88 on partition means, and individual ALT/AST standard deviation ratios can fall to approximately 0.27 relative to validation. These diagnostics limit any claim of faithful distribution reproduction despite lower constraint violations. Full outcomes, seed variation, rule bounds and figures are in the [iteration-4 report](../reports/iteration4/INTERPRETATION.md).

![ILPD synthetic table size and prediction utility](../reports/iteration4/figures/sample_size_utility.png)

Figure: means and SD across three training seeds after averaging three sampling seeds per checkpoint. The error bars are descriptive training-seed variation, not population confidence intervals. Inner partitions overlap.

## 6. Discussion

The mechanism experiments support a specific conclusion: a supplied nonlinear membership structure can improve preservation of that supplied relation relative to the chosen matched dense control. They also demonstrate why constraint compliance is insufficient as the primary measure of a medical generator. A method can preserve imposed relations while losing variability or predictive relationships that were not encoded in its energy. Velocity changes the numerical learning problem and improves several fidelity outcomes, but weakens some rule preservation. Gating does not remove this tradeoff consistently.

The ILPD findings further caution against selecting a favorable partition. The initial forest utility increase reverses on two additional validation partitions, and logistic/forest conclusions differ for velocity-full. Overlapping partitions, small samples and repeated model selection prevent interpreting these differences as formal population effects. The clinical rules themselves are approximate, not clinician-certified validity labels. Missingness eligibility and real-data residuals must accompany any claim of valid records; suppressing rare valid measurements or shifting the modeled population would not be a successful clinical outcome.

The follow-up strength sweep improves the earlier configuration's fidelity and supplied-rule compliance, but does not eliminate the tradeoff against stronger controls. Separating training and sampling variation confirms that fivefold synthesis is insufficient to resolve utility losses; feature-specific checks expose a repeated observation-mask failure. Further work should investigate missingness calibration and compressed numerical variability, and validate clinically reviewed relationships under incorrect-rule sensitivity and external cohorts. The current results support a tradeoff study, not a universal superiority claim.

## 7. Reproducibility and limitations

The code saves split indices, data/source hashes, preprocessing, graph stability, rule registries, configurations, training history, model checkpoints, synthetic samples, downstream prediction scores and metric outputs. Iterations 2–4 use the same archived core package source; experiment scripts and parent-checkpoint hashes are archived separately. All 174 iteration-3 and 216 iteration-4 sampling evaluations pass their provenance/split/prediction audits. The three exported samplers reproduce their evaluated tables, and an independently replayed guided iteration-4 pool matches exactly. Twenty-three software tests pass. The [iteration-2 setup](../docs/ITERATION_2_RUNBOOK.md), [iteration-3 runbook](../docs/ITERATION_3_RUNBOOK.md) and [iteration-4 commands and costs](../docs/ITERATION_4_RUNBOOK.md) give executable commands. Read saved floating-point prediction CSVs with round-trip parsing when reconstructing rank metrics, preserving probability ties.

Limitations include a custom compact diffusion backbone, fixed confidence, categorical treatment of ordinal codes, approximate guidance, incomplete external baseline reproduction, no external cohort in the current iteration, no clinical review of the rule registry and a preliminary nearest-distance privacy attack. Dataset convenience sampling and label timing constrain generalization. No differential privacy, clinical deployment, calibrated uncertainty or statistical superiority claim is made. ILPD sample-size and missingness sensitivity have been measured, but broader replication, correction of the identified failures, wrong-rule sensitivity, confirmatory inference, stronger privacy attacks and researcher declarations remain incomplete.

## 8. Development conclusion

MedHyperDiff is implemented, trained and evaluated as a synthetic medical table generator. Controlled comparisons indicate that supplied hypergraph structure can improve supplied-rule preservation, while numerical fidelity and prediction utility can favor different configurations. The current evidence motivates a carefully controlled validity–utility study and does not yet support submission-level generalization claims.

## Declarations — researcher input required before submission

Specify authorship/contributions, funding, conflicts of interest, dataset access and licensing compliance, applicable institutional ethics determination, data/code availability, and any venue-required disclosure of AI assistance. Do not invent approvals or authors.
