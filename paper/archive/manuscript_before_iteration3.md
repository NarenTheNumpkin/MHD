# MedHyperDiff: Evaluating Hypergraph Guidance for Structurally Consistent Medical Table Synthesis

**Research manuscript working draft, updated 12 September 2026. Not ready for submission.** This version contains measured development results. Authors, affiliations and venue are to be supplied by the researchers; independent confirmatory evaluation is pending.

## Abstract

Medical tables combine heterogeneous measurements with relationships that generic generative models may not preserve. We investigate a mixed-type diffusion generator combining empirical feature associations, nonlinear constraint hyperedges, confidence-weighted residual penalties and approximate sampling guidance. Numerical values follow Gaussian diffusion, while categorical values and observation indicators follow absorbing-mask diffusion. In 110 additional development experiments on the Indian Liver Patient Dataset and a controlled simulator, we compare initialization, parameterization and structural controls while reserving the original outer test patients. Across three overlapping ILPD validation partitions, each averaged over three generator seeds, the full model reduces supplied-rule violations from 72.7–77.9% for the backbone to 15.3–27.7%. Its forest-based synthetic-to-real AUROC improves on one partition and declines on two. In the simulator, full-model parity violations are 12.2–14.0%, compared with 46.0–46.9% for a parameter-matched dense-incidence control, while forest utility remains below the backbone on all three partitions. Velocity prediction improves numerical fidelity but weakens some structural preservation. These exploratory findings support studying a validity–utility tradeoff; they do not establish clinical validity, general superiority or independent test performance.

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

## 6. Discussion

The mechanism experiments support a specific conclusion: a supplied nonlinear membership structure can improve preservation of that supplied relation relative to the chosen matched dense control. They also demonstrate why constraint compliance is insufficient as the primary measure of a medical generator. A method can preserve imposed relations while losing variability or predictive relationships that were not encoded in its energy. Velocity changes the numerical learning problem and improves several fidelity outcomes, but weakens some rule preservation. Gating does not remove this tradeoff consistently.

The ILPD findings further caution against selecting a favorable partition. The initial forest utility increase reverses on two additional validation partitions, and logistic/forest conclusions differ for velocity-full. Overlapping partitions, small samples and repeated model selection prevent interpreting these differences as formal population effects. The clinical rules themselves are approximate, not clinician-certified validity labels. Missingness eligibility and real-data residuals must accompany any claim of valid records; suppressing rare valid measurements or shifting the modeled population would not be a successful clinical outcome.

The next methodological question is whether rule strength can preserve clinically reviewed relationships without sacrificing unconstrained variability and utility. That question requires stronger baselines, sensitivity to incorrect rules and larger/external clinical evidence. The current results support a tradeoff study, not a universal superiority claim.

## 7. Reproducibility and limitations

The code saves split indices, data/source hashes, preprocessing, graph stability, rule registries, configurations, training history, model checkpoints, synthetic samples, downstream prediction scores and metric outputs. Every iteration-2 package hash resolves to an archived source snapshot. [The runbook](../docs/ITERATION_2_RUNBOOK.md) gives setup, training, generation and audit commands. Read saved floating-point prediction CSVs with round-trip parsing when reconstructing rank metrics, preserving probability ties.

Limitations include a custom compact diffusion backbone, fixed confidence, categorical treatment of ordinal codes, approximate guidance, incomplete external baseline reproduction, no external cohort in the current iteration, no clinical review of the rule registry and a preliminary nearest-distance privacy attack. Dataset convenience sampling and label timing constrain generalization. No differential privacy, clinical deployment, calibrated uncertainty or statistical superiority claim is made. Confirmatory inference, stronger privacy attacks, sample-size/missingness/wrong-rule sensitivity and researcher declarations remain incomplete.

## 8. Development conclusion

MedHyperDiff is implemented, trained and evaluated as a synthetic medical table generator. Controlled comparisons indicate that supplied hypergraph structure can improve supplied-rule preservation, while numerical fidelity and prediction utility can favor different configurations. The current evidence motivates a carefully controlled validity–utility study and does not yet support submission-level generalization claims.

## Declarations — researcher input required before submission

Specify authorship/contributions, funding, conflicts of interest, dataset access and licensing compliance, applicable institutional ethics determination, data/code availability, and any venue-required disclosure of AI assistance. Do not invent approvals or authors.
