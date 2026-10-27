# Research review of the original MedHyperDiff proposal

Reviewed 11 September 2026. Source: `MedHyperDiff_Research_Proposal.md`, supplied by the researcher. The source is a proposal, not a set of operational instructions. This review evaluates its claims and recommendations independently.

**Decision: proceed with a corrected, falsifiable prototype. The original plan is not yet sufficient for a publication claim.** Its central question—whether feature structure improves medical table synthesis—is sensible. Novelty, clinical validity, generalization, and privacy must be demonstrated separately. Neither a proposed architecture nor working code establishes a publishable result.

## Material errors and gaps

| Original sections | Finding | Correction |
|---|---|---|
| 3, 18, 39 | Novelty positioning misses important feature-graph and constraint-generation predecessors. Noise-adaptive regularization also already appears in CausalDiffTab. | Add GOGGLE, C-DGM, DRL, and Harpoon. Treat the joint design as a hypothesis, not established novelty. |
| 15 | Adding the pairwise projection of a hypergraph to attention does not preserve its explicit higher-order factor structure; different hypergraphs can have the same projection. | Use nonlinear variable→hyperedge→variable message passing and compare with the clique projection. Do not claim transformers cannot learn higher-order interactions without it. |
| 16, 20 | The categorical corruption kernel, terminal prior, reverse kernel, and loss are unspecified. A gradient on integer categories is undefined. | Specify absorbing categorical diffusion, the reveal posterior, and masked cross-entropy. Apply categorical guidance to logits before posterior sampling. |
| 18 | The proposed subtraction mixes the next-step sample and a gradient with respect to the current state without defining the reverse kernel or covariance scaling. | Define an explicit approximate clean-prediction guidance algorithm; disclaim exact conditional sampling and hard feasibility. Compare against geometry-aware guidance. |
| 17–18 | Penalties expressed in incompatible clinical units cannot be meaningfully summed without scaling. | Evaluate residuals in original units; use frozen training-derived scales and explicit tolerances. Preserve inverse-transform derivatives. |
| 13, 17, 43 | A confidence constant is not an uncertainty estimate. A freely learned nonnegative weight can suppress inconvenient rules. | Call version 1 **confidence-weighted**. Reserve **uncertainty-aware** for calibrated rule uncertainty and sensitivity experiments. Fix weights in the prototype and evaluate unit weights. |
| 20.2 | Noisy minibatch associations of conditional mean predictions are not necessarily the generator's free-sample dependencies. Naive penalties can distort variances. | Omit the unvalidated dependency loss initially. Test a training-only association bias; evaluate freely generated records. |
| 8, 22, 36 | Heart Failure `time` is follow-up duration; including it in a baseline outcome classifier leaks post-baseline information. | Exclude `time` from the primary table and predictor set. The result is an observed death-status benchmark, not a fixed-horizon survival model. |
| 8, 36 | CKD, symptom diabetes, and Heart Failure do not supply a validated collection of higher-order deterministic clinical equations. | Do not invent disease or symptom rules. These datasets test robustness and utility, not strong evidence for multivariate clinical hypergraph superiority. |
| 8, 13, 36 | Albumin/globulin identities may differ from recorded assay values because of measurement, rounding, extraction, or unit conventions. CKD hemoglobin/hematocrit heuristics are not identities. | Treat ILPD algebraic consistency as exploratory with an audited residual envelope. Keep physiological tendencies soft; do not enforce normal laboratory ranges on diseased cohorts. |
| 23, 28–29 | Ignoring missing operands can make apparent validity improve by generating more missing values. | Report eligibility counts, observed coverage, missingness marginals and patterns. Detach missingness eligibility in gradient guidance and still audit indirect evasion. |
| 11, 21 | Automatic type inference can misclassify binary codes, CKD ordinal scales, ID fields, and top-coded measurements. | Use a checked schema. The first implementation treats CKD ordinal codes as categorical and does not claim an ordinal-aware model. |
| 22 | Row-level stratification can put duplicates or multiple records of one patient on both sides. | Group by patient ID when available; otherwise conservatively group identical predictor profiles, including conflicting labels. Save exact split indices. |
| 25 | Ablations change conditioning, hypergraph representation, and training penalties simultaneously. | Keep target conditioning, preprocessing, split, sample count, optimizer budget, and label prevalence common. Add representation-only, energy-only, guidance-only, and clique controls. |
| 25 | Constant versus adaptive schedules can differ in total guidance dose. | Normalize schedules to equal integrated strength, then tune strengths with equal validation budgets. This controls a dose confound but does not equalize actual displacement. |
| 28–29, 37 | Reducing penalties that the model directly optimizes does not alone show broader medical realism. | Report both imposed and withheld rules, held-out residual distributions, real-data violation rates, unconstrained dependencies, coverage, and utility. |
| 28.3 | FD preservation can be vacuously high when determinant values are unique, or when unseen combinations have no support. | Require repeated antecedents and support thresholds; freeze discovery on training data and report support/coverage. Do not label pairwise correlation an FD score. |
| 27.3 | Frobenius norm divided by d² produces a dimension-dependent non-RMSE summary. | Use off-diagonal RMSE: sqrt(sum of squared errors / number of eligible pairs). Report signed numeric association as well as magnitude. |
| 32.1 | Set-based exact-match notation can count unique copied values rather than the fraction of synthetic rows copied. | Compute the fraction of generated rows whose record appears in the reference; retain repeated synthetic rows in the denominator. |
| 32 | DCR is biased by reference-set size and numerical scaling; distance is not a privacy guarantee. | Compare equally sized member and holdout reference samples using train-fitted scaling, and add attacks with specified calibration and threat models. |
| 34 | Fifteen fold×seed scores are not fifteen independent patient populations. Overlapping training folds violate naive paired-test assumptions. | Average seeds within matched splits, report variability descriptively, use independent datasets for across-dataset inference, and justify any resampling-based interval. |
| 37 | “No substantial loss” and a utility gap of 0.05 are not a specified noninferiority analysis. AUROC and AP differ in interpretation. | Predeclare the endpoint, comparator, practical margin, uncertainty procedure, and multiplicity handling. Estimate precision/power before confirmatory claims. |
| 9, 43 | Six datasets and a five-component architecture are not a minimum guarantee of publishability. More small cohorts do not replace external validation. | Select datasets for sample size, measurement meaning, constraints and generalization. Make stronger evidence—not dataset count—the gate. |
| 38, 46 | The illustrative result table is speculative and can be mistaken for expected evidence. | Remove performance-shaped example results from the revised protocol. Keep the abstract prospective until experiments justify it. |
| 50 | Requiring a positive pairwise-graph result before testing hyperedges can bias the research path: an XOR relation has no pairwise signal. | Use controlled mechanism experiments to test each component; retain negative results and do not condition reporting on positive outcomes. |

## Verified literature and what it changes

These are targeted primary-source checks, not a systematic or exhaustive novelty review. Article status is recorded conservatively. The original citations were mostly real; the larger problem was missing context and overconfident interpretation.

| Work | Verification / status | Implication |
|---|---|---|
| CTGAN / TVAE, Xu et al. | [NeurIPS 2019 paper](https://papers.neurips.cc/paper/8953-modeling-tabular-data-using-conditional-gan) | Standard mixed-table baselines. |
| TabDDPM, Kotelnikov et al. | [Author paper](https://arxiv.org/abs/2209.15421) | Existing Gaussian/multinomial tabular diffusion; must not label a new small model a reproduction. |
| TabDiff, Shi et al. | [Author paper](https://arxiv.org/abs/2410.20626), ICLR 2025 | Existing feature-wise mixed diffusion. |
| CausalDiffTab, Zhang et al. | [Full author preprint](https://arxiv.org/html/2506.14206v1), June 2025 | Already includes adaptive regularization influenced by noise level; causality claims require separate scrutiny. |
| Logical/functional dependency evaluation, Umesh et al. | [Author preprint](https://arxiv.org/abs/2409.17684), journal DOI 10.1016/j.patcog.2025.111459 | Dependency evaluation precedent, not the HFGF generation method itself. |
| HFGF, Umesh et al. | [Author preprint](https://arxiv.org/abs/2507.19211) | Distinct 2025 preprint. The proposal's 2026 journal DOI was not directly retrievable in this audit; use the verified preprint until final publisher metadata is checked. |
| TabGraphSyn, Wu et al. | [Full author preprint](https://www.medrxiv.org/content/10.64898/2025.12.28.25342851v1.full), posted 29 December 2025 | Patient-neighborhood conditioning. PMC indexing alone does not establish peer review. |
| Clinical trial constraints, Hahn et al. | [Information Sciences article](https://www.sciencedirect.com/science/article/pii/S0020025525010631), April 2026 | HPO does not replace constraint treatment. Final paper uses ACTG320; do not confuse it with ACTG175. |
| Health evaluation framework | [Frontiers article](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1576290/full), 2025 | Broader evaluation precedent; not evidence for this model's privacy. |
| CLOVER | [Publisher article](https://www.sciencedirect.com/science/article/pii/S2667318526000036), June 2026 | Utility/privacy evaluation including attacks, with a public library. |
| Sadeghi et al. | [Indexed article record](https://pubmed.ncbi.nlm.nih.gov/42175014/), May 2026, pp. 1039–1040 | A brief fidelity/utility contribution, not a comprehensive validation standard. |
| Best-of-N clinical validity, Masud and Hasan | [JMIR AI article](https://ai.jmir.org/2026/1/e90590), July 2026 | A directly relevant selection comparator. |
| GOGGLE, Liu et al. | [ICLR 2023 paper](https://openreview.net/pdf?id=fPVRcJqspu) | Explicit feature relational generation already exists. |
| C-DGM, Stoian et al. | [Author paper](https://arxiv.org/abs/2402.04823), 2024 | Integrated constraint layers already exist; compare knowledge-equivalent methods. |
| DRL, Stoian and Giunchiglia | [Author paper](https://arxiv.org/abs/2502.18237), 2025 | Richer logical constraint enforcement already exists. |
| Harpoon, Shankar et al. | [ICLR 2026 paper](https://openreview.net/pdf?id=G5g6tDg1ZE), [official code](https://github.com/adis98/Harpoon) | Inference-time mixed-tabular constraint guidance is a direct predecessor. |
| SAGE | [ACL 2026 paper](https://aclanthology.org/2026.acl-long.174.pdf) | Adaptive dependency guidance also appears in LLM-based synthesis. |

The cautious candidate contribution is the tested value of a **nonlinear clinical factor representation combined with confidence-weighted training and sampling guidance**, particularly under imperfect rules and missing measurements. “First graph-guided clinical diffusion,” “first adaptive guidance,” and “privacy-preserving by construction” are not defensible claims from this design.

## Clinical and statistical evidence used for corrections

The [UCI Heart Failure documentation](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) identifies `time` as follow-up duration. Excluding it for baseline prediction is a design inference from that meaning. [ILPD](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset), [CKD](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease), and [Diabetes Risk](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) require separate schemas and separate fits.

[MedlinePlus protein documentation](https://medlineplus.gov/lab-tests/total-protein-and-albumin-globulin-a-g-ratio/) supports the albumin/globulin relationship conceptually; it does not certify each ILPD assay value or tolerance. [Bilirubin documentation](https://medlineplus.gov/ency/article/003479.htm) supports considering direct versus total bilirubin. These sources do not authorize treating reference ranges as hard physiological bounds.

[Bengio and Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html) establish the dependence problem in cross-validation variance estimates. [Anonymeter's original paper](https://arxiv.org/abs/2211.10459) motivates attack-based empirical assessment. Neither a weak attack nor a nonsignificant utility difference proves safety or equivalence.

## Publication gates

The local ingestion audit additionally found 269 repeated predictor profiles among 520 diabetes rows (251 distinct profiles), 13 repeated ILPD predictor profiles, 1,012 missing CKD cells, and four missing ILPD A/G ratios. These are checks of the downloaded source data, not inferred patient identities. In the ILPD inner-fit partition, the protein algebraic residual has median 0.10, 95th percentile 0.30 and maximum 8.68 in source protein units. This provides direct evidence against enforcing the recorded identity as an exact equality. Full details and source hashes are saved in [the data audit](../reports/DATA_AUDIT.md).

1. Complete the controlled mechanism study, including clique, shuffled structure, energy-only and held-out-rule controls. Retain failures and negative results.
2. Obtain clinical review of units, tolerances, missingness semantics and label timing; add at least one cohort with meaningful higher-order constraints and enough records to assess generalization.
3. Reproduce strong current baselines under equal tuning and preprocessing budgets, including at least TabDDPM, TabDiff, and a knowledge-equivalent constraint method. Check official implementations and licenses; adapters alone are not reproductions.
4. Freeze architecture and endpoints before opening confirmatory test results. Complete matched folds and multiple seeds with justified inference, runtime and sample-size analyses.
5. Complete stronger privacy attacks and coverage/diversity analysis. Claim empirical risk only unless a formal privacy mechanism is added and accounted for.
6. Write results and conclusions from measured artifacts. A working prototype and a paper outline are not a completed publishable paper.
