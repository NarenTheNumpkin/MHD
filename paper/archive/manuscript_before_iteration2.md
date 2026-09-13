# MedHyperDiff: Evaluating Hypergraph Guidance for Structurally Consistent Medical Table Synthesis

**Research manuscript working draft. Not ready for submission.** Authors, affiliations and venue are to be supplied by the researchers. No performance claims or invented results are included.

## Abstract

Medical tables combine heterogeneous measurements with relationships that generic generative models may not preserve. This study investigates a mixed-type diffusion generator combining empirical feature associations, nonlinear constraint hyperedges, confidence-weighted residual penalties and approximate sampling guidance. Numerical values follow Gaussian diffusion, while categorical values and observation indicators follow absorbing-mask diffusion. The evaluation separates compliance with imposed constraints from unconstrained fidelity, prediction utility, missingness coverage and empirical privacy risk. Controlled simulations isolate higher-order structure, and separate clinical cohorts evaluate the validity–fidelity trade-off under imperfect domain knowledge. **Quantitative results and supported conclusions must be inserted after the frozen evaluation protocol is completed.**

## 1. Introduction

Matching marginal feature distributions does not establish that synthetic records preserve inter-attribute structure. In clinical data, a discrepancy may reflect an invalid generated combination, but it may also expose measurement noise, recording conventions, genuine patient heterogeneity or a poorly specified constraint. A useful evaluation therefore needs to distinguish mathematical/schema constraints from approximate biochemical relations and learned associations.

The study asks whether nonlinear factor structure improves a matched diffusion generator beyond what can be obtained from a pairwise representation, energy guidance alone, or straightforward constraint-aware correction. It also asks when imposed structure harms coverage or utility. The contributions will be stated only after those comparisons are available.

## 2. Related work

CTGAN and TVAE provide established mixed-table baselines. TabDDPM and TabDiff establish diffusion for heterogeneous tabular variables. GOGGLE demonstrates feature-relational generation, while CausalDiffTab studies structural regularization with adaptive weighting. HFGF reconstructs dependent features using functional/logical structure. C-DGM and DRL integrate constraints into generative models, and Harpoon provides tabular inference-time guidance. TabGraphSyn uses patient-neighborhood representations for clinical synthesis. These precedents rule out broad novelty claims based on graphs, constraints or adaptive guidance alone.

The verified source ledger and links appear in [the research review](../docs/RESEARCH_REVIEW.md). Bibliographic metadata must be exported from the final authoritative paper versions for the selected venue. The targeted audit is not an exhaustive literature review.

## 3. Method

Use the explicit formulation in [the revised protocol](../MedHyperDiff_Research_Proposal_Revised.md), sections 3–4. The implementation comprises:

1. Explicit schemas, observed-value numeric scaling and categorical missingness states.
2. Bootstrap-filtered fit-only feature associations.
3. Nonlinear variable–hyperedge–variable message passing inside a feature transformer.
4. Gaussian clean prediction and absorbing categorical reverse kernels.
5. Confidence-weighted physical-unit residual penalties and approximate descent on clean numerical predictions and category logits.

The method is not an exact sampler from a constrained distribution and does not guarantee feasibility. Fixed confidence values are sensitivity parameters, not calibrated clinical uncertainty.

## 4. Experimental protocol

Describe exact dataset versions, inclusion/exclusion rules, units, label timing, source acknowledgments, patient/duplicate grouping, missingness representation and split hashes. Heart Failure excludes post-baseline follow-up duration. The initial other cohorts are ILPD, CKD and symptom-based diabetes risk. Controlled simulations are reported separately from clinical evidence.

The final comparisons must include current diffusion baselines and knowledge-equivalent constraints, matched inner tuning, class prevalence and synthesis counts, multiple generator seeds, and full outer evaluations. Ablations hold target conditioning fixed and separate representation, training penalties and sampling guidance. Include a parameter-matched dense control, clique projection, shuffled incidence, withheld rules, wrong-rule sensitivity and repair.

Report per-rule eligibility and real-data violation rates, normalized and original-unit residuals, marginal/multivariate fidelity, unconstrained higher-order patterns, TSTR/TRTR utility, sample coverage, attack-based privacy diagnostics and runtime. Specify the primary endpoints and inferential units before opening the test set.

## 5. Results — pending confirmatory experiments

No result table is populated here. The [measured development report](../reports/pilot/EXPERIMENT_SUMMARY.md), when generated, contains exploratory validation runs only. It is not this section's final evidence.

Required tables:

- Dataset provenance, effective group counts, missingness, constraints and clinical review status.
- Main held-out fidelity/validity/utility outcomes with prespecified effect estimates and intervals.
- Per-rule severity and coverage, including withheld relationships.
- Matched architecture and guidance ablations with computational costs.
- Membership/attribute/singling-out or linkability attacks and memorization controls.
- Rule quality, sample size, missingness and subgroup sensitivity analyses with appropriate limitations.

## 6. Discussion — to follow results

Explain where explicit structure helps, where a simpler baseline is preferable, and how approximate constraints change the modeled population. Discuss the risks of suppressing rare valid records, hiding violations with missingness, target-defined clinical associations, small convenience cohorts and biased development estimates. Separate biochemical consistency from a clinician's assessment of realistic patient records.

## 7. Reproducibility and limitations

The code saves split indices, data/source hashes, preprocessing, graph stability, rule registries, configurations, training history, model checkpoints, synthetic samples and metric outputs. Current limitations include a custom compact diffusion backbone, fixed confidence, categorical treatment of ordinal codes, approximate guidance, incomplete external baseline reproduction and a preliminary nearest-distance privacy attack. No differential privacy or deployment claim is made.

## 8. Conclusion — pending

Write the conclusion from the observed, appropriately qualified results. Do not presume that the proposed model outperforms simpler methods.

## Declarations — researcher input required before submission

Specify authorship/contributions, funding, conflicts of interest, dataset access and licensing compliance, applicable institutional ethics determination, data/code availability, and any venue-required disclosure of AI assistance. Do not invent approvals or authors.
