# MedHyperDiff: confidence-weighted hypergraph guidance for mixed-type medical table synthesis

**Revised research protocol — 11 September 2026.** This replaces the methodological recommendations in the supplied proposal. Status: implementation and development experiments; no confirmed superiority, clinical validity guarantee, calibrated uncertainty claim, or formal privacy guarantee. See [the research review](docs/RESEARCH_REVIEW.md) for corrections and primary-source verification.

## 1. Question and candidate contribution

Can explicit nonlinear feature-factor structure improve the validity–fidelity trade-off of synthetic medical tables relative to a matched diffusion backbone and knowledge-equivalent constraint baselines?

The candidate contribution is a measured benefit from combining an empirical feature association graph, nonlinear clinical hyperedge message passing, confidence-weighted residual penalties, and noise-dependent guidance. These ingredients individually have predecessors: feature relations in [GOGGLE](https://openreview.net/pdf?id=fPVRcJqspu), constrained generation in [C-DGM](https://arxiv.org/abs/2402.04823) and [DRL](https://arxiv.org/abs/2502.18237), adaptive regularization in [CausalDiffTab](https://arxiv.org/html/2506.14206v1), and tabular inference-time guidance in [Harpoon](https://openreview.net/pdf?id=G5g6tDg1ZE). The contribution must survive direct comparisons and a refreshed novelty search at submission.

Prespecified hypotheses, not expected results:

- Nonlinear hyperedges improve higher-order residual distributions relative to a parameter-controlled clique representation on datasets with supported higher-order relations.
- Guided generation reduces imposed-rule violations while preserving observed coverage, diversity and held-out utility within justified practical margins.
- Noise-dependent guidance improves that trade-off relative to equal-integrated-strength constant guidance.
- Benefits weaken, disappear or reverse with irrelevant, inaccurate or insufficiently supported constraints. This is an informative result.

Pairwise graph improvement is not a prerequisite for testing higher-order structure: a parity mechanism can have no pairwise association.

## 2. Data, scope and estimands

Keep cohorts separate. Start with four UCI datasets, using explicit column schemas rather than type inference.

| Dataset | Source rows | Source predictors | Primary treatment |
|---|---:|---:|---|
| [Heart Failure](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) | 299 | 12 | Exclude follow-up `time`; model 11 baseline predictors and observed death status. |
| [ILPD](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset) | 583 | 10 | Preserve lab missingness; verify protein and bilirubin semantics and top-coded age. |
| [CKD](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | 400 | 24 | Normalize whitespace and missing markers; treat recorded ordinal codes as categorical in version 1. |
| [Diabetes Risk](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) | 520 | 16 | Model symptom categories and age; do not promote symptom associations to deterministic clinical laws. |

Death during varying observed follow-up is not fixed-horizon mortality. CKD and diabetes labels support diagnostic-label utility benchmarks, not prospective treatment or causal claims. No clinical deployment claim follows from these small convenience cohorts.

Record raw-byte checksums, acquisition date, source metadata, source license/acknowledgment requirements, schema, missing counts, duplicate profiles and target mapping. Keep raw files immutable. Group repeated patients by identifier when possible; in these public single-table datasets conservatively group identical predictor profiles. Do not delete duplicate rows merely to improve privacy scores; report the grouping policy and sample counts.

Add a controlled simulator with a known protein-like algebraic relation and a binary XOR triple, plus irrelevant-feature, corrupted-rule, and missingness variants. It is a mechanism test, not clinical evidence. Expand to an independently sourced cohort with clinically reviewed higher-order relationships and substantially more records. [ACTG175](https://archive.ics.uci.edu/dataset/890/aidsclinicaltrialsgroupstudy175) is an option with 2,139 records, but its event/censoring and post-baseline variables require a survival-specific protocol; it is not a drop-in binary mortality dataset. Dataset count alone is not a publication criterion.

## 3. Leakage control and preprocessing

Use a fixed five-fold stratified group outer split, common to all models and generator seeds. Within each outer training fold reserve an inner validation partition, also group-disjoint. Generator fitting, category support, numerical means/scales/imputation, graph estimation and residual envelopes use the **inner fit partition only**. Validation selects hyperparameters/checkpoints; outer test records are opened only after all choices for that evaluation are frozen.

Development CLI runs evaluate inner validation and label every artifact accordingly. This is optimistically selected development evidence. It must not be reused as an unbiased confirmatory result. A final protocol may refit on the full outer training partition using the selected epoch budget; if so, refit all learned preprocessing and structures there consistently for every model and distinguish that procedure from the prototype.

Numerical values use a train-fitted affine standardization, retaining differentiable inversion to physical units. Missing numerical entries use a training median as an internal placeholder and have explicit observed/missing categorical tokens. Numeric reconstruction losses exclude unobserved targets. Categorical missingness has its own state. Categories unseen in the fit partition map to an unknown state during evaluation, whose frequency is reported; that state is not a synthetic category.

Generate the missingness indicators jointly with values. Evaluate missingness prevalence, joint patterns and rule eligibility. A future missing-data study should compare this representation with common training-only imputation using the same policy across baselines; it cannot establish MAR or MNAR identifiability from these tables alone.

## 4. Precisely specified prototype

### 4.1 Association structure

Nodes are feature tokens, including numerical observation indicators; the condition label is provided separately. Estimate pairwise absolute Spearman association for numerical pairs and quantile-binned normalized mutual information for mixed/categorical pairs using only fit records, with pairwise overlap counts. Bootstrap edge inclusion, requiring a prespecified magnitude and stability threshold. Different estimator types are not perfectly calibrated; compare the graph with no graph and shuffled structure and perform a threshold sensitivity analysis.

The adjacency expresses association, not causality. The prototype does not include a minibatch association-matching loss, because conditional clean estimates are not free generator samples and could make that loss misleading.

### 4.2 Constraint registry and hypergraph

Each rule stores name, ordered operands, mathematical relation, status, source, units, tolerance, residual scale, fixed confidence, observed-value applicability and clinical-review status. Bounds/categories are schema checks; soft biochemical relations and training-derived envelopes are reported separately from hard rules. A differentiable penalty cannot guarantee even a hard rule is satisfied.

Use true variable→edge→variable nonlinear message passing: transform feature tokens, aggregate all participating tokens to an edge state, apply a nonlinear edge update with an edge identity, then return messages to incident variables. Feature identity distinguishes operands. Interleave these blocks with attention biased by the empirical association graph. A clique-projection attention model is a separate comparator; it is not renamed a hypergraph model.

For ILPD, candidate residuals include direct bilirubin minus total bilirubin, albumin minus total protein, and

\[
r_{protein}(x)=|\mathrm{A/G}\,(\mathrm{total\ protein}-\mathrm{albumin})-\mathrm{albumin}|.
\]

The last expression avoids division by a potentially near-zero globulin denominator, but still requires companion domain checks and compatible units. [MedlinePlus](https://medlineplus.gov/lab-tests/total-protein-and-albumin-globulin-a-g-ratio/) supports the relationship conceptually, not an exact ILPD tolerance. In development only, use a fit-only 95th-percentile residual envelope and call it descriptive, not a clinical acceptance threshold. Report the full real residual distribution and sensitivity to fixed clinically reviewed tolerances before a paper claim.

No outcome-implies-lab-value, hemoglobin–hematocrit identity, or arbitrary disease rule is imposed on the other three cohorts. These cohorts can legitimately have no higher-order clinical hyperedges. Constraint absence is part of the result.

### 4.3 Mixed forward and reverse diffusion

The compact implementation is an original Gaussian-plus-absorbing-mask backbone, **not a reproduction of TabDiff or TabDDPM**. [TabDDPM](https://arxiv.org/abs/2209.15421), [TabDiff](https://arxiv.org/abs/2410.20626) and [D3PM](https://arxiv.org/abs/2107.03006) are methodological precedents and independent comparison targets.

Numerics use a cosine variance schedule with cumulative product \(\bar\alpha_t\):

\[
q(z_t\mid z_0)=\mathcal N(\sqrt{\bar\alpha_t}z_0,(1-\bar\alpha_t)I).
\]

The denoiser predicts standardized clean values \(\hat z_0\), with observed-only squared-error training. This is a simple clean-prediction surrogate, not a claim to optimize the exact Gaussian ELBO. Reverse sampling uses the DDPM posterior coefficients

\[
\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\hat z_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}z_t,
\qquad
\tilde\beta_t=\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}.
\]

There is no random Gaussian perturbation at the final step.

For a categorical value \(c_0\), forward corruption preserves it with probability \(a_t=1-t/T\) and replaces it with an absorbing mask otherwise. The model predicts logits over the original legal support. Already revealed values copy through; a masked value reveals at reverse step \(t\) with probability

\[
\frac{a_{t-1}-a_t}{1-a_t}=1/t,
\]

and chooses its value from the predicted categorical probabilities. All masks disappear at \(t=1\). With uniformly sampled time, the categorical training term is \((T/t)\mathbf{1}[c_t=MASK]\,CE(c_0,\hat p_0)\), averaged over features and rows. Missing-value states are ordinary categories; they are distinct from temporary diffusion masks.

Every primary arm learns \(p(X\mid Y)\) with the same binary target encoding. Sample labels at the empirical fit prevalence, with identical counts across arms. Rebalanced augmentation is a separate experiment with its own target-prior evaluation. Conditional generation does not imply a causal effect of intervening on a disease label.

### 4.4 Energy and guidance

For each eligible rule, define nonnegative excess residual \(v_k=\max(0,r_k-\delta_k)/s_k\). Apply a smooth-L1 penalty \(\rho\), confidence \(c_k\), and eligibility \(m_k\):

\[
E(x)=\frac{\sum_k c_k m_k\rho(v_k)}{\max(1,\sum_kc_k)}.
\]

Confidence is a prespecified sensitivity weight, not posterior uncertainty. Do not learn it by unconstrained minimization. The training loss is numeric reconstruction plus absorbing categorical loss plus a late-weighted energy term. Categorical rule violations use differentiable probabilities; the product-marginal expectation is a surrogate, not the true joint posterior. Measure validity again on discretely sampled rows.

At sampling, obtain clean numerical predictions and category logits. Take a bounded, per-row gradient step on their energy, then insert adjusted predictions into their respective reverse kernels. Eligibility probabilities are detached so direct guidance cannot reduce energy merely by hiding observations. Sampled coverage is still audited for indirect effects.

The adaptive schedule increases toward the end of reverse diffusion; normalize its discrete sum to equal constant and late-only schedules. Gradient clipping uses the per-row combined numerical/logit gradient norm and does not depend on batch size. This is **approximate clean-prediction guidance**. It does not reproduce the original proposal's undefined state-gradient update, guarantee feasibility, or exactly sample an energy-tilted target. Monitor diversity and compare with projection/repair and Harpoon before arguing that this approximation is beneficial.

## 5. Comparators and ablations

All internal arms share labels, sample counts, width, depth, split, optimizer settings and allowed tuning budget; report parameter and time differences introduced by hypergraph blocks. Add a parameter-matched dense residual control before attributing gains purely to structure.

| Arm | Empirical graph | Nonlinear hyperedges | Training energy | Sampling guidance |
|---|---|---|---|---|
| Base | No | No | No | None |
| Graph | Yes | No | No | None |
| Hypergraph representation | No | Yes | No | None |
| Energy only | No | No | Yes | None |
| Guidance only | No | No | No | Adaptive |
| Full, no guidance | Yes | Yes | Yes | None |
| Full constant / late / adaptive | Yes | Yes | Yes | Matched-strength schedule |
| Clique | Yes + pairwise clique | No | Yes | Adaptive |
| Shuffled structure | Permuted | Permuted incidences | Same semantic energy | Adaptive |
| Unit confidence | Yes | Yes | Unit-weight rules | Adaptive |
| Repair | Base | No | No | Post-generation correction; report changed rows |

Reference generators: conditional independent marginals, empirical Gaussian copula, and bootstrap resampling as a memorization positive control. CTGAN and TVAE use the official `ctgan` implementation with explicit per-class fitting and shared missingness encoding. This adaptation must be reported, not presented as published benchmark replication.

Before submission add verified official TabDDPM and TabDiff runs, and knowledge-equivalent C-DGM/DRL/Harpoon or another strong applicable constraint baseline. Evaluate CTAB-GAN+ as a stronger GAN comparator if feasible; document justified baseline omissions. A basic repair implementation is not HFGF, C-DGM or Harpoon. Record baseline commits, licenses, environments and tuning budgets.

## 6. Evaluation and statistical analysis

Do not collapse clinical validity, fidelity, utility and privacy into an unexplained scalar ranking.

- Fidelity: train-scale-normalized Wasserstein distances, KS statistics, categorical TV and squared Jensen–Shannon distance, missingness error, signed numerical Spearman error, and off-diagonal mixed-association RMSE. Record undefined metrics and support rather than assigning zero. Add real-vs-real sampling floors and coverage/diversity measures before final interpretation.
- Dependencies: report imposed-rule residuals and independent/withheld relationships separately; require repeated antecedents and explicit support for mined FDs. Do not call association-matrix error FD preservation. Include higher-order joint patterns that were not optimized.
- Validity: for every rule report eligible count, coverage, violation rate, normalized and original-unit severity for fit, validation/test and synthetic data. Distinguish schema, biochemical and exploratory rules. Do not count missing operands as a valid record.
- Utility: matched TRTR, TSTR and real+synthetic evaluations with logistic regression and random forest; add a tuned boosted-tree model for final benchmarking. Each downstream imputer and encoder is fitted only to its own training table. Report AUROC, average precision (not mislabeled trapezoidal AUPRC), F1, sensitivity, specificity, balanced accuracy and Brier score. Choose any decision thresholds on inner data, never on test patients.
- Multivariate fidelity: balanced classifier two-sample testing with classifier-train-fitted preprocessing. A discriminator near chance does not prove identical distributions.
- Privacy: exact row-copy rates, equally sized member/holdout DCR reference sets, NNDR with a policy for duplicate second neighbors, and a specified membership attack. The initial nearest-distance attack is a diagnostic, not a complete audit. Before submission add stronger calibrated membership and sensitive-attribute attacks, singling-out/linkability when applicable, and positive/negative controls. [Anonymeter](https://arxiv.org/abs/2211.10459) is one implementation option. No DP claim without a DP mechanism and full accounting of learned preprocessing, graphs, selection and releases.
- Costs: fitting/structure time, sampling time, model parameters, device, peak memory where available, synthetic sample count, convergence/checkpoint behavior, failures and tuning budget.

Use at least three independent generator seeds on common outer splits. Average seeds within splits for matched comparisons; seed variation measures algorithmic instability, not independent patient evidence. Do not use the 15 fold×seed outcomes as 15 independent samples in a t-test or Wilcoxon test. Overlapping folds complicate uncertainty, as shown by [Bengio and Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html).

Before confirmatory evaluation specify one primary structural endpoint and one primary utility endpoint, comparator selection using validation only, clinically justified practical margins, and multiplicity correction. Report effect sizes and intervals with their resampling unit and limitations. Patient-level bootstrap of fixed predictions measures conditional test-sample variation, not full retraining variability. Across-dataset paired inference uses independent datasets, with limited power when there are few. A nonsignificant difference is not a noninferiority finding.

The original 0.05 utility margin is removed as an automatic success threshold. Choose margins with the relevant task/clinical collaborators and quantify achievable precision. Report negative and inconclusive results.

## 7. Execution stages and decision gates

1. **Audit and numerical correctness:** source data checksums, explicit schemas, missingness/duplicate audit; tests for split isolation, categorical posterior normalization, final mask removal, guidance direction, batch invariance and checkpoint reproducibility.
2. **Controlled mechanism development:** base, graph, hypergraph, energy, guidance, clique, shuffled, repair and known-dependency simulations. Test incorrect/missing rules and changing sample sizes. Record validation-only results; do not use simulator wins as medical validation.
3. **Small clinical pilots:** four cohorts with identical development folds and modest budgets. Determine whether added structure helps, over-regularizes, or lacks a suitable constraint set. Do not require a graph win to permit a hypergraph test.
4. **Study completion:** clinical constraint review, suitable larger/external cohort, official strong baseline reproduction, fair tuning, parameter-matched controls and stronger attacks. Freeze the protocol.
5. **Confirmatory runs:** complete frozen outer-fold/seed matrix, verify all results from artifacts, analyze uncertainty and costs, then write the paper's results and conclusions.

Proceed to a methodological submission only if the evidence supports a meaningful contribution beyond simpler knowledge-equivalent methods. If repair or the copula is superior, revise the scientific claim accordingly. Publication cannot be promised in advance.

## 8. Prospective abstract

Medical table generators may reproduce marginal distributions while missing meaningful relationships among variables. We investigate confidence-weighted hypergraph guidance for mixed-type diffusion, combining empirical feature associations with nonlinear multivariate constraint factors. A Gaussian and absorbing categorical backbone supports explicit observation masks, while training penalties and approximate guidance act on clean predictions in physical units and category probabilities. The study separates imposed-rule compliance from held-out fidelity, dependency preservation, utility, missingness coverage and empirical privacy risk. Controlled mechanism tests and heterogeneous clinical cohorts will compare the method with matched ablations, simple generators and knowledge-equivalent constraint approaches. Results will determine whether the representation and guidance improve the validity–fidelity trade-off and identify settings where imperfect or unsupported rules harm generation.

This abstract is prospective. Replace future tense and add quantitative effects only after the registered evaluation is complete.
