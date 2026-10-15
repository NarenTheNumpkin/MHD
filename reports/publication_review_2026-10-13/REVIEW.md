# Updated publication assessment — 13 October 2026

**Verdict: substantially improved, with a publishable empirical contribution, but still requiring major revision before a strong archival submission. A newly added comparator contradicts the headline claim of superior realism on all four small-sample confirmation datasets. Correct that result before submitting.**

This review covers the revised 12-page [manuscript](/Users/naren/Desktop/MedHyperDiff/paper/ieee/main.pdf), the [response to the earlier review](/Users/naren/Desktop/MedHyperDiff/reports/publication_review_2026-10-12/RESPONSE.md), iteration 14 reanalysis, all 160 iteration 15 manifests and their saved metrics, and relevant implementation/reporting code. All 12 PDF pages were rendered and inspected. The manuscript SHA-256 is `cd0914f7017c42f847dbb87584510e27bd0aab637c4095a31bca450dddd19849`.

No generators were trained and no manuscript or model files were changed. I independently recomputed the two membership diagnostics for one consequential new control. Evidence and a reproducible script are saved beside this report: [audit evidence](/Users/naren/Desktop/MedHyperDiff/reports/publication_review_2026-10-13/audit_evidence.json), [audit script](/Users/naren/Desktop/MedHyperDiff/reports/publication_review_2026-10-13/audit_updated_results.py).

## What improved materially

| Earlier concern | What the revision does | Assessment |
|---|---|---|
| The training comparison changed several things simultaneously | Adds eight training variants over 20 paired samples, giving 160 fits | A meaningful new experiment. The saved C2ST, membership and utility values match the report for all 160 runs. Short training raises C2ST by about 0.168 in all 20 comparisons; checkpoint selection changes it by about 0.0035 on average with mixed direction. The revised interpretation is supported. |
| The method lacked a sufficiently explicit specification | Adds residual encoder/decoder equations, a reconstruction-direction objective, a first-order justification and its exponential search cost | Much clearer. The direction rule is appropriately described as a heuristic. There is one equation–implementation discrepancy noted below. |
| Exact chemistry relations were compared against a weak reconstruction control | Adds textbook-direction reconstruction and acknowledges that it is at least as faithful | Scientifically valuable correction. Exact chemistry relation fidelity is no longer a convincing superiority result for the learned model. The new control must also enter the realism/utility comparison. |
| Dataset independence and uncertainty were overstated | Discloses cross-panel participant overlap and overlapping fit samples; uses per-dataset effects and leave-one-cycle-out summaries | The reporting problem is largely repaired. The dependence and limited transportability remain properties of the study design. |
| Privacy language exceeded what the attacks establish | Reports the 0.608 exception, the much worse n=100 results, and explicitly rejects a privacy guarantee | Substantially better. Apply the eligibility threshold symmetrically, and remove remaining uses of “memorized” as a synonym for exceeding an attack threshold. |
| Relation distance ignored sign, dependence and tails | Adds signed residuals, correlations between residuals, covariate dependence, tail exceedance and coverage | Useful diagnostics, correctly labelled exploratory. They reveal a narrower and more interesting result than universal superiority. |
| Negative results and comparator selection were obscured | Reports larger-sample losses, plug-in failures, limited clinical-category gains, and separate realism/utility comparator selection | More credible and easier to assess. These are strengths of the revision. |
| Compute and reproducibility were difficult to assess | Adds parameter counts, updates, passes and timing; archives the iteration 14 source | Good progress. The iteration 14 archive hash verifies; current source matches all iteration 15 source hashes. A public release and a separately archived iteration 15 snapshot still need attention. |

The strongest scientific story is now: **recording resolution affects evaluation; exact formulas often need only straightforward reconstruction; approximate formulas benefit from modelling their residual distribution and dependence.** The paper's evidence is most persuasive when organized around those distinctions.

## 1. Submission blocker: the new canonical control is missing from the headline comparison

**Where:** abstract (page 1); Table VII and “Realism, Utility and Sample Size” (page 9); discussion and conclusion (page 11); the analysis feeding those claims.

The new canonical analysis writes a separate `canonical.csv`. It calculates C2ST and clinical-category distance but omits utility and membership metrics. The comparator selection in [reanalysis_iteration14.py](/Users/naren/Desktop/MedHyperDiff/scripts/reanalysis_iteration14.py:150) reads only `confirm_all.csv`; it never merges the new canonical controls into that selection. This affects a central result, not just wording.

For **chemistry 2021–2023, nominal n=250**, the saved canonical results and the independent membership audit give:

| Mean across the same five samples | MedHyperDiff | Gaussian copula + canonical reconstruction |
|---|---:|---:|
| C2ST, lower better | 0.775005 | **0.760200** |
| Relation distance, lower better | 0.049519 | **0.026554** |
| Membership diagnostic, mean of per-run maximum of the two attacks | 0.565510 | **0.562075** |
| Paired samples with lower C2ST | 1/5 | **4/5** |

The copula qualifies under the paper's **mean-based** membership eligibility rule. One individual copula sample has membership AUROC 0.6014; therefore this should not be described as every run passing, or as establishing privacy. The same aggregation rule used by the reporting script is what matters for this comparison.

For the audit, I reconstructed the canonical outputs from the saved primary synthetic tables using the project's formulas and rounding functions. I reproduced both previously saved primary-control attack AUROCs for all five samples before applying the attack calculation to the canonical outputs. These checks passed to numerical tolerance. C2ST values above come from the project's existing canonical analysis; I did not retrain its classifier or any generator.

**Implication:** “more realistic than every comparator below the membership threshold on all four datasets at n=250” is contradicted by a control already introduced in this revision. Table VII currently reports 0.783 as the comparator here and four MedHyperDiff wins; a qualifying canonical control achieves 0.760 and reverses that paired win count.

**Required correction:** finish utility and membership evaluation for all canonical controls, merge them into one explicit comparator registry, and regenerate the best-comparator results, paired effects, clinical-category comparisons, pooled descriptive summaries and prose. Keep these additions labelled post hoc. This requires saved-output evaluation, not new generator training.

Figure 8 (page 10) also continues to show the weaker “hard derivation” chemistry control without the canonical alternative. Add the stronger curve and distinguish textbook and selected reconstruction directions in its legend. Otherwise the figure visually suggests a relation advantage that the revised text explicitly withdraws.

## 2. Apply the membership criterion to MedHyperDiff as well

**Where:** Table VII and its bolding, page 9; abstract and conclusion.

The 2011–2012 chemistry result at n=250 has mean membership AUROC **0.608460**, above the declared 0.60 threshold. The revised text now discloses this, which is good, but the table still presents the model as a winner against privacy-filtered comparators without marking its own ineligibility.

Report unconstrained realism separately from “best among eligible methods,” or add an eligibility column/dagger and apply the same rule to every method. Under the current evidence, one of the four small-sample cells fails the model's own eligibility criterion and another loses to canonical copula. A privacy-constrained four-dataset sweep is not supported.

The threshold itself is an operational convention, not a safety certificate. That limitation is now correctly stated. Stronger attacks would strengthen a privacy-focused submission, especially for rare records; weak attacks can miss such risk, as discussed in the original [DOMIAS paper](https://proceedings.mlr.press/v206/breugel23a.html).

## 3. The representation has better support than the hypergraph contribution

**Where:** Table VIII (page 9), method (Section VII), discussion (page 11).

For CBC residual coupling, the saved diagnostics give errors of **0.067691 for MedHyperDiff and 0.069310 for TabDiff with the same residual representation**. This supports the usefulness of residual coding, but gives little evidence that hypergraph message passing is responsible for preserving that coupling.

The missing decisive comparison is the **final residual-coded model with and without structure**, using the same preprocessing, decoder, training budget, model-selection opportunities and samples. If graph and hypergraph components remain claimed contributions, separate backbone, graph-only, hypergraph-only and combined variants. Report parameter counts and consider matched capacity. Old ablations of earlier configurations do not isolate the final system.

A fair tuned TabDiff comparison remains necessary for a strong generator-superiority claim. Fewer parameters and faster measured fitting are useful results, but do not compensate for asymmetric tuning. Give the strongest baselines comparable development searches with explicit budgets; choose configurations without consulting the new test outcomes.

The related-work revision now acknowledges that generating independent features and reconstructing dependent ones has precedent in [HFGF](https://arxiv.org/abs/2507.19211). The defensible novelty is the treatment of approximate numerical relations, direction choice and recorded precision, plus their empirical evaluation. An actual constrained-generator comparison would be valuable if broad superiority over constraint-aware generation remains part of the positioning; it is less central to a deliberately narrower residual-representation paper.

## 4. Add a joint residual-resampling control

**Where:** Table VIII and the claim that resampling loses coupling; [iteration13.py](/Users/naren/Desktop/MedHyperDiff/scripts/iteration13.py:95).

The current baseline samples each residual column independently. Losing cross-relation dependence is an expected consequence of that construction. Beating it does not establish that learning a diffusion model is necessary to retain residual coupling.

Add **row-wise joint residual resampling** using one sampled fit-row index for the residual vector, with an explicit missingness policy. A conditional variant can test whether dependence on operands or clinical covariates matters. Evaluate the decoded tables: retaining dependence before reconstruction does not guarantee retaining every desired relationship after rounding and reconstruction. Keep the residual-coded copula baseline as well.

This is a particularly informative, inexpensive next comparison. If a simple joint residual baseline matches the learned model, that remains publishable evidence about which parts of the method are actually needed.

## 5. Clinical usefulness and generalization remain limited

**Where:** Sections IX and XII.

The new reporting correctly shows that better relation scores do not consistently improve clinical categories: MedHyperDiff is better in four of ten settings against the currently reported category-selected comparator pool. That pool also needs refreshing after the canonical controls are included. Utility remains diagnosed-diabetes AUROC from one reported classifier, with small average differences.

For a stronger medical-methods paper, specify an intended use and evaluate an outcome relevant to it. Examples include preservation of clinically important subgroup distributions and a downstream task with discrimination and calibration, evaluated with more than one appropriate predictive model. Residual-distribution fidelity alone does not establish educational suitability, clinical-rule validity or safe data release.

An untouched NHANES cycle with participant-consistent splits would improve the internal confirmation design. It would still be another NHANES evaluation. A cohort from a different source, laboratory or acquisition setting is needed to support claims of transportability to hospital records. More overlapping resamples of the current panels cannot substitute for that change in evidence.

The results at n=2,000 and n=100 should stay visible. They define the observed limits of the method rather than detracting from an otherwise valid narrowly scoped result.

## Smaller corrections before submission

| Location | Correction |
|---|---|
| Conclusion, page 11 | The sentence attributes approximate-relation superiority to four survey cycles. The CBC data cover **three** cycles: 2013–2014, 2017–2018 and 2021–2023. Four is the total across the CBC and chemistry panels. Separate those counts. |
| Equation (3), page 6; [relations.py](/Users/naren/Desktop/MedHyperDiff/src/medhyperdiff/relations.py:84) | The stated coefficient-weighted log-scale objective and implementation differ for BMI. `log(BMI) = log(weight) - 2 log(height) + constant`, so height has coefficient magnitude 2; `Relation.weight` returns 1 for BMI participants. Align the definition and implementation, and check whether any saved reconstruction choices change. This review has not established that it changes the reported empirical results. |
| Exact chemistry language, pages 8 and 11 | “Matched” should mean similar practical fidelity, not demonstrated equivalence. The canonical control has lower mean relation distance in every chemistry cell of Table VII. If claiming equivalence, define and justify a margin; otherwise state the observed comparison directly. |
| Rules and Memorization, page 5 | Replace “memorized in 10 of 20” with “exceeded the membership diagnostic threshold in 10 of 20.” An AUROC threshold is not direct proof of copying. |
| “Common versus corrected practice” | The study demonstrates sensitivity to the selected pipelines. It does not establish how prevalent those choices are in published research. Cite evidence of prevalence or use narrower failure-mode language. |
| Data and code availability, page 11 | Provide a usable repository or archival identifier. Archive iteration 15's source hash `1003bc8c…` and the reanalysis/report scripts too; the verified separate archive currently covers iteration 14's `890ebbee…` source. |
| Model specification | Retain the clearer relation equations and provide a self-contained supplement/configuration table for the final denoiser, graph construction, hyperedges, losses, missingness and sampling. |

The PDF is considerably easier to inspect. Tables VII and VIII expose the substantive tradeoffs well, and the figures are legible in the rendered pages. The main presentation issue is consistency between the figures, comparator pool and revised conclusions, rather than basic layout. The manuscript still contains two substantial stories; prioritize either evaluation methodology or residual-based generation so the contribution is unmistakable.

## Publication judgment and next steps

**The project is worth pursuing for publication.** The strongest current contribution is an empirical methods paper about evaluating and preserving recorded laboratory relationships. The undertraining decomposition is a genuine improvement over the previous draft, and the candid negative results make the paper more credible.

A focused workshop or empirical report becomes plausible after the comparator and claim corrections. A strong archival health-informatics or medical-ML submission would benefit most from the matched final-method ablation, stronger simple residual controls, fair baseline tuning and genuinely untouched evaluation. A top general-ML architecture submission still lacks an isolated architectural contribution and broader validation. These are reviewer-style assessments, not acceptance predictions.

Prioritize work in this order:

1. Complete the canonical-control evaluations and regenerate every affected claim and figure. Apply membership eligibility to all methods.
2. Run the final residual-coded backbone versus full model, joint residual resampling and tuned TabDiff on untouched data. Freeze the analysis beforehand.
3. Add validation tied to the intended clinical use and, for broader claims, a different data source.
4. Package a versioned release with exact source snapshots and reproducible saved-results-to-paper commands.

The response document's proposed Phase 2 is directionally appropriate. It should be expanded to include joint residual resampling and a complete canonical comparator pool. A plan for those experiments is not yet evidence that the remaining limitations have been resolved.

**Verification limit:** I verified saved-run consistency and the targeted membership calculations, not a fresh end-to-end training reproduction. The local virtual environment currently stalls while importing an offloaded NumPy module, so I did not independently reproduce the response's “63 tests pass” statement. This is an environment limitation, not evidence of test failures. The audit script runs with the bundled NumPy/pandas runtime and passed its consistency checks.
