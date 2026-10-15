# MedHyperDiff: revision summary (11 to 15 October 2026)

This document records what changed in the research and the paper after the two publication reviews by the guide (12 and 13 October 2026): why each change was made, what new experiments and analyses were run, how the claims changed, and where every file is.

**Current manuscript:** [paper/ieee/main.pdf](paper/ieee/main.pdf), 13 pages, IEEE conference format.
**Title:** *Evaluating and Preserving Laboratory Relationships in Synthetic Clinical Tables*
**Authors:** Naren Medarametla, Abishek K G, Roshan Parveen, Parkavi K (School of Computer Science and Engineering, VIT Chennai)

---

## 1. In one paragraph

The paper is now a **methods paper**. Its main contributions are:
- an evaluation methodology for synthetic clinical tables;
- a method for handling laboratory formulas when generating them;
- empirical evidence for both.

The diffusion model (MedHyperDiff) is supporting evidence, not the headline.

The paper shows four things:
1. Routine training and evaluation choices reverse 25–34% of generator rankings, and the training damage came from an epoch cap that left models undertrained.
2. Formulas computed exactly from reported values are best recomputed from generated values.
3. Approximate formulas need their residuals kept and sampled jointly.
4. A diffusion model with learned residual coding reproduces approximate blood count relations most closely, by a small margin, and is the most realistic generator below a privacy threshold in 5 of 10 settings, all at 250 or 1,000 records.

---

## 2. Version history

| Version | Date | Title | Pages | Archived copy |
|---|---|---|---|---|
| 1 | 9 Oct or earlier | Common Pitfalls in Evaluating Synthetic Medical Data and a Generator That Preserves Laboratory Relationships | 8 | not archived (its scratch copy was removed by the system's temporary-file cleanup) |
| 2 | 11 Oct | Same title; added the robustness study (iteration 14) | 10 | `paper/ieee/archive/main_2026-10-11.pdf` |
| 3 | 12 Oct | Synthetic Clinical Tables That Keep Laboratory Relationships, and How to Evaluate Them | 12 | `paper/ieee/archive/main_2026-10-12.pdf` |
| 4 | 14 Oct | Evaluating and Preserving Laboratory Relationships in Synthetic Clinical Tables | 13 | `paper/ieee/archive/main_2026-10-14.pdf` |
| **5 (current)** | **15 Oct** | **Same title; six-part IEEE structure and redrawn figures** | **13** | `paper/ieee/main.pdf` |

What triggered each change:
- **Version 3** responds to the first review ([REVIEW.md](reports/publication_review_2026-10-12/REVIEW.md), [RESPONSE.md](reports/publication_review_2026-10-12/RESPONSE.md)).
- **Version 4** responds to the second review ([REVIEW.md](reports/publication_review_2026-10-13/REVIEW.md), [RESPONSE.md](reports/publication_review_2026-10-13/RESPONSE.md)).
- **Version 5** follows the authors' formatting request: only the standard IEEE sections, and figures in the figures4papers style. No result changed (see section 7.1).

---

## 3. What the reviews found (all verified against the saved results)

### First review (12 October)

1. **Not independent populations.** The NHANES datasets share participants within a survey cycle: 4,753 (2017–18 biochemistry vs CBC), 5,019 (2015–16 panels) and 5,263 (2021–23 panels). No participants are shared across cycles.
2. **Privacy overstated.** Membership AUROC was 0.608 on chemistry 2011–12 at n = 250, and 0.70–0.76 at n = 100. "Does not copy patients" could not be claimed.
3. **Training confound.** The early-stopping comparison changed the update budget, learning-rate schedule and weight averaging all at once.
4. **Novelty.** The closest prior work (HFGF, Umesh et al. 2025) was not cited, and the equations were missing.
5. **Relation metric too narrow.** It uses absolute residuals only; sign and dependence were not checked.
6. **Reporting details.** Nominal n vs fit rows, the 598 vs 608 fit count, and the comparator used for utility.

### Second review (13 October)

1. **Blocker.** The new textbook re-derivation control was missing from the realism comparison. Copula + textbook re-derivation beats our model on chemistry 2021–23 at n = 250 (C2ST 0.760 vs 0.775, winning 4 of 5 samples).
2. **Privacy rule applied asymmetrically.** It must also apply to our model, which fails it once (0.608).
3. **Hypergraph not shown to matter.** TabDiff with the same residual coding reaches the same coupling (0.069 vs 0.068).
4. **Missing control.** Joint (row-wise) residual resampling was needed.
5. **Small fixes.**
   - "four survey cycles" should be "three" for blood counts;
   - the BMI coefficient in the code was 1 where Eq. 3 implies 2;
   - "matched", "memorized" and "common practice" needed rewording;
   - the iteration 15 source was not archived.

---

## 4. New experiments and analyses

No generator was retrained for version 4. All new numbers come from saved outputs, apart from the iteration 15 experiment, which was new training run for version 3.

### 4.1 Iteration 15: what in the training recipe matters (160 new fits)

**Setup.**
- **Protocol:** [docs/ITERATION_15_PROTOCOL.md](docs/ITERATION_15_PROTOCOL.md), written before running.
- **Data:** the iteration 12 samples (NHANES 2017–18 biochemistry and UCI thyroid; n = 250 and 1,000; 5 samples each).
- **Variants:** 8.

**Results.** C2ST is the mean of 5 samples (lower is better). The last column counts, out of 20, the samples where each variant was worse or better than V1.

| Variant | NHANES 250 | NHANES 1,000 | Thyroid 250 | Thyroid 1,000 | Worse/better vs V1 |
|---|---:|---:|---:|---:|---:|
| V1 corrected recipe (5,000 updates, learning-rate decay, weight averaging, final weights) | 0.729 | 0.691 | 0.710 | 0.686 | reference |
| V2 + validation-selected checkpoint (same training run) | 0.735 | 0.691 | 0.719 | 0.685 | 6/3 |
| V3 + patience 20 | 0.778 | 0.691 | 0.710 | 0.685 | 5/6 |
| V4 + 16 corruption draws | 0.722 | 0.691 | 0.712 | 0.686 | 4/6 |
| V6 constant rate, raw weights | 0.786 | 0.689 | 0.754 | 0.683 | 13/7 |
| V5 legacy number of updates (400 / 1,300) | 0.943 | 0.896 | 0.851 | 0.798 | **20/0** |
| V7 legacy rebuilt in the same loop | 0.945 | 0.832 | 0.862 | 0.760 | 20/0 |
| V0 legacy recipe | 0.951 | 0.857 | 0.866 | 0.756 | 20/0 |

**Findings.**
- **The training budget causes the damage.** The short budget alone reproduces the whole legacy gap: C2ST is worse in 20 of 20 samples, by +0.168 on average.
- **Checkpoint selection has no consistent effect** (+0.004 on average, mixed signs).
- **Constant learning rate without weight averaging** raises membership AUROC in 20 of 20 samples (+0.035).
- **Pitfall 1 is rewritten:** "an epoch-capped update budget", not "noisy checkpoint selection".

Report: [reports/iteration15/INTERPRETATION.md](reports/iteration15/INTERPRETATION.md).

### 4.2 Reanalysis of iteration 14 (saved outputs only)

Report: [reports/iteration14/REANALYSIS.md](reports/iteration14/REANALYSIS.md). It added:
- **Per-dataset analysis:** paired effects for each dataset, leave-one-survey-cycle-out summaries, and a utility-selected comparator kept separate from the realism-selected one.
- **Relation diagnostics:** signed residual distance, coupling between relations, dependence of residuals on participants, age and label, tail exceedance, coverage and invalid (negative) values.
- **Textbook-direction re-derivation control:** recompute each derived value the way the lab does.
- **Model-only runs at n = 100:** membership 0.703–0.759 and C2ST 0.80–0.85. These are reported, not hidden.

### 4.3 Post hoc controls and the final comparison (version 4)

- **Two new controls** were evaluated with the full metric set (relations, C2ST, utility, membership, clinical categories, diagnostics) for every published-generator output of both relation studies: 280 outputs, 560 evaluations.
  - **Textbook re-derivation:** every derived variable recomputed from the generated operands.
  - **Joint residual resampling:** each synthetic row receives the complete residual vector of one real fit row.
- **One comparator registry** combines all arms (`scripts/report_final.py`).
- **The 0.60 membership threshold applies to every method, including ours.**
- All manuscript table rows are generated by code (`paper/ieee/tables/`), so no number is typed by hand.

---

## 5. How the claims changed

| Topic | Version 2 (11 Oct) | Version 4 (current) |
|---|---|---|
| Data | "Five independent populations" | Five NHANES panel-by-cycle datasets from four survey cycles. Within-cycle overlap is disclosed. Blood counts come from 3 cycles, chemistry from 2. |
| Training pitfall | Noisy validation checkpoint selection | An epoch cap that allows too few updates (iteration 15) |
| Approximate relations (blood counts) | Our model is best | Still the lowest relation distance in all 5 settings, but by a small margin (0.008–0.019) over the best control |
| Exact relations (chemistry) | Our model beats hard derivation | Textbook re-derivation is best in every setting |
| Coupling between formulas | Distinctive property of our model | Comes from sampling residuals jointly; joint resampling of real residuals does it as well |
| Realism at n = 250 | Best on all four datasets | Best eligible on both blood count datasets. Chemistry 2011–12: lowest C2ST but fails the privacy rule. Chemistry 2021–23: beaten by copula + textbook re-derivation. |
| Realism overall | "Best up to 1,000 records" | Best eligible in 5 of 10 settings, all at n = 250 or 1,000; TabDiff or simple controls elsewhere, including at 2,000 |
| Pooled realism gain | −0.020 (n = 250), −0.021 (n = 1,000) | −0.014 and −0.012, against comparators that now include the new controls |
| Privacy | "Does not copy patients" | Membership diagnostics only: one setting at 0.608; 0.70–0.76 at n = 100 |
| Hypergraph | Implied contribution | Not isolated; the evidence supports the residual representation, not the architecture |
| Hypothesis criteria | "Pre-registered claims" | Prospectively specified, locally dated protocols with descriptive success criteria, not significance tests |

---

## 6. Final results (as in the current paper)

### 6.1 Robustness study (paper Table VIII)

All values are means over samples at nominal n.
- **Relation control:** the lowest relation distance among all comparator arms, chosen per setting in hindsight.
- **C2ST and TSTR comparators:** the best arm among those with mean membership AUROC at most 0.60.
- **†:** our model fails the membership threshold.
- **Wins:** samples in which our model was better.

| Dataset | n | Relation: ours | Relation: control | Wins | C2ST: ours | C2ST: comparator | Wins | TSTR: ours | TSTR: comparator | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Blood count 2013–14 | 250 | **0.128** | 0.139 | 4/5 | **0.709** | 0.728 | 4/5 | **0.709** | 0.698 | 5/5 |
| | 1,000 | **0.092** | 0.106 | 5/5 | 0.646 | **0.634** | 1/5 | **0.771** | 0.753 | 5/5 |
| Blood count 2021–23 | 250 | **0.151** | 0.159 | 2/5 | **0.690** | 0.719 | 5/5 | **0.684** | 0.680 | 2/5 |
| | 1,000 | **0.137** | 0.156 | 4/5 | **0.629** | 0.634 | 2/5 | **0.733** | 0.723 | 4/5 |
| | 2,000 | **0.136** | 0.155 | 3/3 | 0.616 | **0.570** | 0/3 | 0.745 | 0.745 | 1/3 |
| Chemistry 2011–12 | 250 | 0.049 | **0.033** | 0/5 | 0.715† | **0.737** | 4/5 | 0.878† | **0.864** | 5/5 |
| | 1,000 | 0.025 | **0.019** | 2/5 | **0.644** | 0.674 | 5/5 | 0.888 | 0.888 | 3/5 |
| Chemistry 2021–23 | 250 | 0.050 | **0.027** | 0/5 | 0.775 | **0.760** | 1/5 | **0.895** | 0.882 | 5/5 |
| | 1,000 | 0.025 | **0.017** | 1/5 | **0.700** | 0.725 | 5/5 | 0.904 | **0.905** | 1/5 |
| | 2,000 | 0.021 | **0.015** | 1/3 | 0.645 | **0.620** | 0/3 | 0.905 | **0.907** | 1/3 |

**Pooled paired differences** (ours minus comparator), with the range when each survey cycle is left out in turn:

| | n = 250 | n = 1,000 |
|---|---|---|
| C2ST vs realism-selected comparator | −0.014 (−0.021 to −0.011) | −0.012 (−0.020 to −0.006) |
| TSTR AUROC vs utility-selected comparator | +0.010 (0.009 to 0.012) | +0.007 (0.003 to 0.009), blood counts only |

At n = 2,000, TabDiff is more realistic and utility is equal.

### 6.2 Relation diagnostics (paper Table IX)

Means over the two datasets of each panel and n = 250 and 1,000; lower is better.

| Arm | Relation distance (blood / chem) | Coupling error (blood / chem) | Dependence error (blood / chem) |
|---|---|---|---|
| MedHyperDiff | **0.127** / 0.037 | 0.068 / **0.023** | 0.042 / 0.030 |
| TabDiff + reparameterization | 0.168 / 0.051 | 0.069 / 0.025 | 0.043 / **0.028** |
| Copula, joint residuals | 0.152 / 0.046 | **0.061** / 0.025 | **0.029** / 0.030 |
| TabDiff, joint residuals | 0.155 / 0.047 | 0.062 / 0.024 | 0.031 / 0.030 |
| Copula, independent residuals | 0.151 / 0.046 | 0.196 / 0.024 | 0.030 / 0.029 |
| TabDiff, textbook re-derivation | 1.006 / **0.025** | 0.197 / **0.023** | 0.030 / 0.028 |
| TabDiff, rule-direction re-derivation | 0.867 / 0.327 | 0.202 / 0.024 | 0.031 / 0.029 |
| TabDiff, as generated | 10.56 / 4434 | 0.132 / 0.041 | 0.058 / 0.058 |

**Other findings.**
- Textbook re-derivation of the blood count indices raised TabDiff's C2ST to 0.892 (n = 250) and 0.738 (n = 1,000), against 0.754 and 0.638 with our direction.
- It also left a negative basophil fraction in 17% of rows. This is experimental support for the reconstruction-direction rule.

### 6.3 First test, 2017–18 blood counts (paper Table VII)

| n | Measure | Ours | TabDiff | Best other arm | Wins |
|---|---|---:|---:|---:|---:|
| 250 | Relation distance | **0.187** | 12.14 | 0.199 (TVAE, joint residuals) | 3/5 |
| | C2ST | **0.691** | 0.762 | 0.731 (copula, rule-direction re-derivation) | 5/5 |
| | TSTR AUROC | **0.717** | 0.670 | 0.687 (copula, independent residuals) | 5/5 |
| | Membership | 0.555 | 0.619 | 0.528 | |
| 1,000 | Relation distance | **0.173** | 3.92 | 0.176 (TVAE, joint residuals) | 2/5 |
| | C2ST | 0.664 | 0.645 | **0.633** (TabDiff, rule-direction re-derivation) | 0/5 |
| | TSTR AUROC | 0.735 | 0.735 | **0.742** (TabDiff + reparameterization) | 1/5 |
| | Membership | 0.509 | 0.560 | 0.561 | |

With the training budget scaled to sample size (iteration 14), the n = 1,000 result reverses: C2ST 0.627 against 0.633.

### 6.4 Computation (paper Table X)

| Model | Parameters | Updates (n = 250 / 1k / 2k) | Passes over fit rows | Median fit minutes |
|---|---:|---|---:|---|
| MedHyperDiff | 39,925 | 5k / 20k / 40k | 1,600 | 5 / 20 / 42 |
| TabDiff | 10.8 M | 8k | 8,000 | 36 / 95 / 157 |
| TabDDPM | 0.66 M | 30k | 30,000 | 7 / 18 / 34 |

Tuning is not matched: our training budget was chosen on development data, while the comparators used fixed configurations.

---

## 7. Changes to the manuscript, section by section

| Section | Change |
|---|---|
| Title | Now "Evaluating and Preserving Laboratory Relationships in Synthetic Clinical Tables" |
| Abstract | Rewritten around the three relation regimes and corrected results; 247 words (364 in version 2) |
| Introduction | Contributions restated as a methods paper; comparison with simple controls made explicit |
| Related work | Added HFGF (arXiv:2507.19211) and Umesh et al. 2024 (arXiv:2409.17684); states that our re-derivation controls are not reimplementations of C-DGM or Harpoon; full Harpoon citation |
| Pitfalls (III) | Pitfall 1 now "under-specified training budget"; evaluation-size thresholds labelled as empirical heuristics |
| Case study (IV) | Ablation row relabelled "Corrected, original training loop"; "pre-registered" changed to "specified in advance" |
| Controlled study (V, VI) | "Common practice" changed to "uncorrected pipeline", stating that no prevalence is claimed; nominal n vs fit rows; privacy-aware reversals (20–36%) added; training section rewritten with the decomposition table (Table V); "memorized" changed to "exceeded the membership threshold" |
| Method (VII, new) | Equations: residual encoding (1), decoding (2), reconstruction-direction objective (3) with its first-order justification, BMI coefficient and combinatorial cost; exact-relation rule; relation metric (4); new configuration table (Table VI) |
| First test (VIII) | New controls added; Table VII and Fig. 7 regenerated |
| Robustness (IX) | Panel-by-cycle wording; post hoc controls; privacy rule for every method; Tables VIII and IX generated from code; Fig. 8 shows both re-derivation directions and joint residuals; 598 / 10 / 20 / 560 counts; three regimes |
| Protocol (X) | Recompute exact formulas; sample approximate residuals jointly; compare with simple controls that use the same knowledge; same privacy threshold for all |
| Discussion (XI) | Three regimes; architecture contribution not isolated; next tests |
| Limitations (XII) | Post hoc controls, untuned comparators, NHANES only, no intended clinical use evaluated, local protocols |
| Conclusion (XIII) | Rewritten to match the evidence |
| End matter | Data and Code Availability updated; new Ethics Statement |
| Figures | All re-rendered at print size (7 pt serif); first-test and robustness figures include the new controls |

---

### 7.1 Version 5 (15 October): section structure and figures

**Sections.** The manuscript now has only these numbered sections: I Introduction, II Literature Review, III Methodology, IV Results and Discussion, and V Conclusion. The abstract, index terms and references are kept. Subsections are unnumbered.

| Former section | Now in |
|---|---|
| Related Work | II Literature Review (tabular generators; dependency- and constraint-aware generation; evaluation) |
| Method, data, model configuration, comparators, pitfalls, metrics, design | III Methodology |
| Case study, controlled study, first test, robustness, discussion, limitations, corrected protocol | IV Results and Discussion |
| Data and Code Availability, Ethics Statement | One paragraph in III Methodology, under Datasets |
| Acknowledgment | Removed |

Two references were added for the UCI cohorts and the thyroid data. A check script confirmed that no reported number was lost. One sentence on rule scoring was also made more precise: 11% to 22% of pairs among all variant and post-processing combinations, and 24% to 39% among the eight generator identities of Fig. 3. The script that performed the restructuring is archived as `reproducibility/final_scripts/restructure_sections.py`.

**Figures.** All eight figures were redrawn by `scripts/figures_paper.py` in the figures4papers house style (github.com/ChenLiu-1996/figures4papers):
- Helvetica, no top or right spines, frameless legends and no grid;
- one semantic palette: blue for the proposed method, reds for published generators as generated, greens for residual-based controls, neutral for references;
- drawn at twice the print width and embedded as vector PDF.

The pipeline diagram (Fig. 1) was redesigned as five stages with a panel showing the encode and decode equations and a blood count example. Its inputs are in `reproducibility/figure_inputs/`:

```
python scripts/figures_paper.py reproducibility/figure_inputs paper/ieee/figures
```

## 8. Code changes

| File | Change |
|---|---|
| `src/medhyperdiff/relations.py` | Linear relations (osmolality); `relations_for`; reported-exact detection and re-derivation; BMI log-scale coefficient fixed (height = 2), with saved choices unchanged in all 778 runs |
| `src/medhyperdiff/constraints.py` | Linear rule kind; chemistry relation rules |
| `src/medhyperdiff/evaluation.py` | `clinical_categories`, `clinical_category_tv` |
| `src/medhyperdiff/diffusion.py` | Decomposition options for the fixed-budget loop (`checkpoint`, `validation_draws`, `selection_patience`, `check_every`, `lr_schedule`, `use_ema`); defaults are byte-identical to before |
| `src/medhyperdiff/data.py` | NHANES cycle builders (2011–12, 2013–14, 2015–16, 2021–23) with pinned SHA-256 |
| `scripts/iteration14.py`, `report_iteration14.py` | Robustness study |
| `scripts/iteration15.py` | Training decomposition |
| `scripts/reanalysis_iteration14.py` | Per-dataset effects, diagnostics, textbook control |
| `scripts/controls_iteration14.py` | Textbook and joint-resampling controls, full metrics |
| `scripts/report_final.py` | Comparator registry, symmetric eligibility, LaTeX table rows |
| `scripts/figure_final.py`, `figure_first_test.py`, `figures_print.py` | Print-size figures |
| `scripts/fetch_data.py` | Rebuilds every dataset from the public sources |
| `tests/test_iteration14_methods.py` | Tests for linear relations, exact re-derivation, clinical categories, size scaling, the training-decomposition trajectory and the BMI coefficient; **64 tests pass** |

---

## 9. Reproducibility

| Study | Source snapshot (`reproducibility/source_snapshots/`) |
|---|---|
| Case study / iteration 11 | `9739e858…`, `721c3bbb…` |
| Controlled study (iteration 12) | `3aafec1c…` |
| First relation test (iteration 13) | `36533bf5…` |
| Robustness study (iteration 14) | `890ebbee…` |
| Training decomposition (iteration 15) | `1003bc8c…` |
| Post hoc controls and final comparison | `1626362e…` |

**Release bundle.** [release/medhyperdiff-paper-release.zip](release/medhyperdiff-paper-release.zip).
- **Size:** 8.9 MB.
- **SHA-256:** `11ecefc8703d478d129d86189a8d968ff98f4ebde2824aff38e5544b9fc30ac7`.
- **Contents:** code, tests, configs, protocols, snapshots, result tables, the paper, and a `RELEASE_README.md` with setup and commands.
- **Not included:** run records (`runs/`, about 1 GB).

**Clean-copy check.**
- `scripts/fetch_data.py` rebuilt all 13 processed datasets **byte-identically** from NHANES and UCI.
- All 64 tests pass with the pinned TabDiff and TabDDPM sources.

**Environment note.** The project folder is in iCloud-synced Desktop with storage optimization, so files can be offloaded ("dataless"). Run code from a copy outside Desktop.

---

## 10. What kind of paper this is, and where to send it

| Type | Is it this? |
|---|---|
| Methods paper (evaluation methodology plus a relation-handling technique, tested empirically) | **Yes, the main identity** |
| Model or architecture paper ("our network is best") | No |
| Benchmark study | Partly, as supporting evidence |

**Suggested venues:**
- Journal of Biomedical Informatics (methodological research article);
- BMC Medical Research Methodology;
- PLOS Digital Health.

At 13 pages it is too long for most IEEE conferences (6–8 pages); a journal will need its own template.

---

## 11. What is still open

### Required from the authors

Also listed in [paper/ieee/SUBMISSION_CHECKLIST.md](paper/ieee/SUBMISSION_CHECKLIST.md).
- Upload the release bundle publicly (GitHub or Zenodo), add the link to "Data and Code Availability", and choose a code license.
- Funding, conflict-of-interest and author-contribution statements.
- Confirm VIT's ethics requirements for public de-identified data.
- Co-author and guide approval of the final claims.
- Reformat for the chosen venue.

### Optional experiments that journal reviewers will likely ask for (about 1–2 days of computing)

- A matched ablation of the final model with and without the graph and hypergraph.
- Tuned TabDiff and residual baselines on untouched NHANES cycles (2009–10, 2007–08) with participant-consistent splits.
- Stronger membership attacks.
- A downstream task tied to an intended clinical use.
- An external hospital cohort, which needs data access.

---

## 12. File map

| What | Where |
|---|---|
| Manuscript (PDF, LaTeX, figures, generated tables) | `paper/ieee/` |
| Earlier versions | `paper/ieee/archive/` |
| Submission checklist | `paper/ieee/SUBMISSION_CHECKLIST.md` |
| Reviews and responses | `reports/publication_review_2026-10-12/`, `reports/publication_review_2026-10-13/` |
| Final comparison tables | `reports/final/` |
| Robustness study and reanalysis | `reports/iteration14/` (`INTERPRETATION.md`, `REANALYSIS.md`, `reanalysis/`) |
| Training decomposition | `reports/iteration15/` |
| Protocols and runbooks | `docs/ITERATION_14_PROTOCOL.md`, `docs/ITERATION_14_RUNBOOK.md`, `docs/ITERATION_15_PROTOCOL.md` |
| Run records | `runs/iteration14/`, `runs/iteration15/`, `runs/iteration14_controls/` |
| Source snapshots and archived scripts | `reproducibility/` |
| Release bundle | `release/medhyperdiff-paper-release.zip` |
