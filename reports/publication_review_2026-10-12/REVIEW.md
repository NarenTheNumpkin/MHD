**Publication assessment of MedHyperDiff — 12 October 2026**

**Judgment: there is a publishable research contribution here, but the present draft needs major scientific and reporting revisions before an archival submission.** The best-supported contribution is careful evaluation of recorded laboratory tables, coupled with a method for preserving supplied approximate relationships. The current evidence does not establish general superiority for medical data, privacy protection, or a necessary benefit from the hypergraph architecture.

I reviewed all ten pages of the [draft](/Users/naren/Desktop/MedHyperDiff/paper/ieee/main.pdf), including the rendered figures and tables, and inspected the current generation, preprocessing, evaluation, experiment and reporting code. I traced the latest confirmation results back to saved run metrics and reconstructed cohort membership from the cached NHANES source files. I also checked relevant primary literature. This is a research assessment and an artifact audit, not a reproduction of model training.

**What I verified.** C2ST and relation-distance entries in the latest confirmation CSV agree with saved JSON metrics for all **598 analyzed confirmation fits**, including their recorded post-processing variants. The current package source hash matches the source hash in these runs. Reconstructing NHANES cohort eligibility reproduced the processed datasets' row counts and laboratory values, allowing participant overlap to be checked without exposing individual records. There were no fit/reference index intersections in the ten audited iteration-14 dataset/size cells. I parsed 66 locally available Python source files; 15 older scripts were marked dataless and were skipped. The repository declares 62 tests, but a fresh pytest run was interrupted before any tests ran because local PyTorch source files were offloaded/dataless. Historical test-pass claims were not independently re-established.

The aggregate audit evidence and a script to recompute it are saved beside this report: [audit_evidence.json](/Users/naren/Desktop/MedHyperDiff/reports/publication_review_2026-10-12/audit_evidence.json), [reproduce_audit.py](/Users/naren/Desktop/MedHyperDiff/reports/publication_review_2026-10-12/reproduce_audit.py). The script reads local artifacts and needs NumPy and pandas; it does not fit models or change experimental results.

**What is good about the project**

- **The research question matters.** A table can have plausible individual columns while breaking relationships between those columns. Distinguishing exact definitions from approximate relationships is useful, especially when reporting precision creates nonzero residuals in real records.
- **The evaluation work contains concrete findings.** Table II's 25–34% ordering reversals and 18/20 changes in the top-ranked generator agree with the saved fidelity-ranking summary. Table III's large change after resolution-aware decoding is a useful demonstration that representation details affect benchmark conclusions. These results support sensitivity to the tested pipelines, subject to the causal-attribution caveat below.
- **The approximate-relation result is substantial.** For CBC 2013–2014 at nominal n=1,000, the reported relation distance is about 0.092 for MedHyperDiff versus 0.741 for the best hard-derivation comparator. The result is consistent across additional panels and sample sizes. This is more distinctive than simply making generated rows satisfy an exact formula.
- **There are meaningful controls and failures.** The project includes statistical baselines, a bootstrap copy control, hard derivation, resampled residuals and reparameterized competitors. It reports the failed universal plug-in hypothesis and TabDiff's stronger realism at n=2,000. Retain those negative results.
- **The implementation shows research discipline.** Fit-only transforms and relation selection, grouped splits, declared development choices, stored metrics, source hashes and archived iterations provide a strong foundation. The latest source hash reconciles with the confirmation manifests. These practices are valuable even though they do not by themselves establish statistical validity or public preregistration.

**The strongest results and their limits**

| Aspect | Evidence in this project | What it supports |
|---|---|---|
| Evaluation sensitivity | 25–34% pair reversals in the specified common/corrected fidelity pipelines | Pipeline choices can materially alter conclusions on these datasets |
| Approximate relations | CBC advantages against raw generation and hard derivation across tested settings | Strong reproduction of the evaluated supplied residual distributions |
| Small-sample realism | Pooled C2ST difference about −0.0196 at nominal n=250 and −0.0206 at n=1,000 | An advantage over the tested comparator configurations; uncertainty needs reanalysis |
| Larger-sample realism | At n=2,000, C2ST 0.616 vs 0.577 on CBC and 0.645 vs 0.625 on chemistry | TabDiff is more realistic under this discriminator at those settings |
| Utility | Positive pooled differences, but comparator selection is based on C2ST; some cells tie or lose against the utility-best arm | Useful predictive performance for the tested diabetes task, not universal utility superiority |
| Privacy | Two empirical attacks, with a 0.60 selection threshold; one n=250 cell reaches 0.608 | Limited attack diagnostics, not proof of no memorization or safe disclosure |
| Clinical relevance | Clinical-category distance is available but not consistently best | Relation fidelity has not yet demonstrated consistent improvement in clinically meaningful decisions |

**Priority 1 — Correct the independence and “fresh population” claims.**

The patient identifiers in the raw files show extensive overlap:

| Datasets compared | Rows in the two datasets | Shared participants |
|---|---:|---:|
| 2017–2018 chemistry from iteration 12 and CBC from iteration 13 | 4,756 / 4,865 | **4,753** |
| 2015–2016 development CBC and chemistry | 5,078 / 5,028 | **5,019** |
| 2021–2023 confirmation CBC and chemistry | 5,555 / 5,284 | **5,263** |

These are different feature panels with useful replication across survey cycles; they are not all independent patient populations. The 2017–2018 CBC panel contains new laboratory features but largely the same people as the earlier study. This does **not** establish within-run train/test leakage: the audited fit and reference indices were disjoint. It does mean that “five independent populations” and a completely fresh patient population overstate independence.

Change the abstract, Introduction, Sections VII–VIII, Discussion and Conclusion to describe **NHANES panel-by-cycle datasets**, disclose participant overlap and retain SEQN-based provenance for cross-panel splits. For future studies, assign a participant's split consistently across all panels and use an external source or an untouched cycle for confirmation. Relevant builders: [data.py](/Users/naren/Desktop/MedHyperDiff/src/medhyperdiff/data.py:201), with CBC at line 240 and further cycles at line 306. The exact aggregate overlaps are in the accompanying audit.

**Priority 2 — Rework uncertainty and noninferiority claims.**

[The pooled analysis](/Users/naren/Desktop/MedHyperDiff/scripts/report_iteration14.py:127) bootstraps individual population-by-replicate differences as exchangeable observations. It does not resample survey cycles or participants, model cross-panel participant overlap, or include uncertainty in selecting the best comparator. The n=1,000 fit samples share about 25–34% of their rows; at n=2,000 the overlap is about 56–61%. Sharing a reference set further limits the variation these intervals represent. Such results can describe repeated draws conditional on the fixed datasets, but should not be presented as uncertainty about performance across independent clinical populations.

Report paired effects separately for every dataset and specify the estimand. Add leave-one-cycle-out sensitivity and uncertainty that respects the participant/cycle structure; with so few independent cycles, avoid treating a cluster bootstrap as a complete solution. Additional genuinely independent data are more valuable than more seeds from the same finite pool. Where feasible, repeat the full splitting, fitting and evaluation procedure, preserving participant groups.

The [hypothesis implementation](/Users/naren/Desktop/MedHyperDiff/scripts/report_iteration14.py:62) checks win counts or point-estimate margins. For example, chemistry checks whether the mean relation distance is within 0.10, and H4 checks whether mean C2ST is within 0.02. These are predeclared descriptive success criteria, not confidence-bound noninferiority tests. Either call them that or justify the margins and apply an appropriate paired uncertainty procedure. A 4/5-win rule alone is not a conventional significance test.

**Priority 3 — Remove claims of “no memorization,” “non-copying,” and general safety.**

The limitations correctly state that privacy is not guaranteed, but the abstract and conclusion still say the generator did not copy patients. Passing two attacks at an arbitrary AUROC threshold cannot establish that. The nearest-distance attack and the [DOMIAS-style attack](/Users/naren/Desktop/MedHyperDiff/src/medhyperdiff/evaluation.py:307) are useful diagnostics; the latter uses a specific eight-component PCA/KDE implementation, not a comprehensive attack suite. The original [DOMIAS paper](https://proceedings.mlr.press/v206/breugel23a.html) specifically warns that weak attacks can give a false impression of privacy.

Use language such as “below the prespecified threshold on the two evaluated membership attacks,” with the exception reported prominently. Chemistry 2011–2012 at n=250 has mean membership AUROC **0.60846**, above the project's threshold. Also, the retained n=100 MedHyperDiff-only runs have mean membership AUROCs about **0.703–0.759** across the four panels. They lack comparator runs, so no relative-performance conclusion is possible, but they are relevant descriptive evidence against extending the favorable claim to arbitrarily small samples. The recorded amendment is not evidence that they were removed because of their results; the correct response is transparent reporting.

Add calibrated low-FPR attack performance with uncertainty, rare-subgroup attacks, exact and near-copy diagnostics, and attribute/linkability attacks appropriate to the intended release. More powerful membership attacks or shadow-model attacks would strengthen a privacy-oriented submission. Formal differential privacy is optional if privacy guarantees are not claimed; it becomes a separate substantial methods requirement if a guarantee is the objective. Changing the wording is essential even without new attacks.

**Priority 4 — Make the latest baseline comparison fair enough to support superiority.**

Iteration 14 selects among MedHyperDiff training/width candidates on development datasets and increases its optimizer steps from 5,000 to 20,000 and 40,000 as nominal n grows. Comparator settings remain fixed from iteration 12. This is transparently acknowledged, but it weakens claims about the best available generators. It does not mean competitors necessarily received less computation: their batch sizes, parameter counts and epoch definitions differ substantially.

Give TabDiff, TabDDPM and the strongest copula/residual variants a declared validation-only search and comparable tuning opportunities. Report optimizer steps, effective data passes, parameter counts, training time and sampling time. Include both a tuning-budget comparison and performance-versus-compute curves rather than equating a step in one model with a step in another. See [iteration14.py](/Users/naren/Desktop/MedHyperDiff/scripts/iteration14.py:60), [iteration12.py](/Users/naren/Desktop/MedHyperDiff/scripts/iteration12.py:34), and [the TabDiff adaptation](/Users/naren/Desktop/MedHyperDiff/src/medhyperdiff/external_baselines.py:234).

Hard derivation is a useful knowledge-aware baseline, but it is not an implementation of the cited constrained methods. [C-DGM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/887932131fddf943e8fe3310b62c0147-Abstract-Conference.html) integrates a constraint layer and studies training-time effects. [Harpoon](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e785d64baa2cde06c69d9b7a14636ef2-Abstract-Conference.html) uses inference-time manifold guidance. Run an applicable actual constrained method, adapting only constraints its formulation supports, or explicitly limit claims to the implemented post-processing controls. Do not claim those papers were experimentally defeated by substituting deterministic re-derivation for their algorithms.

**Priority 5 — Separate checkpoint selection from training-budget and optimizer changes.**

The “early stopping” comparison changes several things together. The early-stopping model has a 100-epoch cap, while the corrected model takes 5,000 optimizer steps and uses learning-rate decay and EMA. At 200 fit rows and batch size 64, even 100 full legacy epochs produce only 400 updates; at 800 rows, the cap is 1,300 updates. Thus the experiment changes training duration substantially as well as checkpoint selection.

Moreover, [the legacy loop](/Users/naren/Desktop/MedHyperDiff/src/medhyperdiff/diffusion.py:237) reuses fixed validation corruption. The empirical validation objective can still be a poor finite-sample proxy, but fresh noise resampling each epoch is not the implemented mechanism. Describe exactly what was tested.

Add a controlled comparison holding total updates, learning-rate schedule and EMA fixed: final checkpoint versus validation-selected checkpoint, followed by separate budget and corruption-averaging experiments. Until then, the supported claim is that the tested **training recipe bundle** improves results. It does not establish that validation-based selection itself should generally be abandoned. In Section X, remove the statement that all seven pitfalls alter rankings without changing learned distributions: the training corrections plainly change what is learned.

**Priority 6 — Establish the incremental novelty and the role of the hypergraph.**

The strongest plausible methodological novelty is the combination of learned residuals, fit-only selection of a numerically suitable reconstruction direction, and preservation of approximate relationships at recorded precision. A general “generate independent variables, reconstruct dependent variables” claim overlaps directly with [Dependency-aware synthetic tabular data generation / HFGF](https://arxiv.org/abs/2507.19211), published as a preprint in July 2025. The earlier [dependency-preservation study](https://arxiv.org/abs/2409.17684) is also relevant. Both are missing from the draft's related work. Their existence does not negate your residual-based contribution; it makes a precise comparison necessary.

Add equations for the encoding and decoder, the exact-relation rule, the reconstruction-direction objective, the diffusion objective and the reported relation metric. Give pseudocode for choosing an acyclic reconstruction, define its assumptions and discuss its combinatorial scaling. [choose_derivations](/Users/naren/Desktop/MedHyperDiff/src/medhyperdiff/relations.py:126) enumerates candidate target assignments; it is a heuristic based on spread, not a demonstrated guarantee of optimal numerical conditioning.

The current backbone-only comparison is part of iteration-13 development. It does not isolate the graph and hypergraph contributions on the latest confirmation data. Use the same corrected training and residual coding in a matched matrix: backbone only; graph only; hypergraph only; full structure; shuffled or dense matched-capacity structure. Compare these with the same residual representation added to the strongest alternatives. Earlier ablations of an older training recipe do not establish why the final system wins.

**Priority 7 — Show that better relation distance means more useful medical data.**

The primary metric compares each relation's **absolute** residual distribution. This discards residual sign and does not establish the joint or conditional distribution of residuals given age, disease status or laboratory operands. A simple residual resampler can therefore approach this metric without preserving those dependencies. Indeed, the draft reports that resampled-residual comparators slightly outperform MedHyperDiff on chemistry 2011–2012 relation distance. The huge raw-to-corrected ratios for exact relations also partly reflect normalization by a very small residual scale; they should not be interpreted as equivalently huge gains in clinical usefulness.

Add signed and conditional residual checks, joint residual dependence, tail behavior, relation coverage and numerical-invalidity rates. The [evaluation helper](/Users/naren/Desktop/MedHyperDiff/src/medhyperdiff/evaluation.py:206) excludes missing/nonfinite residuals, so coverage must accompany the score. Assess the encoder/decoder's handling of partially missing operands and targets: an observed target with a missing operand can become an unavailable residual, making missingness modeling a substantive issue.

Use several downstream tasks and models, class-sensitive metrics, calibration, subgroup performance and real-only versus real-plus-synthetic comparisons at controlled sample sizes. One random-forest AUROC for diagnosed diabetes is a limited utility demonstration. Show the existing clinical-category endpoint in the manuscript: MedHyperDiff is not consistently best on it. For example, at n=2,000 in CBC 2021–2023 its category TV distance is about 0.047 versus about 0.020 for an eligible comparator, despite its much better relation distance. This is a valuable boundary of the method, not a result to hide.

Evaluate a genuinely external hospital or clinical cohort with justified relations and an appropriate data-access process. Multiple NHANES panels and cycles provide useful robustness checks but do not demonstrate portability to hospital EHRs, longitudinal records, rare diseases, pediatric cohorts, different laboratories or different missingness mechanisms. A clinical or laboratory collaborator can help distinguish analyzer/reporting identities from medically important abnormal cases. Survey weights need not be mandatory for a clearly scoped algorithmic convenience-sample benchmark, but population-representative conclusions require addressing NHANES's survey design.

**Priority 8 — Fix exact reporting and reproducibility details.**

| Location | Issue | Concrete correction |
|---|---|---|
| Abstract, figures and Tables V–VI | Nominal “training n” includes a held-out 20% validation partition | State sampled n and actual fit n: 250→200, 1,000→800, 2,000→1,600 |
| Protocol's synthetic sample count | The code tiles fit labels five times | Report 5×fit rows, approximately 4×nominal n, rather than 5×nominal n |
| Table II and the common/corrected narrative | The quoted 25–34% numbers refer to the fidelity-only comparison | Distinguish it from privacy-aware rankings; their top method and reversal rates differ |
| Section VIII pooled utility claim | “Best published” is selected by C2ST before its utility is compared | Label it the realism-selected comparator, and separately compare against the utility-best eligible arm |
| Section VIII job count | The analyzed confirmation artifacts contain 598 unique fits | Distinguish 598 confirmation fits, 10 secondary fits and 20 excluded n=100 fits; 608 combines primary and secondary stages |
| “Up to 1,000” and a universal crossover | Only discrete sizes were tested; results differ by panel | Say “at the tested nominal sizes 250 and 1,000”; describe losses at 2,000 without asserting a universal threshold |
| Protocol recommendations | 500 evaluation rows and a 0.02 unresolved zone are treated as general rules | Present them as empirical heuristics for these discriminators/datasets, not universal sample-size or significance requirements |
| Preregistration and release statements | Local protocols exist, but the draft supplies no immutable public registration/release identifier | Provide verifiable dated protocol versions and amendments, or call them prospectively specified local protocols; add a real repository/archive link |
| Packaging | README contains older contradictory status statements; latest source snapshot is not separately archived | Publish a clean versioned release of the source that currently matches the runs, with dependency lock, data retrieval, figure/table generation and a compact smoke run |

The document is visually organized, but the abstract is approximately 364 whitespace-delimited words, the title is long and several figure labels are small at page size. Shorten the abstract to the chosen venue's requirement, enlarge the crowded plots, define abbreviations in tables and complete references, particularly the abbreviated Harpoon entry. As a methods submission, it needs more explicit mathematics and a self-contained specification of the final model. The current narrative has the feel of two substantial papers compressed together: a benchmark-sensitivity study and a generator study. Choose one central contribution and use the other as supporting evidence, unless a journal-length format provides room to develop both properly.

**A practical revision sequence**

1. **Before new training:** correct independence/privacy/sample-count wording; make all claims traceable to exact result cells; clarify the pooled comparator; add the missing literature and method equations; report clinical-category and unpaired n=100 diagnostics; regenerate a consistent set of tables. Label current pooled intervals and win-count criteria appropriately.
2. **For a stronger archival submission:** freeze a new study with common participant splits across panels, an untouched external source, comparable tuning opportunities for strong baselines, the final-method ablation matrix and better attacks. Select models using development data; do not turn already-inspected confirmation panels into a new unseen test set by renaming them.
3. **If computation is limited:** prioritize tuned TabDiff, copula plus residual resampling, copula with reparameterization, the residual-coded backbone and the full model on two justified sample sizes. These directly test whether the added complexity earns its advantage. Hundreds more untuned runs or seeds on the same panels would be less informative.
4. **Package after the scientific choices are stable:** archive exact code and protocols, provide a compact runnable example and a deterministic saved-results-to-tables workflow, and confirm tests once the offloaded local dependencies are available.

**Publication positioning**

My reviewer-style judgment is **major revision**, rather than a recommendation to abandon the project. A focused workshop or short empirical paper is plausible after the reporting and inference issues are repaired. An archival health-informatics methods paper is a credible target after the baseline, ablation and external-validation work. A top general-ML submission would currently face a difficult novelty and experimental-rigor case; many runs do not substitute for an isolated methodological contribution.

For scope, [Journal of Biomedical Informatics](https://amia.org/news-publications/journals/journal-biomedical-informatics) explicitly values methodological innovation and generalizable lessons. [PLOS Digital Health](https://journals.plos.org/digitalhealth/s/journal-information) includes synthetic-dataset generation and validation. [CHIL](https://chil.ahli.cc/submit/call-for-papers/) is relevant to health-focused ML research. These are topic-fit suggestions, not acceptance predictions or statements that a submission window is currently open.

The most defensible central claim is: **preserving the distribution of discrepancies in supplied laboratory relationships can improve synthetic recorded clinical tables, and evaluation details substantially affect how those improvements are measured.** Making that claim precise would produce a stronger paper than claiming a generally best, non-memorizing medical-data generator.
