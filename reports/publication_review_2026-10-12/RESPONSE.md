# Response to the publication review of 12 October 2026

Every factual claim in [REVIEW.md](REVIEW.md) that could be checked against the code, splits and saved runs was re-verified, and all held.
- **Participant overlap:** 4,753, 5,019 and 5,263 shared participants within cycles, and none across cycles.
- **Fit counts:** 598 confirmation fits, 10 secondary and 20 at n = 100.
- **Membership:** 0.608 on chemistry 2011–12 at n = 250, and 0.703–0.759 at n = 100.
- **Category distance on CBC 2021–23 at n = 2,000:** 0.047 against 0.020.
- **Replicate overlap:** 28–32% of fit rows at n = 1,000 and 59–61% at n = 2,000.
- **Legacy training confound:** confirmed.

The revised manuscript is [paper/ieee/main.pdf](../../paper/ieee/main.pdf) (12 pages); the earlier version is kept in `paper/ieee/archive/`.

## What was done, by priority

### 1. Independence

**Changes.**
- The manuscript, README and iteration 14 interpretation now describe "five NHANES panel-by-cycle datasets from four survey cycles".
- Within-cycle participant overlap is disclosed (5,263, 5,019 and 4,753).
- "Fresh population" was removed for the 2017–18 CBC, which shares 4,753 participants with the iteration 12 biochemistry population.

**Remaining.** Participant-consistent splits across panels for future studies (Phase 2 below).

### 2. Uncertainty and noninferiority

**Changes.**
- Paired effects are reported for every dataset × size with win counts (Table VII, `reports/iteration14/reanalysis/per_dataset_effects.csv`).
- Leave-one-cycle-out pooled estimates were added (`leave_one_cycle_out.csv`).
- Pooled intervals were dropped from the paper. The text states that the averages describe repeated sampling from fixed datasets.
- The prespecified criteria are called "descriptive criteria rather than significance tests".

**Remaining.** A design with more independent cycles.

### 3. Privacy wording

**Changes.**
- "Did not copy patients" and "without memorization" were removed everywhere in the paper.
- The 0.608 exception and the n = 100 results (0.70–0.76) are reported in the Clinical Categories and Membership subsection.
- The paper states the attacks are diagnostics only.

**Remaining.** Stronger attacks (low-FPR, shadow models, attribute inference).

### 4. Fair baselines

**Changes.**
- A computation table was added: parameters, updates, passes and fit time.
- MedHyperDiff has 40 thousand parameters against 10.8 M for TabDiff, and makes 1,600 passes against TabDiff's 8,000.
- The paper states that tuning is not matched.
- The text now says the deterministic re-derivation controls are not implementations of C-DGM or Harpoon.

**Remaining.** Tuned TabDiff and copula/residual baselines (Phase 2).

### 5. Training confound

**Changes.**
- New iteration 15 decomposition: 160 fits, protocol written before running.
- The update budget explains the whole legacy gap (20 of 20 samples).
- Validation-based selection has no consistent effect.
- A constant learning rate without averaging raises membership in 20 of 20 samples.
- Pitfall 1, the Discussion sentence about "no change in learned distributions" and the ablation label were corrected.
- The fixed validation corruption is described exactly.

**Remaining.** Nothing.

### 6. Novelty and the hypergraph

**Changes.**
- HFGF (arXiv:2507.19211) and Umesh et al. 2024 (arXiv:2409.17684) were added, with a precise distinction: learned residuals, data-driven reconstruction direction, recording resolution.
- Equations were added for the encoder, decoder, exact-relation rule, direction objective with its first-order justification and combinatorial cost, training budget and metric.
- **Experimental support for the direction choice.** Textbook-direction re-derivation raises TabDiff's C2ST:

| CBC 2013–14 | Textbook re-derivation | Rule-direction re-derivation |
|---|---:|---:|
| n = 250 | 0.892 | 0.754 |
| n = 1,000 | 0.738 | 0.638 |

**Remaining.** A matched ablation of the graph and hypergraph on untouched data (Phase 2).

### 7. Clinical meaning of relation fidelity

**Changes.**
- Diagnostics were added: signed distance, coupling between relations, dependence on participants, age and label, tails, coverage and invalid values (Table VIII).
- **Distinctive property found.** Residual coding preserves the coupling between relations that share participants (error 0.068 against about 0.20 for resampling and re-derivation).
- **Weakness found.** Slightly worse residual dependence on covariates at n = 250.
- **Textbook-direction re-derivation control added.** For exact chemistry relations it is as faithful as our model, so H1x had passed only against a weak control. This is stated.
- Clinical categories are reported (better in 4 of 10 settings).

**Remaining.**
- More downstream tasks and models.
- Calibration.
- An external clinical cohort, which needs data access.

### 8. Reporting details

**Changes.**
- **Sample sizes:** nominal n vs fit rows, and synthetic size = 5 × fit rows.
- **Ranking definitions:** fidelity vs privacy-aware rankings (20–36%) distinguished.
- **Comparators:** realism-selected vs utility-selected comparators distinguished.
- **Counts:** 598 / 10 / 20 fits.
- **Size claims:** "at the tested sizes" language.
- **Heuristics:** the 500-record and 0.02 thresholds are presented as empirical for this setting.
- **Protocols:** "prospectively specified" and "dated and hashed locally rather than registered publicly".
- **Text:** abstract shortened to 249 words, and the title shortened.
- **Figures and references:** all figures re-rendered at print size (7 pt), and the Harpoon reference completed.
- **Archived source:** the iteration 14 source was archived (`reproducibility/source_snapshots/890ebbee…`, hash verified).
- **Tests:** 63 tests pass in a fresh environment, including a new test that validation-based selection leaves the training trajectory unchanged.

**Remaining.** A public versioned release with an archive link.

## Phase 2: the focused confirmation the review recommends (not yet run)

The review's step 2 and step 3 recommendations need new data and new compute.
- **Data:** confirmation on untouched NHANES cycles (2009–10 and 2007–08 have CBC and chemistry files) with participant-consistent splits across panels.
- **Comparators and ablations:**
  - a validation-only search for TabDiff;
  - copula with residual resampling and with reparameterization;
  - a residual-coded backbone without graph or hypergraph, with graph only and with hypergraph only;
  - the full model;
  - two sample sizes.
- **Stronger membership attacks.**
- **Estimated compute:** one to two days on the study laptop, dominated by TabDiff tuning.
- **External cohort:** this requires a data-access agreement and a clinical collaborator, and cannot be done from public files.
