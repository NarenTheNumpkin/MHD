# Response to the publication review of 13 October 2026

The revised manuscript is [paper/ieee/main.pdf](../../paper/ieee/main.pdf) (13 pages). The 12 October version is in `paper/ieee/archive/`.

Every quantitative claim in the review was re-verified against saved results before any change:
- **Copula with textbook re-derivation, chemistry 2021–23, n = 250:** C2ST 0.7602 against 0.7750, winning 4 of 5 samples, membership 0.5621.
- **MedHyperDiff membership, chemistry 2011–12, n = 250:** 0.60846.
- **Coupling error:** 0.067691 against 0.069310.
- **BMI coefficient:** the code used 1 where Eq. (3) implies 2.
- **Unarchived source:** hash `1003bc8c`.

No generator was retrained for this revision. All new numbers come from saved outputs.

## 1. Canonical control missing from the headline comparison (blocker)

**Done.** `scripts/controls_iteration14.py` evaluates textbook re-derivation for every published-generator output of iterations 13 and 14 with the full metric set, including utility and membership. This covers 280 outputs and 560 evaluations together with item 4.

`scripts/report_final.py` merges these into one comparator registry and regenerates:
- Table VII (first test) and Table VIII (per-dataset confirmation);
- the paired and leave-one-cycle-out summaries;
- the clinical-category comparisons;
- Table IX (diagnostics);
- the prose.

Figure 8 now shows textbook and rule-direction re-derivation separately, plus joint residual resampling. The headline claim is corrected: at n = 250 the model is the most realistic eligible generator on the two blood count datasets, not on all four.

## 2. Membership criterion applied to MedHyperDiff

**Done.** Every arm, including the model, is eligible only when its mean membership AUROC is at most 0.60. Table VIII marks the model's ineligible cell with a dagger and bolds the best eligible value.

Under this rule the model is the most realistic eligible generator in 5 of 10 settings:
- both blood count datasets at n = 250;
- blood count 2021–23 at n = 1,000, narrowly;
- both chemistry datasets at n = 1,000.

All five are reported in the abstract, results, discussion and conclusion.

## 3. Representation vs hypergraph

**Stated, not yet tested.** The paper now says the evidence supports the residual representation and joint sampling rather than the architecture. Joint residual resampling and TabDiff with reparameterization reach similar coupling. The matched final-model ablation is listed as the most direct next test.

## 4. Joint residual-resampling control

**Done.** Row-wise joint resampling assigns each synthetic row the residual vector of one fit row with all residuals observed. A residual stays missing where the synthetic row lacks its participants. The decoded tables were evaluated.

| Blood count measure | Joint resampling (copula / TabDiff) | MedHyperDiff |
|---|---|---|
| Coupling error | 0.061 / 0.062 | 0.068 |
| Dependence error | 0.029 / 0.031 | 0.042 |
| Relation distance | 0.152 / 0.155 | 0.127 (closest in all five blood count settings, by 0.008–0.019) |

Joint resampling preserves the coupling at least as well as the model. The "distinctive coupling" claim was withdrawn and replaced by the three-regime conclusion (exact, approximate, coupled).

## 5. Clinical usefulness and generalization

**Stated as a limitation.** No intended clinical use was evaluated directly, and an external cohort is needed for claims beyond NHANES. The category comparison now uses the refreshed comparator pool: lower in 4 of 10 settings.

## Smaller corrections

| Item | Change |
|---|---|
| Conclusion attributed approximate relations to four cycles | Now "three survey cycles" for blood counts; the Design section states three CBC cycles and two chemistry cycles |
| Eq. (3) vs implementation for BMI | `Relation.weight` returns 2 for height, and the text states the coefficient. Reconstruction choices are unchanged in all 778 saved runs, and a new test was added. |
| "Matched" for exact chemistry | Replaced by the direct comparison: the strongest control was closer in every chemistry setting |
| "Memorized in 10 of 20" | Now "exceeded the membership threshold in 10 of 20" |
| "Common versus corrected practice" | Now "uncorrected versus corrected pipelines", with an explicit statement that no prevalence is claimed |
| Archive iteration 15 source and scripts | `reproducibility/source_snapshots/1003bc8c…` and `iteration15_scripts/`. The final analysis source `1626362e…` and `final_scripts/` are also archived. |
| Model specification | New Table VI with the complete final configuration |
| Repository identifier | A release bundle is prepared in `release/medhyperdiff-paper-release.zip`. A public URL must be added by the authors. |
| Two stories | Retitled "Evaluating and Preserving Laboratory Relationships in Synthetic Clinical Tables". The abstract and conclusion are organized around the evaluation lessons and the three relation regimes, with the generator as supporting evidence. |

## Verification

The suite runs in a clean copy of the release:
- `scripts/fetch_data.py` rebuilds all 13 processed datasets byte-identically from the public sources;
- all 64 tests pass with both pinned baselines.

## Not done (requires new experiments or author input)

- Matched ablation of the final model with and without graph and hypergraph.
- Tuned TabDiff and other baselines on untouched data with participant-consistent splits.
- Stronger membership attacks.
- External cohort.
- Repository URL, license, funding, conflict-of-interest and author-contribution statements.
