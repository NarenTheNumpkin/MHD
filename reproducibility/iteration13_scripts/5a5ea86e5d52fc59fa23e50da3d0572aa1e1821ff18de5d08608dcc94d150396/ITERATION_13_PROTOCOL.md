# Iteration 13 protocol: relation-faithful generation

Written 7 October 2026 after building and auditing the NHANES complete-blood-count population (counts, missingness, relation residuals) and before fitting any generator to it.

## Question

Can MedHyperDiff reproduce supplied *approximate* clinical relations better than published tabular generators, with or without the constraint enforcement used in the literature, while remaining as realistic overall and not memorizing?

## Population and relations

NHANES 2017–2018 adults with a complete blood count (`build_nhanes_cbc`; 4,865 records; label diagnosed diabetes; 16 numerical predictors and sex). The analyser computes haematocrit = RBC × MCV / 10, MCH = 10 × haemoglobin / RBC and MCHC = 100 × haemoglobin / haematocrit, and the five white-cell differential percentages sum to about 100. All four relations hold only up to reporting rounding (real absolute residuals: medians 0.03–0.10; 95th percentiles 0.09–0.20), so forcing them exact and ignoring them are both wrong.

## Method under test

**Relation reparameterization** (`medhyperdiff.relations`): the generator models operands plus one residual column per relation (`derived − f(operands)`); after sampling, operands are decoded to recorded resolution, then `derived = f(operands) + residual` is reconstructed and decoded. In MedHyperDiff, each residual and its operands form a structure-only hyperedge. Candidates for the primary MedHyperDiff variant, chosen on development data only:

- `mhd_unguided`: corrected MedHyperDiff with relation energy in training, no guidance;
- `mhd_guided`: the same with guidance strength calibrated per fit on its validation rows (grid 0, 1, 2, 5, 10; minimum relation residual distance);
- `mhd_reparam`: relation reparameterization with structure hyperedges;
- `mhd_reparam_backbone`: reparameterization without graph or hyperedges.

**Choice rule (fixed now):** lowest mean development relation-residual distance among variants whose mean development C2ST is within 0.02 of the lowest; ties to the earlier variant in the list above.

## Splits

The population is split once (grouped by predictor profile, stratified, seed 2027) into a reference half and a pool. 30% of the pool forms a development pool; development samples (n = 250 and 500, three seeds) are evaluated on the development-pool rows they do not contain. Confirmation samples (n = 250 and 1,000, five replicates) are drawn from the remaining 70% and evaluated on the never-fitted reference half. All samples split 80/20 into fit and validation rows.

## Comparators (confirmation)

Published generators with fixed configurations as in iteration 12: TabDiff, TabDDPM (upstream recipe), CTGAN, TVAE and the Gaussian copula; the bootstrap copy control. For each, two knowledge-equivalent post-processings of the same raw output (no refitting): **hard derivation** (derived := f(decoded operands), the exact-constraint enforcement of constraint-layer methods) and **derivation plus resampled residuals** (independent draws from the fit residuals). The generic reparameterization is also applied to TabDiff (`tabdiff_reparam`) to separate the transform from the MedHyperDiff architecture. Corrected MedHyperDiff without the chosen mechanism (`mhd_unguided`) is the ablation.

## Endpoints

**Primary (relation fidelity):** mean over the four relations of the Wasserstein distance between synthetic and reference absolute residuals, divided by the fit residual SD. **Guardrails:** C2ST against the reference (banded decoding) and membership AUROC (maximum of nearest-distance and density-ratio attacks). **Secondary:** per-relation violation-rate gap, numeric Wasserstein, association RMSE, forest TSTR utility.

## Pre-registered claim and how it is judged

The claim "MedHyperDiff reproduces approximate clinical relations better than published generators" is supported if, at both training sizes, the primary MedHyperDiff variant (i) has a lower primary endpoint than every published generator in raw form and with hard derivation in at least four of five replicates, (ii) has mean C2ST no more than 0.02 above the best non-memorizing published generator (raw or post-processed), and (iii) has mean membership AUROC at most 0.60. Comparisons with derivation-plus-resampling and with reparameterized TabDiff are reported as secondary analyses that bound how much of any advantage comes from the transform rather than the architecture. All outcomes are reported, including failure of the claim.

Secondary populations (UCI thyroid FTI; NHANES biochemistry globulin and BMI) reuse iteration-12 samples and literature outputs; because their reference halves were already examined in iteration 12, they are supporting evidence only.

## Amendment 1 (7 October 2026, development stage; reference half untouched)

The first development run reconstructed the basophil percentage as 100 minus four generated differential percentages plus a residual. On development data this was ill-conditioned: errors of the four large components accumulated in a quantity whose real range is 0–2%, producing negative values (basophil KS 0.33 versus at most 0.09 for every other feature) and raising development C2ST to 0.84–0.88. The differential relation now reconstructs the largest-variance component (neutrophil percentage) from the other four, and the general rule is recorded: reconstruct the component best conditioned by its operands. Evaluation rules are unchanged (the differential residual |sum − 100| is symmetric in the components). All 24 development jobs are rerun with the corrected relation before the choice rule is applied; the first run is kept under `runs/iteration13/develop_v1` for the record.

## Amendment 2 (7 October 2026, development stage; reference half untouched)

The second development run (neutrophil percentage reconstructed) removed the basophil artefact but MCHC became the main discriminating feature (development KS 0.12–0.17), because it was reconstructed from haemoglobin and a reconstructed haematocrit although its real spread is very narrow (log-SD 0.028 versus 0.11–0.12 for haemoglobin, RBC and haematocrit). The fixed manual choice is replaced by a general, fit-only rule in `RelationCoder.fit`: choose one reconstructed participant per relation, all distinct and admitting an acyclic reconstruction order, maximizing the summed spread of the reconstructed participants on the scale where the relation is additive (log scale for products and ratios, raw scale for sums and differences). On the development fit rows this reconstructs RBC, haemoglobin, MCH and neutrophil percentage and generates haematocrit, MCV, MCHC and the other differential percentages directly. The same chosen derivations are used for the hard-derivation and resampling post-processing of every comparator. The development stage is rerun a third time (runs 1 and 2 retained as `develop_v1`, `develop_v2`); the choice rule above is unchanged. Three development rounds on one development pool are disclosed as a source of optimism for the development estimates only; the confirmation uses the untouched reference half.
