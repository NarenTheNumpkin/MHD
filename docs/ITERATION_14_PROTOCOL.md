# Iteration 14 protocol: is relation-faithful generation robust?

Written 10 October 2026, after building and auditing the new populations (counts, missingness, relation residuals, exactness) and before fitting any generator to them.

## Why

Iteration 13 supported a narrow claim: on one population (NHANES 2017–2018 CBC), at n = 250, the reparameterized MedHyperDiff reproduced four approximate relations better than every published generator and was also the most realistic and useful non-memorizing generator. At n = 1,000 TabDiff was more realistic. Four questions decide whether this is a general result or a niche one:

1. Does the relation result replicate on independent populations?
2. Does it extend to a different family of relations?
3. Is the reparameterization specific to MedHyperDiff, or a plug-in that improves any generator?
4. Where is the crossover with TabDiff when the training size grows, and does scaling the MedHyperDiff training budget with n move it?

## Populations (NHANES public files, SHA-256 pinned in `medhyperdiff.data`)

| Role | Name | Cycle | Panel | Rows | Relations |
|---|---|---|---|---:|---|
| Development | `nhanes_cbc_i` | 2015–2016 | CBC | 5,078 | approximate |
| Development | `nhanes_chem_i` | 2015–2016 | chemistry | 5,028 | three exact, osmolality approximate |
| Confirmation | `nhanes_cbc_h` | 2013–2014 | CBC | 5,229 | approximate |
| Confirmation | `nhanes_cbc_l` | 2021–2023 | CBC | 5,555 | approximate |
| Confirmation | `nhanes_chem_g` | 2011–2012 | chemistry | 4,778 | exact |
| Confirmation | `nhanes_chem_l` | 2021–2023 | chemistry | 5,284 | exact |
| Secondary | `nhanes_cbc` | 2017–2018 | CBC | 4,865 | approximate (iteration 13 population; reference half already examined) |

**Inclusion and label.** All populations include adults aged 20 or over with a definite diabetes answer and at least one panel value. The label is diagnosed diabetes.

**CBC panel.** It is the iteration 13 panel: 16 numerical variables and sex, with four relations (haematocrit, MCH, MCHC and the differential sum). They hold up to analyser rounding.

**Chemistry panel.** It has 20 numerical variables and sex, covering anthropometry, the standard biochemistry profile and urine albumin and creatinine. NHANES reports four relations computed from the reported values:
- globulin = total protein − albumin;
- BMI = weight / height²;
- urine albumin-to-creatinine ratio = 100 × urine albumin / urine creatinine;
- serum osmolality = 1.86 × sodium + glucose / 18 + BUN / 2.8 + 9.

**Exactness audit.** In cycles G and L, every chemistry relation is exact at reported precision: all records lie within half a recording step of the formula. In cycle I, osmolality is approximate (44% of records lie further than 0.5 from the formula). No CBC relation is exact in any cycle (at most 72% of records within half a step).

## Method changes fixed before any fitting

1. **Linear relations.** A `linear` relation kind (derived = constant + Σ wᵢ xᵢ) is added. In the conditioning rule, spread on the additive scale uses the coefficient-weighted contribution of each operand.
2. **Reported-exact relations.** `RelationCoder.fit` flags a relation as reported-exact when at least 99% of complete fit rows lie within half a recording step of the formula. `decode` then re-derives the reported value from the decoded operands after reconstruction, which is the laboratory's own computation. Without this, rounding a reconstructed operand (urine albumin is recorded to 0.1 mg/L) would add relation error that real records do not have. Decoding is shared, so the step applies equally to MedHyperDiff, to every plug-in and to the hard-derivation and resampling post-processing of comparators. No CBC relation is flagged, so CBC decoding is unchanged from iteration 13.
3. **Relation distance scale.** Per relation, the Wasserstein distance between synthetic and reference absolute residuals is divided by the fit SD of the absolute residual, as in iteration 13. That SD is floored at the SD of the absolute rounding error of the derived value (step/√48), because exact relations such as NHANES globulin have zero residual SD. The floor never binds for the CBC relations, so iteration 13 values are unchanged; both versions are stored.
4. **Clinical category agreement (new secondary endpoint).** It is the total-variation distance between reference and synthetic distributions over a joint clinical classification that depends on the relations:
   - CBC: WHO anaemia × MCV class (< 80, 80–100, > 100 fL) × MCH < 27 pg × MCHC < 32 g/dL.
   - Chemistry: WHO BMI class × KDIGO albuminuria (< 30, 30–300, > 300 mg/g) × albumin/globulin < 1 × osmolality (< 275, 275–295, > 295 mmol/kg).

## Development: scaling MedHyperDiff with training size

Candidates are declared before development as `mhd_params` in `scripts/iteration14.py`. All use relation reparameterization with structure hyperedges.
- **c0:** the iteration 13 configuration (5,000 optimizer steps, width 32).
- **c1:** steps proportional to n (5,000 × max(1, n/250)), so the number of passes over the fit rows is constant.
- **c2:** c1 with width 64.
- **c3:** c1 with width 32 × √max(1, n/250), rounded to a multiple of 8 (32, 64 and 88 at n ≤ 250, 1,000 and 2,000).

**Development runs.** Both development populations use n = 250, 1,000 and 2,000 with three seeds. Each population is split once into an evaluation half and a pool, and samples come from the pool. Identical configurations are fitted once. TabDiff at n = 1,000 (three seeds per population) is fitted for information only.

**Choice rule.** The lowest mean development C2ST over both populations and three sizes wins. Candidates are eligible only if their mean development membership AUROC is at most 0.60 in every population × size cell. Candidates within 0.002 of the lowest are tied, and the earliest in the list wins.

## Confirmation design

**Splits.** Each confirmation population is split once into a reference half (never fitted; all evaluation) and a pool. The split is grouped by predictor profile and stratified. Samples of n = 100, 250 and 1,000 (five replicates) are drawn from the pool. n = 2,000 (three replicates) is drawn in `nhanes_cbc_l` and `nhanes_chem_l`, where pools allow it; these samples overlap heavily, and their replicate spread understates sampling variability. Every sample is split 80/20 into fit and validation rows. Synthetic tables have 5 × n rows with the fit-row class balance.

**Arms (14 population × size cells).**
- **MedHyperDiff-primary:** the chosen candidate.
- **MedHyperDiff c0:** reference arm, the iteration 13 configuration.
- **Published generators:** TabDiff, TabDDPM (upstream recipe), CTGAN, TVAE and the Gaussian copula, with iteration 12 configurations. Each is reported as generated, with hard derivation, and with derivation plus resampled real residuals.
- **Plug-in:** each published generator fitted on the reparameterized table (`<generator>_rp`).
- **Control:** the bootstrap copy control.

**Execution.** Comparator, plug-in, control and c0 arms are launched in parallel with development. Their outputs are not examined until the choice is frozen, and the report script refuses to run without the choice file. All fits run on CPU, single-threaded, eight jobs at a time.

## Endpoints

**Primary.** Relation residual distance (item 3 above).

**Guardrails.**
- C2ST AUROC against the reference half, with banded decoding for generators that do not decode precision themselves.
- Membership AUROC: the maximum of the nearest-distance and density-ratio attacks.

**Secondary.** Forest TSTR AUROC, clinical category distance, per-relation distances, numeric Wasserstein and association RMSE.

**Definition used below.** A comparator arm is non-memorizing in a cell when its mean membership AUROC is at most 0.60.

## Pre-registered hypotheses

**H1. Approximate relations replicate** (CBC, 7 cells). MedHyperDiff-primary must have a lower relation distance than every published generator, both as generated and with hard derivation, in at least 4 of 5 replicates (3 of 3 at n = 2,000). It must hold in every cell.

**H1x. Exact relations are reproduced** (chemistry, 7 cells). MedHyperDiff-primary must have a lower relation distance than every published generator as generated, in at least 4 of 5 replicates (3 of 3). Its mean must also be no more than 0.10 above the best hard-derived published generator. Hard derivation is already the right post-processing for exact relations, so superiority over it is not claimed.

**H2. Reparameterization is a general plug-in** (5 generators × 14 cells = 70). For each generator and cell, three conditions are tested:
- **(a)** the plug-in beats the same generator as generated on relation distance in at least 4 of 5 replicates (3 of 3);
- **(b)** for CBC it also beats the generator with hard derivation in at least 4 of 5 replicates (3 of 3); for chemistry its mean is within 0.10 of the hard-derived mean;
- **(c)** its mean C2ST is at most the better of the generator as generated and with hard derivation, plus 0.02.

H2 is supported if (a) and (b) hold in all 70 and (c) holds in at least 63.

**H3. Small-sample advantage** (n = 100 and 250, 8 cells). MedHyperDiff-primary has the lowest mean C2ST among all non-memorizing arms. These include every published generator in every processing and every plug-in. H3 is supported if this holds in at least 6 of 8 cells, with MedHyperDiff-primary membership at most 0.60 in all 8.

**H4. Larger samples** (n = 1,000 and 2,000, 6 cells). MedHyperDiff-primary mean C2ST is within 0.02 of the best non-memorizing arm. H4 is supported if this holds in at least 4 of 6 cells.

**Reporting.**
- All cells, hypotheses and failures are reported.
- The pooled analysis gives, per size, the mean paired difference of MedHyperDiff-primary against (i) the best published arm of each cell and (ii) TabDiff with the plug-in, on C2ST and forest TSTR AUROC. Each comes with a 95% bootstrap interval over population × replicate.
- The secondary population (iteration 13 CBC, n = 250 and 1,000) is reported against the stored iteration 13 comparators and is not part of any hypothesis.

## Amendments

Any change after development starts is recorded here with its reason and the data it was based on. Nothing is changed after confirmation outputs have been examined.

### Amendment 1 (10 October 2026, during confirmation; no confirmation output examined)

The n = 100 condition is removed from confirmation to shorten the run, at the investigator's request. TabDiff fits made the full design take more than a day on the study laptop. When the amendment was made:
- the development stage and the MedHyperDiff-primary arm (including its n = 100 runs) were complete;
- 44 comparator jobs were complete, all TabDiff at n = 1,000 and 2,000;
- no comparator or plug-in job at n = 100 had run;
- no confirmation metric had been read.

The MedHyperDiff n = 100 runs stay on disk and are not reported, because they have no comparators. The hypotheses keep their criteria with the reduced cell counts:
- **H1 and H1x:** 5 cells each, at n = 250, 1,000 and 2,000.
- **H2:** 50 generator-cells. Condition (c) must hold in at least 90% of them (45).
- **H3:** n = 250 only, 4 cells. Supported in at least 3 of 4.
- **H4:** unchanged.

The queue was restarted with the amended runner. Completed jobs are reused, and their manifests record the runner hash they ran with.
