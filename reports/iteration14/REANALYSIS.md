# Iteration 14: corrections and post hoc reanalysis (12 October 2026)

Prompted by the publication review in `reports/publication_review_2026-10-12/REVIEW.md`. Nothing here refits a generator. Every number comes from saved synthetic tables, saved metrics and the frozen splits, via `scripts/reanalysis_iteration14.py` (`relations`, `canonical`, `report`). All analyses below are exploratory, added after the confirmation outputs were examined.

## Corrections to earlier statements

**Independence.** The datasets are panel-by-cycle datasets, not independent populations. Participant identifiers (SEQN) reconstructed from the raw files show:

| Datasets compared | Shared participants |
|---|---:|
| 2017–18 biochemistry (iteration 12) and 2017–18 CBC (iteration 13) | 4,753 |
| 2015–16 development CBC and chemistry | 5,019 |
| 2021–23 CBC and chemistry (confirmation) | 5,263 |
| Datasets from different cycles | 0 |

The confirmation datasets come from four distinct survey cycles: 2011–12, 2013–14, 2017–18 and 2021–23.

**Sample sizes.** Nominal n = 250, 1,000 and 2,000 correspond to 200, 800 and 1,600 fit rows. Synthetic tables have 5 × fit rows (4 × nominal n), not 5 × n as the protocol said.

**Replicate overlap.** Pairwise fit-row overlap between replicate samples is 4–10% at n = 250, 28–32% at n = 1,000 and 59–61% at n = 2,000.

**Fit counts.**

| Fits | Count |
|---|---:|
| Confirmation (sizes 250, 1,000 and 2,000) | 598 |
| Secondary, on the 2017–18 CBC | 10 |
| MedHyperDiff only at n = 100, run before Amendment 1 | 20 |

**Privacy wording.** The model does not "never copy patients".
- On 2011–12 chemistry at n = 250, membership AUROC is 0.608, above the 0.60 threshold.
- At n = 100 (model only), membership AUROC is 0.703–0.759 and C2ST 0.80–0.85.
- The two attacks are diagnostics, not guarantees.

**Hypothesis criteria.** The prespecified criteria (win counts and point-estimate margins) are descriptive success criteria, not significance tests.

## Per-dataset effects and leave-one-cycle-out

All values are MedHyperDiff (c1) minus the comparator, as paired means over replicates. The comparator is the best arm among comparators with membership ≤ 0.60, including plug-ins, as in H3 and H4.

| Comparison | n = 250 | n = 1,000 |
|---|---:|---:|
| C2ST vs realism-selected comparator, pooled | −0.020 | −0.018 |
| Same, leaving one cycle out | −0.019 to −0.021 | −0.009 to −0.028 |
| TSTR vs utility-selected comparator, pooled | +0.010 | +0.007 |
| Same, leaving one cycle out | +0.009 to +0.012 | +0.003 to +0.009 |

**C2ST by dataset:**
- **n = 250:** better on all four datasets, winning 4–5 of 5 samples.
- **n = 1,000, chemistry:** better on both datasets (−0.030 and −0.049, winning 5 of 5).
- **n = 1,000, blood counts:** a tie on CBC 2021–23 (−0.005, winning 3 of 5) and a loss on CBC 2013–14 (+0.012, winning 1 of 5).
- **n = 2,000:** TabDiff is better.

**Correction to the pooled numbers.** The earlier pooled −0.021 at n = 1,000 used a comparator set that excluded the plug-ins. Utility against the utility-best comparator is smaller than against the realism-selected one.

## Strongest relation controls

The strongest control is chosen per cell in hindsight from all post-processings, including textbook-direction re-derivation, and from all plug-ins.
- **Blood count:** MedHyperDiff has the lowest mean relation distance in all 5 cells. Margins are small: 0.128 vs 0.144, 0.092 vs 0.109, 0.151 vs 0.159, 0.137 vs 0.156 and 0.136 vs 0.155. Sample-level wins are 2, 5, 2, 4 and 3 of 3.
- **Chemistry (exact relations):** textbook re-derivation (`canonical.csv`) is at least as faithful, 0.015–0.033 against 0.021–0.050. The "rule-direction" re-derivation control used in the protocol (0.28–0.38) is weak for exact relations, because rounding the reconstructed operand removes real rounding noise. H1x passed against a weak control and should not be read as superiority over re-derivation.
- **Direction matters for realism.** Textbook re-derivation of the CBC indices raises TabDiff's C2ST:

| CBC 2013–14 | Textbook re-derivation | Rule-direction re-derivation |
|---|---:|---:|
| n = 250 | 0.892 | 0.754 |
| n = 1,000 | 0.738 | 0.638 |

## Relation diagnostics beyond the absolute residual distance

Means are over both datasets of each panel at n = 250 and 1,000 (`relations.csv`). Each cell gives blood / chemistry.

| Arm | Signed distance | Coupling error between relations | Dependence error (participants, age, label) |
|---|---:|---:|---:|
| MedHyperDiff | 0.101 / 0.033 | 0.068 / 0.023 | 0.042 / 0.030 |
| TabDiff + reparameterization | 0.127 / 0.039 | 0.069 / 0.025 | 0.043 / 0.028 |
| Copula + resampled residuals | 0.110 / 0.037 | 0.196 / 0.024 | 0.030 / 0.029 |
| TabDiff, textbook re-derivation | 0.750 / 0.025 | 0.197 / 0.023 | 0.030 / 0.028 |

**Coupling between relations.** The distinctive property of residual coding is reproducing the coupling between relations that share participants. Haematocrit, haemoglobin and RBC enter three CBC relations.

**Dependence on covariates.** The model reproduces residual dependence on covariates slightly worse than the controls on CBC, mainly at n = 250.

**Other diagnostics.**
- **Tail exceedance** (target 0.01): 0.019–0.024 for residual-based arms on CBC, about 0.007 on chemistry, and 0 after re-derivation.
- **Coverage gap:** at most 0.01.
- **Negative laboratory values:** none for MedHyperDiff, TabDiff or the copula. Up to 68% of CTGAN rows and 41% of TVAE rows had one.

## Clinical categories

The model's category TV distance was lower than that of the category-best eligible arm in 4 of 10 settings and higher in 6, by up to 0.027 (CBC 2021–23, n = 2,000: 0.047 against 0.020 for TabDDPM).

## Computation

| Model | Parameters | Updates | Passes over fit rows | Median fit minutes (n = 250 / 1,000 / 2,000) |
|---|---:|---:|---:|---:|
| MedHyperDiff (c1) | 39,925 | 5,000 / 20,000 / 40,000 | 1,600 | 5 / 20 / 42 |
| TabDiff | 10.8 M | 8,000 | 8,000 | 36 / 95 / 157 |
| TabDDPM | 0.66 M | 30,000 | 30,000 | 7 / 18 / 34 |

Times are with 8 concurrent single-threaded jobs.

**Tuning is not matched.** MedHyperDiff's training budget was selected on development data, while the comparators used fixed configurations.
