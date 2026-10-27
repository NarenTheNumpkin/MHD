# Iteration 14: is relation-faithful generation robust?

> **Corrections, 12 October 2026 (after the publication review; see [REANALYSIS.md](REANALYSIS.md)).**
> - **Not independent populations.** The four confirmation datasets are panel-by-cycle datasets from three survey cycles, and the two 2021–23 panels share 5,263 participants.
> - **Privacy wording.** "Does not copy patients" overstates the evidence. Membership AUROC is 0.608 on chemistry 2011–12 at n = 250 and 0.70–0.76 at n = 100, and the attacks are diagnostics only.
> - **Exact chemistry relations.** Textbook-direction re-derivation is at least as faithful as MedHyperDiff, so H1x passed only against a weak control.
> - **Realism at n = 1,000.** The advantage holds on chemistry, not on blood counts.
> - **Pooled n = 1,000 difference.** With plug-ins included among comparators it is −0.018, not −0.021.
> - **Sample sizes.** Nominal sizes include a 20% validation split, giving 200, 800 and 1,600 fit rows.

**Bottom line.**
- **Relation fidelity is not a one-dataset effect.** On four independent NHANES populations, MedHyperDiff with relation reparameterization reproduced clinical relations better than every published generator (TabDiff, TabDDPM, CTGAN, TVAE, Gaussian copula), whether as generated or with hard derivation. This holds in all 10 population × size cells, across two relation families (approximate CBC indices and exact chemistry formulas) and up to n = 2,000.
- **The training-size problem from iteration 13 is fixed.** With the training budget scaled to sample size (configuration c1, chosen on development data), MedHyperDiff is the most realistic non-memorizing generator at n = 250 and n = 1,000 on the pooled data:
  - C2ST is 0.020 to 0.021 lower than the best published arm of each cell, with 95% intervals excluding zero.
  - Forest TSTR AUROC is 0.012 to 0.015 higher.
- **TabDiff is still more realistic at n = 2,000.**

Pre-registered verdicts: H1 supported (5/5), H1x supported (5/5), H3 supported (3/4), H4 supported (4/6). H2, the general plug-in claim, is **not supported** (37/50 units).

Protocol: [ITERATION_14_PROTOCOL.md](../../docs/ITERATION_14_PROTOCOL.md), including Amendment 1, which dropped n = 100. 608 confirmation jobs completed, none failed.

## Hypotheses

| Hypothesis | Units | Holding | Verdict |
|---|---:|---:|---|
| H1 approximate (CBC) relations beat all published generators, raw and hard-derived | 5 | 5 | supported |
| H1x exact (chemistry) relations: beat raw, within 0.10 of hard derivation | 5 | 5 | supported (MedHyperDiff was lower than hard derivation in every cell) |
| H2 reparameterization as a general plug-in | 50 | 37 | not supported: (a) 48/50, (b) 44/50, (c) 44/50 |
| H3 lowest C2ST of all non-memorizing arms at n = 250 | 4 | 3 | supported; the miss is chemistry 2011–12, where MedHyperDiff had the lowest C2ST (0.715 vs 0.737) but membership 0.608 > 0.60 |
| H4 within 0.02 of best at n ≥ 1,000 | 6 | 4 | supported; both misses are at n = 2,000 |

## Where MedHyperDiff wins, and by how much

**Relation distance.** Values are means; lower is better.

| Population | n | MedHyperDiff | Best published, hard derivation | TabDiff as generated |
|---|---:|---:|---:|---:|
| CBC 2013–14 | 250 / 1,000 | 0.128 / 0.092 | 0.743 / 0.741 | 24.8 / 3.1 |
| CBC 2021–23 | 250 / 1,000 / 2,000 | 0.151 / 0.137 / 0.136 | 0.968 / 0.972 / 0.974 | 10.5 / 3.8 / 2.5 |
| Chemistry 2011–12 | 250 / 1,000 | 0.049 / 0.025 | 0.295 / 0.284 | 3,677 / 2,225 |
| Chemistry 2021–23 | 250 / 1,000 / 2,000 | 0.050 / 0.025 / 0.021 | 0.360 / 0.354 / 0.351 | 6,063 / 5,771 / 1,700 |

**C2ST.** Values are means; lower is better. The comparison column is the best non-memorizing arm, which includes every published generator in every processing and every plug-in.

| Population | n | MedHyperDiff c1 | Best other arm |
|---|---:|---:|---:|
| CBC 2013–14 | 250 | **0.709** | 0.728 copula, hard derivation |
| CBC 2013–14 | 1,000 | 0.646 | **0.634** TabDiff, derivation + resampled |
| CBC 2021–23 | 250 | **0.690** | 0.719 copula, derivation + resampled |
| CBC 2021–23 | 1,000 | **0.629** | 0.634 TabDiff, hard derivation |
| CBC 2021–23 | 2,000 | 0.616 | **0.577** TabDiff, hard derivation |
| Chemistry 2011–12 | 250 | **0.715** | 0.737 copula, hard derivation |
| Chemistry 2011–12 | 1,000 | **0.644** | 0.674 TabDiff + reparameterization |
| Chemistry 2021–23 | 250 | **0.775** | 0.783 copula, as generated |
| Chemistry 2021–23 | 1,000 | **0.700** | 0.749 TabDDPM, as generated |
| Chemistry 2021–23 | 2,000 | 0.645 | **0.625** TabDiff, as generated |

**Pooled paired differences** (MedHyperDiff minus comparator; 95% bootstrap interval over population × replicate):

| n | Comparison | C2ST | Forest TSTR AUROC |
|---|---|---:|---:|
| 250 | best published arm of each cell | −0.020 (−0.028, −0.011) | +0.015 (0.010, 0.020) |
| 250 | TabDiff + reparameterization | −0.044 (−0.055, −0.034) | +0.025 (0.013, 0.036) |
| 1,000 | best published arm of each cell | −0.021 (−0.033, −0.008) | +0.012 (0.008, 0.017) |
| 1,000 | TabDiff + reparameterization | −0.025 (−0.038, −0.012) | +0.010 (0.005, 0.015) |
| 2,000 | best published arm of each cell | +0.030 (0.020, 0.039) | +0.004 (0.002, 0.008) |

**Scaling the training budget is what changed the larger-sample result.** The iteration 13 configuration (c0) fits 5,000 steps whatever n is, so it undertrains on larger samples. c1 scales the steps with n:

| Population | n | c0 C2ST | c1 C2ST |
|---|---:|---:|---:|
| CBC 2021–23 | 1,000 | 0.661 | 0.629 |
| CBC 2021–23 | 2,000 | 0.656 | 0.616 |
| Chemistry 2021–23 | 1,000 | 0.765 | 0.700 |
| Chemistry 2021–23 | 2,000 | 0.763 | 0.645 |

Membership rises slightly (about +0.02 to +0.03) and stays at most 0.56 from n = 1,000. Making the network wider did not help in development.

**Secondary check on the iteration 13 population** (2017–18; its reference half had already been examined). c1 reverses the iteration 13 failure at n = 1,000: C2ST 0.627 against 0.633 for TabDiff with hard derivation, with relation distance 0.157 and utility 0.747, against 0.733 for TabDiff.

## What did not work

**The plug-in claim (H2).** Reparameterization reliably fixes relations for the strong generators:
- TabDiff, TabDDPM and the copula improved over the raw output in every cell.
- For approximate relations they also beat hard derivation; for exact relations they stayed within the margin.

It is not a universal plug-in, for two reasons:
- **CTGAN and TVAE on the chemistry panel.** They generate implausible operands, for example near-zero urine creatinine. Because the albumin-to-creatinine ratio divides by it, every processing of their output, including the plug-in, has astronomically large relation distances. The relation metric is meaningless for these generators there.
- **Realism cost for some fits.** TabDiff and TabDDPM with the plug-in lost more than 0.02 C2ST in 6 cells. The worst case is chemistry 2021–23 at n = 2,000, where one TabDiff replicate reached C2ST 0.878, so the plug-in can destabilize TabDiff training.

The plug-in is therefore reported as working for the stronger generators, not as a general result.

**n = 2,000.** TabDiff is more realistic, by 0.039 on CBC and 0.020 on chemistry. Utility is about equal. MedHyperDiff still has far better relation fidelity and slightly lower membership. The crossover in realism therefore lies between 1,000 and 2,000 training records.

**Clinical category distance.** It was not consistently better:
- MedHyperDiff was best on CBC 2013–14 (0.045 vs 0.075 at n = 250).
- It was comparable or worse elsewhere. Hard derivation already places most records in the right category, so the categories are less sensitive than the relation distance itself.

## What it means

The advantage is no longer niche:
- It holds on four independent populations, two relation families, and every size tested for relation fidelity.
- With the scaled budget it is the most realistic and most useful generator up to n = 1,000. Most clinical data-sharing requests (single sites, subgroups, rare conditions) fall in that range.
- Beyond about 2,000 records, TabDiff produces more realistic tables. Even there, MedHyperDiff remains the choice when laboratory relations must hold, since TabDiff misses them by one to three orders of magnitude and hard derivation over-corrects approximate relations by about sevenfold.

## Limitations

- **Survey design:** NHANES only, without survey weights.
- **Configurations:** comparator configurations are fixed upstream or iteration 12 defaults, not tuned.
- **n = 2,000:** it rests on three heavily overlapping samples in two populations.
- **n = 100:** it was dropped (Amendment 1).
- **Membership attacks:** they are diagnostics, not privacy guarantees.
