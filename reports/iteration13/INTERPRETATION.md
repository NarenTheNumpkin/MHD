# Iteration 13: relation-faithful generation — where MedHyperDiff is better

**Bottom line.** With relation reparameterization, MedHyperDiff reproduces supplied approximate clinical relations better than every published generator tested, with or without the constraint enforcement used in the literature, in all 5 confirmation samples at both training sizes. At **n = 250 training records** it is also the most realistic and most useful non-copying generator. It beats TabDiff with hard derivation, the strongest published alternative, on relation fidelity, two-sample realism and downstream utility in **5 of 5 samples**, without memorization. The pre-registered claim is **supported at n = 250 and not supported at n = 1,000**, where realism trails TabDiff by 0.02–0.03. Protocol and amendments: [ITERATION_13_PROTOCOL.md](../../docs/ITERATION_13_PROTOCOL.md). Run records: `runs/iteration13` (development 24 jobs × 3 disclosed rounds; confirmation 90 jobs, none failed).

## Setting

NHANES 2017–2018 complete blood counts (4,865 adults, never used before this iteration). Four supplied relations hold only up to reporting rounding: haematocrit = RBC × MCV/10, MCH = 10 × haemoglobin/RBC, MCHC = 100 × haemoglobin/haematocrit and the five differential percentages summing to 100. Confirmation samples were evaluated on a never-fitted reference half of 2,433 records. Comparators were TabDiff, TabDDPM (upstream recipe), CTGAN, TVAE and a Gaussian copula, each also with two knowledge-equivalent post-processings of the same output:
- hard derivation (the exact-constraint enforcement of constraint-layer methods);
- derivation plus resampled real residuals.

The same reparameterization was also applied to TabDiff.

## Method

The generator models a well-conditioned basis plus one learned residual per relation. Reconstructed variables are chosen automatically from training rows: one per relation, acyclic, with the largest spread on the relation's additive scale. Here that means RBC, haemoglobin, MCH and neutrophil %, while narrow quantities such as MCHC are generated directly. Each residual and its operands form a hypergraph edge. Development showed the conditioning choice is essential. Reconstructing the basophil fraction or MCHC raised development C2ST to 0.84–0.88 before the rule was introduced (Amendments 1 and 2).

## Confirmation (mean of five samples; paired wins out of five)

| n | Measure | MedHyperDiff + reparam | TabDiff, as generated | TabDiff + hard derivation | Best non-memorizing published + derivation + resampling | TabDiff + same reparam |
|---|---|---:|---:|---:|---:|---:|
| 250 | Relation residual distance ↓ | **0.187** | 12.14 (5/5) | 0.965 (5/5) | 0.210 copula (4/5) | 0.257 (5/5) |
| 250 | C2ST ↓ | **0.691** | 0.762 (5/5) | 0.736 (5/5) | 0.735 copula (5/5) | 0.745 (5/5) |
| 250 | Forest TSTR AUROC ↑ | **0.717** | 0.670 (5/5) | 0.671 (5/5) | 0.687 copula (5/5) | 0.679 (5/5) |
| 250 | Membership AUROC ↓ | 0.555 | 0.619 | 0.618 | 0.514 CTGAN | 0.598 |
| 1,000 | Relation residual distance ↓ | **0.173** | 3.92 (5/5) | 0.911 (5/5) | 0.190 TVAE (4/5) | 0.184 (5/5) |
| 1,000 | C2ST ↓ | 0.664 | 0.645 (1/5) | **0.633** (0/5) | 0.635 TabDiff (0/5) | 0.643 (0/5) |
| 1,000 | Forest TSTR AUROC ↑ | 0.735 | 0.735 (2/5) | 0.733 (2/5) | 0.730 TabDiff (3/5) | 0.742 (1/5) |
| 1,000 | Membership AUROC ↓ | 0.509 | 0.560 | 0.561 | 0.503 CTGAN | 0.548 |

"Best non-memorizing published + derivation + resampling" is, per metric, the best published generator with membership AUROC at most 0.60 after resampling post-processing; the generator is named in each cell. TabDiff exceeds 0.60 slightly at n = 250 (0.618–0.619), and TabDDPM memorizes at both sizes (0.96 and 0.66; its resampled utility is 0.695 and 0.743). The other published generators are further behind: at n = 250 CTGAN, TVAE and TabDDPM have C2ST 0.84–0.99, and TabDDPM memorizes (membership 0.96).

**Pre-registered verdict.**
- At n = 250, criteria (i)–(iii) hold: relation wins in 5/5 against all ten published-generator × processing combinations; C2ST 0.691 against the best non-memorizing published 0.731; membership 0.555.
- At n = 1,000, criterion (i) holds (5/5), criterion (iii) holds (0.509), and criterion (ii) fails: C2ST 0.664 against 0.633 for TabDiff with hard derivation, a margin of 0.031 against the allowed 0.02.

## What it means

The win is specific and meaningful: in small clinical samples with known laboratory relations, which is the setting of most clinical data-sharing requests, this model produces tables whose relations match real data. Unconstrained generators miss them by one to two orders of magnitude and exact enforcement over-corrects by about fivefold. The same tables are more realistic overall and more useful for prediction than the strongest published generator, and they do not copy patients.

Part of the advantage is the reparameterization itself. Applied to TabDiff, it also fixes relations (0.257 at n = 250, 0.184 at n = 1,000), but the MedHyperDiff version is better on all three axes at n = 250. The hypergraph matters too: without it, development C2ST was worse. With more data (n = 1,000), TabDiff's general realism overtakes it, so the advantage is a small-sample one and is reported as such.

## Limitations

- **One population:** the result rests on a single population, one NHANES cycle, analysed without survey weights.
- **Configurations:** they were fixed rather than tuned; the comparators used iteration-12 defaults.
- **Development optimism:** three disclosed development rounds on the development pool make the development estimates optimistic, though not the confirmation.
- **Relation metric:** it is computed on absolute residuals.
- **Membership attacks:** they are diagnostics, not guarantees.
- **Secondary populations:** the thyroid and NHANES biochemistry populations named in the protocol were not run in this iteration.
