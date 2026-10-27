# Iteration 15: what in the training recipe matters?

**Bottom line.** The legacy recipe's damage comes from the number of updates its 100-epoch cap allows on small tables (400 at 200 fit rows, 1,300 at 800), not from validation-based checkpoint selection.

- **Short budget.** Giving the corrected recipe the legacy number of updates reproduced or exceeded the whole legacy gap. C2ST was worse in 20 of 20 samples (mean +0.168) and association error was worse in 20 of 20.
- **Checkpoint selection.** Returning the validation-selected checkpoint along the same trajectory instead of the final weights changed C2ST by +0.004 on average, with no consistent sign (6 worse, 3 better, 11 identical). Stopping by patience and averaging the validation loss over 16 corruption draws were also inconsistent.
- **Optimizer and averaging.** A constant learning rate with raw weights did not consistently change realism (13 of 20 worse, mean +0.024). It raised membership AUROC in 20 of 20 samples (mean +0.035).
- **Membership of short runs.** The undertrained variants had membership AUROC about 0.04 lower, consistent with the realism–memorization trade-off.
- **Reconstruction check.** Rebuilding the legacy recipe inside the fixed-budget loop reproduced the legacy results (paired C2ST difference −0.007, 9 vs 11 samples).

Protocol: [ITERATION_15_PROTOCOL.md](../../docs/ITERATION_15_PROTOCOL.md). It covers 160 fits (2 populations × 2 sizes × 5 samples × 8 variants), none failed. Tables: `means.csv`, `effects.csv`, `all.csv`. These are the iteration 12 samples, whose reference halves had been examined before, so this is a mechanism study, not a confirmation.

## C2ST, mean of five samples

| Variant | NHANES 250 | NHANES 1,000 | Thyroid 250 | Thyroid 1,000 | vs V1 (worse/better of 20) |
|---|---:|---:|---:|---:|---:|
| V1 corrected (5,000 updates, decay, averaging, final) | 0.729 | 0.691 | 0.710 | 0.686 | reference |
| V2 + validation-selected checkpoint | 0.735 | 0.691 | 0.719 | 0.685 | 6/3 |
| V3 + patience 20 | 0.778 | 0.691 | 0.710 | 0.685 | 5/6 |
| V4 + 16 corruption draws | 0.722 | 0.691 | 0.712 | 0.686 | 4/6 |
| V5 legacy number of updates | 0.943 | 0.896 | 0.851 | 0.798 | 20/0 |
| V6 constant rate, raw weights | 0.786 | 0.689 | 0.754 | 0.683 | 13/7 |
| V7 legacy rebuilt in the fixed-budget loop | 0.945 | 0.832 | 0.862 | 0.760 | 20/0 |
| V0 legacy recipe | 0.951 | 0.857 | 0.866 | 0.756 | 20/0 |

## Consequence for the manuscript

The first pitfall is restated as an under-specified training budget, meaning an epoch cap that leaves small tables undertrained, rather than noisy checkpoint selection.

The case-study ablation row formerly labelled "all but original selection" restored the whole original training loop: its epoch cap plus selection. It is relabelled accordingly.
