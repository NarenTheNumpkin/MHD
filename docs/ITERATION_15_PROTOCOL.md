# Iteration 15 protocol: what in the training recipe matters?

Written 12 October 2026, before any fit of this iteration, in response to the publication review of the same date (Priority 5).

## Question

The paper's first pitfall compared two bundles on identical training rows. The legacy recipe and the corrected recipe differ in four ways at once, so the earlier comparison cannot say which change mattered:

| | Legacy recipe | Corrected recipe |
|---|---|---|
| Training budget | 100-epoch cap (about 400 updates at 200 fit rows, 1,300 at 800) | 5,000 updates |
| Learning rate | constant | linear decay |
| Weights returned | raw weights | exponential moving average |
| Checkpoint | the lowest validation diffusion loss, with patience 20 | the final weights |

The legacy validation loss uses one fixed corruption draw, so "noisy" in the earlier text referred to the finite-sample variability of a single draw, not to fresh noise at each epoch. This iteration separates the four changes on the same rows.

## Data

**Populations.** The iteration 12 populations: NHANES 2017–2018 biochemistry and UCI thyroid. Their frozen samples are nominal n = 250 and 1,000 (200 and 800 fit rows), with five replicates each.

**Disclosure.** The reference halves were examined in iteration 12. This is a mechanism study, not a confirmation of generator superiority.

## Variants

All variants are MedHyperDiff with the iteration 12 corrected settings apart from the listed change: quantile coding, v-prediction, 100 diffusion steps, width 32, batch 64, learning rate 0.001.

**Bundles.**
- **V0 legacy:** the original epoch loop (100-epoch cap, patience 20, constant learning rate, raw weights, validation-selected checkpoint).
- **V1 corrected:** 5,000 updates, linear decay, weight averaging (decay 0.999), final checkpoint.

**Checkpoint selection, holding the V1 trajectory fixed.**
- **V2:** V1, but returning the averaged-weight checkpoint with the lowest validation loss among checks made once per pass over the fit rows (ceil(fit rows / 64) updates). Training is identical to V1 because validation uses a separate random generator.
- **V3:** V2 with patience 20 checks, as in the legacy rule. Training stops early.
- **V4:** V2 with the validation loss averaged over 16 fixed corruption draws.

**Training budget.**
- **V5:** V1 with the legacy number of updates (100 × ceil(fit rows / 64)). The averaging decay is 1 − 5/updates, so the averaging horizon stays at 20% of training, as in V1.

**Optimizer.**
- **V6:** V1 with a constant learning rate and raw final weights.

**Legacy rebuilt in the fixed-budget loop.**
- **V7:** the legacy budget, constant learning rate, raw weights, and per-pass validation selection with patience 20.
- This checks that the decomposition reproduces V0.

Each variant is fitted on the 20 samples (2 populations × 2 sizes × 5 replicates), giving 160 fits.

## Measures

The measures are the same as in iteration 12, without the C2ST curve:
- C2ST AUROC against the reference half after banded decoding (primary);
- association RMSE, numerical Wasserstein, forest TSTR AUROC;
- membership AUROC (the maximum of the nearest-distance and density-ratio attacks);
- the selected and executed updates.

## Analysis (fixed now)

For each population × size, report the mean over the five samples. For each variant, report the paired difference from V1 with its paired win count. The effects are read as follows:
- **Selection:** V2 − V1.
- **Early termination:** V3 − V1.
- **Corruption averaging:** V4 − V2.
- **Budget:** V5 − V1.
- **Optimizer and averaging:** V6 − V1.
- **Legacy reproduction:** V7 against V0.

No superiority hypothesis is tested. The paper will state only the components whose paired effect has the same sign in at least 18 of the 20 samples. Any other effect is reported as inconsistent.
