# Iteration 3: an improved development candidate, with remaining tradeoffs

**Completed: 46 new training runs and 174 distinct sampling/evaluation configurations.** The evaluations used 94 fitted checkpoints: 46 new and 48 reused. Changing generation-time guidance does not require training the same weights again. Counts here are separate from the earlier 60 and 110 training/evaluation runs. Automatic zero-guidance evaluations emitted by new-training commands remain in their training directories and are not counted again as independent sweep configurations.

The [complete results](RESULTS.md), [selection record](selection.json), [replication differences](replication_deltas.csv), and [artifact audit](ARTIFACT_AUDIT.json) retain favorable and unfavorable outcomes. All experiments use the original reserved-test boundary and evaluation version 2.

## What improved relative to the previous proposed model

The useful candidate combines the full architecture with **velocity prediction, training-energy coefficient 0.01 and sampling-guidance strength 10**. These settings were selected in the declared seed-0 screen and then held fixed. The previous full configuration used x0 prediction, coefficient 0.1 and guidance 0.2. Thus this comparison concerns the combined configuration; it cannot attribute the whole gain to one change.

Each table entry averages three generator seeds within one ILPD validation partition. Numerical W is mean Wasserstein distance normalized by fit-partition feature standard deviation; lower is better.

| Inner seed | Previous full: rule violation | Updated full: rule violation | Previous numeric W | Updated numeric W | Previous RF AUROC | Updated RF AUROC |
|---|---:|---:|---:|---:|---:|---:|
| 2027 | 15.3% | 5.8% | 0.391 | 0.294 | 0.726 | 0.716 |
| 3027 | 19.1% | 6.0% | 0.408 | 0.227 | 0.683 | 0.680 |
| 4027 | 27.7% | 6.6% | 0.303 | 0.242 | 0.661 | 0.712 |

Numerical W improves by approximately 25%, 44% and 20%. Forest AUROC changes by −0.010, −0.003 and +0.051. Logistic AUROC changes from 0.753/0.676/0.639 to 0.753/0.703/0.694. This is a meaningful improvement over the previous configuration in supplied-rule compliance and numerical fidelity, with limited forest losses on the first two partitions and a larger gain on the third. It is not a statistical superiority or clinical noninferiority finding.

## Stronger controls prevent an overall superiority claim

| Inner seed | Velocity backbone RF | Guided velocity backbone RF | Updated full RF | Updated full logistic | Guided backbone logistic |
|---|---:|---:|---:|---:|---:|
| 2027 | 0.694 | 0.701 | 0.716 | 0.753 | 0.683 |
| 3027 | 0.723 | 0.722 | 0.680 | 0.703 | 0.725 |
| 4027 | 0.702 | 0.715 | 0.712 | 0.694 | 0.694 |

The guided backbone receives the same rule registry and guidance strength 10. It also greatly reduces rule violations, to 9.6%, 6.8% and 8.4%. Full has lower numerical W and fewer violations on all three partitions, but clearly loses utility on partition 3027. A substantial part of the compliance gain therefore comes from generation-time guidance. Matched dense controls also have mixed utility rankings; clinical incidence is not consistently the best representation.

Against its zero-guidance velocity backbone, the coefficient-0.01 candidate passes the declared utility/fidelity engineering conditions on seed means in **two of three** partitions. The coefficient-0.1 candidate passes on **one of three**. Neither satisfies them everywhere. The 0.02 AUROC tolerances and 10% fidelity tolerances are development screening choices, not justified clinical acceptance margins or confidence bounds. Repeatedly reused, overlapping validation partitions cannot establish external generalization.

## Simulator result and missingness limitation

No structured simulator candidate passed all seed-0 screening conditions. Only the x0 backbone with guidance 0.2 qualified. In replication it decreases overall supplied-rule violations from approximately 99% to 87–89%, but parity remains near 46–47% violations. It passes the engineering conditions on two partitions and fails the association-error condition on the third. The previous full x0 model still preserves supplied parity better, with a predictive-utility tradeoff. This iteration does not solve higher-order preservation and utility simultaneously on the simulator.

Rule compliance is conditional on observation eligibility. Protein-rule coverage for the coefficient-0.01 ILPD candidate ranges from 93.4% to 100% across the nine replicated outputs; unobserved protein relationships are not counted as verified valid. Read the per-rule coverage in `all_rule_results.csv` and per-feature missingness/variance in `all_feature_results.csv`. The screening guard uses average missingness error and can miss a concentrated increase in one feature; this is a limitation to tighten in a future protocol, not a reason to silently change the completed selection. The exported samples are research outputs, not clinically certified records.

## Reusable outputs and verification

Both selected ILPD candidates and the simulator comparator are packaged in `artifacts/iteration3/`, with parent-checkpoint provenance. Exports use the original screening split and seed 0; they are convenience artifacts, not independently validated or ensemble models. All three exports exactly reproduce their evaluated generated tables with the recorded class order and seed.

The audit passed for all 174 evaluations: parent weights and training configurations are intact, source/script snapshots resolve, saved predictions reproduce AUROC, class counts match, all 18 full/dense training pairs match parameter count, and no reserved outer-test record was used for fitting or validation. The 20 existing model/invariant tests also passed. No core model code changed in this iteration; improvements arise from the measured configuration changes and better controlled experiments.

The next research milestone is to address the remaining utility loss and feature-specific missingness, separate sampling variation from training variation, and evaluate a frozen candidate with successful external baselines and clinically reviewed rules on independent data. The final test set remains unopened.
