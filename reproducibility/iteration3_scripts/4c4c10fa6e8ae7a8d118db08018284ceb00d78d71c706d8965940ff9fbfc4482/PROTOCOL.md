# Iteration 3: separate training constraints from sampling strength

Declared before new results. Development only; not preregistration or confirmatory testing.

Keep outer fold 0 and split seed 2026 fixed. Screen only inner seed 2027 and generator seed 0 on ILPD and the existing simulator. Replicate at most two selected settings per dataset across generator seeds 0/1/2 and inner seeds 2027/3027/4027. These overlapping development partitions have already influenced earlier research decisions and are not independent test cohorts.

## Questions and bounded search

Does weaker/stronger training energy help? Can stronger sampling guidance preserve the numerical-fidelity advantage of velocity prediction? Does a structured model outperform guidance applied to its unstructured backbone?

- Full models: x0 and velocity prediction, training-energy coefficients 0, 0.01, 0.1 and 1.0; sampling strengths 0, 0.2, 2 and 10 with the same adaptive schedule.
- Backbone controls: x0 and velocity, no training energy or structural blocks, each with the same sampling-strength grid.
- Repair references: x0 and velocity backbone samples at zero guidance followed by the existing deterministic repair.
- Keep paired initialization, width 32, two layers, 30 diffusion steps, 100-epoch maximum, patience 20, batch 64, learning rate .001 and 10 graph bootstraps. No gating changes or new clinical rules.

Reuse existing checkpoints when training configuration, source, dataset and split match exactly. A guidance-only change does not require retraining. Record **new training runs and sampling/evaluation runs separately**, with parent checkpoint hashes and effective sampling settings. Do not count resampling as new model training. Sampling uses the same seed and requested class counts within each comparison.

## Candidate selection before replication

Compare candidates with their matching zero-guidance backbone parameterization on the screening split. A candidate qualifies for the utility/fidelity-preserving shortlist if both RF and logistic AUROC drops are no more than 0.02, normalized numeric Wasserstein and association RMSE are no more than 10% above baseline, and mean missingness error increases by no more than 0.01. Require a decrease in supplied-rule violation. These are engineering screening tolerances, **not clinical noninferiority margins**. They have no confirmatory statistical interpretation.

Select at most two non-repair candidates per dataset by smallest supplied-rule violation among qualifying settings (break ties by numeric Wasserstein, then mean classifier AUROC). If none qualifies, retain the two candidates with the lowest maximum normalized penalty across those utility/fidelity conditions for diagnostic replication, explicitly label them non-qualifying. Preserve every rejected setting. Do not silently retune after viewing replication results.

Replicate selected settings together with their zero-guidance backbone controls. For any selected structured setting, add a dense-incidence model at the same energy/parameterization/guidance to control capacity and membership. Include original full x0 (energy .1, guidance .2) as a reference. Reuse identical sampling/evaluation artifacts across stages instead of counting duplicates.

## Reporting

Report rule eligibility/severity, numerical and categorical fidelity, per-feature variance, both classifiers, copy/membership diagnostics and timing. A single combined violation score cannot establish clinical improvement. Show all partitions and selection failures. Save predictions with round-trip precision, verify reserved-test exclusion and parent provenance, and update the manuscript only with supported development findings. No final test evaluation or claim of a validated external-baseline victory is authorized by favorable development results.
