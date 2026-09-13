# Development iteration 2 — declared before new model comparisons

Date: 12 September 2026. This is a development protocol, not a registered or confirmatory study.

## Objective

Improve the prototype's validity–fidelity–utility trade-off and strengthen the comparisons. Do not presume the full architecture will win. Preserve old results and report rejected candidates.

## Development boundary

Keep outer fold 0, split seed 2026 fixed. Its test indices are reserved. Screen on the existing inner validation split of ILPD and the controlled simulator; then repeat across generator seeds and additional inner partitions **within the same outer development pool**. Do not vary outer folds while tuning architecture: doing so would expose reserved test patients in another fold's training data. Previously suggested all-fold development commands are superseded by this stricter policy for this iteration.

## Candidate changes and controls

1. Add a scalar, zero-initialized gate to each nonlinear hypergraph residual and start the empirical attention bias at zero. The gate allows the model to learn how strongly to inject structure. The original unscaled normalized residual can dominate its input at initialization; this is a diagnostic hypothesis, not an established cause of utility loss.
2. Add optional velocity prediction for Gaussian diffusion, with the appropriate training target and clean reconstruction formula. This is an established diffusion parameterization, not a novelty claim (Salimans and Ho, https://arxiv.org/abs/2202.00512).
3. Match common backbone initialization and training corruption randomness across the gated structural controls. Add a dense-incidence control with the same learned parameter count as the clinical hypergraph.
4. Preserve the original x0/ungated configuration as a reference. Test gating and velocity separately before their combination. Training and sampling guidance remain separate ablations.
5. Add official-source TabDDPM and inspect official TabDiff integration. Record upstream commits, license, all adaptations and budgets. Small CPU development runs must not be called a replication of published benchmark numbers.

## Screening and replication

Use the same data split, conditioning, synthesis counts, width 32, depth 2, batch size 64, 30 diffusion steps and 100-epoch maximum for the first screen. Include 100-epoch legacy base/full references so extra training is not mistaken for an architectural improvement. Preserve fixed validation-corruption checkpoint selection; identify its limitations. Use seed 0 for screening and seeds 1/2 for replication, then vary only the inner split for sensitivity.

Report numeric fidelity, variance/coverage, pairwise and higher-order dependence, each rule's violation/severity/eligibility, logistic and forest utility, missingness, costs and empirical privacy. Do not select a model using outer-test outcomes. Use a Pareto comparison: a lower imposed penalty alone does not establish improvement. Any practical utility tolerance used for screening is an engineering choice and not a clinical noninferiority margin.

## Evidence and manuscript policy

Add measured results, failures, and limitations to the manuscript as development results. Clinical expert review, meaningful external cohorts, stronger formal comparison and independent confirmatory testing remain gates for a submission claim. No invented authors, ethics approvals, DP protection or statistical significance.

## Replication decision after seed-0 screen

Retain paired base/full, gated full, velocity full, gated dense and paired repair for seeds 1 and 2 on both ILPD and the simulator. This retains a candidate that improves numerical fidelity as well as the gating hypothesis, and does not select only the largest AUROC. All rejected screen arms remain reported. The initial official TabDDPM screen was poor with a tiny model/30 steps. Add a separate sensitivity budget: width 128, two hidden layers, 1,000 diffusion steps, 500 maximum epochs, patience 100, batch size 64, seeds 0/1/2. This is a larger-budget baseline sensitivity, not a matched-compute comparison or complete upstream benchmark replication.

## Additional-partition decision after three-seed comparison

Gating did not consistently dominate the ungated full model, so it is not promoted to the default. Keep paired base/full and velocity full for inner split seeds 3027 and 4027, using generator seeds 0/1/2 on both datasets. Add an ungated paired dense-incidence control, matching the selected ungated full model, on the original and both additional partitions. The gated dense control alone cannot identify the ungated full model's structural benefit.

TabDDPM's larger-budget runs still show severe numeric fidelity problems. Treat this as an unresolved baseline training limitation, not evidence of MedHyperDiff superiority. Run an additional diagnostic at 8,500 epochs with early stopping disabled (51,000 minibatch updates for ILPD; 42,500 for the simulator), width 128 and 1,000 steps, seed 0, on both datasets. Check whether increasing training resolves the issue; keep this diagnostic separate from the three-seed benchmark table.
