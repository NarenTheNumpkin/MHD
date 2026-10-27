# Optional stability changes in development iteration 2

These options retain the original architecture as the default for compatibility. They are established stabilization/parameterization ideas under evaluation, not standalone novelty claims.

## Gated structural residual

For feature tokens H and an existing nonlinear incidence update U(H), the optional block computes

`H_next = H + tanh(g) * I(degree > 0) * LayerNorm(U(H))`,

where each layer has a learned scalar `g` initialized to zero. The learned coefficient of the empirical graph attention bias also starts at zero. At initialization the structured model therefore has the same predictions as its shared backbone. Gradients can immediately update the gate; gradients through the gated branch develop as the gate moves away from zero. Negative coefficients are permitted by tanh and are not interpreted as calibrated clinical confidence.

With `--paired-initialization`, construction of hypergraph parameters does not advance the RNG used to initialize common output heads. Training uses a seed reset independent of architecture size. Models with the same generator seed see identical shuffle orders and diffusion corruptions for overlapping training epochs. Early stopping can still produce different numbers of updates and selected epochs; report those counts.

`--mode dense` retains the same edge slots, learnable edge identities and network modules but replaces incidence membership with all ones. It controls parameter count and tests clinical membership against unrestricted factor mixing. It is not a universal control for all dense neural architectures. Gated full/dense models have equal parameter counts; base/full do not.

## Gaussian velocity prediction

For `x_t = sqrt(a_bar_t) * x_0 + sqrt(1-a_bar_t) * epsilon`, velocity uses

`v_target = sqrt(a_bar_t) * epsilon - sqrt(1-a_bar_t) * x_0`,

and reconstructs

`x_0_hat = sqrt(a_bar_t) * x_t - sqrt(1-a_bar_t) * v_hat`.

The observed-numeric loss is MSE against `v_target`; clinical energies and reverse Gaussian transitions still use reconstructed `x_0_hat`. Categorical diffusion and guidance schedules are unchanged. Velocity changes the effective time weighting of clean reconstruction error, so validation loss values across x0/v arms are not directly comparable. This parameterization follows established diffusion work, including [Salimans and Ho, ICLR 2022](https://arxiv.org/abs/2202.00512).

## Evaluation and split changes

- `--inner-seed` changes only the inner development partition. Outer fold 0 and split seed 2026 remain fixed during this iteration. Default `None` preserves all earlier split indices.
- The discriminator now groups identical full joint records across its cross-validation folds. This is evaluation version 2; its discriminator AUC must not be silently pooled with version 1.
- Per-feature numerical standard-deviation ratios expose variance loss that average rule compliance can hide.
- Every new run saves patient-row-indexed logistic and forest prediction scores for TRTR, TSTR and augmentation. These enable later paired analyses, but do not turn validation results into independent test evidence.
- Summaries distinguish inner partitions and evaluation versions. Repeated seeds and overlapping inner partitions are not independent patient cohorts.
