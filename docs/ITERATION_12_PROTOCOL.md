# Iteration 12 protocol: a controlled study of training and evaluation pitfalls in synthetic clinical tables

Written 7 October 2026, after building and auditing the two new populations (counts, missingness, recording resolution and real rule-violation rates only) and before fitting any generator to them. Iterations 2–11 are the motivating case study; their test folds are already opened and are not reused for any claim here.

## Aim

Quantify, across generators, datasets and training-set sizes, how much each of seven common practices changes (a) measured quality and privacy and (b) the **ranking of generators**, and test a corrected protocol. The paper is a methods paper: generators are instruments, not contestants.

## Populations

| Population | Records | Label (positive rate) | Supplied relations | Real violation |
|---|---:|---|---|---:|
| NHANES 2017–2018 adults with a biochemistry record (`build_nhanes`) | 4,756 | diagnosed diabetes (16.6%) | globulin = total protein − albumin; BMI = weight/height²; albumin ≤ total protein; nonnegativity | 0% (definitional) |
| UCI thyroid0387 (`build_thyroid`) | 9,172 | any thyroid diagnosis (26.2%) | FTI ≈ TT4/T4U (fit 95th-percentile envelope); nonnegativity | ≈5% by construction |

Source files are pinned by SHA-256 (NHANES) or hashed at download (UCI zip) and recorded in `data/raw/*.manifest.json`. NHANES is used without survey weights as a convenience population. Thyroid source float artifacts (e.g. 9.599999) are rounded to four decimals; ages above 120 are set missing.

## Design

Each population is split once (seed 2026) into a **reference** half and a **pool** half, grouped by identical predictor profile and stratified by label. From the pool, training samples of n = 250 and n = 1,000 are drawn, five independent replicates per size (stratified, without replacement within a replicate; replicates may overlap). Each sample is split 80/20 into fit and validation rows (stratified). All generators fit the same fit rows; only the early-stopping variants use the validation rows. Synthetic tables contain five times the fit rows with the fit label vector repeated. The reference half is never used for fitting: it is the evaluation population for utility and two-sample tests, and supplies non-members and background data for membership attacks.

## Generators (fixed configurations; no per-size tuning)

| Variant | Purpose | Configuration |
|---|---|---|
| `mhd_corrected` | corrected diffusion | iteration-11 corrected default: fixed 5,000 steps, linear LR decay, EMA 0.999, quantile coding, velocity, T = 100, width 32, two layers, energy weight 0.01, **guidance 0** (strict definitional tolerances made strong guidance destructive in the simulator) |
| `mhd_earlystop` | pitfall 1 | identical, but original loop: validation-loss checkpoint, patience 20, 100-epoch cap |
| `mhd_affine` | pitfall 2 | identical to corrected but affine numeric coding |
| `tabddpm_upstream` | baseline, upstream recipe | iteration-11 most-selected configuration (MLP 512–1024, LR 1.68e-4, 30k steps, full batch, T = 100, linear annealing, final weights) |
| `tabddpm_adapter` | pitfall 5 | same architecture and LR, iteration-2 adapter loop: constant LR, batch 64, clipping, validation-loss early stopping (patience 20, cap 300 epochs) |
| `tabdiff` | baseline | upstream default trainer and sampler |
| `ctgan`, `tvae` | baselines | ctgan-package defaults (300 epochs), per class |
| `copula`, `copula_repair`, `independent`, `bootstrap` | statistical references and copy control | as in iteration 11; repair re-derives supplied identities |

Every continuous-output generator emits raw values; three decodings are evaluated post hoc from fit rows only: **raw**, **global** recorded grid and **banded** (per order of magnitude, 99% coverage). Banded is the primary decoding.

## Measurements

For the primary decoding: the iteration-11 evaluator against the full reference (fidelity, association, utility with forest and logistic TSTR, rule violation rates and realism, nearest-distance membership attack); C2ST at evaluation sizes 50, 100, 200, 500 and 1,000 per class, ten random subsamples each; and a density-ratio membership attack (DOMIAS-style: log density of synthetic data minus log density of attacker background data, kernel estimates after fitting reference-only normal scores and PCA). For raw and global decodings: full-reference C2ST and per-feature Wasserstein distance. Training records: selected epoch, executed epochs and budget for early-stopping variants.

## Pre-registered analyses

1. **Checkpoint selection (P1).** Distribution of selected epochs (fraction within the first 20% of the cap; fraction at the cap) and paired replicate differences in C2ST, numeric W and association RMSE for `mhd_earlystop` vs `mhd_corrected` and `tabddpm_adapter` vs `tabddpm_upstream`.
2. **Representation (P2).** `mhd_affine` vs `mhd_corrected`: per-feature W for the five most skewed features and C2ST.
3. **Recording resolution (P3).** C2ST raw vs global vs banded for every continuous generator; generator ranking under raw vs banded C2ST.
4. **Rule scoring (P4).** Rankings by synthetic violation rate vs violation gap vs residual Wasserstein, separately for definitional (NHANES) and envelope (thyroid) relations.
5. **Memorization (P6).** Both attacks against C2ST across generators and sizes; how often the lowest-C2ST generator has membership AUROC above 0.60. The iteration-11 search records (160 configurations) are analysed the same way.
6. **Evaluation size (P7).** For each pair of generators, agreement between the C2ST order at evaluation size m and at the largest size; the smallest m with at least 90% agreement.
7. **Overall.** A "common practice" pipeline (early-stopping diffusion variants, adapter TabDDPM, raw outputs, violation-rate rule score, C2ST at 100 per class, no privacy check) versus the corrected pipeline (fixed-budget variants, upstream TabDDPM, banded decoding, rule realism, C2ST on the full reference, membership AUROC reported and >0.60 flagged). Outcomes: Kendall τ between the two generator rankings, top-1 agreement, and the fraction of generator pairs whose order reverses, per dataset and size, with replicate bootstrap intervals.

The primary quantitative results are the pairwise reversal rates in analyses 3, 4, 6 and 7 and the effect sizes in 1 and 2. All jobs, failures and outcomes are reported. Uncertainty uses the five replicates (training-sample variation); evaluation-subsample variation is reported separately in analysis 6.
