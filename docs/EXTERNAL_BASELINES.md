# External diffusion baseline provenance

## Iteration 11: upstream training recipes and TabDiff

The earlier TabDDPM "instability" came from the adapter, not from TabDDPM: it trained with a constant learning rate, gradient clipping and validation early stopping, whereas upstream `scripts/train.py` trains a fixed number of steps with linear learning-rate annealing, no clipping and no validation selection, and `scripts/pipeline.py` samples the final `model.pt`. `training="steps"` in `TabDDPMAdapter` now reproduces that loop, including the upstream 128-dimensional time embedding and explicit MLP widths (`mlp_layers`). The legacy adapter path is unchanged for archived runs. Iteration 11 tunes TabDDPM with the upstream `tune_ddpm.py` space (configuration 0 is upstream `exp/diabetes/ddpm_cb_best`, the closest small medical table in the TabDDPM benchmark).

`TabDiffAdapter` imports the unmodified model (`UniModMLP`, `Model`), learnable per-column noise schedules and stochastic second-order sampler from [TabDiff](https://github.com/MinkaiXu/TabDiff/tree/5ecdb3356261aea72716cc9a779f31d7ad083bf4) (MIT licence, commit verified at import, `tabdiff/` must be clean) and reproduces the upstream Trainer: AdamW over all diffusion parameters, ReduceLROnPlateau (factor 0.9, patience 50) on the epoch loss, linearly annealed continuous-loss weight, EMA 0.997 of the denoiser and both schedules, and the best EMA state by training loss after epoch 4000. The class label is generated jointly as a categorical column, as upstream does for classification tables, and exact class counts are drawn by oversampling. Numerical features use a fit-only normal quantile transform; numeric observation flags are explicit categorical columns. All continuous-output generators (MedHyperDiff, TabDDPM, TabDiff, CTGAN, TVAE) decode to the fit partition's recorded measurement resolution in iteration 11. Both adapters are documented adaptations, not reproductions of published benchmark scores.

## Current matched development comparison (iteration 7)

Native `ctgan==0.11.1` CTGAN/TVAE were fitted successfully with both compact/100-epoch and native-default/300-epoch presets, using the current ILPD fit/validation splits, class labels and 5x evaluation counts. This adds 36 adapters (72 class-specific networks) and retains all outcomes. See the [protocol](ITERATION_7_PROTOCOL.md), [results](../reports/iteration7/INTERPRETATION.md) and [commands](ITERATION_7_RUNBOOK.md). Utility rankings are mixed; full beats each default preset on two of three partitions, but compact TVAE beats full on two. Simple repaired copula has stronger numeric/association fidelity and zero observed supplied-rule violations. These class-specific native adaptations are not an exact published benchmark replication or a matched-compute/tuning comparison. TabDDPM remains an unsuccessful reproduction and TabDiff is not trained by this stage.

Iteration 8 did not change the external baselines. Its fit-only asinh-MAD model was compared against the saved full-prior and repaired-copula records and rejected after extreme inverse-decoded bilirubin tails. See the [iteration-8 interpretation](../reports/iteration8/INTERPRETATION.md).


## Official-source TabDDPM adapter

The adapter imports `GaussianMultinomialDiffusion` and `MLPDiffusion` from the authors' [official repository](https://github.com/yandex-research/tab-ddpm/tree/b476257dd460b778ba09eb97f7a51d6490fa17f8), pinned to commit `b476257dd460b778ba09eb97f7a51d6490fa17f8`. The upstream model files are unmodified. The MIT license is retained at `external/tab-ddpm/LICENSE.md`. Each experiment records the commit and SHA-256 hashes of imported source files. The adapter rejects a changed commit or modified upstream package.

The method is described by Kotelnikov et al., [TabDDPM: Modelling Tabular Data with Diffusion Models](https://proceedings.mlr.press/v202/kotelnikov23a.html), ICML 2023. **Our runs are small-budget adaptations, not reproductions of the published benchmark scores.**

Adaptations made in `src/medhyperdiff/external_baselines.py`:

- Fit the shared missingness-aware preprocessor on the fit partition. Apply an empirical normal quantile transform to median-imputed numerical variables, fitted only there. Decode back through both transforms. Quantile inversion bounds numerical outputs to the observed training range; this affects comparisons with the affine-scaled MedHyperDiff backbone and must be disclosed.
- Compact categorical states to those observed during fitting, including numeric observation indicators. Map unseen validation categorical states to the fit mode for validation loss, and record their cell count. This mapping does not affect evaluation on the original validation records. No unknown category is deliberately generated.
- Use the official conditional MLP, Gaussian epsilon prediction, multinomial diffusion loss and reverse sampler. Invoke the sampler separately with a degenerate label distribution to produce the exact common class counts and row order.
- Use AdamW at the requested constant learning rate, weight decay 1e-4, gradient clipping at 1, and fixed-corruption validation checkpoint selection. These choices differ from upstream training, including its EMA/scheduling choices. Do not compare loss magnitudes across methods as a performance metric.
- For the initial screen, use width 32, two MLP hidden layers, 30 steps, 100 maximum epochs, batch size 64 and patience 20. Equal width/epochs do **not** imply equal parameter count, FLOPs or a fair final tuning budget. Report the larger-budget sensitivity separately.
- Save safe-loadable model state, numerical quantiles, categorical maps, preprocessing, history and provenance. The common `sample` command supports these checkpoints.

Set up the source once, from the project root:

```bash
git clone https://github.com/yandex-research/tab-ddpm.git external/tab-ddpm
git -C external/tab-ddpm checkout b476257dd460b778ba09eb97f7a51d6490fa17f8
```

The clone is already present in the supplied workspace. For another location, set `TABDDPM_SOURCE` to its absolute path. The imported modules require only the project's installed NumPy and PyTorch; installing the entire upstream historical environment is unnecessary for this adapter. Upstream emits deprecation warnings in the verified modern environment; successful execution does not establish future compatibility.

Example (use a new output directory):

```bash
PYTHONPATH=src .venv/bin/python -m medhyperdiff.cli run \
  --dataset ilpd --mode tabddpm --epochs 100 --steps 30 --width 32 \
  --layers 2 --batch-size 64 --bootstraps 10 --output runs/my_tabddpm
```

## TabDiff inspection

The [official TabDiff source](https://github.com/MinkaiXu/TabDiff/tree/5ecdb3356261aea72716cc9a779f31d7ad083bf4) was cloned at commit `5ecdb3356261aea72716cc9a779f31d7ad083bf4`; its license is retained at `external/TabDiff/LICENSE`. The [ICLR 2025 paper](https://arxiv.org/abs/2410.20626) describes joint continuous-time diffusion with feature-wise learnable schedules. The inspected source implements EDM preconditioning, numerical/categorical schedules and a second-order stochastic sampler, with separate handling for conditional imputation.

TabDiff has **not** been trained in this project. A fair integration must preserve that method and specify whether synthesis is joint-label generation or conditional imputation. Renaming the current compact backbone to TabDiff would be incorrect. TabDiff and knowledge-equivalent constrained baselines remain requirements for the final comparative study.
