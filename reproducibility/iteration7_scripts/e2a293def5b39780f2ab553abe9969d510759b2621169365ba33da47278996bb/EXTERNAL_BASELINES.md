# External diffusion baseline provenance

## Current matched development comparison (iteration 7)

Native `ctgan==0.11.1` CTGAN/TVAE were fitted successfully with both compact/100-epoch and native-default/300-epoch presets, using the current ILPD fit/validation splits, class labels and 5x evaluation counts. This adds 36 adapters (72 class-specific networks) and retains all outcomes. See the [protocol](ITERATION_7_PROTOCOL.md), [results](../reports/iteration7/INTERPRETATION.md) and [commands](ITERATION_7_RUNBOOK.md). Utility rankings are mixed; full beats each default preset on two of three partitions, but compact TVAE beats full on two. Simple repaired copula has stronger numeric/association fidelity and zero observed supplied-rule violations. These class-specific native adaptations are not an exact published benchmark replication or a matched-compute/tuning comparison. TabDDPM remains an unsuccessful reproduction and TabDiff is not trained by this stage.


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
