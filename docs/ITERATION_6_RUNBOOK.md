# Iteration 6: setup, sampler experiments and measured costs

**162 new sampling/evaluation runs, 162 reused references and zero new generator fits.** The sampler tests two fixed reverse-variance settings using 27 existing prior-initialized models. None met every correction criterion; the export command produced no replacement recipe.

## Setup

The current project already contains the environment, data, fitted models and frozen plan. For a fresh environment with those project artifacts available:

```bash
cd /Users/naren/Desktop/MedHyperDiff
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
```

The study requires the exact iteration-4 reference tables and iteration-5 prior-initialized models/tables listed in `configs/iteration6.json`. Dependency installation alone does not recreate those artifacts. Refer to the [iteration-5 training commands](ITERATION_5_RUNBOOK.md) for their construction. The core package is unchanged; the experimental sampler lives in `scripts/iteration6_sampler.py`.

## Commands executed

Run from the project root. Completed matching evaluations are reused; a resumed run is not a fresh execution-time benchmark.

| Task | Command | Behavior |
|---|---|---|
| Freeze reference hashes and design | `PYTHONPATH=src .venv/bin/python scripts/iteration6.py prepare` | Already performed; refuses to replace the frozen plan |
| Run/resume the study | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/iteration6.py run` | Generates 162 tables, evaluates them and reanalyzes shape diagnostics for 162 references |
| Rebuild tables and selection | `PYTHONPATH=src .venv/bin/python scripts/report_iteration6.py` | Applies both-reference gates, including tails and the AST target |
| Audit the study | `PYTHONPATH=src .venv/bin/python scripts/audit_iteration6.py` | Checks 324 records, all AUROCs, hashes, split isolation, variance/quantile/tail/missingness measures and six exact replays |
| Rebuild figures | `MPLCONFIGDIR=/private/tmp/medhyperdiff-mpl XDG_CACHE_HOME=/private/tmp/medhyperdiff-cache PYTHONPATH=src .venv/bin/python scripts/plot_iteration6.py` | Exports PNG, PDF and SVG |
| Conditional recipe export | `PYTHONPATH=src .venv/bin/python scripts/export_iteration6.py` | Correctly exports nothing because no full setting qualifies |
| Run tests | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m pytest -q --disable-warnings` | 31 passed, with 21 upstream deprecation warnings |

## Explicit experimental sampling

The ordinary CLI does not implement this variance override. Use the separate command below to investigate it; these settings worsened supplied-rule compliance and are not recommended as a replacement model:

```bash
PYTHONPATH=src .venv/bin/python scripts/sample_iteration6.py \
  --checkpoint runs/iteration5/training/full_prior_i2027_t0/checkpoint.pt \
  --variance-mix 0.5 --guidance 10 \
  --positive-count 200 --negative-count 100 --seed 70123 \
  --output runs/my_variance_experiment_300.csv
```

Use a fresh filename. The command writes predictors plus `condition_positive`, where 1 corresponds to original ILPD `disease == "1"`, and a companion provenance manifest. `--variance-mix 0` reproduces the existing sampler; 0.5 and 1 are the tested alternatives. Positive/negative labels are ordered in blocks for this CLI example, so it does not reproduce a study table, which repeats the fit-label vector. Six exact study replays are checked by the audit.

A real CLI smoke run generated `reports/iteration6/examples/full_m0p5_10.csv` with five rows per class and seed 61123; this demonstration is separate from the 162 research evaluations. No fitting occurs in this command, and weights are not rewritten.

## Measured resources

| Operation | Count / size | Time per operation | Sum | GPU VRAM |
|---|---|---:|---:|---:|
| New generator fitting | 0 | — | 0 | 0 GiB |
| Generate 5x tables | 162 × 1,745/1,750 rows | 1.02–1.44 s | 193.6 s | 0 GiB |
| Evaluate new tables and shape | 162 | 0.86–1.02 s | 147.2 s | 0 GiB |
| Reused reference records | 162 | Read and recompute shape diagnostics | No model refitting/resampling | 0 GiB |
| Software suite | 31 tests | 3.38 s total | 3.38 s | 0 GiB |

Recorded new-generation/evaluation components sum to about **5.7 minutes** on the local CPU with two PyTorch threads. This excludes loading, reference-shape reanalysis, audits, reports and other overhead; it is not total elapsed study wall time. Peak system RAM and GPU performance were not measured. GPU VRAM being zero does not mean CPU memory use is zero.

[Interpretation](../reports/iteration6/INTERPRETATION.md), [complete results](../reports/iteration6/RESULTS.md), [audit](../reports/iteration6/ARTIFACT_AUDIT.json), [frozen protocol](ITERATION_6_PROTOCOL.md).
