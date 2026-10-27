# Iteration 4 setup, experiments and measured costs

This stage uses previously fitted ILPD checkpoints. **There are zero new generator training runs, 108 new sample pools and 216 evaluations.** It tests sample size and separates training/sampling randomness. It does not train on reserved test records.

## Setup

The current workspace already has a working environment, data, 36 required parents and the frozen plan. Do not reinstall to inspect existing results. For a fresh environment with the project and its checkpoint artifacts available:

```bash
cd /Users/naren/Desktop/MedHyperDiff
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
```

The actual local experiments use Python 3.13, CPU and two PyTorch threads. All commands below run from the project root. Prefixing with `PYTHONPATH=src` selects the checked project source. Downloading data or installing dependencies alone does not reproduce trained parent checkpoints; their paths and SHA-256 hashes are listed in `configs/iteration4.json`. The prior training commands are in [iteration 3](ITERATION_3_RUNBOOK.md) and the exact parent manifests. This runner fails if required parents are missing; it never silently trains replacements.

## Commands executed and commands to use

| Purpose | Command | Behavior now |
|---|---|---|
| Freeze parents and protocol | `PYTHONPATH=src .venv/bin/python scripts/iteration4.py prepare` | Already executed; deliberately refuses to replace the frozen plan |
| Run/resume the study | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/iteration4.py run` | Already complete; checks and reuses finished outputs |
| Rebuild result tables | `PYTHONPATH=src .venv/bin/python scripts/report_iteration4.py` | Recomputes summaries and declared gates from saved metrics |
| Audit all 216 evaluations | `PYTHONPATH=src .venv/bin/python scripts/audit_iteration4.py` | Checks parent/source hashes, exact nested rows, original test exclusion, all saved AUROCs and rule bounds; regenerates one pool exactly |
| Rebuild scientific figures | `MPLCONFIGDIR=/private/tmp/medhyperdiff-mpl XDG_CACHE_HOME=/private/tmp/medhyperdiff-cache PYTHONPATH=src .venv/bin/python scripts/plot_iteration4.py` | Writes PNG, PDF and SVG |
| Run software tests | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m pytest -q --disable-warnings` | 23 passed; 21 upstream deprecation warnings |

For a new scientifically distinct experiment, create a separate plan/output stage and archive its protocol before sampling; do not edit the frozen iteration-4 plan or mix new results into these 216 records. Existing cached runs are not a fresh execution-time benchmark.

## Generate a new table using the trained proposed model

The sampler exported in iteration 3 remains available. This stage found no consistent benefit from making 5x synthesis its default and did not export newly trained weights.

```bash
PYTHONPATH=src .venv/bin/python -m medhyperdiff.cli sample \
  --checkpoint artifacts/iteration3/ilpd_full_v_l0p01_g10_i2027_s0.pt \
  --positive-count 200 --negative-count 100 --seed 70123 \
  --output runs/my_ilpd_iteration4_300.csv
```

Use an unused output filename. This generates 300 rows from fitted weights; it does not retrain. The CLI orders positive and negative labels in blocks, whereas the study repeats the original fit-label order. Therefore the CLI command is a usage example, not an exact replay of a study table. The study audit verifies exact replay using the archived label order and batch size.

## Measured resource use

| Operation | Count / size | Time per operation | Sum | GPU VRAM |
|---|---|---:|---:|---:|
| New generator training | 0 | — | 0 | 0 GiB |
| Generate a five-fit-size pool | 108 pools; 1,745 or 1,750 rows each | 1.01–1.31 s | 131.1 s | 0 GiB |
| Evaluate 1x or 5x table | 216 tables; 349/350 or 1,745/1,750 rows | 0.54–0.95 s | 155.8 s | 0 GiB |
| Entire software test suite | 23 tests | 2.86 s total | 2.86 s | 0 GiB |

Generation plus evaluation component times sum to about **4.8 minutes**. This excludes startup, checkpoint loading, report/audit work and other overhead; it is not measured end-to-end elapsed study time. Hardware and concurrent load change runtime. Peak system RAM and CUDA/MPS performance were not measured; CPU memory use is not zero simply because GPU VRAM is zero.

Results: [interpretation](../reports/iteration4/INTERPRETATION.md), [all partition means](../reports/iteration4/RESULTS.md), [artifact audit](../reports/iteration4/ARTIFACT_AUDIT.json).
