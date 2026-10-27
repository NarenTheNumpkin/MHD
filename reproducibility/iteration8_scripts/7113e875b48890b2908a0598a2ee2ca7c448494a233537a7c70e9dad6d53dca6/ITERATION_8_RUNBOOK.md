# Iteration 8: setup, commands and measured costs

This stage tested one fit-only asinh-MAD numerical representation. It trained nine experimental MedHyperDiff models, generated 27 new tables and reused 36 existing references. It does not alter the core package or exported sampler.

## Setup

```bash
cd /Users/naren/Desktop/MedHyperDiff
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
```

The historical iteration-5 full_prior checkpoints and iteration-7 repaired-copula tables listed in `configs/iteration8.json` are prerequisites. Dependency installation alone does not recreate them.

## Commands

| Operation | Command | Result |
|---|---|---|
| Freeze protocol | `PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/iteration8.py prepare` | Frozen before fitting; refuses overwrite |
| Run/resume models | `PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/iteration8.py run` | 9 fits and 27 evaluations |
| Aggregate results | `PYTHONPATH=src:scripts .venv/bin/python scripts/report_iteration8.py` | Partition/feature tables and timings |
| Audit artifacts | `PYTHONPATH=src:scripts .venv/bin/python scripts/audit_iteration8.py` | Hashes, split isolation, physical-rule metrics, 9 exact replays |
| Run tests | `PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m pytest -q` | 36 passed, 21 upstream warnings |

The experiment stores trusted whole-object CPU checkpoints because the robust preprocessor and physical-unit energy are experimental classes. Their checksums are verified before loading; the same Python/module versions are required.

## Measured components

| Operation | Count | Component time | GPU VRAM |
|---|---:|---:|---:|
| Transformed diffusion fits | 9 | 52.5 s total | 0 GiB |
| Sampling | 27 tables | 38.4 s total | 0 GiB |
| Evaluation | 27 tables | 25.7 s total | 0 GiB |
| Combined measured components | — | about 1.9 min | 0 GiB |

Peak system RAM and GPU performance were not measured. These times are local CPU components, not guaranteed wall time on another machine.

The experiment is a retained negative result: source-unit inverse decoding produced extreme bilirubin values and supplied-rule violations around 21–25%. Do not export it or treat its mixed forest AUROC as a win.
