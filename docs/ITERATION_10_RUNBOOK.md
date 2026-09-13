# Iteration 10: setup and commands

This stage runs the current full model on CKD, heart failure and diabetes. It fits nine models, generates 27 tables and keeps all outer tests closed.

```bash
cd /Users/naren/Desktop/MedHyperDiff
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
```

The processed public tables under `data/processed/` are prerequisites. Run:

| Operation | Command | Result |
|---|---|---|
| Freeze design | `PYTHONPATH=src:scripts .venv/bin/python scripts/iteration10.py prepare` | Freezes 9 fits, 27 evaluations and dataset hashes |
| Run/resume fits | `PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/iteration10.py run` | Fits and evaluates all three cohorts |
| Aggregate results | `PYTHONPATH=src:scripts .venv/bin/python scripts/report_iteration10.py` | Dataset/seed tables and timings |
| Audit artifacts | `PYTHONPATH=src:scripts .venv/bin/python scripts/audit_iteration10.py` | Hashes, replay, labels and split isolation |
| Run software checks | `PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m pytest -q` | Full suite |

GPU VRAM is 0 GiB in the recorded CPU run. Peak system RAM and GPU time are not measured; local component timings are reported in `reports/iteration10/timing.json`.

The first CKD table was an incomplete preflight attempt: numeric-looking categorical levels were inferred as floats after CSV round-trip, and the evaluator rejected mixed categorical types. The table is preserved under `runs/iteration10/preflight_failed/`, and the accepted rerun forces every categorical column to string before evaluation. This is recorded in `configs/iteration10.json`; it is not counted as a completed research evaluation.
