# Iteration 13 runbook: relation-faithful generation

Protocol: [ITERATION_13_PROTOCOL.md](ITERATION_13_PROTOCOL.md). Results: [interpretation](../reports/iteration13/INTERPRETATION.md).

```bash
PYTHONPATH=src .venv/bin/python -c "from medhyperdiff.data import build_nhanes_cbc; build_nhanes_cbc('data')"
export PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 OMP_NUM_THREADS=2
python scripts/iteration13.py prepare             # configs/iteration13_splits.json (reference / development / confirmation)
python scripts/iteration13.py develop --workers 6 # 24 development jobs
python scripts/iteration13.py choose              # configs/iteration13_choice.json (pre-registered rule)
python scripts/iteration13.py confirm --workers 5 # 90 confirmation jobs on the reference half
python scripts/report_iteration13.py              # claim test, tables, figure
```

`build_nhanes_cbc` pins `CBC_J.xpt` by SHA-256. Confirmation used source snapshot `36533bf5…`, runner `059e5808…` and choice file `afd4798b…`; development rounds 1 and 2 are kept under `runs/iteration13/develop_v1` and `develop_v2`. TabDiff fits took about 60 minutes each at 800 fit rows on the development Mac; the full confirmation took about 5 hours with five workers.
