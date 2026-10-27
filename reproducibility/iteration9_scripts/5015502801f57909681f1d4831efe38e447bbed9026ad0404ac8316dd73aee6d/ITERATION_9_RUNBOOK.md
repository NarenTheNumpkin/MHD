# Iteration 9: setup and commands

Iteration 9 performs no fitting and no sampling. It re-evaluates the 225 saved iteration-7 synthetic tables under five frozen rule registries.

```bash
cd /Users/naren/Desktop/MedHyperDiff
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
```

The iteration-7 outputs listed in `configs/iteration7.json` are prerequisites. Run:

| Operation | Command | Result |
|---|---|---|
| Freeze hashes and variants | `PYTHONPATH=src:scripts .venv/bin/python scripts/iteration9.py prepare` | Records 225 source tables and five rule variants |
| Re-evaluate saved tables | `PYTHONPATH=src:scripts .venv/bin/python scripts/iteration9.py run` | 1,125 variant-table evaluations; no fitting/sampling |
| Aggregate sensitivity | `PYTHONPATH=src:scripts .venv/bin/python scripts/report_iteration9.py` | Partition/arm changes and rule bounds |
| Audit outputs | `PYTHONPATH=src:scripts .venv/bin/python scripts/audit_iteration9.py` | Hashes, labels, splits and correct-variant replay |
| Run software checks | `PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m pytest -q` | Full suite |

The local CPU cost is dominated by reading 225 tables and evaluating five small rule sets. GPU VRAM is 0 GiB; peak system RAM is not measured. These figures are analysis-component time, not a model-training benchmark.
