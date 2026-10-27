# Iteration 7: setup, executed commands and measured costs

This stage trains reference generators and compares the existing full_prior model. It performs zero new MedHyperDiff fits. The default exported sampler is unchanged.

## Setup

The current workspace already has its environment, data, frozen protocol and completed outputs. For a fresh environment with the project and historical reference artifacts available:

```bash
cd /Users/naren/Desktop/MedHyperDiff
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
```

Exact iteration-5 full_prior/dense_prior/base_prior checkpoints, splits and evaluation artifacts listed in `configs/iteration7.json` are prerequisites; installing dependencies does not recreate them. Their construction is documented in the [iteration-5 runbook](ITERATION_5_RUNBOOK.md). `ctgan==0.11.1` is installed and included in the recorded environment. Whole-object CPU baseline checkpoints require the same module/dependency versions and trusted local provenance; the loader verifies the saved checksum before deserializing.

## Commands executed and available to rerun

Run from the project root. Completed outputs are checked and reused; a resumed command does not measure fresh training cost.

| Operation | Command | Result |
|---|---|---|
| Freeze protocol and reference hashes | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/iteration7.py prepare` | Already completed; refuses to overwrite the frozen plan |
| Run or resume all baseline fits/evaluations | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/iteration7.py run` | 36 neural adapter fits, 9 statistical fits, 144 new evaluations |
| Rebuild comparison tables | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/report_iteration7.py` | 225 comparison records and separate fit/sampling variation |
| Audit artifacts and replay checkpoints | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/audit_iteration7.py` | 225 records, 1,350 recalculated AUROCs, 45 exact checkpoint replays and 9 repairs |
| Recompute supplemental copy/boundary checks | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/supplement_iteration7.py` | Numerical tolerance diagnostics for all 225 records; bootstrap positive control passes |
| Rebuild scientific figures | `MPLCONFIGDIR=/private/tmp/medhyperdiff-mpl XDG_CACHE_HOME=/private/tmp/medhyperdiff-cache PYTHONPATH=src .venv/bin/python scripts/plot_iteration7.py` | PNG, PDF and SVG |
| Run software checks | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m pytest -q` | 34 passed, 21 upstream warnings; 2.77 seconds |

The study runner first generates raw copula tables, then repairs their saved round-trip values. Native generators use separate per-label models and restore the exact requested class-label order. The 100-epoch compact and 300-epoch native-default presets intentionally have different network dimensions and batch sizes; report both rather than selecting an easier comparator. No new data download, GPU rental or external service is needed to rerun the retained local study.

## Measured CPU costs

Each neural adapter contains two class-specific networks; times below include fitting both. Two PyTorch CPU threads were used. Statistical fitting stores fit/preprocessing state; copula correlation work also occurs during sampling.

| Operation | Count | Measured time each | Total component time | GPU VRAM |
|---|---:|---:|---:|---:|
| CTGAN compact, 100 epochs | 9 adapters | 1.81–2.44 s | 17.1 s | 0 GiB |
| CTGAN native defaults, 300 epochs | 9 adapters | 7.42–8.27 s | 68.3 s | 0 GiB |
| TVAE compact, 100 epochs | 9 adapters | 1.39–1.45 s | 12.8 s | 0 GiB |
| TVAE native defaults, 300 epochs | 9 adapters | 1.82–1.91 s | 16.7 s | 0 GiB |
| Copula / independent / bootstrap fit | 9 objects | 0.0015–0.0019 s | 0.015 s | 0 GiB |
| Sampling and repair | 135 tables + 9 derivatives | Model dependent | 4.8 s | 0 GiB |
| New table evaluation | 144 | About 0.88 s average | 127.0 s | 0 GiB |
| New MedHyperDiff training | 0 | — | 0 | 0 GiB |

Recorded components sum to approximately **4.1 minutes**. This excludes auditing, reporting, deserialization and other overhead; full elapsed wall time was not measured. More epochs with a larger batch can yield fewer optimizer updates, especially on these small per-label subsets. Peak system RAM and GPU time/VRAM requirements were not measured; do not extrapolate these ILPD CPU timings to larger datasets or architectures.

[Findings](../reports/iteration7/INTERPRETATION.md), [complete tables](../reports/iteration7/RESULTS.md), [frozen protocol](ITERATION_7_PROTOCOL.md), [diagnostic addendum](ITERATION_7_DIAGNOSTIC_ADDENDUM.md), [current publication priorities](PUBLICATION_READINESS.md).
