# Iteration 12 runbook: controlled pitfall study

Protocol: [ITERATION_12_PROTOCOL.md](ITERATION_12_PROTOCOL.md). Results: [interpretation](../reports/iteration12/INTERPRETATION.md), [tables](../reports/iteration12/RESULTS.md). Manuscript: [methods_manuscript.md](../paper/methods_manuscript.md).

## Data

```bash
PYTHONPATH=src .venv/bin/python -c "from medhyperdiff.data import build_nhanes, build_thyroid; build_nhanes('data'); build_thyroid('data')"
```

`build_nhanes` downloads the four NHANES 2017–2018 public files (DEMO_J, BMX_J, BIOPRO_J, DIQ_J) and refuses files whose SHA-256 differs from the pinned values; `build_thyroid` downloads the UCI thyroid archive and records its hash. Both write `data/processed/*.csv` and `data/raw/*.manifest.json` with inclusion counts. Cite NCHS for NHANES (public domain) and the Garvan Institute / Ross Quinlan and the UCI repository (CC BY 4.0) for the thyroid data.

## Runs

```bash
export PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 OMP_NUM_THREADS=2
python scripts/iteration12.py prepare            # writes configs/iteration12_splits.json once
python scripts/iteration12.py smoke              # one tiny job per variant under runs/iteration12/smoke
python scripts/iteration12.py run --workers 5    # 240 jobs; resumable
python scripts/report_iteration12.py             # analyses 1-7, tables and figures in reports/iteration12
```

The pinned TabDDPM and TabDiff sources must be present (`TABDDPM_SOURCE`, `TABDIFF_SOURCE` override their paths). All 240 jobs used source snapshot `3aafec1cb9300f8efc734d407ee48307ceea619ddcf73a4dabff1822501a4e76`, runner `61a93edf…` and splits `3b60a107…`; `report_iteration12.py` computes the evaluation-only raw-output C2ST at 100 records per source from saved tables and caches it.

## Measured cost

On a 10-core Apple-silicon Mac with five concurrent 2-thread jobs: TabDiff ≈ 48 minutes per fit at 800 fit rows; upstream TabDDPM 9–12 minutes; corrected MedHyperDiff a few minutes; evaluation ≈ 0.6 minutes per job; about 6 hours wall time for all 240 jobs, plus 8 minutes for the report. Per-job times are in each manifest.
