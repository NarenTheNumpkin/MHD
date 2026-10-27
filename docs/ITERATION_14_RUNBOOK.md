# Iteration 14 runbook: robustness of relation-faithful generation

Protocol: [ITERATION_14_PROTOCOL.md](ITERATION_14_PROTOCOL.md). Results: [interpretation](../reports/iteration14/INTERPRETATION.md).

```bash
export PYTHONPATH=src:scripts OMP_NUM_THREADS=1
python scripts/iteration14.py prepare               # builds six NHANES populations (SHA-256 pinned) and configs/iteration14_splits.json
python scripts/iteration14.py queue --workers 8     # 60 development jobs first, then 792 comparator, plug-in, control and c0 jobs
python scripts/iteration14.py choose                # configs/iteration14_choice.json (pre-registered rule)
python scripts/iteration14.py confirm --workers 4 --arms mhd   # chosen configuration on 66 cells + iteration 13 population
python scripts/report_iteration14.py                # hypotheses, pooled differences, tables and figures
```

**Data.** `build_nhanes_cycle(root, panel, cycle)` downloads the public XPT files for cycles G (2011–2012), H (2013–2014), I (2015–2016) and L (2021–2023). It checks the SHA-256 values recorded on 10 October 2026.

**Execution.**
- All fits run on CPU with one thread per job.
- TabDiff is the bottleneck. On the development Mac (Apple M4, 4 performance and 6 efficiency cores), total throughput saturates at about 16 training epochs per second at 800 fit rows, whatever the number of concurrent jobs.
- Jobs are resumable: completed manifests are skipped.
