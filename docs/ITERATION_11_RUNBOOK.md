# Iteration 11 runbook

Protocol: [ITERATION_11_PROTOCOL.md](ITERATION_11_PROTOCOL.md). Frozen plan: `configs/iteration11.json`. Source snapshot used by the ablation and four-method search: `reproducibility/source_snapshots/9739e858bc267fee430689d74988ad5e86e8ea54b1e788d30bcf6a1f8e399bd8`. The only later source change appends `TabDiffAdapter` to `external_baselines.py` (verified by diff; no existing line changed).

## Environment

Python 3.13 with `requirements-lock.txt`. The pinned upstream baselines must be present and unmodified:

```bash
git -C external/tab-ddpm rev-parse HEAD   # b476257dd460b778ba09eb97f7a51d6490fa17f8
git -C external/TabDiff rev-parse HEAD    # 5ecdb3356261aea72716cc9a779f31d7ad083bf4
```

On the development Mac, opening many files under `~/Desktop` was extremely slow (≈0.4 s per file; importing pandas from the project `.venv` took ≈6 minutes). The study was therefore executed from a byte-identical copy of `src/`, `scripts/` and `data/` outside the Desktop, with a fresh environment built from the lock file (`uv venv` then `uv pip install -r requirements-lock.txt`, about one minute). Outputs can be redirected with `MHD_OUT` (runs), `MHD_REPORT` (reports) and `MHD_LEGACY_RUNS` (archived iteration-5/7 tables for re-scoring). Every job manifest records the source, runner and data SHA-256.

## Stages

All runners skip completed job directories, so an interrupted stage can be relaunched with the same command.

```bash
export PYTHONPATH=src:scripts LOKY_MAX_CPU_COUNT=2 OMP_NUM_THREADS=2
python scripts/iteration11.py ablation --workers 5          # 72 fits, ILPD development
python scripts/iteration11.py search --workers 5            # 384 fits: MedHyperDiff, TabDDPM, CTGAN, TVAE
python scripts/iteration11.py search_tabdiff --workers 2    # 96 fits: TabDiff (Amendment 1)
python scripts/iteration11_confirm.py mechanism --workers 1 # 54 fits, controlled simulator
python scripts/iteration11_confirm.py select                # writes configs/iteration11_selection.json once
python scripts/iteration11_confirm.py guidance              # writes configs/iteration11_guidance.json once
python scripts/iteration11_confirm.py confirm --workers 5   # opens outer test fold 0 of every dataset
python scripts/report_iteration11.py ablation mechanism search confirm
python -m pytest -q
```

`select` and `guidance` refuse to overwrite their frozen outputs, and `confirm` refuses to start until both exist; its manifests record their hashes. Only `confirm` evaluates on the reserved outer test partition.

## Measured cost (CPU, 2 threads per job, Apple M-series, 10 cores shared by 5–8 concurrent jobs)

- Corrected MedHyperDiff (5,000 steps, width 32, two layers, T = 100): ≈85–95 s per fit; legacy recipe ≈9 s.
- TabDDPM upstream recipe: from ≈2 minutes (5k steps) to ≈29 minutes (30k steps, 8×1024 MLP) per fit under load.
- TabDiff upstream defaults: ≈62 ms per epoch under load, ≈10 minutes per 8,000-epoch fit plus about a minute of sampling.

Exact per-job fit and sampling times are in each `manifest.json` and in `reports/iteration11/*_all.csv`.
