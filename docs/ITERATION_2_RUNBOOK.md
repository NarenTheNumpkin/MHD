# Setup and reproduction for iteration 2

Run from `/Users/naren/Desktop/MedHyperDiff`. This workspace already has the environment, data, trained checkpoints and pinned external source. Setup below is for a fresh checkout/environment.

```bash
cd /Users/naren/Desktop/MedHyperDiff
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
PYTHONPATH=src .venv/bin/python -m pytest -q --disable-warnings
PYTHONPATH=src .venv/bin/python -m medhyperdiff.cli download
```

For the optional official-source baseline, follow [the pinned source setup](EXTERNAL_BASELINES.md). No GPU is required for these small cohorts. CUDA/MPS training has not been validated here, and GPU memory/time estimates are not measurements.

## What was actually run

The immutable configuration files specify every argument and output directory. All 110 experiments below completed, using inner validation only. Source and split isolation were audited after completion.

| Experiment stage | Command executed with `PYTHONPATH=src` | Runs | Measured fit time per run | GPU VRAM |
|---|---|---:|---:|---:|
| Initial 12-arm screen on ILPD and simulator | `.venv/bin/python -m medhyperdiff.cli batch --config configs/iteration2_screen.json` | 24 | 1.0–6.5 s | 0 |
| Six retained arms, generator seeds 1/2 | `.venv/bin/python -m medhyperdiff.cli batch --config configs/iteration2_replication.json` | 24 | 1.5–7.8 s | 0 |
| Larger TabDDPM, both datasets × three seeds | `.venv/bin/python -m medhyperdiff.cli batch --config configs/iteration2_external.json` | 6 | 2.9–6.4 s | 0 |
| Additional inner partitions and matched dense controls | `.venv/bin/python -m medhyperdiff.cli batch --config configs/iteration2_inner.json` | 54 | 1.4–8.4 s | 0 |
| Extended TabDDPM diagnostic, 8,500 epochs | `.venv/bin/python -m medhyperdiff.cli batch --config configs/iteration2_external_diagnostic.json` | 2 | 72.8–94.3 s | 0 |

Fit includes graph estimation, excludes sampling/evaluation, and was measured with two CPU threads per process. Some stages ran concurrently. Summed fit time was 684.4 seconds and summed sampling time was 32.7 seconds; these sums are not elapsed wall time. Setup/download time is separate. Most internal samplers took around 0.1–0.3 seconds for one fit-sized table. A previous single ILPD command measured 9.98 seconds end-to-end, but that is not a guarantee for other hardware or configurations. Peak system RAM was not measured.

Do not rerun the original immutable configs directly into completed output directories: the runner intentionally refuses to overwrite them. To prepare fresh paths without training anything:

```bash
.venv/bin/python scripts/prepare_iteration2_runs.py
PYTHONPATH=src .venv/bin/python -m medhyperdiff.cli batch \
  --config configs/user_runs/iteration2_screen.json
```

The helper also prepares `iteration2_replication.json`, `iteration2_external.json`, `iteration2_inner.json` and `iteration2_external_diagnostic.json` under `configs/user_runs/`. The extended diagnostic is optional; its observed instability means running it is not a shortcut to a valid baseline.

## Train one proposed model yourself

Choose a fresh output path. This is the paired full x0 configuration used in the comparisons, not a declared final winner:

```bash
PYTHONPATH=src .venv/bin/python -m medhyperdiff.cli run \
  --dataset ilpd --mode full --paired-initialization \
  --epochs 100 --steps 30 --width 32 --layers 2 --batch-size 64 \
  --bootstraps 10 --seed 0 --fold 0 --split-seed 2026 \
  --output runs/my_ilpd_full_iteration2
```

Use `--mode base` for its backbone control, `--mode dense` for the matched-incidence-capacity control, or add `--numeric-prediction v` to test velocity-full. `--gated-structure` is optional and did not consistently improve the model. Set `--inner-seed 3027` to use the specified alternative development split while preserving reserved test patients. Do not add `--evaluate-test` during tuning.

## Generate synthetic records from an already trained model

This uses an existing trained full-model checkpoint and does **not** train again:

```bash
PYTHONPATH=src .venv/bin/python -m medhyperdiff.cli sample \
  --checkpoint runs/iteration2/screen/ilpd_paired_full_s0/checkpoint.pt \
  --positive-count 200 --negative-count 100 --seed 9 \
  --output runs/my_300_synthetic_ilpd_rows.csv
```

Output contains predictors and `condition_positive`: 1 means ILPD's positive class (`disease == "1"`), 0 its negative class. These research samples are not clinically certified patient records. Class counts are chosen by the user in `sample`; experiment runs use fit-partition prevalence.

## Verify and regenerate the retained report

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_iteration2.py
PYTHONPATH=src .venv/bin/python scripts/report_iteration2.py
PYTHONPATH=src .venv/bin/python scripts/diagnose_tabddpm.py
```

The report script reads the original `runs/iteration2` tree. For fresh-path reruns, the generic `scripts/summarize_runs.py --root PATH_TO_STAGE --output PATH_TO_REPORT` summarizes an individual stage. Read [interpretation and decisions](../reports/iteration2/INTERPRETATION.md) before treating any metric as evidence for a paper.
