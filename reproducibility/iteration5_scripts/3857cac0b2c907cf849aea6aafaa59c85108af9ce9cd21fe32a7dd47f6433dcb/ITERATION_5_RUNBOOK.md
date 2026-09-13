# Iteration 5: training corrections, setup and reproducibility

This stage trains **36 new generator models from scratch** and generates/evaluates **108 new synthetic tables**. It also reuses 81 matched reference evaluations from iteration 4, for 189 evaluated tables total. New models initialize variable observation-head probabilities from fit-only frequencies; one full-model variant additionally multiplies observation-token CE by four. Numerical sampling, clinical rules and the core package are unchanged.

## Setup

The current workspace has the required environment and cached artifacts. For a fresh environment with the project data and archived reference checkpoints available:

```bash
cd /Users/naren/Desktop/MedHyperDiff
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
```

The experimental training class is in `scripts/iteration5_model.py`; the existing core CLI does not expose its training options. The study runner imports it explicitly. Existing matched parents are used to recover the original preprocessing, graph, rules, architecture and training settings; their learned weights are not used as initialization. Installing dependencies alone does not recreate these parent artifacts. See `configs/iteration5.json` for paths and checksums.

## Commands

Run from the project root. The frozen plan is already prepared and must not be overwritten.

| Task | Command | What it does |
|---|---|---|
| Freeze study | `PYTHONPATH=src .venv/bin/python scripts/iteration5.py prepare` | Already performed before fitting; refuses to replace the plan |
| Train, generate and evaluate | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python scripts/iteration5.py run` | Runs/resumes the 36-fit, 108-evaluation matrix and references the 81 historical controls |
| Rebuild all comparisons | `PYTHONPATH=src .venv/bin/python scripts/report_iteration5.py` | Computes seed summaries, per-feature/rule results and the frozen correction gates |
| Audit artifacts | `PYTHONPATH=src .venv/bin/python scripts/audit_iteration5.py` | Checks fit-only frequencies, checkpoint/source hashes, matched settings, test exclusion, all saved AUROCs, bounds and variance ratios; replays three tables |
| Rebuild figures | `MPLCONFIGDIR=/private/tmp/medhyperdiff-mpl XDG_CACHE_HOME=/private/tmp/medhyperdiff-cache PYTHONPATH=src .venv/bin/python scripts/plot_iteration5.py` | Writes scientific PNG, PDF and SVG files |
| Test software | `PYTHONPATH=src LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m pytest -q --disable-warnings` | Checks the existing model plus the isolated training changes |

Completed matching outputs are reused; rerunning the matrix is not a fresh benchmark. Keep new study designs in separate directories with a new frozen protocol. All generation uses five repetitions of the original fit-label vector, 256-row sampling batches, seeds 70001–70003 and evaluator seed 2026. The generated tables contain 1,745 or 1,750 rows. Fivefold size is a matched evaluation budget here, not a change to the proposed model's recommended size.

## Saved models and interpretation

New fitted weights are in `runs/iteration5/training/<variant>_i<inner>_t<training_seed>/checkpoint.pt`. Their experimental metadata records empirical initialization counts and observation-loss weight. The unmodified sampler can load these weights; the study then explicitly sets guidance 10 and the adaptive schedule. A raw training checkpoint can retain its original zero-guidance training configuration, so do not assume that sampling it with default CLI settings reproduces the evaluated guided table.

`reports/iteration5/selection.json` records whether either full variant passes all declared correction gates. Missingness improvement alone does not justify a replacement: utility, fidelity, rule coverage and numerical variability must also pass. This remains development evidence on overlapping inner partitions, without independent test results, clinical validation or formal privacy guarantees.

Neither full variant passed every gate. `PYTHONPATH=src .venv/bin/python scripts/export_iteration5.py` was executed and correctly exported no replacement. The new checkpoints remain accessible for research; the previous exported development sampler remains separate.

To generate from the newly trained, prior-initialized full model with the evaluated guidance settings, use the following research example. This setting failed one numerical-variability guardrail, so this is not a recommendation to replace the existing model:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
import torch
from medhyperdiff.diffusion import MixedDiffusion, guidance_schedule

torch.set_num_threads(2)
output = Path('runs/my_prior_initialized_ilpd_300.csv')
if output.exists():
    raise FileExistsError(output)
model = MixedDiffusion.load('runs/iteration5/training/full_prior_i2027_t0/checkpoint.pt')
model.config.guidance = 10.0
model.config.schedule = 'adaptive'
model.schedule = guidance_schedule(model.config.steps, 'adaptive')
labels = np.array([1]*200 + [0]*100)
table = model.sample(labels, seed=70123, batch_size=256)
table['condition_positive'] = labels
table.to_csv(output, index=False)
PY
```

The output is 300 synthetic predictor rows plus `condition_positive`, where 1 denotes original `disease == "1"`. This example has a different class order and sampling seed from the study, so it does not replay an evaluated table. The audit command performs exact study replays. No training occurs in this generation command.

## Measured resources

| Operation | Count / size | Time per operation | Sum | GPU VRAM |
|---|---|---:|---:|---:|
| New model fitting | 36 fits, up to 100 epochs | 1.58–8.30 s | 201.7 s | 0 GiB |
| Generate guided 5x tables | 108 × 1,745/1,750 rows | 1.23–2.35 s | 169.3 s | 0 GiB |
| Evaluate new tables | 108 | 0.97–1.26 s | 113.4 s | 0 GiB |
| Reused reference evaluations | 81 | Loaded existing artifacts | No refitting or resampling | 0 GiB |
| Software tests | 27 tests | 4.05 s total | 4.05 s | 0 GiB |

Fit, sample and evaluation components sum to about **8.1 minutes** on the local CPU, with two PyTorch threads in the study runner. This excludes initialization, checkpoint loading, reports/audits and other overhead; it is not total elapsed study wall time. Fit timing excludes reused structure estimation. Peak system RAM and GPU performance were not measured. Different hardware and model sizes change these costs.

Results: [interpretation](../reports/iteration5/INTERPRETATION.md), [all results](../reports/iteration5/RESULTS.md), [artifact audit](../reports/iteration5/ARTIFACT_AUDIT.json).
