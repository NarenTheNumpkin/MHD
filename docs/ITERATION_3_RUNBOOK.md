# Iteration 3 commands and trained samplers

Run from `/Users/naren/Desktop/MedHyperDiff`. The existing `.venv`, cached data and checkpoints are sufficient. For a fresh installation, use the [setup instructions](ITERATION_2_RUNBOOK.md). CPU was used throughout; GPU VRAM consumption by training was zero. Peak system RAM and GPU performance were not measured.

## Generate data now, without retraining

The coefficient-0.01 velocity candidate improves numerical fidelity and rule compliance relative to the earlier full model. It is a development candidate with remaining utility tradeoffs.

```bash
cd /Users/naren/Desktop/MedHyperDiff
PYTHONPATH=src .venv/bin/python -m medhyperdiff.cli sample \
  --checkpoint artifacts/iteration3/ilpd_full_v_l0p01_g10_i2027_s0.pt \
  --positive-count 200 --negative-count 100 --seed 9 \
  --output runs/my_iteration3_ilpd_300.csv
```

Choose a new output filename. The output contains ILPD predictors plus `condition_positive`: 1 corresponds to original `disease == "1"`, 0 to its negative class. This command uses fitted weights and the saved guidance setting; it does not train a model.

## Train that configuration yourself

```bash
PYTHONPATH=src .venv/bin/python -m medhyperdiff.cli run \
  --dataset ilpd --mode full --numeric-prediction v \
  --clinical-weight 0.01 --guidance 10 --paired-initialization \
  --epochs 100 --steps 30 --width 32 --layers 2 --batch-size 64 \
  --bootstraps 10 --seed 0 --inner-seed 2027 \
  --fold 0 --split-seed 2026 --output runs/my_iteration3_ilpd_training
```

Use a new output directory. Training energy and sampling guidance are separate: the former affects fitting, while the latter changes generation. The run saves a checkpoint, synthetic table and validation metrics. Do not add `--evaluate-test` while making model choices.

## Experiments and audits performed

All commands below use `PYTHONPATH=src .venv/bin/python` before the displayed script and arguments.

| Stage | Script and arguments | Purpose |
|---|---|---|
| Screen | `scripts/iteration3.py screen` | 84 settings on seed 0; reuse matching fitted checkpoints |
| Freeze shortlist | `scripts/report_iteration3.py --select` | Apply declared screening conditions and save immutable selection |
| Replicate | `scripts/iteration3.py replicate` | Selected settings, references and dense controls across three seeds and three inner partitions |
| Guided controls | `scripts/iteration3_controls.py` | Backbones with the same guidance and the original x0 backbone |
| Report | `scripts/report_iteration3.py` | Full CSVs, partition means, seed variation and figures |
| Replication figure | `scripts/plot_iteration3_replication.py` | Means and seed SD across all three ILPD partitions |
| Audit | `scripts/audit_iteration3.py` | Verify weights, splits, source hashes, counts and saved prediction metrics |
| Export | `scripts/export_iteration3.py` | Package the selected samplers; already completed, refuses overwrites |

Screen/replication/control commands resume completed evaluations with matching configuration, package source and parent checkpoint hashes. They do not overwrite earlier completed evaluations. Training commands likewise reuse exactly matching configurations, data and splits. The saved stage registries distinguish shared evaluations from newly generated ones. Do not describe the 174 evaluations as 174 new model trainings.

| Measured operation | Count | Time per operation | Sum of recorded time |
|---|---:|---:|---:|
| New model fitting, including graph estimation | 46 | 2.5–7.1 s | 274.1 s |
| Sampling for distinct sweep evaluations | 174 | 0.11–0.67 s | 40.5 s |
| Metrics for those evaluations | 174 | 0.50–0.95 s | 98.9 s |

These are local CPU measurements with two threads per process. Sums exclude process startup and the additional automatic zero-guidance evaluation written by each training command, and are not total elapsed study wall time. Reuse of 48 previously fitted checkpoints saved retraining. There were 94 distinct fitted checkpoints used in total. Different hardware, sample counts and model sizes change costs.

For exact reproduction of rank metrics, read `utility_predictions.csv` with Pandas `float_precision="round_trip"`. For interpretation, including the failed screening/replication conditions, read [the development report](../reports/iteration3/INTERPRETATION.md).
