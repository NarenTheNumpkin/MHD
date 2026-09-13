# MedHyperDiff

Research prototype for **confidence-weighted hypergraph-guided mixed-type medical table synthesis**. The project includes a corrected research protocol, public-data audit, runnable models, leakage controls, tests and an exploratory experiment runner.

**Status:** implemented and tested prototype. Publication-quality superiority, clinical validity, calibrated uncertainty and privacy protection are not established. Development runs use inner validation; outer test metrics require an explicit flag.

**Latest development:** [iteration-7 findings](reports/iteration7/INTERPRETATION.md), [all matched baseline results](reports/iteration7/RESULTS.md), and [setup, commands and measured costs](docs/ITERATION_7_RUNBOOK.md). This stage adds **36 CTGAN/TVAE adapter fits (72 class networks), nine statistical fits and 144 new evaluations against 81 reused references; zero new MedHyperDiff fits**. Utility rankings are mixed. Explicitly repaired copula has better numeric/association fidelity and zero observed supplied-rule violations across all three development partitions. A supplemental numerical-tolerance check corrects misleading literal copy/boundary diagnostics: bootstrap resampling is correctly identified as 100% training copies. **34 tests and all 225 artifact checks pass.** Reserved tests remain untouched; no new sampler is promoted.

The [iteration-6 variance study](reports/iteration6/INTERPRETATION.md) added 162 sampling evaluations without new fits and rejected uniform variance inflation. The [iteration-5 training correction](reports/iteration5/INTERPRETATION.md) trained 36 new models and added 108 evaluations, correcting excess marginal missingness but exposing an AST shape tradeoff. The [iteration-4 size study](reports/iteration4/INTERPRETATION.md) added 216 evaluations without new training. The [iteration-3 strength sweep](reports/iteration3/INTERPRETATION.md) added 46 fits and 174 sampling evaluations and produced the [available exported samplers](docs/ITERATION_3_RUNBOOK.md). Earlier studies remain separate. The model is trained; [publication requirements and current evidence](docs/PUBLICATION_READINESS.md) distinguish completed work from remaining scientific gaps.

## Start with the review

- [Revised research proposal](MedHyperDiff_Research_Proposal_Revised.md)
- [Errors, gaps, verified literature and publication gates](docs/RESEARCH_REVIEW.md)
- [Dataset and constraint audit](reports/DATA_AUDIT.md)
- [Clinical pilot results](reports/pilot/EXPERIMENT_SUMMARY.md)
- [Interpretation of clinical results](reports/PILOT_INTERPRETATION.md)
- [Controlled mechanism results and interpretation](reports/mechanism/MECHANISM_INTERPRETATION.md)
- [Working manuscript, with results clearly pending](paper/manuscript.md)

The original proposal is preserved in `docs/source/MedHyperDiff_Research_Proposal_Original.md`; the Desktop source file is unchanged.

## Environment and tests

Python 3.11–3.13 is supported. The verified local environment uses Python 3.13; `requirements-lock.txt` records its installed dependency versions. PyTorch CPU is the default. CUDA/MPS are configurable but were not validated by these CPU tests.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps .
.venv/bin/python -m pytest -q
```

Alternatively install package dependencies using `uv pip install --python .venv/bin/python '.[dev,baselines]'`. For active source edits, prefix commands with `PYTHONPATH=src` or reinstall the package afterward. A writable UV cache can be set with `UV_CACHE_DIR=/private/tmp/medhyperdiff-uv` in restricted environments. Some joblib versions query physical CPU counts on macOS; `LOKY_MAX_CPU_COUNT=2` avoids that unnecessary probe in restricted sessions.

## Public data

```bash
.venv/bin/medhyperdiff download
.venv/bin/python scripts/audit_data.py
```

This downloads only the four named UCI tables and source metadata, retains raw bytes, normalizes known source aliases/whitespace and writes checksums. Repeat calls reuse the raw cache and can reprocess it offline. Read each dataset's source metadata and license/acknowledgment terms before redistribution. Data are ignored by Git.

Schema decisions:

- Heart Failure excludes follow-up duration `time`; its binary label is observed death status, not fixed-horizon survival.
- ILPD retains missing A/G ratios and uses an exploratory fit-only residual tolerance; there is no exact clinical assay guarantee.
- CKD retains missingness and treats its coded ordinal features as categories in this prototype.
- Diabetes groups identical predictor profiles across splits. The grouping is conservative and does not establish that repeated profiles are the same person.

## Run a pilot

```bash
.venv/bin/medhyperdiff run --dataset ilpd --mode full \
  --epochs 60 --steps 30 --width 32 --layers 2 --bootstraps 10 \
  --output runs/my_ilpd_full
```

Default: outer fold 0 of 5, fixed split seed 2026, generator seed 0, inner-validation evaluation, synthetic count and class prevalence equal to the fit partition. Output directories must be new; completed runs are never overwritten. The program fails on unsupported source values rather than silently recoding them.

Run the saved experiments (choose new output paths if they already exist):

```bash
.venv/bin/medhyperdiff batch --config configs/pilot.json
.venv/bin/medhyperdiff batch --config configs/mechanism.json
.venv/bin/python scripts/summarize_runs.py --root runs/pilot --output reports/pilot
.venv/bin/python scripts/summarize_runs.py --root runs/mechanism --output reports/mechanism
.venv/bin/python scripts/mechanism_report.py
```

The pilot configuration covers 4 datasets × 5 model families × 1 fold × 1 seed, with 60 epochs for the neural models. It is a pipeline/development comparison with unequal model-specific optimization behavior, **not** a tuned final benchmark. The controlled configuration covers 12 arms × 3 seeds, at one development fold and 100 epochs. Bootstrapped graph estimation is repeated reproducibly from the same fit data and fixed split seed.

`configs/targeted_controls.json` adds three parity-withholding experiments and one ILPD repair control. [The reproducibility log](reproducibility/README.md) documents source snapshots and the corrected copula reruns.

## Model arms

| Name | Purpose |
|---|---|
| `base` | Gaussian numeric + absorbing categorical diffusion, conditioned on labels |
| `graph` | Association bias only |
| `hypergraph` | Nonlinear incidence representation only |
| `energy` | Training constraints without graph/hypergraph/guidance |
| `guidance` | Sampling guidance without graph/hypergraph/training constraints |
| `full_no_guidance` | Graph, hypergraph and training energy |
| `full`, `constant`, `late` | Same full model with different equal-integrated-strength guidance schedules |
| `clique` | Pairwise hypergraph projection in attention, replacing nonlinear incidence blocks |
| `shuffled` | Permuted graph/incidence as a structural negative control, keeping semantic energy |
| `unit_confidence` | Full model with unit rule weights |
| `repair` | Base samples followed by disclosed simple repair; saves uncorrected samples |
| `copula`, `independent` | Conditional statistical references |
| `bootstrap` | Training resampling as a memorization positive control |
| `ctgan`, `tvae` | Official ctgan-package models fitted separately per label |
| `dense` | Same hyperedge slots/parameters as full, with all-to-all incidence membership |
| `tabddpm` | Pinned official TabDDPM diffusion/MLP with a documented local training adapter |

All arms keep label conditioning and missingness encoding. Hypergraph parameters change model size; compare full and dense using the same stabilization flags for a parameter-matched structural control. `--paired-initialization` matches common backbone weights and training corruption randomness. `--gated-structure` and `--numeric-prediction v` are experimental options; neither is assumed superior. See [method details](docs/ITERATION_2_METHOD.md). On cohorts with only unary rules, the full model has no nonlinear clinical hyperedges. It must not be interpreted as evidence for higher-order structure on those cohorts.

Hold out a known rule from the model while retaining it in evaluation:

```bash
.venv/bin/medhyperdiff run --dataset toy --mode full --withhold-rule parity \
  --output runs/toy_withheld_parity
```

The rule is excluded from architecture, loss, and guidance. Other rules can imply a withheld relation, so verify independence of the evaluation target before interpreting generalization.

## Artifacts and reuse

Each run records `manifest.json`, `splits.json`, `preprocessing.json`, graph estimates/support, training/evaluation constraint registries, `synthetic.csv`, `metrics.json` and (from iteration 2) `utility_predictions.csv`. Neural diffusion runs also save history and a CPU-loadable checkpoint. TabDDPM checkpoints require the pinned upstream source. A failed run retains its failure manifest. The source/data hashes support exact version auditing.

```bash
.venv/bin/medhyperdiff sample --checkpoint runs/my_ilpd_full/checkpoint.pt \
  --positive-count 200 --negative-count 100 --seed 9 --output runs/new_samples.csv
```

The sample command returns predictors plus `condition_positive` (1 = the schema's positive label). The run command returns the original target labels. CTGAN/TVAE and statistical pilot adapters currently regenerate by refitting the deterministic run configuration; their fitted objects are not serialized.

## Evaluation boundary

The implementation measures numerical/categorical fidelity, association errors, constraint severity and coverage, logistic/forest TRTR/TSTR/augmentation utility, balanced RF two-sample discrimination, exact copy rates, matched-size DCR/NNDR and a basic calibrated nearest-synthetic membership attack. Aggregate summaries do not run naive significance tests on dependent folds/seeds.

Remaining publication requirements include a tuned and validated external diffusion benchmark (the official-source TabDDPM adapter is implemented; TabDiff is inspected but untrained), strong knowledge-equivalent constraint baselines, externally reviewed clinical constraints, appropriate larger/external cohorts, full held-out study execution, rule-corruption controls, stronger privacy attacks, and justified inference/noninferiority analysis. The prototype implements no ordinal-specific kernel, causal learning, formal DP or exact constrained sampler.

During current development keep `--fold 0 --split-seed 2026` fixed. Use `--inner-seed 3027` or another declared inner seed to vary validation partitions inside that development pool. Changing outer folds during model development would expose reserved test patients through other folds' training partitions. The updated `scripts/prepare_user_runs.py` follows this boundary for its expanded validation plans.

Only after freezing architecture, metrics, tuning budgets and endpoint definitions should a confirmatory run use `--evaluate-test`. The CLI cannot prevent a researcher from repeatedly inspecting test metrics; the scientific freeze must be documented. The initial runner reserves inner validation rather than refitting on the full outer training fold; report this consistently for all models.

The working manuscript deliberately leaves confirmatory results and conclusions pending. Do not convert the illustrative research hypothesis into a reported finding.
