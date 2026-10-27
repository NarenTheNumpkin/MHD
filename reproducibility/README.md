# Reproducibility record

`requirements-lock.txt` at the project root records the exact Python dependencies installed for the CPU pilots. Each run additionally records Python, platform, package versions, thread setting, source/data SHA-256 hashes, split indices, seed, hyperparameters, checkpoints and metrics.

`source_snapshots/<source_sha256>/medhyperdiff/` contains the exact Python package source matching retained run hashes. To run a snapshot instead of the installed package, use:

```bash
PYTHONPATH=reproducibility/source_snapshots/HASH .venv/bin/python -m medhyperdiff.cli --help
```

Replace `HASH` with the corresponding manifest's value. The package-directory hash is computed by concatenating each sorted top-level `.py` filename and its bytes, then applying SHA-256. Tests and reporting scripts live in the main project.

## Development correction log

- Early ingestion smoke runs detected UCI source-name variations (`death_event`, `wbcc`/`rbcc`, and lowercase underscored diabetes fields). The schema adapters were corrected before the retained pilot matrix.
- Early implementation checks corrected categorical guidance to use exact known states for already revealed categories. The retained matrix uses this correction.
- An audit of the simple copula reference found that tied categorical normal scores should be centered before covariance estimation. Only that baseline's implementation changed. All four clinical and three controlled copula runs were repeated after the fix; previous artifacts remain in `runs/archive_pre_copula_centering/`. The scientific comparison/report uses the corrected baseline. Other model code and evaluation logic are identical across those two source snapshots.
- Editable installation's `.pth` was not honored by this session's Python startup configuration. A normal wheel installation was verified, and the README also documents `PYTHONPATH=src` for development. No global Python configuration was changed.

The 20 clinical pilot configurations, 36 mechanism configurations, three withheld-rule configurations and one ILPD repair control are development experiments. All retained primary manifests report `outer_test_opened: false`. There are no confirmatory outer-test results or formal privacy guarantees.

Training artifacts may contain information about the fit data even when they do not contain the original rows. This project has not implemented differential privacy. Raw UCI data and run artifacts are ignored by Git; review dataset acknowledgments before sharing data or checkpoints.

## Development iteration 2 (12 September 2026)

The 110 new training/evaluation runs in `runs/iteration2` use package snapshot `1a9e99d67a8131df426c2292c785cfe2b7c0f9d99d465fe4f476278d69a5645c`. They comprise 24 screen, 24 seed replication, six larger-budget TabDDPM, 54 inner-partition/control and two extended TabDDPM diagnostic runs. The original 60 development runs remain separate. Unit tests and ingestion/sampling smoke checks are not counted as research runs.

This snapshot adds optional structural gates, Gaussian velocity prediction, paired initialization, dense-incidence controls, an official-source TabDDPM adapter, inner-only split variation, grouped discrimination and downstream prediction artifacts. Evaluation version 2 changes the discriminator's duplicate grouping; do not silently pool that score with older reports. The general summarizer distinguishes inner partitions and evaluator versions.

TabDDPM imports unmodified upstream source at commit `b476257dd460b778ba09eb97f7a51d6490fa17f8`, retained under `external/tab-ddpm` with its MIT license. Each baseline manifest records imported source hashes. See `docs/EXTERNAL_BASELINES.md` for adaptations and the failed numerical-fidelity diagnostics. TabDiff was cloned/inspected, not trained.

All 110 runs passed `scripts/audit_iteration2.py`: the original reserved outer fold-0 patients were absent from every fit/validation set, saved scores reproduce the reported AUROC with round-trip CSV parsing, class counts match the fit partitions, source hashes resolve, and dense/full parameter counts match. No confirmatory test metrics were opened. The protocol and adaptive screening decisions are in `docs/ITERATION_2_PROTOCOL.md`; these are development decisions, not preregistration.

## Development iteration 3

The strength sweep uses the unchanged core package snapshot `1a9e99d67a8131df426c2292c785cfe2b7c0f9d99d465fe4f476278d69a5645c`. The experiment runner is archived under `iteration3_scripts/4c4c10fa6e8ae7a8d118db08018284ceb00d78d71c706d8965940ff9fbfc4482/`; per-evaluation manifests record its hash, fitted parent hash and separate training/sampling configurations. Report/control scripts are also archived by hash.

There are 46 new training runs and 174 distinct sampling/evaluation configurations, referencing 94 fitted checkpoints (48 reused from iteration 2). Checkpoint matching checks data and source hashes, exact split indices and training configuration. Guidance-only changes reuse fitted weights. The 84-setting screen, frozen candidate selection, replication and comparator registries are stored under `reports/iteration3`; duplicate settings shared across stages are not counted twice. All source/script/parent provenance and reserved-test isolation passed the artifact audit.

The previous manuscript is preserved in `paper/archive/manuscript_before_iteration3.md`. Exported samplers and their provenance are under `artifacts/iteration3`; they are development artifacts, not independently validated winners. Setup and commands are in `docs/ITERATION_3_RUNBOOK.md`.

## Development iteration 4

The unchanged core package is reused for 216 evaluations of 108 nested synthetic pools from 36 fitted checkpoints. No new generator is trained. `configs/iteration4.json` freezes the parent hashes, three independent sampling seeds, evaluator seed, multipliers and protocol hash before sampling. The runner, protocol and plan are archived under `iteration4_scripts/<runner_sha256>/`; audit/report/plot/test files are separately archived by hash. Per-pool and per-evaluation manifests record both training and sampling configurations and data/source/parent provenance.

Sampling uses three repeats per checkpoint, independent of the three training seeds. A fivefold pool repeats the fit-label vector; its first fit-size rows form the exact 1x comparator. Counts and prevalence therefore match by design. `AUGMENT_5x` denotes real fit rows plus five synthetic fit sizes. The core evaluator's legacy key is renamed explicitly in the experiment wrapper, including saved predictions.

The full audit verifies all 216 artifact records, recomputes missingness and rule bounds, reproduces all saved AUROC values, checks nine matched-capacity pairs and independently regenerates one guided pool exactly. Outer fold 0 and its original reserved test remain untouched. Run/report/audit instructions and component timings are in `docs/ITERATION_4_RUNBOOK.md`; all outcomes and gate failures are in `reports/iteration4/`. The previous draft is archived in `paper/archive/manuscript_before_iteration4.md`.

## Development iteration 5

This stage trains 36 new models from scratch using an experimental observation-head training class in `scripts/iteration5_model.py`. The core package source remains unchanged. Before fitting, `configs/iteration5.json` freezes the protocol, model/runner/helper script hashes, exact reference checkpoint hashes, four variants, three training seeds, three sampling seeds and three inner partitions. It references 81 existing 5x evaluations without counting them as new experiments. New fits produce 108 additional evaluations, for 189 comparison records and 63 fitted models total.

`reproducibility/iteration5_scripts/<file_sha256>/<filename>` archives the initial runner, model module, protocol and plan; a manifest separately lists final audit/report/plot/export/test/document snapshots. New training manifests record fit-only initialization frequencies, observation-loss weight, inherited preprocessing/graph/rule settings, selected epoch, executed epochs, unchanged parameter counts and checkpoint hashes. They explicitly record that learned reference weights were not used as a warm start. New checkpoints retain experimental training metadata and load through the ordinary sampler; the study sets guidance 10 explicitly after fitting.

The full audit passes all fitted/evaluated artifacts, recomputes numerical variance ratios and unknown-rule bounds, reproduces every saved AUROC and replays three pools exactly. All original reserved test rows remain excluded. Twenty-seven software tests pass. Neither full variant meets every frozen correction criterion; no replacement sampler is exported. See `docs/ITERATION_5_RUNBOOK.md` for research sampling commands and measured component costs. The previous manuscript is preserved in `paper/archive/manuscript_before_iteration5.md`.

## Development iteration 6

The core package remains unchanged. The separately archived `scripts/iteration6_sampler.py` changes only reverse variance, with exact original-sampler equivalence at mixture zero. `configs/iteration6.json` freezes the two mixtures, 27 fitted parents, original/prior reference hashes, script/protocol hashes, split/training/sampling/evaluator seeds, label order and sample size before generation. There are 162 new evaluations and 162 reused reference records, no new fits, and 54 fitted models represented across the comparisons.

Quantile/IQR/fit-tail diagnostics for historical tables are written under `reports/iteration6/reference_shapes`; their original metrics and samples remain unchanged. Each reanalysis points to the source synthetic-table hash. New samples retain all metrics and classifier scores. The audit recomputes all 324 records and verifies six exact sampler replays, including the zero-mixture equivalence. The CLI smoke table and audit replays are not counted as new research runs.

Initial sampler/runner/protocol/plan files and final report/audit/plot/export/CLI/test/document files are archived by SHA-256 under `iteration6_scripts/`, with a manifest of final snapshots. The previous paper is archived in `paper/archive/manuscript_before_iteration6.md`. No candidate meets all gates, and no replacement recipe is exported. `docs/ITERATION_6_RUNBOOK.md` records commands and CPU component times; `docs/PUBLICATION_READINESS.md` distinguishes scientific gaps from completed implementation checks.

## Development iteration 7 (13 September 2026)

The frozen `configs/iteration7.json` records core/runner/adapter/helper/protocol/data hashes, installed CTGAN Python-source hashes, package versions, native defaults and every reused reference. The stage trains 36 CTGAN/TVAE adapters (72 native class networks) and fits nine deterministic references, adding 144 evaluations against 81 reused records. Every preset is retained. Native whole-object CPU checkpoints are trusted local artifacts and require matching module/dependency versions; checksum validation precedes loading. Per-class loss CSVs, preprocessing, rules, splits, samples and utility scores remain available.

All 225 artifact records pass, with 45 exact first-seed checkpoint replays, nine exact repair replays and 1,350 reconstructed AUROCs. Thirty-four tests pass. `docs/ITERATION_7_DIAGNOSTIC_ADDENDUM.md` discloses a post-protocol numerical-tolerance supplement: float32 preprocessing made strict numeric hashes miss bootstrap copies and perturbed strict boundary ties. The supplement preserves original metrics, records its own hashes and correctly recovers all bootstrap training profiles. This is neither a prespecified primary endpoint nor a privacy guarantee. Final stage scripts/documents and figure outputs are listed in `iteration7_scripts/manifest.json`.

See `docs/ITERATION_7_RUNBOOK.md` for exact setup/run commands and measured CPU costs. No core source change, new MedHyperDiff fit, outer-test evaluation or replacement sampler occurs in this stage. Strong repaired-copula performance and mixed native-baseline utility constrain the manuscript's claims.

## Development iteration 8 (13 September 2026)

The frozen `configs/iteration8.json` records the runner, experimental model helper, protocol, helpers, normalized data, splits and reused-reference hashes. Nine full diffusion models use a fit-only `asinh_mad_v1` numerical coordinate with a differentiable physical-unit inverse in the constraint energy; 27 new tables are compared with 27 full-prior and nine repaired-copula records. The outer test stays closed and the core package is unchanged.

All 63 records pass the artifact audit, including 378 saved AUROC reconstructions and nine exact first-seed checkpoint replays. The representation is retained as a negative result: source-unit inverse decoding produces extreme bilirubin outliers and 21–25% supplied-rule violations. The fit-only transform and physical-energy unit tests are in `tests/test_iteration8_model.py`. Component CPU times and rerun commands are in `docs/ITERATION_8_RUNBOOK.md`. No new default sampler is exported.

## Development iteration 9 (13 September 2026)

The frozen `configs/iteration9.json` records hashes for every 225 iteration-7 source table and five rule registries. The stage performs 1,125 evaluation-only analyses: correct supplied rules, protein omission, reversed bilirubin inequality, reversed albumin inequality and both reversed. No fitting, sampling or output modification occurs. All fit/validation/synthetic metrics and lower/upper unknown-rule bounds are retained. The audit verifies source hashes, labels, split isolation and recomputed rule metrics, with zero outer-test rows. The result shows that the exploratory A/G identity drives most current compliance and is violated by real validation partitions, so clinical/laboratory review is required.

See `docs/ITERATION_9_RUNBOOK.md` for commands. This sensitivity stage changes no core source and exports no model.

## Development iteration 10 (13 September 2026)

The frozen `configs/iteration10.json` records three additional public cohorts, their normalized data hashes, locked outer/inner splits, nine current-full training configurations and three sampling seeds. Nine fits and 27 evaluations run on CKD, heart failure and diabetes; all outer test rows remain excluded. A preflight table exposed mixed string/float categorical types for numeric-looking CKD levels after CSV round-trip. The incomplete first table is retained under `runs/iteration10/preflight_failed/`; accepted evaluation reads force all categorical columns to strings, and the correction is recorded in the plan. All 27 accepted records, 162 AUROCs and nine first-seed replays pass the audit. CKD missingness and heart-failure utility failures remain visible; no clinical generalization or external superiority claim is made.

See `docs/ITERATION_10_RUNBOOK.md` for commands and costs. No core source or default sampler changes.

## Development and confirmation iteration 11

Source snapshot `9739e858bc267fee430689d74988ad5e86e8ea54b1e788d30bcf6a1f8e399bd8` ran the 72 ablation jobs and the 384-job four-method search; snapshot `721c3bbb31a36d9533147a33c952502f0e1ff49d6363592c863844c618e77bb1` differs only by the appended `TabDiffAdapter` and ran the 96 TabDiff search, 54 simulator and 255 confirmation jobs. Both snapshots were re-hashed and match every manifest. Runner, stage and report scripts are archived by hash under `iteration11_scripts/`; the frozen plan, selection and guidance files are in `configs/iteration11*.json`, and every confirmation manifest records their hashes. Only the 255 confirmation manifests have `outer_test_opened: true`. Under legacy settings the modified core reproduces the archived iteration-5 `full_prior_i2027_t0_s70001_n5` synthetic table byte-for-byte, and a guidance-stage refit reproduces its search counterpart exactly. Because file access under `~/Desktop` was extremely slow on the development machine, the study ran from byte-identical copies outside the Desktop and results were copied back; see `docs/ITERATION_11_RUNBOOK.md`.

## Controlled study iteration 12

All 240 jobs in `runs/iteration12` record source snapshot `3aafec1cb9300f8efc734d407ee48307ceea619ddcf73a4dabff1822501a4e76` (adds NHANES/thyroid builders, definitional rule kinds, banded precision decoding, the C2ST curve and the density-ratio attack; legacy defaults unchanged), runner `61a93edf…` and splits `3b60a107…` (`configs/iteration12_splits.json`). Runner, report and protocol are archived by hash under `iteration12_scripts/`. NHANES source files are pinned by SHA-256 in `medhyperdiff.data.NHANES_FILES`; the UCI thyroid archive hash is recorded in `data/raw/thyroid.manifest.json`. The study ran from byte-identical copies outside `~/Desktop` (see the iteration-11 note) and results were copied back.

## Relation study iteration 13

All 90 confirmation jobs in `runs/iteration13/confirm` record source snapshot `36533bf58efea275a386096353e98ecac98db6631004c2279e301559c4dc333d` (adds `medhyperdiff.relations`, the NHANES CBC builder and relation rule kinds), runner `059e5808…`, splits `configs/iteration13_splits.json` and choice `configs/iteration13_choice.json` (`afd4798b…`). Development rounds 1–2 (superseded by Amendments 1–2) are retained. Runner, report and protocol are archived by hash under `iteration13_scripts/`.

## Robustness study iteration 14

All 618 confirmation fits and 10 secondary fits in `runs/iteration14` and all 60 development fits record source snapshot `890ebbee41f16b5dee0aa50ce8f051d32345db35b3d3bbac09c8e21415dff718`. This snapshot adds the NHANES cycle builders, linear relations, reported-exact detection and re-derivation, and the clinical-category metric. The runner and report scripts are archived in `iteration14_scripts/`. Of the 618 confirmation fits, 598 are analyzed; 20 MedHyperDiff fits at n = 100 finished before Amendment 1 and are reported separately.

## Training decomposition iteration 15

All 160 fits in `runs/iteration15` record source snapshot `1003bc8c1049287a67612a72c8f7681ba27547cfdc097e5a4be6331ea333320c`. It adds options to the fixed-budget loop (checkpoint selection, corruption draws, patience, constant rate, no averaging); default behaviour is byte-identical. Scripts are in `iteration15_scripts/`.

## Post hoc controls and final comparison (14 October 2026)

`runs/iteration14_controls/controls.csv` evaluates textbook re-derivation and joint residual resampling for every published generator output of iterations 13 and 14 (280 outputs, 560 evaluations, no refitting). It used source snapshot `1626362e15411c2e9016f490f82976b9f722d658c056a37f1c60aa8756902148`.

This snapshot also aligns the BMI log-scale coefficient with Eq. (3) of the manuscript. A check over all 778 saved iteration 13 and 14 runs found that no reconstruction choice changes.

`scripts/report_final.py` builds the comparator registry, applies the membership threshold to every arm and writes the manuscript's LaTeX table rows. Scripts are archived in `final_scripts/`.

**Clean-copy check.** In a clean copy, `scripts/fetch_data.py` rebuilds all 13 processed datasets byte-identically from the public sources. All 64 tests pass with both pinned baselines in `external/`.
