# Verification record

## Development iteration 8

- **36 tests passed in 3.08 seconds**, with 21 upstream deprecation warnings. New tests cover asinh-MAD round trips, mask preservation and agreement between transformed and affine physical-unit constraint energies.
- **Nine transformed diffusion fits and 27 new evaluations against 36 reused references.** The outer test remains closed and no default sampler or core package source changes.
- All 63 records pass the artifact audit; 378 saved AUROCs are recalculated and all nine transformed checkpoints exactly replay their first sampling seed. The audit confirms physical-unit metric recomputation and zero outer-test overlap.
- The asinh-MAD representation is retained as a negative result: inverse decoding creates extreme bilirubin outliers, numeric Wasserstein errors up to 1e14 and 21–25% supplied-rule violations. Missingness errors remain below 0.003. No clipping or favorable postprocessing was added.
- Component times are 52.5 seconds fitting, 38.4 seconds sampling and 25.7 seconds evaluation on CPU; GPU VRAM is zero and peak system RAM is unmeasured. The runbook and protocol are archived by hash.

## Development iteration 7

- **34 tests passed in 2.77 seconds**, with 21 upstream deprecation warnings. Three new diagnostic tests cover float roundoff copies, true numeric changes, category/missingness mismatch and boundary perturbations. A first test fixture incorrectly requested a one-row binary dataset; it was corrected before the passing rerun.
- **36 native neural adapter fits (72 class networks), nine statistical fits, 135 sampled tables and nine repaired derivatives**. The 144 new evaluations plus 81 reused records total 225. No new MedHyperDiff training or exported default.
- All 225 artifact records pass; 1,350 saved AUROCs are recalculated, all 45 newly fitted objects exactly replay their first sampling seed, and all nine repairs reproduce exactly. Original outer-test rows remain excluded.
- A post-protocol numerical diagnostic supplements all 225 records without modifying frozen metrics. Bootstrap is correctly detected as 100% training near-copies with zero tolerance-aware extrapolation. Exact numeric hashes alone had missed preprocessing roundoff copies. Supplemental source/output hashes and bootstrap checks pass.
- Protocol/source/reference hashes were frozen before fitting. Preflight corrected the location of reused split files before freeze; no research output was discarded. A reporting-path typo was corrected before successful aggregation. The core package remains unchanged. All native fits/evaluations completed, and full negative results remain retained.
- CPU fitting/sampling/evaluation components total 115.0/4.8/127.0 seconds (about 4.1 minutes, excluding overhead). GPU VRAM is zero; peak system RAM and GPU performance unmeasured. Scientific PNG/PDF/SVG figure visually inspected.


## Latest update — development iteration 6

- **31 tests passed in 3.38 seconds**, with 21 upstream deprecation warnings. Four new tests cover variance endpoints/log interpolation, exact original-sampler equivalence through the complete loop, deterministic last-step decoding, unchanged sampled model weights, and fit-threshold shape diagnostics.
- **162 new sampling/evaluation records plus 162 reused references; zero new generator fits.** New generation uses 27 prior-initialized checkpoints, and references span 54 fitted models.
- All 324 records passed the audit: parent/source/sampler/plan hashes, matching label order and sampling settings, original reserved-test isolation, all saved AUROCs, missingness and unknown-rule bounds, numerical SD ratios and new quantile/IQR/tail metrics.
- Six exact replays covered mixtures 0, 0.5 and 1 at the earlier missingness case and an AST-failure partition. Mixture zero reproduces the prior sampler exactly. Nine full/dense checkpoint pairs retain equal capacity.
- The separate sampling CLI generated a verified 10-row example with five rows per class and a provenance manifest. This smoke example and audit replays are excluded from the 162 research-evaluation count.
- No full variance setting passed every frozen gate; conditional export produced no replacement recipe. Failed outcomes and all synthetic tables are retained.
- Generation/evaluation component times sum to 193.6/147.2 seconds on CPU. GPU VRAM use is zero; peak system RAM and GPU performance are unmeasured. These are not end-to-end study wall times.
- Core source remains unchanged. Initial protocol, plan, sampler and runner were archived before sampling; final scripts, reports and manuscript are archived by hash. The final figure was visually inspected and exported as PNG/PDF/SVG.

The added noise improves one AST dispersion metric but worsens rule compliance and other laboratory tails. Low SD coexists with excessive central spread, so uniform variance inflation is not a validated improvement. The paper remains a development draft; see the evidence-based publication readiness document.

## Development iteration 5

- **27 tests passed in 4.05 seconds**, with 21 upstream deprecation warnings. New tests check exact original loss/gradient equivalence at unit observation weight, fit-frequency initialization changing only variable observation heads, isolated mask-CE weighting, and saved-model compatibility with the unchanged sampling loader.
- **36 new generator fits from scratch, 108 new evaluations and 81 reused references**, representing 63 fitted models in the comparison. Unit-test fits are not counted as research runs.
- All 36 fitted-model artifacts and 189 evaluation records passed `scripts/audit_iteration5.py`: source/parent/plan hashes, fit-only initialization counts, matched preprocessing/graph/rules/capacity, unchanged sample configurations and class-label order, and original reserved-test exclusion.
- Every saved AUROC, feature missingness, rule bound and numerical variance ratio was independently recomputed from retained artifacts. Three exact replays covered the original failing cell and both new full variants using the ordinary core sampler.
- The experimental training module is isolated under `scripts/iteration5_model.py`; the core source hash remains `1a9e99d67a8131df426c2292c785cfe2b7c0f9d99d465fe4f476278d69a5645c`. The plan, protocol, runner and model module were archived before fitting.
- Fitting, sampling and evaluation component times sum to 201.7, 169.3 and 113.4 seconds. All runs use CPU; GPU VRAM use is zero. Peak system RAM and GPU performance were not measured.
- No full variant passed every frozen correction gate. The conditional export command correctly produced no replacement sampler. All trained weights and synthetic tables remain retained.
- The final figure was visually inspected and exported as PNG/PDF/SVG. The previous manuscript is archived before incorporating measured positive and negative outcomes.

Frequency initialization removed the observed missingness flags, but full-prior failed the AST variability guardrail on one partition. Successful software checks and a corrected marginal metric do not establish clinical validity, formal privacy, independent generalization or submission readiness.

## Development iteration 4

- **23 tests passed in 2.86 seconds**, with 21 upstream deprecation warnings. Three new tests check unknown-rule union bounds, fully unevaluable rules, and concentrated missingness that a feature average can hide.
- **216 evaluations, 108 generated pools and 36 reused checkpoints; zero new generator training runs.** The generator/core evaluator package remains unchanged from iterations 2–3.
- The frozen plan was archived before sampling. All 216 artifact audits passed: exact source/parent/script/plan hashes, unchanged fitted configurations, original reserved test exclusion, independent sampling seeds and fixed evaluator seed.
- Saved 1x tables are exact prefixes of their 5x pools, with exact fit class proportions. All six classifier/experiment AUROCs per table reproduce from saved prediction scores. Fivefold augmentation outputs are correctly labelled `AUGMENT_5x`.
- Every feature's missingness and every rule's observed/unknown bounds were recomputed from saved tables. Nine matched dense/full checkpoint pairs have equal capacity. One guided updated-model fivefold pool was independently regenerated exactly.
- Sampling took 131.1 seconds summed across 108 pools; evaluation took 155.8 seconds summed across 216 tables. These CPU component times are not total elapsed wall time. GPU VRAM used is zero; peak system RAM and GPU performance are not measured.
- The final PNG figure was visually inspected; PDF/SVG exports are retained. The prior manuscript is archived before updating the current draft with the new positive and negative findings.

The updated model does not pass the declared larger-table adoption requirements. Three 5x and three 1x outputs from one fitted checkpoint exceed the feature-wise missingness threshold. No clinical validity, privacy, statistical superiority or publication-readiness claim is inferred from successful software checks.

Completed in the local Python 3.13 CPU environment on 11 September 2026.

- `PYTHONPATH=src .venv/bin/python -m pytest -q`: **14 passed** (final run: 3.63 seconds).
- Source, scripts and tests compiled successfully with `compileall`.
- Normal package installation and the `medhyperdiff` CLI were exercised.
- A saved ILPD checkpoint was loaded through the installed CLI and generated a new 10-row sample file.
- All four UCI datasets downloaded, normalized and audited. Raw-byte checksums and source metadata are retained.
- **60 retained development runs completed:** 20 clinical pilots, 36 controlled mechanism runs, three withheld-rule runs and one ILPD repair control.
- Every retained primary run manifest reports `status: complete` and `outer_test_opened: false`.
- Each retained run's package-source hash resolves to an archived source snapshot. Corrected copula reruns replace the earlier baseline values in the measured reports; the previous artifacts are retained separately.
- The original supplied proposal remains unchanged and its project archive was verified byte-for-byte against the Desktop source.
- The clinical comparison PNG was rendered and visually inspected for legible labels and unclipped layout.

The tests cover split/profile isolation, train-only transforms, unknown categories and missing values, absorbing posterior probabilities and terminal reveal, known categorical states during guidance, Gaussian schedule endpoints, equal-integrated-strength schedules, energy gradient direction and batch invariance, nonlinear incidence behavior, training/sampling/checkpoint determinism, identity fidelity metrics, a memorization positive control, signed dependency reversal, follow-up exclusion, all-missing numerical columns and constant/imbalanced copula columns.

These checks establish software behavior and development execution. They do not establish clinical validity, formal privacy, a calibrated uncertainty method, statistical superiority or publication readiness. CUDA/MPS, official external diffusion baselines and the final confirmatory protocol were not validated here.

## Update — 12 September 2026, development iteration 2

- `PYTHONPATH=src .venv/bin/python -m pytest -q --disable-warnings`: **20 passed in 2.84 seconds**. Upstream TabDDPM generated 21 deprecation warnings; no test failed.
- Added tests verify velocity reconstruction across all noise levels, identical base/full initial predictions with gating and paired initialization, learnable gate gradients, equal dense/full parameter counts, unchanged reserved test patients under inner reseeding, grouped discriminator behavior and official-source TabDDPM checkpoint reuse.
- **110 additional research runs completed**; their full artifact audit passed. The original 60 runs remain separate.
- Every new manifest reports `outer_test_opened: false`; the exact original reserved test indices are unchanged. No reserved test record occurs in any new fit/validation partition.
- Saved prediction scores reproduce reported AUROC with Pandas `float_precision="round_trip"`. Default CSV parsing initially changed a few probability ties; the audit now preserves exact floats. Stored model scores and metric calculations were unchanged.
- All new package-source hashes resolve to archived snapshot `1a9e99d67a8131df426c2292c785cfe2b7c0f9d99d465fe4f476278d69a5645c`.
- Full/dense parameter counts agree (ILPD 37,600; simulator 36,633). Gated pairs add two scalars.
- Official-source TabDDPM trained and sampled on CPU, including extended training. Its learned ILPD sampling remains unstable; a successful software run is not a valid performance reproduction. An analytic standard-Gaussian oracle checks the upstream sampler independently; diagnostics are retained.
- New result figures were generated as PNG/SVG/PDF. Their development-only interpretation is stated in the report and manuscript.
- A normal offline package reinstall succeeded; the installed CLI loaded both a new full-model checkpoint and an official-source TabDDPM checkpoint to generate separate 10-row samples. Source/scripts/tests compiled successfully.
- Fresh-path user configurations were regenerated; all 300 expanded-validation plans now preserve outer fold 0 and vary only inner seeds. The supplied original proposal remains byte-identical to its archive.

The earlier 14-test/60-run record remains an accurate historical record for the prior iteration. The current work still does not establish clinical validity, formal privacy, external generalization or publication readiness. GPU execution and peak system memory remain unmeasured.

## Update — development iteration 3

- Existing core-model tests: **20 passed in 2.54 seconds**; 21 upstream deprecation warnings. The core model package was unchanged.
- **46 new fits and 174 distinct sampling/evaluation configurations**, using 94 fitted checkpoints (48 reused). Training and resampling counts are kept separate. New-training commands also emit automatic zero-guidance diagnostic evaluations; these are not counted again as independent sweep configurations.
- All 174 evaluation artifacts passed `scripts/audit_iteration3.py`: exact parent/source/script hashes, unchanged training configurations, split isolation, class counts and saved RF/logistic AUROC values. Eighteen full/dense training pairs matched parameter count.
- Reserved test patients remain excluded from every fit/validation partition. No outer-test performance was calculated.
- Reusing the original ILPD full checkpoint/settings exactly reproduced its iteration-2 synthetic table.
- Three frozen screen-selected samplers were exported with parent provenance. All three exported checkpoints exactly reproduced their evaluated generated tables. A CLI generation example uses the exported weight-0.01 velocity candidate.
- Result tables retain all screen candidates, immutable selection, additional controls, seed variation and failed replication conditions. The selected weight-0.01 candidate passes the engineering conditions on two of three ILPD partition means, not all three. These conditions are not clinical noninferiority tests.
