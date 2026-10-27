# Data behind the tables and figures

All files are comma-separated values with a header row. Each row is an aggregated evaluation result for one training sample, generator variant and post-processing, or a summary of such rows; no participant-level records are included.

Common columns: `c2st` is the AUROC of the classifier two-sample test (0.5 is indistinguishable); `relation_w1` is the relation distance; `membership` is the larger of the nearest-distance and density-ratio membership AUROCs; `rf` is the AUROC of a random forest trained on synthetic and tested on real records; `numeric_w` is the scaled numerical Wasserstein distance; `association` is the association error; `fit_seconds` is the wall-clock fit time.

| File | Supports |
|---|---|
| `ablation_partition_means.csv` | Table 3 and Fig. 2: corrections on the liver dataset development partitions |
| `reference_rescored.csv`, `reference_rescored_means.csv` | Reference arms of Fig. 2 and the development case study |
| `a7_pipeline_summary.csv` | Table 4 and Fig. 3: pairs reversed, Kendall's tau and top-1 agreement |
| `all_jobs.csv` | Controlled study, one row per fit: training choices, rules and memorization (Fig. 5) |
| `it15_means.csv` | Table 5: decomposition of the training correction |
| `a3_precision_c2st.csv` | Table 6 and Fig. 4: C2ST by decoding |
| `a6_agreement_by_size.csv` | Table 7 and Fig. 6: evaluation size |
| `first_test.csv` | Table 8: 2017 to 2018 blood counts |
| `registry_iteration13.csv`, `confirm_summary.csv` | Fig. 7 and the first relation test |
| `registry_iteration14.csv` | Fig. 8, Table 11 (`fit_seconds`) and the robustness study, one row per evaluation |
| `per_dataset.csv` | Table 9: confirmation on four NHANES datasets |
| `pooled.csv` | Paired C2ST differences pooled over datasets |
| `diagnostics_by_panel.csv` | Table 10: relation diagnostics by panel |

The leave-one-survey-cycle-out ranges and per-dataset paired differences are in `reports/iteration14/reanalysis/`. The figures are drawn from these files by `scripts/figures_paper.py`.
