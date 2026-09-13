# Iteration 9: wrong-rule sensitivity of supplied clinical checks

This stage tests how much the reported compliance conclusions depend on the supplied ILPD rule registry. It is an evaluation-only sensitivity analysis: no generator is fit, no synthetic table is changed, and the outer test remains closed.

## Frozen variants

For every one of the 225 iteration-7 comparison tables (current full, dense/backbone controls, CTGAN/TVAE presets and statistical references), evaluate the saved table under five fit-derived rule sets:

1. `correct`: the existing rule registry.
2. `omit_protein_identity`: remove the exploratory A/G identity relation while retaining all other rules.
3. `reverse_bilirubin_order`: replace `direct_bilirubin <= total_bilirubin + 0.1` by the intentionally reversed relation `total_bilirubin <= direct_bilirubin + 0.1`.
4. `reverse_albumin_order`: replace `albumin <= total_protein` by the intentionally reversed `total_protein <= albumin`.
5. `reverse_both`: apply both reversed relations.

The reversed variants are stress tests of metric dependence, not plausible clinical models. Their rule names and status are explicit. Fit-derived tolerances are recalculated from each record's fit partition; observed and unevaluable rows remain separate through the existing lower/upper bounds. Evaluate fit, validation and synthetic populations for every variant. The supplied rules are only algebraic/schema checks and are not clinical validity labels.

## Retention and interpretation

Reuse the exact table, manifest, split and sample hashes from iteration 7. Confirm row order, labels and zero outer-test overlap. Recalculate the original correct-variant metric from each table before inspecting reversed variants. Report arm rankings and absolute changes by partition descriptively; there are no independent cohorts, significance tests or clinical noninferiority margins. A sensitivity change in a compliance metric does not demonstrate that either rule set is clinically correct.

The analysis is intended to answer whether a claimed compliance advantage persists under reasonable rule omission and whether obviously wrong relations can produce misleading scores. It does not replace clinician review, assay/unit validation or an external cohort. No rule variant is used to select or export a model.
