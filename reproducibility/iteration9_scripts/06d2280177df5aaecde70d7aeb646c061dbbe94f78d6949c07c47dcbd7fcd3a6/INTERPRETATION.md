# Iteration 9: compliance depends strongly on the rule registry

This stage re-evaluated the 225 saved iteration-7 comparison tables under five frozen rule registries. It performed **1,125 rule analyses, zero fits and zero sampling runs**. No table or model output was changed, and the reserved test remained closed.

The correct registry contains nonnegativity rules, the albumin/total-protein inequality, a fit-calibrated A/G identity residual, and a soft direct/total bilirubin inequality. The sensitivity variants deliberately omit the exploratory protein identity, reverse the bilirubin inequality, reverse the albumin inequality, or reverse both. Reversed variants are intentionally wrong stress tests, not clinical alternatives.

For the current full_prior arm, the synthetic any-rule violation mean is 5.50%, 4.97% and 5.96% under the correct registry. Omitting protein identity lowers it to 1.25%, 2.23% and 1.99%. The same omission lowers validation any-rule violations from 9.40%, 5.98% and 7.76% to 0.85%, 0% and 0.86%. This means the supplied A/G identity rule drives most of the apparent compliance penalty, and the real validation partitions themselves violate the rule at materially higher rates than the synthetic full_prior tables. The rule is therefore not a defensible clinical validity label without domain review; it may be an assay/rounding or schema mismatch.

The deliberately reversed albumin inequality produces 100% any-rule violations for current full and repaired-copula tables in every partition. Reversing bilirubin produces approximately 96–98% current-full and 91–93% repaired-copula violations. These controls show that a high or low compliance score can be created by the rule definition itself. Reversing both reaches 100% throughout. The correct-variant scores reproduce the original supplied-rule outcomes, while all lower/upper bounds retain unevaluable rows separately.

The repaired copula remains zero-violation under the correct registry, but its apparent advantage is not enough to establish clinical validity: it explicitly changes approximately 99% of rows and follows rules whose clinical meaning remains unreviewed. Rule omission and reversals do not identify the correct medical relation. They establish a publication requirement: a clinician/laboratory review must verify units, assay rounding, measurement timing, applicability and tolerances before rule-guided results are described as medically consistent.

All source table hashes, labels, fit/validation split indices, correct and sensitivity metrics and outer-test exclusions pass the artifact audit. Use `rule_sensitivity_metrics.csv` and `rule_sensitivity_bounds.csv` for every arm, population and variant; do not report only the favorable correct-rule column.

This result closes the current engineering sensitivity gap. The next work requiring researcher/domain input is selection of a reviewed external cohort and a frozen confirmatory protocol. No new rule-guided model should be promoted from this analysis.

[Results](RESULTS.md), [artifact audit](ARTIFACT_AUDIT.json), [protocol](../../docs/ITERATION_9_PROTOCOL.md), [runbook](../../docs/ITERATION_9_RUNBOOK.md).
