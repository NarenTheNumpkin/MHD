# Publication readiness after iteration 6

The project has a working, trained synthetic medical-table generator and reproducible development evidence. **It is not ready for a superiority or clinical-validity paper.** Successful code checks do not substitute for the scientific requirements below. The present evidence supports studying the tradeoffs of supplied constraints, missingness and distribution shape.

| Requirement | Evidence now | Work still needed |
|---|---|---|
| Executable, reproducible method | Implemented conditional generator; retained weights, data/source hashes, split indices, samples and downstream scores; 31 tests; latest 324-record audit passed | Package the eventual frozen method and minimal end-to-end reproduction for reviewers |
| Clear scientific contribution | Matched simulator structure improves supplied parity; clinical compliance/fidelity/utility tradeoffs measured; marginal missingness correction tested | Establish a specific contribution beyond combining existing components; a consistent method advantage is not established |
| Successful strong baselines | CTGAN/TVAE and simpler pilots exist; pinned official-source TabDDPM adapter and diagnostics exist | Reproduce a stable external diffusion baseline successfully and give strong baselines matched data, tuning opportunity, budgets and sampling/evaluator seeds; current unstable TabDDPM runs cannot support a win claim |
| Clinical relevance and rule quality | Public-data schemas audited; rule sources/status, fit-only tolerances, coverage and unknown-rule bounds saved | Domain review of rule meaning, units, tolerances and applicability; explicit incorrect-rule sensitivity; no invented clinical approvals |
| Generalization and evaluation integrity | Original reserved fold-0 records excluded throughout; duplicate predictor profiles grouped; sampling/training seeds separated | Additional suitable clinical cohorts and tasks; freeze final choices before opening reserved tests; design inference around independent data rather than overlapping development partitions |
| Broad distribution fidelity | Numeric W, associations, missingness, SD/IQR ratios, quantiles and tail frequencies measured | Resolve central-versus-tail mismatch without worsening supplied relations, missingness or utility; low SD alone is an inadequate diagnosis |
| Privacy evidence | Exact-copy and limited nearest-distance diagnostics retained | Stronger membership/attribute/linkability analyses with realistic threat models; no differential privacy or privacy guarantee is implemented |
| Researcher and submission information | Working manuscript reports negative findings and preserves prior drafts | Authors/contributions, funding/conflicts, applicable institutional ethics determination, dataset terms/acknowledgments and venue-specific requirements from the researchers |

## Priority for the next experiments

1. **Establish credible benchmark references before further broad tuning.** Match the current fit/validation partitions, label conditions and 5x evaluation budget for strong reference generators, diagnose unsuccessful external diffusion sampling, and retain unsuccessful attempts. A failed external reproduction is not evidence that MedHyperDiff is better.
2. **Test a fit-only numerical representation or denoising change aimed at distribution shape.** Keep the observation-frequency initialization, use the existing controls, and verify physical-unit constraint calculations under any new transform. The observed combination of low SD and excessive IQR means that adding uniform noise is not an adequate correction.
3. **Close the clinical and generalization gaps.** Choose additional cohorts with meaningful reviewed relationships, quantify wrong-rule sensitivity, freeze a defensible confirmatory protocol and improve privacy evaluation before making submission-level claims.

These are research priorities, not completed experiments or a claim that any proposed change will work. Do not keep adjusting thresholds to make a favored model pass. The current engineering gates are descriptive development tolerances, not clinical noninferiority margins or formal hypothesis tests. Repeated development comparisons must be disclosed.

## Shape of the eventual paper

The main paper should state one precise contribution and show the final matched comparisons, central-versus-tail fidelity, utility and rule coverage together. The simulator mechanism test can isolate supplied structure. Move exhaustive development grids, failed settings, implementation audits and chronological experiment counts to appendices or a reproducibility supplement. Preserve them rather than discarding negative results. The current manuscript is a working record; it still requires consolidation once the final method and confirmatory experiment are fixed.

[Current manuscript](../paper/manuscript.md), [research review](RESEARCH_REVIEW.md), [external baseline status](EXTERNAL_BASELINES.md), [latest measured findings](../reports/iteration6/INTERPRETATION.md), [reproducibility record](../reproducibility/README.md).
